"""
Data Loader Module

Handles loading historical price and pool data (TVL, volume, fees)
directly from DEX subgraphs using the SubgraphClient.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Union, Any

# Set up logger
logger = logging.getLogger(__name__)

# Use absolute imports assuming 'src' is the root package or accessible in PYTHONPATH
from src.api.subgraph_client import get_client, SubgraphClient
from src.utils.helpers import parse_date # Keep date parsing
# Removed interpolate_missing_values import - will do interpolation here directly if needed
# from src.simulation import config as cfg # Config not directly needed here anymore

class DataLoader:
    """
    Loads historical pool data directly from subgraphs.
    No caching is implemented.
    """

    def __init__(self):
        """Initialize the data loader."""
        # No configuration needed directly in init anymore
        logger.info("Initialized DataLoader for direct subgraph fetching.")

    def _get_subgraph_client(self, exchange_name: str) -> Optional[SubgraphClient]:
        """Gets the subgraph client for the specified exchange using the factory."""
        client = get_client(exchange_name)
        if not client:
            logger.error(f"Failed to get subgraph client for exchange: {exchange_name}. Check configuration.")
        return client

    def load_historical_data(
        self,
        exchange_name: str,
        pool_address: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
        """
        Load pool details and historical data (price, TVL, volume, fees) for a pool.

        Args:
            exchange_name (str): The exchange identifier (e.g., 'aerodrome', 'shadow').
            pool_address (str): The address of the liquidity pool.
            start_date (datetime): Start date for historical data.
            end_date (datetime): End date for historical data.

        Returns:
            Tuple[Optional[Dict], Optional[pd.DataFrame]]:
                - Pool details dictionary or None if failed.
                - DataFrame with historical data (timestamp, price, volumeUSD, tvlUSD, feesUSD)
                  or None if loading failed or data is insufficient.
                  Data is returned raw from the client, interpolation happens downstream if needed.
        """
        client = self._get_subgraph_client(exchange_name)
        if not client:
            return None, None # Client creation failed

        # --- 1. Fetch Pool Details ---
        logger.info(f"Fetching pool details for {exchange_name}/{pool_address}")
        pool_details = client.get_pool_details(pool_address)
        if not pool_details:
            logger.error(f"Failed to fetch pool details for {pool_address} on {exchange_name}. Backtest may lack fee tier info.")
            # Continue to fetch history, but signal that details are missing
            # The backtest engine might fail later if feeTier is essential
            # Alternatively, return None, None here to fail fast:
            # return None, None
            pass # Allow history fetching attempt

        # --- 2. Fetch Historical Data ---
        logger.info(f"Fetching historical data for {pool_address} from {start_date.date()} to {end_date.date()}")

        historical_df = client.get_historical_data(pool_address, start_date=start_date, end_date=end_date)

        if historical_df is None:
             logger.error(f"Subgraph client returned None for historical data for {pool_address}. Query likely failed.")
             return pool_details, None # Return details if fetched, but signal history failure

        if historical_df.empty:
            logger.warning(f"No historical data found for {pool_address} in the specified date range.")
            # Return details if fetched, but history is empty
            return pool_details, pd.DataFrame() # Return empty DataFrame

        # --- 3. Basic Validation and Processing (Interpolation moved downstream) ---
        required_cols = ['timestamp', 'price', 'volumeUSD', 'tvlUSD', 'feesUSD']
        if not all(col in historical_df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in historical_df.columns]
             logger.error(f"Historical data fetched for {pool_address} is missing essential columns: {missing}. Cannot use for backtesting.")
             return pool_details, None # Data is unusable

        # Ensure data is sorted by timestamp
        historical_df = historical_df.sort_values('timestamp').reset_index(drop=True)

        # Optional: Check for large gaps or very few data points
        if len(historical_df) > 1:
            time_diffs = historical_df['timestamp'].diff().dt.days
            max_gap = time_diffs.max()
            if max_gap > 7: # Example threshold: warn if gap is more than 7 days
                 logger.warning(f"Large gap detected in historical data for {pool_address}. Maximum difference between points: {max_gap} days. Results may be affected.")

        min_required_points = 5 # Example: require at least 5 data points for a meaningful backtest
        if len(historical_df) < min_required_points:
             logger.warning(f"Insufficient historical data points ({len(historical_df)}) found for {pool_address} in the range. Backtest might be unreliable.")
             # Decide whether to return the insufficient data or None
             # return pool_details, None # Option: Fail if too few points
             # Option: Return the few points we have
             pass


        # --- 4. Clean Zero/NaN Prices & Interpolate ---
        logger.debug(f"Preprocessing data for {pool_address}: Handling zero/NaN prices and interpolating.")
        
        # Create expected date range based on actual fetched min/max to avoid excessive NaNs
        # Get the date part of timestamps by converting to date objects
        min_date = historical_df['timestamp'].min().date()
        max_date = historical_df['timestamp'].max().date()
        
        # Use these date objects to create the date range
        full_range_index = pd.date_range(start=min_date, end=max_date, freq='D')

        # Set timestamp as index for resampling/reindexing
        historical_df = historical_df.set_index('timestamp')
        
        # --- Start Change: Zero Price Handling ---
        # Explicitly handle zero prices in the 'price' column before interpolation
        if 'price' in historical_df.columns:
            initial_zeros = (historical_df['price'] == 0).sum()
            if initial_zeros > 0:
                 logger.warning(f"Found {initial_zeros} zero price entries for {pool_address}. Replacing with NaN before interpolation.")
                 historical_df['price'] = historical_df['price'].replace(0, pd.NA)
        # --- End Change: Zero Price Handling ---
        
        # Select only numeric columns suitable for interpolation
        cols_to_interpolate = ['price', 'volumeUSD', 'tvlUSD', 'feesUSD']
        numeric_df = historical_df[[col for col in cols_to_interpolate if col in historical_df.columns]]

        # Resample to daily frequency (if multiple points per day exist, take mean) and reindex to fill gaps
        # Note: Subgraph usually provides daily data ('poolDayDatas'), so resample might not change much unless data is patchy
        numeric_df = numeric_df.resample('D').mean()
        numeric_df = numeric_df.reindex(full_range_index)

        initial_nan_count = numeric_df.isnull().sum().sum()

        # Interpolate missing values (linear method) - now handles NaNs from original data AND replaced zeros
        interpolated_df = numeric_df.interpolate(method='linear', limit_direction='both', limit_area='inside')
        # 'limit_direction=both' fills NaNs at ends too, 'limit_area=inside' only interpolates between valid points

        final_nan_count = interpolated_df.isnull().sum().sum()
        interpolated_count = initial_nan_count - final_nan_count

        if interpolated_count > 0:
             logger.info(f"Interpolated {interpolated_count} missing/zero data points (using linear method) for {pool_address}.")
             # Warn if a large percentage was interpolated
             total_points = len(interpolated_df)
             if total_points > 0 and (interpolated_count / total_points) > 0.2: # Example: Warn if > 20% interpolated
                  logger.warning(f"Significant data interpolation ({interpolated_count}/{total_points} points) occurred for {pool_address}. Backtest accuracy may be affected.")

        # Handle remaining NaNs (e.g., at the very start/end if limit_direction='inside')
        # Option 1: Forward fill then backward fill
        interpolated_df = interpolated_df.ffill().bfill()
        # Option 2: Drop rows with any NaNs left (might shorten the period)
        # interpolated_df.dropna(inplace=True)

        # Reset index to get timestamp column back
        interpolated_df = interpolated_df.reset_index().rename(columns={'index': 'timestamp'})

        # Filter final dataframe to the originally requested start/end dates (inclusive)
        # Convert datetime objects to date objects for comparison
        start_date_as_date = start_date.date()
        end_date_as_date = end_date.date()
        
        # Filter based on the date part of timestamp
        final_df = interpolated_df[
             (interpolated_df['timestamp'].dt.date >= start_date_as_date) &
             (interpolated_df['timestamp'].dt.date <= end_date_as_date)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        logger.info(f"Data loading complete for {pool_address}. Pool details fetched: {'Yes' if pool_details else 'No'}. Historical data points after processing: {len(final_df)}")

        if final_df.empty:
             logger.warning(f"Final historical dataset is empty for {pool_address} after processing and date filtering.")
             return pool_details, pd.DataFrame()


        return pool_details, final_df


# Convenience function to be called from main.py
def load_data(
    exchange_name: str,
    pool_address: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime]
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """
    Loads pool details and historical data using the DataLoader.

    Args:
        exchange_name (str): Network identifier (e.g., 'aerodrome').
        pool_address (str): Pool address.
        start_date (Union[str, datetime]): Start date (string or datetime).
        end_date (Union[str, datetime]): End date (string or datetime).

    Returns:
        Tuple[Optional[Dict], Optional[pd.DataFrame]]: Pool details and historical data.
    """
    # Parse dates if they are strings
    try:
        start_date_dt = parse_date(start_date) if isinstance(start_date, str) else start_date
        end_date_dt = parse_date(end_date) if isinstance(end_date, str) else end_date
    except ValueError as e:
        logger.error(f"Invalid date format provided: {e}")
        return None, None

    # Basic date validation
    if start_date_dt >= end_date_dt:
         logger.error(f"Start date ({start_date_dt.date()}) must be before end date ({end_date_dt.date()}).")
         return None, None

    loader = DataLoader()
    return loader.load_historical_data(exchange_name, pool_address, start_date_dt, end_date_dt)


if __name__ == "__main__":
    # Test the module if run directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Example: Load data for a specific Aerodrome pool
    exchange = "aerodrome"
    # WETH/USDC pool on Base
    pool_addr = "0xb2cc224c1c9fee385f8ad6a55b4d94e92359dc59"
    # Use string dates for parsing test
    start_str = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_str = datetime.now().strftime('%Y-%m-%d')

    print(f"\nTesting data loading for {exchange}/{pool_addr} from {start_str} to {end_str}")
    pool_details_res, historical_data_res = load_data(
        exchange_name=exchange,
        pool_address=pool_addr,
        start_date=start_str,
        end_date=end_str
    )

    print("\n--- Pool Details ---")
    if pool_details_res:
        # Print selected details
        print(f"  ID: {pool_details_res.get('id')}")
        print(f"  Fee Tier: {pool_details_res.get('feeTier')}")
        print(f"  Token0: {pool_details_res.get('token0', {}).get('symbol')}")
        print(f"  Token1: {pool_details_res.get('token1', {}).get('symbol')}")
    else:
        print("No pool details loaded or loading failed.")

    print("\n--- Historical Data ---")
    if historical_data_res is not None and not historical_data_res.empty:
        print(f"Shape: {historical_data_res.shape}")
        print("Head:")
        print(historical_data_res.head())
        print("Tail:")
        print(historical_data_res.tail())
        # Check for NaNs after processing
        print("\nNaN check after processing:")
        print(historical_data_res.isnull().sum())
    elif historical_data_res is not None and historical_data_res.empty:
         print("Historical data loaded, but the DataFrame is empty (no data in range or filtered out).")
    else: # historical_data_res is None
        print("Failed to load historical data.")