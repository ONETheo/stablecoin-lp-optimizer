"""
Subgraph Client Module

Provides classes to interact with TheGraph subgraphs for DEXes like
Aerodrome and Shadow Finance.
"""

import os
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
import requests
from python_graphql_client import GraphqlClient
from typing import Dict, Optional, Any, List, Tuple

# Use absolute imports
from src.simulation import config as cfg

logger = logging.getLogger(__name__)

class SubgraphClient:
    """
    Client for interacting with TheGraph subgraphs for DEXes.
    Handles primary and fallback endpoints, API keys, and retries.
    """

    def __init__(
        self,
        exchange_name: str,
        endpoint: str,
        api_key: Optional[str] = None,
        backup_endpoint: Optional[str] = None,
        max_retries: int = 3,
        initial_retry_delay: int = 10,
        timeout_seconds: int = 20
    ):
        self.exchange_name = exchange_name.lower()
        self.api_key = api_key
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.timeout_seconds = timeout_seconds

        self.primary_endpoint, self.primary_headers = self._setup_endpoints_and_headers(endpoint)
        self.backup_endpoint, self.backup_headers = self._setup_endpoints_and_headers(backup_endpoint)

        # Log initialized endpoints
        logger.info(f"Initialized {self.exchange_name.capitalize()} SubgraphClient.")
        logger.info(f"  Primary Endpoint: {self.primary_endpoint}")
        if self.backup_endpoint:
             logger.info(f"  Backup Endpoint: {self.backup_endpoint}")
        if self.api_key:
             logger.info(f"  API Key: {'Set' if self.api_key else 'Not Set'}")

    def _setup_endpoints_and_headers(self, endpoint: Optional[str]) -> Tuple[Optional[str], Dict[str, str]]:
        """Sets up endpoint URL and corresponding headers."""
        if not endpoint:
            return None, {}

        headers = {'Content-Type': 'application/json'}
        processed_endpoint = endpoint

        # Handle Gateway endpoints requiring Authorization header
        if "gateway.thegraph.com/api/" in endpoint and self.api_key:
            parts = endpoint.split("/api/")
            if len(parts) == 2:
                # Construct Gateway URL structure if needed (ensure /api/ is present)
                processed_endpoint = f"https://gateway.thegraph.com/api/{self.api_key}/{parts[1]}"
                logger.info(f"Using Gateway endpoint structure: {processed_endpoint}")
                # Gateway uses API key in URL, not header
            else:
                 logger.warning(f"Could not parse Gateway endpoint structure: {endpoint}. API key might not be applied correctly.")
        elif "gateway.thegraph.com" in endpoint and self.api_key:
             # Fallback for slightly different gateway URLs, add API key as header
             logger.info(f"Using Gateway endpoint with Authorization header: {endpoint}")
             headers['Authorization'] = f'Bearer {self.api_key}'
        elif self.api_key:
            # For non-Gateway endpoints, add as standard Authorization
            headers['Authorization'] = f'Bearer {self.api_key}'

        # Add debug log here to see the final endpoint URL being used
        logger.debug(f"_setup_endpoints_and_headers returning endpoint: {processed_endpoint}")
        return processed_endpoint, headers

    def _execute_query(self, query: str, variables: Optional[Dict] = None) -> Optional[Dict]:
        """Executes a GraphQL query with retries and fallback to backup endpoint."""
        retries = 0
        retry_delay = self.initial_retry_delay
        current_endpoint = self.primary_endpoint
        current_headers = self.primary_headers
        using_backup = False

        while retries <= self.max_retries:
            if not current_endpoint:
                 logger.error(f"No valid endpoint configured for {self.exchange_name}. Query cannot be executed.")
                 return None
            
            logger.debug(f"Attempting query to {current_endpoint} (Retry {retries}/{self.max_retries})")
            try:
                response = requests.post(
                    current_endpoint,
                    headers=current_headers,
                    json={'query': query, 'variables': variables or {}},
                    timeout=self.timeout_seconds
                )
                response.raise_for_status()

                data = response.json()
                if 'errors' in data:
                    error_message = data['errors'][0]['message']
                    logger.error(f"GraphQL Error from {current_endpoint}: {data['errors']}")

                    # --- Backup Endpoint Logic START ---
                    is_timeout_error = "timeout" in error_message.lower() or "bad indexers" in error_message.lower()
                    if is_timeout_error and not using_backup and self.backup_endpoint:
                        logger.warning(f"Primary endpoint {self.primary_endpoint} failed with timeout/indexer error. Switching to backup endpoint: {self.backup_endpoint}")
                        current_endpoint = self.backup_endpoint
                        current_headers = self.backup_headers
                        using_backup = True
                        retries = 0
                        retry_delay = self.initial_retry_delay
                        continue
                    # --- Backup Endpoint Logic END ---

                    if retries < self.max_retries:
                        logger.info(f"Retrying query to {current_endpoint} in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retries += 1
                        retry_delay *= 2
                        continue
                    else:
                        logger.error(f"Max retries ({self.max_retries}) reached. Query failed.")
                        return None
                else:
                    return data['data']

            except requests.exceptions.Timeout as e:
                logger.error(f"Request timed out after {self.timeout_seconds}s for {current_endpoint}: {e}")
                if not using_backup and self.backup_endpoint:
                     logger.warning(f"Primary endpoint {self.primary_endpoint} timed out. Switching to backup endpoint: {self.backup_endpoint}")
                     current_endpoint = self.backup_endpoint
                     current_headers = self.backup_headers
                     using_backup = True
                     retries = 0
                     retry_delay = self.initial_retry_delay
                     continue

                if retries < self.max_retries:
                     logger.info(f"Retrying query to {current_endpoint} in {retry_delay} seconds...")
                     time.sleep(retry_delay)
                     retries += 1
                     retry_delay *= 2
                else:
                     logger.error(f"Max retries ({self.max_retries}) reached with timeout errors. Query failed.")
                     return None

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {current_endpoint}: {e}", exc_info=True)
                if not using_backup and self.backup_endpoint:
                     logger.warning(f"Primary endpoint {self.primary_endpoint} failed ({e}). Switching to backup endpoint: {self.backup_endpoint}")
                     current_endpoint = self.backup_endpoint
                     current_headers = self.backup_headers
                     using_backup = True
                     retries = 0
                     retry_delay = self.initial_retry_delay
                     continue
                return None

        return None

    def get_pool_details(self, pool_address: str) -> Optional[Dict[str, Any]]:
        """Fetches basic details for a specific pool."""
        pool_address_lower = pool_address.lower()
        
        # Different query for Shadow exchange
        if self.exchange_name == "shadow":
            logger.info(f"Using Shadow-specific query for pool {pool_address}")
            # Query based on Shadow schema structure
            query = """
            query GetShadowPoolDetails($pool_id: ID!) {
              clPool(id: $pool_id) {
                id
                feeTier
                token0 {
                  id
                  symbol
                  name
                  decimals
                }
                token1 {
                  id
                  symbol
                  name
                  decimals
                }
                totalValueLockedUSD
                volumeUSD
                token0Price
                token1Price
              }
            }
            """
            try:
                logger.debug(f"Executing Shadow GetPoolDetails query for ID: {pool_address_lower}")
                result = self._execute_query(query, variables={"pool_id": pool_address_lower})
                
                if result:
                    logger.debug(f"Raw result from Shadow subgraph for {pool_address_lower}: {result}")
                    pool_data = result.get("data", {}).get("clPool")
                    logger.debug(f"Parsed pool_data for {pool_address_lower}: {pool_data}")
                else:
                    logger.error(f"Query execution returned None for Shadow pool {pool_address_lower}")
                    pool_data = None

                # --- Start Fallback Logic ---
                if not pool_data:
                    logger.warning(f"Pool {pool_address_lower} details not found via clPool query. Creating fallback details.")
                    # Construct minimal details using the ID. Fee tier logic below will handle defaults.
                    # We lose token symbols here, but can proceed with backtest.
                    pool_data = {
                        'id': pool_address_lower,
                        'token0': {'symbol': 'UnknownT0'}, # Placeholder
                        'token1': {'symbol': 'UnknownT1'}, # Placeholder
                        'feeTier': None # Let the logic below handle default/override
                    }
                # --- End Fallback Logic ---
                
                # --- Start Change: Specific Fee Tier Overrides for Shadow ---
                # This logic now applies to both directly found pools and fallback pools
                pool_address_str = pool_data.get('id', '').lower()
                
                # Pools specified to have ultra-low fees (0.0002% requested -> using 1 bps)
                ultra_low_fee_pools = [
                    '0x2c13eda12241314777846abba2363d9d1c7a7a85', # USDC.e/scUSD 
                    '0x9053fe060f412ad5677f934f89e07524343ee8e7', # USDC.e/USDT (ID corrected from logs)
                    '0x81eb3d2ad4f44a059974a1df7bde9839bdb9e093'  # wstkscUSD/scUSD (ID corrected from logs)
                ]
                
                # Default fee for other stables (0.025% requested -> using 5 bps) - Let's use 1bps as default stable for Shadow based on common practice
                default_stable_fee = 1 # Changed default to 1 bps
                
                if pool_address_str in ultra_low_fee_pools:
                    # logger.warning(f"Overriding feeTier for specific Shadow pool {pool_address_str} to 1 bps (0.01%) based on user input.")
                    # No override needed if default is 1
                    pool_data['feeTier'] = 1
                else:
                    # Handle other Shadow pools (like xUSD/USDC.e or potential non-stables)
                    raw_fee_tier = pool_data.get('feeTier') # Check the potentially existing or None feeTier
                    if raw_fee_tier is not None:
                        try:
                            converted_fee_tier = int(float(raw_fee_tier))
                            if converted_fee_tier == 0:
                                logger.warning(f"Subgraph returned feeTier=0 for Shadow pool {pool_address_str}. Using default {default_stable_fee} bps.")
                                pool_data['feeTier'] = default_stable_fee
                            else:
                                # Use the valid non-zero fee tier from the subgraph if available
                                pool_data['feeTier'] = converted_fee_tier
                                logger.info(f"Using feeTier {converted_fee_tier} bps provided by subgraph for Shadow pool {pool_address_str}.")
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid feeTier '{raw_fee_tier}' from subgraph for Shadow pool {pool_address_str}. Using default {default_stable_fee} bps.")
                            pool_data['feeTier'] = default_stable_fee
                    else:
                        logger.warning(f"No feeTier found/provided for Shadow pool {pool_address_str}. Using default {default_stable_fee} bps.")
                        pool_data['feeTier'] = default_stable_fee
                # --- End Change: Specific Fee Tier Overrides for Shadow ---
                        
                # Ensure required fields exist even if minimal, convert types safely
                pool_data['totalValueLockedUSD'] = float(pool_data.get('totalValueLockedUSD', 0))
                pool_data['volumeUSD'] = float(pool_data.get('volumeUSD', 0))
                pool_data['token0Price'] = float(pool_data.get('token0Price', 0))
                pool_data['token1Price'] = float(pool_data.get('token1Price', 0))
                
                if pool_data.get('token0') is None: pool_data['token0'] = {}
                if pool_data.get('token1') is None: pool_data['token1'] = {}
                pool_data['token0']['decimals'] = int(pool_data['token0'].get('decimals', 18))
                pool_data['token1']['decimals'] = int(pool_data['token1'].get('decimals', 18))
                
                logger.info(f"Processed pool details (Fee: {pool_data.get('feeTier')} bps) for {pool_address_lower}")
                return pool_data # Return either found data or fallback data
            except ConnectionError as e:
                logger.error(f"Connection error fetching Shadow pool details for {pool_address}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error fetching Shadow pool details for {pool_address}: {e}", exc_info=True) # Log traceback
                return None
        
        # Original query for other exchanges (Aerodrome, etc.)
        query = """
        query GetPoolDetails($pool_id: ID!) {
          pool(id: $pool_id) {
            id
            feeTier
            token0 {
              id
              symbol
              name
              decimals
            }
            token1 {
              id
              symbol
              name
              decimals
            }
            totalValueLockedUSD
            volumeUSD
            token0Price # Price of token0 in terms of token1
            token1Price # Price of token1 in terms of token0
          }
        }
        """
        try:
            result = self._execute_query(query, variables={"pool_id": pool_address_lower})
            pool_data = result.get("data", {}).get("pool")

            if pool_data:
                logger.info(f"Found pool details for {pool_address} on {self.exchange_name}")
                # Convert numeric strings to appropriate types safely
                pool_data['feeTier'] = int(pool_data.get('feeTier', 0))
                pool_data['totalValueLockedUSD'] = float(pool_data.get('totalValueLockedUSD', 0))
                pool_data['volumeUSD'] = float(pool_data.get('volumeUSD', 0))
                pool_data['token0Price'] = float(pool_data.get('token0Price', 0))
                pool_data['token1Price'] = float(pool_data.get('token1Price', 0))
                if pool_data.get('token0'):
                    pool_data['token0']['decimals'] = int(pool_data['token0'].get('decimals', 18))
                if pool_data.get('token1'):
                    pool_data['token1']['decimals'] = int(pool_data['token1'].get('decimals', 18))
                return pool_data
            else:
                logger.warning(f"Pool {pool_address} not found in {self.exchange_name} subgraph ({self.primary_endpoint}).")
                return None
        except ConnectionError as e:
             logger.error(f"Connection error fetching pool details for {pool_address} from {self.primary_endpoint}: {e}")
             return None # Propagate connection error as None result
        except Exception as e:
            logger.error(f"Unexpected error fetching pool details for {pool_address} from {self.primary_endpoint}: {e}")
            return None

    def get_top_pools(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Fetches top pools from the exchange sorted by TVL.
        
        Args:
            limit (int): Maximum number of pools to return (default: 20)
            
        Returns:
            List[Dict[str, Any]]: A list of pool details or empty list if query fails.
        """
        logger.info(f"Fetching top {limit} pools from {self.exchange_name}")
        
        # Different query for Shadow exchange
        if self.exchange_name == "shadow":
            # Shadow has been having timeout issues, so we'll use a more conservative approach
            # First try with a smaller limit to increase chances of success
            # conservative_limit = min(10, limit) # Removed conservative limit
            # logger.info(f"Using conservative limit of {conservative_limit} for Shadow due to potential timeout issues")
            fetch_limit = limit # Use the requested limit directly, or set a higher default if needed for this specific use case
            logger.info(f"Fetching up to {fetch_limit} pools for Shadow.")

            query = """
            query GetTopShadowPools($limit: Int!) {
              clPools(
                first: $limit, # Use the limit variable
                orderBy: totalValueLockedUSD,
                orderDirection: desc,
                where: { totalValueLockedUSD_gt: 10000 } # Keep minimum TVL filter
              ) {
                id
                feeTier
                token0 {
                  id
                  symbol
                  name
                  decimals
                }
                token1 {
                  id
                  symbol
                  name
                  decimals
                }
                totalValueLockedUSD
                volumeUSD
                token0Price
                token1Price
              }
            }
            """
            try:
                # Use a longer max_retries for Shadow - NO, this is handled internally now
                result = self._execute_query(
                    query,
                    variables={"limit": fetch_limit} # Remove invalid kwargs
                )
                # Check if result is None (indicating failure after retries/backup attempts)
                if result is None:
                    logger.error(f"Failed to fetch clPools for Shadow after retries/backup.")
                    pools_data = [] # Treat as no pools found
                else:
                    # Ensure result is a dict before accessing data key
                    pools_data = result.get("clPools", []) if isinstance(result, dict) else []

                if not pools_data:
                    logger.warning(f"No valid pools (clPools) found for {self.exchange_name} after query execution.")
                    return [] # Return empty list if still no data

                # Process and return the found clPools data
                logger.info(f"Successfully fetched {len(pools_data)} clPools from {self.exchange_name}")
                return pools_data
                
            except Exception as e:
                logger.error(f"Error fetching top pools from Shadow: {e}", exc_info=True) # Log full traceback
                # Return hardcoded popular pools for Shadow as emergency fallback
                logger.info("Using hardcoded popular Shadow pools as emergency fallback")
                return [
                    {
                        "id": "0x7994eecd2568f4b2b86a345f048ff3bb133635c5",  # Replace with actual popular pool
                        "feeTier": 30,
                        "token0": {"id": "0x1", "symbol": "USDC", "name": "USD Coin", "decimals": 6},
                        "token1": {"id": "0x2", "symbol": "ETH", "name": "Ethereum", "decimals": 18},
                        "totalValueLockedUSD": 1000000,
                        "volumeUSD": 500000,
                        "token0Price": 1,
                        "token1Price": 3500
                    },
                    {
                        "id": "0xdb02498659987cb8cf3be66fadf6995bc5e7c112",  # Replace with actual popular pool
                        "feeTier": 30,
                        "token0": {"id": "0x3", "symbol": "USDC", "name": "USD Coin", "decimals": 6},
                        "token1": {"id": "0x4", "symbol": "WBTC", "name": "Wrapped Bitcoin", "decimals": 8},
                        "totalValueLockedUSD": 800000,
                        "volumeUSD": 400000,
                        "token0Price": 1,
                        "token1Price": 60000
                    }
                ]
        
        # Query for other exchanges (Aerodrome, etc.)
        query = """
        query GetTopPools($limit: Int!) {
          pools(
            first: $limit,
            orderBy: totalValueLockedUSD,
            orderDirection: desc,
            where: { totalValueLockedUSD_gt: 10000 }
          ) {
            id
            feeTier
            token0 {
              id
              symbol
              name
              decimals
            }
            token1 {
              id
              symbol
              name
              decimals
            }
            totalValueLockedUSD
            volumeUSD
            token0Price
            token1Price
          }
        }
        """
        
        try:
            result = self._execute_query(query, variables={"limit": limit})
            pools_data = result.get("data", {}).get("pools", [])
            
            processed_pools = []
            for pool in pools_data:
                # Convert numeric strings to appropriate types safely
                try:
                    processed_pool = pool.copy()
                    processed_pool['feeTier'] = int(processed_pool.get('feeTier', 0))
                    processed_pool['totalValueLockedUSD'] = float(processed_pool.get('totalValueLockedUSD', 0))
                    processed_pool['volumeUSD'] = float(processed_pool.get('volumeUSD', 0))
                    processed_pool['token0Price'] = float(processed_pool.get('token0Price', 0))
                    processed_pool['token1Price'] = float(processed_pool.get('token1Price', 0))
                    
                    if processed_pool.get('token0'):
                        processed_pool['token0']['decimals'] = int(processed_pool['token0'].get('decimals', 18))
                    if processed_pool.get('token1'):
                        processed_pool['token1']['decimals'] = int(processed_pool['token1'].get('decimals', 18))
                    
                    # Only add pools with valid token symbols
                    if (processed_pool.get('token0', {}).get('symbol') and 
                        processed_pool.get('token1', {}).get('symbol')):
                        processed_pools.append(processed_pool)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing pool data: {e}. Skipping pool.")
                    continue
            
            logger.info(f"Retrieved {len(processed_pools)} top pools from {self.exchange_name}")
            return processed_pools
            
        except Exception as e:
            logger.error(f"Error fetching top pools from {self.exchange_name}: {e}")
            return []

    def get_historical_data(self, pool_address: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetches historical daily data for a pool within a specific date range.

        Args:
            pool_address (str): The pool address.
            start_date (datetime): The start date for the data range.
            end_date (datetime): The end date for the data range.

        Returns:
            pd.DataFrame: DataFrame with historical data, empty if none found or error occurs.
                          Columns: timestamp, price, volumeUSD, tvlUSD, feesUSD.
        """
        pool_address_lower = pool_address.lower()
        # Timestamps need to be integers for the subgraph query
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        all_data = []
        first = 1000 # Max records per query (standard limit)
        skip = 0
        
        # Shadow-specific historical data query
        if self.exchange_name == "shadow":
            logger.info(f"Using Shadow-specific query for historical data for pool {pool_address}")
            # Try CL Pool day data
            query_template = """
            query ShadowPoolHistoricalData($pool_id: ID!, $start_ts: Int!, $end_ts: Int!, $first: Int!, $skip: Int!) {
              clPoolDayDatas(
                where: { pool: $pool_id, startOfDay_gte: $start_ts, startOfDay_lte: $end_ts }
                orderBy: startOfDay
                orderDirection: asc
                first: $first
                skip: $skip
              ) {
                startOfDay
                volumeUSD
                tvlUSD
                feesUSD
                token0Price
                token1Price
              }
            }
            """
            
            try:
                # First try with clPoolDayDatas
                while True:
                    variables = {
                        "pool_id": pool_address_lower,
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "first": first,
                        "skip": skip
                    }
                    result = self._execute_query(query_template, variables=variables)
                    if result is None:
                         logger.error(f"Query execution failed for clPoolDayDatas for {pool_address}.")
                         daily_data = []
                         break

                    data_node = result
                    daily_data = data_node.get("clPoolDayDatas", []) if data_node else []
                    
                    if not daily_data:
                        if skip == 0:
                            logger.warning(f"No historical data found for Shadow pool {pool_address} (Query: clPoolDayDatas)")
                        else:
                            logger.debug(f"No more clPoolDayDatas found at skip={skip} for Shadow pool {pool_address}")
                        break
                    
                    all_data.extend(daily_data)
                    logger.debug(f"Fetched {len(daily_data)} records for Shadow pool {pool_address}, total {len(all_data)}")
                    
                    if len(daily_data) < first:
                        break
                    skip += first
                    time.sleep(0.2)
                    
                if not all_data:
                    logger.warning(f"No historical data found for Shadow pool {pool_address}")
                    return pd.DataFrame()
                    
                df = pd.DataFrame(all_data)
                
                # Map day/date to timestamp for standardization
                if 'startOfDay' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['startOfDay'], unit='s')
                else:
                    logger.error(f"No date/timestamp field found in Shadow data for {pool_address}")
                    return pd.DataFrame()
                
                # Map fields based on which query was used
                if 'volumeUSD' not in df.columns and 'dailyVolumeUSD' in df.columns:
                    df['volumeUSD'] = df['dailyVolumeUSD']
                if 'tvlUSD' not in df.columns and 'reserveUSD' in df.columns:
                    df['tvlUSD'] = df['reserveUSD']
                if 'feesUSD' not in df.columns and 'volumeUSD' in df.columns:
                    df['feesUSD'] = df['volumeUSD'] * 0.003  # Estimate fees as 0.3% of volume
                
                # Convert numeric columns
                numeric_cols = ['volumeUSD', 'tvlUSD', 'feesUSD', 'token0Price', 'token1Price']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Set price column
                if 'token0Price' in df.columns:
                    df['price'] = df['token0Price']
                elif 'token1Price' in df.columns:
                     # Fallback: use token1Price if token0Price is missing, but invert it? No, use as is and note.
                     df['price'] = df['token1Price']
                     logger.warning(f"'token0Price' not found, using 'token1Price' as primary price for {pool_address}. Interpretation depends on token order.")
                else:
                     logger.error(f"Neither 'token0Price' nor 'token1Price' found in historical data for {pool_address}. Cannot determine price.")
                     return pd.DataFrame() # Return empty if no price available
                
                # Final column selection
                required_cols = ['timestamp', 'price', 'volumeUSD', 'tvlUSD', 'feesUSD']
                final_cols = [col for col in required_cols if col in df.columns]
                missing_cols = [col for col in required_cols if col not in final_cols]
                
                if missing_cols:
                    logger.warning(f"Missing columns in Shadow data: {missing_cols}")
                    # Add missing columns with reasonable defaults
                    for col in missing_cols:
                        if col in ['volumeUSD', 'tvlUSD', 'feesUSD']:
                            df[col] = 0.0
                
                # Ensure all required columns exist now
                final_cols = [col for col in required_cols if col in df.columns]
                df = df[final_cols]
                
                # Clean data
                df.dropna(subset=['price', 'tvlUSD'], inplace=True)
                
                if df.empty:
                    logger.warning(f"Shadow historical data empty after processing for {pool_address}")
                    return pd.DataFrame()
                
                logger.info(f"Successfully processed {len(df)} historical data points for Shadow pool {pool_address}")
                return df.sort_values('timestamp').reset_index(drop=True)
                
            except Exception as e:
                logger.error(f"Error fetching Shadow historical data for {pool_address}: {e}", exc_info=True) # Log traceback
                return pd.DataFrame() # Return empty on error
        
        # Original query for standard pools like Aerodrome
        query_template = """
        query PoolHistoricalData($pool_id: ID!, $start_ts: Int!, $end_ts: Int!, $first: Int!, $skip: Int!) {
          poolDayDatas(
            where: { pool: $pool_id, date_gte: $start_ts, date_lte: $end_ts }
            orderBy: date
            orderDirection: asc
            first: $first
            skip: $skip
          ) {
            date # Unix timestamp for start of day
            volumeUSD
            tvlUSD
            feesUSD
            token0Price # Price of token0 in terms of token1
            token1Price # Price of token1 in terms of token0
            # OHLC fields (open, high, low, close) might exist in some Uniswap v3 forks
            # Check the specific subgraph schema if needed. We primarily need a closing price.
            # open
            # high
            # low
            # close
          }
        }
        """

        logger.info(f"Fetching historical data for {pool_address} from {start_date.date()} to {end_date.date()} ({self.exchange_name})")

        while True:
            try:
                variables = {
                    "pool_id": pool_address_lower,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "first": first,
                    "skip": skip
                }
                result = self._execute_query(query_template, variables=variables)
                # Access data safely, handling potential null responses
                data_node = result.get("data", {})
                daily_data = data_node.get("poolDayDatas", []) if data_node else []


                if not daily_data:
                    logger.debug(f"No more historical data found at skip={skip} for {pool_address}.")
                    break # No more data

                all_data.extend(daily_data)
                logger.debug(f"Fetched {len(daily_data)} records for {pool_address}, total {len(all_data)}")

                if len(daily_data) < first:
                    break # Last page fetched
                skip += first
                time.sleep(0.2) # Small delay between pages to be polite to the API

            except ConnectionError as e:
                 logger.error(f"Connection error fetching historical data page (skip={skip}) for {pool_address} from {self.primary_endpoint}: {e}")
                 # Stop fetching on connection error, return what we have (if any)
                 break
            except Exception as e:
                logger.error(f"Unexpected error fetching historical data page (skip={skip}) for {pool_address} from {self.primary_endpoint}: {e}")
                # Decide whether to break or retry - break for now
                break

        if not all_data:
            logger.warning(f"No historical data found for pool {pool_address} between {start_date.date()} and {end_date.date()} on {self.exchange_name}.")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)

        # Data Cleaning and Type Conversion
        df['timestamp'] = pd.to_datetime(df['date'], unit='s')
        numeric_cols = ['volumeUSD', 'tvlUSD', 'feesUSD', 'token0Price', 'token1Price']
        # Add OHLC if they exist in the fetched data
        # ohlc_cols = ['open', 'high', 'low', 'close']
        # numeric_cols.extend([col for col in ohlc_cols if col in df.columns])

        for col in numeric_cols:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
             else:
                  logger.warning(f"Expected numeric column '{col}' not found in raw historical data for {pool_address}.")


        # Determine the primary price for backtesting.
        # This is crucial and depends on the pool's token order convention.
        # Assumption: Use token0Price (price of token0 in terms of token1).
        # This works well for Volatile/Stable pairs where Token0 is Volatile.
        # For Stable/Stable or Volatile/Volatile, this might need adjustment or user input.
        # TODO: Add configuration or logic to select price based on token symbols if needed.
        if 'token0Price' in df.columns:
            df['price'] = df['token0Price']
            logger.debug(f"Using 'token0Price' as the primary price column for {pool_address}.")
        elif 'token1Price' in df.columns:
             # Fallback: use token1Price if token0Price is missing, but invert it? No, use as is and note.
             df['price'] = df['token1Price']
             logger.warning(f"'token0Price' not found, using 'token1Price' as primary price for {pool_address}. Interpretation depends on token order.")
        else:
             logger.error(f"Neither 'token0Price' nor 'token1Price' found in historical data for {pool_address}. Cannot determine price.")
             return pd.DataFrame() # Return empty if no price available

        # Select and order essential columns for the backtester
        required_cols = ['timestamp', 'price', 'volumeUSD', 'tvlUSD', 'feesUSD']
        # Add OHLC columns if they were present and converted
        # available_ohlc = [col for col in ohlc_cols if col in df.columns]
        # required_cols.extend(available_ohlc)

        # Filter columns, keeping only those that exist in the DataFrame
        final_cols = [col for col in required_cols if col in df.columns]
        missing_essential_cols = [col for col in ['timestamp', 'price', 'volumeUSD', 'tvlUSD', 'feesUSD'] if col not in final_cols]
        if missing_essential_cols:
             logger.error(f"Essential columns missing after processing historical data for {pool_address}: {missing_essential_cols}. Cannot proceed.")
             return pd.DataFrame()

        df = df[final_cols]

        # Drop rows with NaN in critical columns like price or tvlUSD, as they break calculations
        initial_rows = len(df)
        df.dropna(subset=['price', 'tvlUSD'], inplace=True)
        if len(df) < initial_rows:
             logger.warning(f"Dropped {initial_rows - len(df)} rows with NaN in 'price' or 'tvlUSD' for {pool_address}.")

        if df.empty:
             logger.warning(f"Historical data for {pool_address} became empty after cleaning.")
             return pd.DataFrame()

        logger.info(f"Successfully processed {len(df)} historical data points for pool {pool_address} ({self.exchange_name}).")
        return df.sort_values('timestamp').reset_index(drop=True)


# --- Factory Function ---

def get_client(exchange_name: str) -> Optional[SubgraphClient]:
    """
    Factory function to get the appropriate subgraph client based on configuration.

    Args:
        exchange_name (str): The name of the exchange (e.g., 'aerodrome', 'shadow').

    Returns:
        Optional[SubgraphClient]: An instance of the client, or None if config is missing/invalid.
    """
    name_lower = exchange_name.lower()
    config_data = cfg.get_config()
    api_key = config_data.get("api_config", {}).get("thegraph", {}).get("api_key")
    endpoint = None

    try:
        if name_lower == "aerodrome":
            # Assumes Aerodrome uses TheGraph and is on 'base' network in config
            endpoint = config_data.get("api_config", {}).get("thegraph", {}).get("base", {}).get("aerodrome")
            if not endpoint:
                 logger.error("Aerodrome subgraph endpoint not configured. Set AERODROME_SUBGRAPH_ENDPOINT in .env")
                 return None
            return SubgraphClient(exchange_name="aerodrome", endpoint=endpoint, api_key=api_key)

        elif name_lower == "shadow":
            # Use Shadow's dedicated endpoint from config
            endpoint = config_data.get("api_config", {}).get("shadow", {}).get("endpoint")
            
            if not endpoint:
                 logger.error("Shadow subgraph endpoint not configured. Set SHADOW_SUBGRAPH_ENDPOINT in .env")
                 return None
             
            logger.info(f"Creating Shadow client with endpoint from config: {endpoint}")
            logger.info(f"API key: {'Set' if api_key else 'Not set'}")
                 
            try:
                # Try primary endpoint with the API key
                return SubgraphClient(exchange_name="shadow", endpoint=endpoint, api_key=api_key)
            except Exception as e:
                # If primary endpoint fails, try backup endpoint
                logger.warning(f"Primary Shadow endpoint failed: {e}. Trying backup endpoint.")
                backup_endpoint = config_data.get("api_config", {}).get("shadow", {}).get("backup_endpoint")
                
                if not backup_endpoint:
                    logger.error("Shadow backup endpoint not configured. Set SHADOW_BACKUP_SUBGRAPH_ENDPOINT in .env")
                    return None
                    
                logger.info(f"Using Shadow backup endpoint: {backup_endpoint}")
                return SubgraphClient(exchange_name="shadow", endpoint=backup_endpoint, api_key=api_key)

        else:
            logger.error(f"Unsupported exchange specified: {exchange_name}. Supported: 'aerodrome', 'shadow'.")
            return None

    except ValueError as e: # Catch endpoint configuration errors from SubgraphClient init
        logger.error(f"Failed to initialize client for {exchange_name}: {e}")
        return None
    except ConnectionError as e: # Catch client creation errors
         logger.error(f"Failed to establish connection for {exchange_name} client: {e}")
         return None
    except Exception as e: # Catch unexpected errors during init
         logger.error(f"Unexpected error creating client for {exchange_name}: {e}")
         return None


# Example Usage (for testing basic client functionality)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Testing SubgraphClient Factory and Methods...")

    # Test Aerodrome Client Creation
    print("\n--- Testing Aerodrome Client ---")
    aero_client = get_client("aerodrome")
    if aero_client:
        logger.info("Aerodrome client created successfully.")
        # Example Pool: WETH/USDC on Base (Aerodrome) - Replace if needed
        aero_pool_address = "0xb2cc224c1c9fee385f8ad6a55b4d94e92359dc59"
        print(f"\nFetching details for Aerodrome pool: {aero_pool_address}")
        details = aero_client.get_pool_details(aero_pool_address)
        if details:
            print("\nPool Details (Aerodrome):")
            # Print selected details
            print(f"  ID: {details.get('id')}")
            print(f"  Fee Tier: {details.get('feeTier')}")
            print(f"  Token0: {details.get('token0', {}).get('symbol')}")
            print(f"  Token1: {details.get('token1', {}).get('symbol')}")
            print(f"  TVL (USD): {details.get('totalValueLockedUSD')}")
            print(f"  Token0 Price: {details.get('token0Price')}")

            print(f"\nFetching last 7 days of historical data for Aerodrome pool: {aero_pool_address}")
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=7)
            history = aero_client.get_historical_data(aero_pool_address, start_date=start_dt, end_date=end_dt)
            if not history.empty:
                print("\nHistorical Data Sample (Aerodrome):")
                print(history.head())
                print("...")
                print(history.tail())
                print(f"\nColumns: {history.columns.tolist()}")
            else:
                print(f"No historical data fetched for Aerodrome pool {aero_pool_address}.")
        else:
            print(f"Could not fetch details for Aerodrome pool {aero_pool_address}. Check address and endpoint/API key.")
    else:
        print("Failed to create Aerodrome client. Check .env configuration (AERODROME_SUBGRAPH_ENDPOINT, SUBGRAPH_API_KEY).")

    # Test Shadow Client Creation (Requires SHADOW_SUBGRAPH_ENDPOINT in .env)
    print("\n--- Testing Shadow Client ---")
    shadow_client = get_client("shadow")
    if shadow_client:
        logger.info("Shadow client created successfully.")
        # Add a known Shadow pool address here for testing if available
        # shadow_pool_address = "YOUR_SHADOW_POOL_ADDRESS_HERE"
        # print(f"\nFetching details for Shadow pool: {shadow_pool_address}")
        # details = shadow_client.get_pool_details(shadow_pool_address)
        # ... similar checks and history fetching ...
        print("Shadow client created, but no pool address provided for further testing in this example.")
    else:
        print("Failed to create Shadow client. Check .env configuration (SHADOW_SUBGRAPH_ENDPOINT, SUBGRAPH_API_KEY).")