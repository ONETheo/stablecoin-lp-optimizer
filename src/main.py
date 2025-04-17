"""
LP Optimizer CLI - Main Entry Point

Command-line interface for backtesting vfat.io LP strategies on DEX pools
using historical data fetched directly from subgraphs.
"""

import logging
import argparse
from datetime import datetime, timedelta, timezone
import os
import sys
import pandas as pd
from types import SimpleNamespace
import concurrent.futures # For parallel backtesting
from typing import List, Dict, Any
from tabulate import tabulate

# Use absolute imports assuming 'src' is the root package or accessible in PYTHONPATH
from src.simulation.backtest_engine import BacktestEngine
from src.simulation.result_processor import ResultProcessor, plot_volatility
from src.data.data_loader import DataLoader
from src.simulation import config as cfg
from src.utils.helpers import ensure_dir_exists, parse_date, format_percentage, format_currency, calculate_volatility
from src.api.subgraph_client import SubgraphClient

# --- Configuration ---
# Load config early to access defaults and API settings
config = cfg.get_config()

# Configure logging
log_level = logging.INFO # Default level
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=log_level, format=log_format)
logger = logging.getLogger(__name__) # Logger for this module

# Silence overly verbose libraries if needed
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("python_graphql_client").setLevel(logging.WARNING)

# --- Exchange Configuration ---
# Define supported exchanges with display symbols
config["exchanges"] = {
    "aerodrome": {"symbol": "1", "display": "Aerodrome (Base)"},
    "shadow": {"symbol": "2", "display": "Shadow (Sonic)"}
}

# --- Predefined Fee Tiers for Specific Stable Pairs (Shadow Exchange) ---
# Fees are in Basis Points (bps) as strings, using user-provided exact values
PREDEFINED_STABLE_FEES_BPS = {
    # Canonical name (sorted symbols) -> fee_tier_bps (string)
    "USDC.e/aSonUSDC": "0.5",  # 0.005%
    "USDC.e/scUSD":    "0.02", # 0.0002%
    "USDC.e/USDT":     "0.01", # 0.0001%
    "USDC.e/xUSD":     "2.5",  # 0.025%
    "scUSD/wstkscUSD": "1",    # 0.01%
    "EURC.e/USDC.e":   "2.5",  # 0.025%
    "frxUSD/scUSD":    "1",    # 0.01%
}
# Create a set of the canonical names for quick checking
TARGET_STABLE_PAIRS = set(PREDEFINED_STABLE_FEES_BPS.keys())

# Strategy parameters to test for USD/USD pairs (typically narrower)
USD_PAIR_STRATEGIES = [
    # Original Strategies
    {"width": 0.02, "rebalance_buffer": 0.1, "cutoff_buffer": 0.2},
    {"width": 0.02, "rebalance_buffer": 0.2, "cutoff_buffer": 0.4},
    {"width": 0.05, "rebalance_buffer": 0.1, "cutoff_buffer": 0.2},
    {"width": 0.05, "rebalance_buffer": 0.2, "cutoff_buffer": 0.4},
    {"width": 0.1, "rebalance_buffer": 0.1, "cutoff_buffer": 0.2},
    {"width": 0.1, "rebalance_buffer": 0.2, "cutoff_buffer": 0.4},

    # Wider Width Strategies
    {"width": 0.2, "rebalance_buffer": 0.1, "cutoff_buffer": 0.2},
    {"width": 0.2, "rebalance_buffer": 0.2, "cutoff_buffer": 0.4},
    {"width": 0.5, "rebalance_buffer": 0.1, "cutoff_buffer": 0.2},
    {"width": 0.5, "rebalance_buffer": 0.2, "cutoff_buffer": 0.4},

    # No Rebalance/Cutoff Strategies (Passive)
    {"width": 0.02, "rebalance_buffer": 0, "cutoff_buffer": 0},
    {"width": 0.05, "rebalance_buffer": 0, "cutoff_buffer": 0},
    {"width": 0.1, "rebalance_buffer": 0, "cutoff_buffer": 0},
    {"width": 0.2, "rebalance_buffer": 0, "cutoff_buffer": 0},
    {"width": 0.5, "rebalance_buffer": 0, "cutoff_buffer": 0},
]

# --- Helper Functions ---

def get_pools_for_exchange(exchange_name: str, limit: int = 20):
    """Fetches the top pools for a given exchange.

    Args:
        exchange_name (str): The exchange identifier (e.g., 'aerodrome', 'shadow')
        limit (int): Number of pools to fetch.

    Returns:
        list: A list of pool objects with pool details or empty list if failed.
    """
    logger.info(f"Fetching top {limit} pools for {exchange_name}")
    print(f"\n⏳ Fetching top {limit} pools from {exchange_name.capitalize()}...")

    client = get_client(exchange_name)
    if not client:
        logger.error(f"Failed to create client for {exchange_name}")
        print(f"\n❌ ERROR: Could not create subgraph client for {exchange_name}.")
        return []

    try:
        pools = client.get_top_pools(limit=limit)
        if not pools:
            logger.warning(f"No pools returned for {exchange_name}")
            print(f"\n⚠️ WARNING: No pools found for {exchange_name}.")
            return []

        logger.info(f"Retrieved {len(pools)} pools for {exchange_name}")
        return pools
    except Exception as e:
        logger.error(f"Error fetching pools for {exchange_name}: {e}", exc_info=True)
        print(f"\n❌ ERROR: An unexpected error occurred while fetching pools: {e}")
        return []

def is_usd_pair_pool(pool_details: dict) -> bool:
    """Checks if a pool is a USD pair based on token symbols containing 'usd'."""
    token0_symbol = pool_details.get('token0', {}).get('symbol', '').lower()
    token1_symbol = pool_details.get('token1', {}).get('symbol', '').lower()
    # Check if 'usd' is a substring in both symbols
    return "usd" in token0_symbol and "usd" in token1_symbol


# --- Command Functions ---

def run_single_backtest_wrapper(args_tuple):
    """Wrapper function to run a single backtest, suitable for parallel execution."""
    # Unpack arguments, now including historical_data, daily_volatility_pct, AND original symbols
    (
        pool_details, strategy, investment, simulate_tx_costs, config,
        gas_override, slippage_override, exchange_name,
        historical_data, daily_volatility_pct, # Existing added arguments
        t0_sym_orig, t1_sym_orig # NEW original symbols for naming
    ) = args_tuple

    pool_address = pool_details.get('id')
    # Construct pool name using ORIGINAL symbols passed in
    pool_name = f"{t0_sym_orig}/{t1_sym_orig}"
    logger.info(f"Starting backtest for pool {pool_address} ({pool_name}) with strategy: {strategy}")

    try:
        # 1. Data is already loaded and passed in
        if historical_data is None or historical_data.empty:
            logger.warning(f"Received empty historical data for pool {pool_address}. Skipping.")
            return None

        # Calculate pool-level stats from the passed historical data
        total_volume_period = historical_data['volumeUSD'].sum() if 'volumeUSD' in historical_data else 0.0
        total_pool_fees_period = historical_data['feesUSD'].sum() if 'feesUSD' in historical_data else 0.0
        average_tvl_period = historical_data['tvlUSD'].mean() if 'tvlUSD' in historical_data and not historical_data['tvlUSD'].empty else 0.0

        # Calculate average daily stats from historical data
        avg_daily_tvl = historical_data['tvlUSD'].mean() if 'tvlUSD' in historical_data and not historical_data['tvlUSD'].empty else 0.0
        avg_daily_volume = historical_data['volumeUSD'].mean() if 'volumeUSD' in historical_data and not historical_data['volumeUSD'].empty else 0.0
        avg_daily_fees = historical_data['feesUSD'].mean() if 'feesUSD' in historical_data and not historical_data['feesUSD'].empty else 0.0

        # Add exchange name to pool_details if not present
        if 'exchange' not in pool_details:
             pool_details['exchange'] = exchange_name

        # 2. Initialize Engine with pre-loaded data
        engine = BacktestEngine(
            pool_details=pool_details,
            historical_data=historical_data, # Use passed data
            config=config,
            gas_cost_usd=gas_override,
            slippage_pct=slippage_override
        )

        # 3. Run Backtest
        result = engine.run_vfat_backtest(
            initial_width_pct=strategy['width'],
            rebalance_buffer_pct=strategy['rebalance_buffer'],
            cutoff_buffer_pct=strategy['cutoff_buffer'],
            initial_investment=investment,
            simulate_tx_costs=simulate_tx_costs
        )

        if not result or "metrics" not in result or "error" in result:
            logger.error(f"Backtest failed for pool {pool_address}, strategy {strategy}. Error: {result.get('error', 'Unknown')}")
            return None

        # 4. Return key results including pool stats and volatility
        metrics = result["metrics"]
        # pool_name already constructed from original symbols

        # Calculate Pool Fees APR based on total fees reported by subgraph for the period
        initial_investment = result.get("parameters", {}).get("initial_investment", investment) # Use investment from results if available
        duration_days = metrics.get('duration_days', 0)
        if duration_days > 0 and average_tvl_period > 0:
             # Estimate pool fee APR based on average TVL and total fees for the period
             pool_fees_apr = (total_pool_fees_period / average_tvl_period) * (365.0 / duration_days) * 100
        else:
             pool_fees_apr = 0.0

        strategy_fees_apr = metrics.get('fees_apr', 0.0) # Strategy's estimated fee APR

        return {
            "pool_address": pool_address,
            "pool_name": pool_name, # Uses name from original symbols
            "exchange": exchange_name,
            "strategy": strategy,
            "net_apr": metrics.get('net_apr', 0.0),
            "pool_fees_apr": pool_fees_apr, # APR based on total pool fees / avg TVL
            "strategy_fees_apr": strategy_fees_apr, # Strategy's own fee APR estimate
            "rebalance_count": metrics.get('rebalance_count', 0),
            "time_in_position_pct": metrics.get('time_in_position_pct', 0.0),
            "duration_days": duration_days,
            "daily_volatility_pct": daily_volatility_pct, # Include pre-calculated volatility
            "total_volume_period": total_volume_period, # Use period-specific name
            "total_pool_fees_period": total_pool_fees_period, # Use period-specific name
            "average_tvl_period": average_tvl_period, # Use period-specific name
            # Add new daily averages
            "avg_daily_tvl": avg_daily_tvl,
            "avg_daily_volume": avg_daily_volume,
            "avg_daily_fees": avg_daily_fees
        }

    except Exception as e:
        logger.error(f"Error running backtest for pool {pool_address} ({pool_name}), strategy {strategy}: {e}", exc_info=True)
        return None


def run_top_usd_pairs(args):
    """Finds specific USD/USD pairs, pre-loads data, calculates volatility, optionally plots, and runs backtests using PREDEFINED fee tiers."""
    logger.info(f"Starting 'top-usd-pairs' analysis for exchange: {args.exchange}, targeting specific pairs with predefined fees.")

    # --- Step 1: Initialize Client and Loader ---
    endpoint = None
    api_key = None
    backup_endpoint = None
    api_config = config.get('api_config', {})

    if args.exchange == 'shadow':
        shadow_config = api_config.get('shadow', {})
        endpoint = shadow_config.get('endpoint')
        backup_endpoint = shadow_config.get('backup_endpoint')
        # Retrieve the general TheGraph API key for Shadow as well
        api_key = api_config.get('thegraph', {}).get('api_key')
    elif args.exchange == 'aerodrome':
        thegraph_config = api_config.get('thegraph', {})
        api_key = thegraph_config.get('api_key')
        base_config = thegraph_config.get('base', {})
        endpoint = base_config.get(args.exchange)
    else:
         logger.warning(f"Endpoint lookup logic not fully defined for exchange '{args.exchange}'. Attempting generic lookup.")
         # Add more robust lookup if needed
         generic_config = api_config.get(args.exchange, api_config.get('thegraph', {}).get('base', {}))
         endpoint = generic_config.get('endpoint', generic_config.get(args.exchange))
         api_key = api_config.get('thegraph', {}).get('api_key') # Assume thegraph key for others

    if not endpoint:
         logger.error(f"Endpoint configuration could not be determined for exchange '{args.exchange}'. Please check config.py.")
         print(f"\n❌ ERROR: Endpoint configuration missing or lookup failed for exchange '{args.exchange}'.")
         return

    # Add debug logging for endpoint and API key
    logger.debug(f"Initializing SubgraphClient for {args.exchange} with endpoint: {endpoint}")
    # Use more careful API key logging
    api_key_status = "Set" if api_key and api_key.strip() else "Not Set or Empty"
    logger.debug(f"API Key for SubgraphClient: {api_key_status}")
    if not api_key or not api_key.strip():
        logger.warning(f"API Key is NOT SET or empty for {args.exchange}. Gateway queries might fail or be rate-limited.")


    try:
        client = SubgraphClient(
            exchange_name=args.exchange,
            endpoint=endpoint,
            api_key=api_key,
            timeout_seconds=60
        )
    except Exception as e:
        logger.error(f"Failed to initialize SubgraphClient for {args.exchange}: {e}", exc_info=True)
        print(f"\n❌ ERROR: Failed to initialize subgraph client for {args.exchange}.")
        return

    loader = DataLoader() # DataLoader takes no args
    processor = ResultProcessor(output_dir="./output")
    backtest_results = []

    # --- Step 2: Fetch Top Pools using Client ---
    try:
        logger.info(f"Fetching up to {args.pool_limit} pools from {args.exchange}...")
        print(f"\n⏳ Fetching top {args.pool_limit} pools from {args.exchange.capitalize()} (will filter for target pairs later)...")
        # Fetch a broader set initially, filtering by TVL/Volume if desired
        top_pools = client.get_top_pools(limit=args.pool_limit)

        if not top_pools:
            logger.warning("No pools returned from the subgraph query.")
            print(f"\n⚠️ WARNING: No pools found. Check subgraph status.")
            return

        logger.info(f"Retrieved {len(top_pools)} pools initially.")

        # --- Step 3: Filter for Target Volume, Min Liquidity, and Specific Pairs ---
        # Filter by minimum liquidity first
        initial_count_liq = len(top_pools)
        top_pools = [p for p in top_pools if float(p.get('totalValueLockedUSD', p.get('reserveUSD', 0))) >= args.min_liquidity]
        if len(top_pools) < initial_count_liq:
            logger.info(f"Filtered pools by min_liquidity >= {args.min_liquidity}: {initial_count_liq} -> {len(top_pools)}")

        if not top_pools:
             logger.warning("No pools remaining after applying liquidity filter.")
             print(f"\n⚠️ WARNING: No pools remaining after filtering by TVL >= ${args.min_liquidity:,.0f}.")
             return

        # Filter by minimum volume
        initial_count_vol = len(top_pools)
        top_pools = [p for p in top_pools if float(p.get('volumeUSD', 0)) >= args.min_volume]
        if len(top_pools) < initial_count_vol:
            logger.info(f"Filtered pools by min_volume >= {args.min_volume}: {initial_count_vol} -> {len(top_pools)}")

        if not top_pools:
             logger.warning("No pools remaining after applying volume filter.")
             print(f"\n⚠️ WARNING: No pools remaining after filtering by Volume >= ${args.min_volume:,.0f}.")
             return

        # Now filter for the SPECIFIC target pairs using canonical names
        target_usd_pairs = []
        target_usd_pairs_details = {} # Store original symbols with pool id
        for pool in top_pools:
            t0_sym = pool.get('token0', {}).get('symbol')
            t1_sym = pool.get('token1', {}).get('symbol')

            if not t0_sym or not t1_sym:
                logger.debug(f"Skipping pool {pool.get('id', 'N/A')} due to missing symbol(s).")
                continue

            # Create canonical pair name (alphabetical order)
            symbols = sorted([t0_sym, t1_sym])
            canonical_name = f"{symbols[0]}/{symbols[1]}"

            if canonical_name in TARGET_STABLE_PAIRS:
                target_usd_pairs.append(pool) # Keep the original pool dict for data loading
                # Store original symbols associated with the pool ID for later use
                target_usd_pairs_details[pool.get('id')] = {'t0': t0_sym, 't1': t1_sym}
                logger.info(f"Found target pool: {canonical_name} ({pool.get('id')})")
            else:
                 logger.debug(f"Skipping pool {pool.get('id', 'N/A')} ({t0_sym}/{t1_sym}) - not in target list.")


        if not target_usd_pairs:
            logger.warning("None of the fetched & filtered pools match the predefined target stable pairs.")
            print("\n⚠️ WARNING: No pools matching the specific target pairs found after filtering. Check target list or pool availability.")
            return

        logger.info(f"Filtered down to {len(target_usd_pairs)} target pools. Pre-loading data...")
        print(f"\n⏳ Found {len(target_usd_pairs)} target pools. Pre-loading data and calculating volatility for {args.days} days...")

        # --- Step 5: Data Loading Loop (using target_usd_pairs) ---
        # Use datetime with timezone awareness for start/end dates
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=args.days)
        valid_pools_for_backtest = [] # Initialize list for pools with valid data

        # Loop through the filtered TARGET pairs to load data
        for pool in target_usd_pairs:
            pool_id = pool['id']
            # Get original symbols from our stored details
            original_symbols = target_usd_pairs_details.get(pool_id, {'t0': 'ErrT0', 't1': 'ErrT1'})
            t0_sym_orig = original_symbols['t0']
            t1_sym_orig = original_symbols['t1']
            pool_name_log = f"{t0_sym_orig}/{t1_sym_orig}"
            logger.info(f"Processing target pool: {pool_id} ({pool_name_log})")

            # Use the loader instance correctly
            pool_data, historical_data = loader.load_historical_data(
                exchange_name=args.exchange,
                pool_address=pool_id,
                start_date=start_date,
                end_date=end_date
            )

            # Check results from load_historical_data
            if historical_data is None or historical_data.empty or 'price' not in historical_data.columns or historical_data['price'].isnull().all():
                logger.warning(f"Skipping pool {pool_id} ({pool_name_log}): Failed to retrieve valid historical data or price data for the period.")
                continue

            # Calculate Daily Volatility for the period
            try:
                # Ensure price column is numeric and drop NaNs for calculation
                prices = pd.to_numeric(historical_data['price'], errors='coerce').dropna()
                if len(prices) > 1:
                     # Calculate DAILY volatility for the period, not annualized
                     daily_volatility = calculate_volatility(prices, window=len(prices), annualize=False)
                     daily_volatility_pct = daily_volatility * 100 # Convert to percentage
                     logger.info(f"Calculated {args.days}-day Daily Volatility for {pool_id}: {daily_volatility_pct:.4f}%")
                else:
                     logger.warning(f"Not enough valid price points ({len(prices)}) to calculate volatility for {pool_id}.")
                     daily_volatility_pct = 0.0 # Or None? Let's use 0 for now.
            except Exception as e:
                 logger.error(f"Error calculating volatility for {pool_id}: {e}", exc_info=True)
                 daily_volatility_pct = 0.0 # Default on error

            # --- Pool Data and Fee Tier Handling --- 
            if pool_data is None:
                logger.warning(f"Pool details NOT FOUND for {pool_id} ({pool_name_log}). Creating fallback.")
                # Create a minimal placeholder using ORIGINAL symbols
                pool_data = {'id': pool_id, 'exchange': args.exchange, 'token0': {'symbol': t0_sym_orig}, 'token1': {'symbol': t1_sym_orig}, 'feeTier': None}

            # Ensure exchange is present
            pool_data['exchange'] = args.exchange
            # Ensure symbols in pool_data match originals if needed (though fee lookup uses originals now)
            if 'token0' not in pool_data or not pool_data.get('token0', {}).get('symbol'): pool_data['token0'] = {'symbol': t0_sym_orig}
            if 'token1' not in pool_data or not pool_data.get('token1', {}).get('symbol'): pool_data['token1'] = {'symbol': t1_sym_orig}

            # --- OVERRIDE Fee Tier with Predefined Value --- 
            symbols_sorted = sorted([t0_sym_orig, t1_sym_orig]) # Use original symbols
            canonical_name_lookup = f"{symbols_sorted[0]}/{symbols_sorted[1]}"
            predefined_fee = PREDEFINED_STABLE_FEES_BPS.get(canonical_name_lookup)

            if predefined_fee is not None:
                original_fee = pool_data.get('feeTier')
                pool_data['feeTier'] = predefined_fee # Update pool_data for BacktestEngine
                if original_fee != predefined_fee:
                     logger.info(f"Overriding fee tier for {pool_id} ({pool_name_log}). Original: '{original_fee}', Predefined: '{predefined_fee}' bps.")
                else:
                     logger.info(f"Using predefined fee tier {predefined_fee} bps for {pool_id} ({pool_name_log}).")
            else:
                logger.error(f"Target pool {pool_id} ({pool_name_log}) with canonical name '{canonical_name_lookup}' not found in PREDEFINED_STABLE_FEES_BPS. Skipping backtest.")
                print(f"\n❌ ERROR: Mismatch for pool {pool_name_log}. Cannot find predefined fee '{canonical_name_lookup}' in predefined list. Skipping.")
                continue

            # --- Plotting (if requested) ---
            if args.plot:
                 try:
                     ensure_dir_exists(processor.output_dir)
                     plot_t0_sym = t0_sym_orig # Use original symbols for plot name
                     plot_t1_sym = t1_sym_orig

                     now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                     t0_filename = plot_t0_sym.replace('/', '-').replace('.', '') # Sanitize more
                     t1_filename = plot_t1_sym.replace('/', '-').replace('.', '')
                     short_id = pool_id[:8]
                     # Changed plot filename to reflect content
                     plot_filename = f"{args.exchange}_{t0_filename}_{t1_filename}_{short_id}_daily_pct_change_{now_str}.png"
                     output_file_path = os.path.join(processor.output_dir, plot_filename)

                     logger.info(f"Generating daily price % change plot for {pool_id} ({pool_name_log}) -> {output_file_path}")
                     # Pass original symbols to plotting function if needed for title
                     plot_pool_details = pool_data.copy()
                     plot_pool_details['token0']['symbol'] = t0_sym_orig
                     plot_pool_details['token1']['symbol'] = t1_sym_orig
                     plot_volatility(
                         historical_data=historical_data,
                         pool_details=plot_pool_details, # Pass details with correct names
                         output_file=output_file_path
                     )
                 except Exception as e:
                      logger.error(f"Error generating plot for {pool_id}: {e}", exc_info=True)
                      print(f"\n⚠️ WARNING: Failed to generate plot for {pool_id} ({pool_name_log}): {e}")


            # Add pool to the list for backtesting
            valid_pools_for_backtest.append({
                 'pool_details': pool_data, # Has potentially overridden fee tier
                 'historical_data': historical_data,
                 'daily_volatility_pct': daily_volatility_pct,
                 't0_sym_orig': t0_sym_orig, # Pass original symbols
                 't1_sym_orig': t1_sym_orig
            })
            logger.info(f"Successfully pre-loaded data and calculated volatility for {pool_id} ({pool_name_log})")
            # --- End of Loop ---

        if not valid_pools_for_backtest:
            logger.error("No target pools with valid historical data found after loading. Cannot proceed with backtesting.")
            print("\n❌ ERROR: No target pools with valid data found. Check logs or try different filters.")
            return

        logger.info(f"Proceeding to backtest {len(valid_pools_for_backtest)} pools with {len(USD_PAIR_STRATEGIES)} strategies each.")

        # --- Step 6: Backtesting Setup for Parallel Execution ---
        tasks = []
        for pool_info in valid_pools_for_backtest:
            for strategy in USD_PAIR_STRATEGIES:
                # Prepare tuple arguments for run_single_backtest_wrapper
                args_tuple = (
                    pool_info['pool_details'],
                    strategy,
                    args.investment,
                    not args.no_tx_costs,
                    config,
                    args.gas_cost,
                    args.slippage_pct,
                    args.exchange,
                    pool_info['historical_data'],
                    pool_info['daily_volatility_pct'],
                    pool_info['t0_sym_orig'], # Pass original symbols
                    pool_info['t1_sym_orig']
                )
                tasks.append(args_tuple)

        logger.info(f"Prepared {len(tasks)} backtest tasks for parallel execution.")

        # --- Step 7: Run Backtests in Parallel ---
        print(f"\n⏳ Running {len(tasks)} backtests across {len(valid_pools_for_backtest)} pools using up to {args.max_workers} workers...")
        all_results = []
        failed_count = 0
        # Use concurrent.futures for parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
             # Map the wrapper function to the tasks
             # executor.map preserves order of inputs in results
             future_results = executor.map(run_single_backtest_wrapper, tasks)

             # Process results as they complete (or after all finish)
             for result in future_results:
                  if result is not None:
                      all_results.append(result)
                  else:
                      failed_count += 1

        logger.info(f"Parallel backtesting finished. Received {len(all_results)} successful results, {failed_count} tasks failed or returned None.")
        if not all_results:
            logger.error("No successful backtest results were obtained.")
            print("\n❌ ERROR: All backtest tasks failed. Check logs for details.")
            return

        # --- Step 8: Process and Save Results ---
        print(f"\n✅ Backtesting complete. Processing {len(all_results)} results...")
        try:
            # Convert list of result dictionaries to DataFrame
            results_df = pd.DataFrame(all_results)

            # Define columns for potential display/sorting (though not strictly needed if only plotting)
            output_columns = [
                 "pool_address", "pool_name", "exchange",
                 "strategy", "net_apr", "pool_fees_apr", "strategy_fees_apr",
                 "daily_volatility_pct", # Added volatility
                 "rebalance_count", "time_in_position_pct", "duration_days",
                 "total_volume_period", "total_pool_fees_period", "average_tvl_period",
                 # Add new daily averages to expected columns
                 "avg_daily_tvl", "avg_daily_volume", "avg_daily_fees"
            ]
            # Ensure only existing columns are selected
            results_df_display = results_df[[col for col in output_columns if col in results_df.columns]].copy() # Use a copy for display modifications

            # --- Save Full Results to CSV (COMMENTED OUT) --- 
            # # Sort all results by Net APR descending for the CSV
            # results_df_sorted_full = results_df_display.sort_values(by="net_apr", ascending=False)

            # ensure_dir_exists(processor.output_dir) # Ensure output dir exists before saving
            # timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            # # Add investment amount to filename for clarity
            # investment_k = int(args.investment / 1000)
            # filename_base = f"{args.exchange}_top_usd_pairs_{args.days}d_{investment_k}k_{timestamp_str}"
            # csv_filename = os.path.join(processor.output_dir, f"{filename_base}.csv")

            # logger.info(f"Saving FULL top USD pair results ({len(results_df_sorted_full)} rows) to {csv_filename}")
            # processor.save_results_to_csv(results_df_sorted_full, filename=csv_filename)
            # print(f"\n✅ Full results ({len(results_df_sorted_full)} strategies) saved to: {csv_filename}")
            logger.info("CSV output is currently disabled.") # Log that CSV is skipped

            if args.plot:
                 print(f"\n✅ Plots saved to directory: {os.path.abspath(processor.output_dir)}")

            # --- Prepare Console Output: Top Strategy per Top 5 Pools (Keep this for info) ---
            print("\n--- Top 5 Pools & Their Best Strategy (Sorted by Best Strategy Net APR) ---")

            # Find the index of the row with the highest net_apr for each pool_name
            idx = results_df.groupby('pool_name')['net_apr'].idxmax()
            best_results = results_df.loc[idx]

            # Sort by net_apr descending and take top 5
            best_results = best_results.sort_values(by='net_apr', ascending=False).head(5)

            # Select the columns needed for display using original names
            best_results_display = best_results[[
                'pool_name',
                'strategy',  # Use original strategy column name
                'net_apr',
                'strategy_fees_apr', # Use CORRECT original fees_apr column name from engine
                'daily_volatility_pct', # Use original volatility column name
                'rebalance_count', # Use original rebalance column name
                'time_in_position_pct',
                # Add new columns for display
                'avg_daily_tvl',
                'avg_daily_volume',
                'avg_daily_fees'
            ]].copy()

            # Format the strategy dictionary, applying conditional RB/CB=0 if rebalances=0
            def format_strategy_conditional(row):
                strategy_dict = row['strategy']
                rebalances = row['rebalance_count']
                if isinstance(strategy_dict, dict):
                    width = strategy_dict.get('width', 'N/A')
                    rb = strategy_dict.get('rebalance_buffer', 'N/A')
                    cb = strategy_dict.get('cutoff_buffer', 'N/A')
                    if rebalances == 0:
                        return f"W:{width} | RB:0 | CB:0" # Override RB/CB if 0 rebalances
                    else:
                        return f"W:{width} | RB:{rb} | CB:{cb}" # Standard format
                return str(strategy_dict) # Fallback if not a dict

            best_results_display['formatted_strategy'] = best_results_display.apply(format_strategy_conditional, axis=1)

            # Now, rename columns for the final display table
            best_results_display.rename(columns={
                'pool_name': 'Pool',
                # Rename the *newly formatted* strategy string column
                'formatted_strategy': 'Best Strategy (W|RB|CB)',
                'net_apr': 'Net APR',
                'strategy_fees_apr': 'Strat Fees APR', # RENAME the CORRECT fees_apr column
                'daily_volatility_pct': '90d Daily Vol', # Rename volatility
                'rebalance_count': 'Rebalances', # Rename rebalances
                'time_in_position_pct': 'Time In Pos',
                # Add renames for new columns
                'avg_daily_tvl': 'Avg Daily TVL',
                'avg_daily_volume': 'Avg Daily Vol',
                'avg_daily_fees': 'Avg Daily Fees'
            }, inplace=True)

            # Select final columns AFTER renaming
            final_display_columns = [
                 'Pool', 'Best Strategy (W|RB|CB)', 'Net APR', 'Strat Fees APR',
                 '90d Daily Vol', 'Rebalances', 'Time In Pos',
                 'Avg Daily TVL', 'Avg Daily Vol', 'Avg Daily Fees'
            ]
            best_results_display = best_results_display[final_display_columns]


            # Format percentages
            best_results_display['Net APR'] = best_results_display['Net APR'].map('{:.2f}%'.format)
            best_results_display['Strat Fees APR'] = best_results_display['Strat Fees APR'].map('{:.2f}%'.format)
            best_results_display['90d Daily Vol'] = best_results_display['90d Daily Vol'].map('{:.2f}%'.format)
            best_results_display['Time In Pos'] = best_results_display['Time In Pos'].map('{:.2f}%'.format)

            # Format currency columns (Assumes format_currency helper exists)
            best_results_display['Avg Daily TVL'] = best_results_display['Avg Daily TVL'].apply(format_currency)
            best_results_display['Avg Daily Vol'] = best_results_display['Avg Daily Vol'].apply(format_currency)
            best_results_display['Avg Daily Fees'] = best_results_display['Avg Daily Fees'].apply(format_currency)


            # Define the headers *after* renaming and selecting final columns
            headers = final_display_columns # Use the list of final column names
            print(tabulate(
                best_results_display,
                headers=headers, # Use the defined headers list
                tablefmt='orgtbl',
                showindex=False
            ))
            print("------------------------------------------------------------")

        except Exception as e:
             logger.error(f"Error processing or saving results: {e}", exc_info=True)
             print(f"\n❌ ERROR: Failed to process or save results: {e}")

    except Exception as e:
        logger.error(f"An error occurred during the top USD pairs analysis: {e}", exc_info=True)
        print(f"\n❌ ERROR: An unexpected error occurred: {e}")


    logger.info(f"'top-usd-pairs' analysis finished for exchange: {args.exchange}")

# --- Identify USD Pairs Helper ---
# Ensure this function exists and is accessible
def identify_usd_pairs(pools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identifies potential USD/USD stablecoin pairs from a list of pools."""
    usd_pairs = []
    seen_pairs = set()
    # Looser check: contains 'usd', 'dai', 'mim', 'frax' etc. - Case Insensitive
    stablecoin_substrings = ['usd', 'dai', 'mim', 'frax', 'lusd']

    for pool in pools:
        t0 = pool.get('token0', {})
        t1 = pool.get('token1', {})
        t0_sym = t0.get('symbol', '').lower()
        t1_sym = t1.get('symbol', '').lower()

        # Check if both symbols contain any of the stablecoin substrings
        t0_is_stable = any(sub in t0_sym for sub in stablecoin_substrings)
        t1_is_stable = any(sub in t1_sym for sub in stablecoin_substrings)

        if t0_is_stable and t1_is_stable:
            # Avoid adding duplicate pairs (token0/token1 vs token1/token0)
            pair_key = tuple(sorted((t0.get('id', ''), t1.get('id', ''))))
            if pair_key not in seen_pairs and t0.get('id') and t1.get('id'):
                usd_pairs.append(pool)
                seen_pairs.add(pair_key)
                logger.debug(f"Identified potential USD pair: {pool.get('id')} ({t0_sym}/{t1_sym})")

    return usd_pairs


def run_single_backtest(args):
    """Runs a single vfat strategy backtest based on CLI arguments."""
    # Initial setup information
    print("\n" + "="*60)
    print(f"  Backtesting {args.width}% vfat.io Strategy on {args.exchange.upper()}")
    print(f"  Pool: {args.pool_address}")
    print(f"  Date Range: {args.start_date.strftime('%Y-%m-%d')} to {args.end_date.strftime('%Y-%m-%d')}")
    print("="*60)

    logger.info(f"Initiating backtest for {args.exchange.upper()} pool: {args.pool_address}")
    logger.info(f"Date Range: {args.start_date.strftime('%Y-%m-%d')} to {args.end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Strategy Params: Width={args.width}%, RebalanceBuffer={args.rebalance_buffer}%, CutoffBuffer={args.cutoff_buffer}%")
    logger.info(f"Investment: ${args.investment:.2f}")
    logger.info(f"Simulate Tx Costs: {not args.no_tx_costs}")
    if not args.no_tx_costs:
         # Use CLI args if provided, otherwise they are None and engine uses config defaults
         gas_override = args.gas_cost
         slippage_override = args.slippage_pct
         logger.info(f"Tx Costs: Gas=${gas_override if gas_override is not None else 'config_default'}, Slippage={slippage_override if slippage_override is not None else 'config_default'}%")

    # --- 1. Load Data ---
    print("\n⏳ Fetching pool details and historical data...")
    logger.info("Loading data from subgraph...")
    try:
        pool_details, historical_data = load_data(
            exchange_name=args.exchange,
            pool_address=args.pool_address,
            start_date=args.start_date,
            end_date=args.end_date
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        print(f"\n❌ ERROR: Failed to load data due to an unexpected error: {e}")
        sys.exit(1) # Exit with error code

    # Validate loaded data
    if pool_details is None:
        # Data loader logs specific errors, provide a user-friendly message
        logger.error("Failed to load pool details. Cannot proceed without essential info like fee tier.")
        print("\n❌ ERROR: Could not load pool details for the specified address.")
        print("Please verify the pool address and exchange, and check network/API connectivity.")
        sys.exit(1)

    if historical_data is None:
        logger.error("Failed to load historical data (returned None).")
        print("\n❌ ERROR: Could not load historical data. Subgraph query might have failed.")
        print("Please check the pool address, date range, and network/API connectivity.")
        sys.exit(1)

    if historical_data.empty:
        logger.error("Historical data is empty for the specified range.")
        print("\n❌ ERROR: No historical data found for the specified pool and date range.")
        print("Please check the pool address and date range. The pool might be new or data unavailable.")
        sys.exit(1)

    # Add exchange name to pool_details if not present (useful for parameters log)
    if 'exchange' not in pool_details:
         pool_details['exchange'] = args.exchange

    print(f"✅ Data loaded successfully: {len(historical_data)} historical data points.")
    pool_name = f"{pool_details.get('token0', {}).get('symbol', 'Token0')}/{pool_details.get('token1', {}).get('symbol', 'Token1')}"
    print(f"   Pool: {pool_name}")
    print(f"   Fee Tier: {pool_details.get('feeTier', 'Unknown')} bps")
    logger.info(f"Loaded {len(historical_data)} historical data points.")

    # --- 2. Initialize Backtest Engine ---
    print("\n⏳ Initializing backtest engine...")
    logger.info("Initializing backtest engine...")
    try:
        engine = BacktestEngine(
            pool_details=pool_details,
            historical_data=historical_data,
            config=config,
            gas_cost_usd=args.gas_cost, # Pass overrides (can be None)
            slippage_pct=args.slippage_pct # Pass overrides (can be None)
        )
    except ValueError as e:
        logger.error(f"Failed to initialize BacktestEngine: {e}")
        print(f"\n❌ ERROR: Could not initialize backtest engine: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during engine initialization: {e}", exc_info=True)
        print(f"\n❌ ERROR: Failed to initialize engine due to an unexpected error: {e}")
        sys.exit(1)

    # --- 3. Run Backtest ---
    print("\n⏳ Running backtest simulation...")
    print(f"   - Width: {args.width}%")
    print(f"   - Rebalance Buffer: {args.rebalance_buffer}%") # Changed label
    print(f"   - Cutoff Buffer: {args.cutoff_buffer}%") # Changed label
    print(f"   - Initial Investment: ${args.investment:,.2f}")
    print(f"   - Transaction Costs: {'Included' if not args.no_tx_costs else 'Excluded'}")

    logger.info("Starting vfat backtest simulation...")
    try:
        result = engine.run_vfat_backtest(
            initial_width_pct=args.width,
            rebalance_buffer_pct=args.rebalance_buffer,
            cutoff_buffer_pct=args.cutoff_buffer,
            initial_investment=args.investment,
            simulate_tx_costs=not args.no_tx_costs
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during backtest execution: {e}", exc_info=True)
        print(f"\n❌ ERROR: Backtest simulation failed unexpectedly: {e}")
        sys.exit(1)


    # --- 4. Process Results ---
    if not result or "metrics" not in result or "parameters" not in result:
        logger.error("Backtest failed to produce valid results.")
        print("\n❌ ERROR: Backtest simulation did not return expected results.")
        sys.exit(1)
    if "error" in result:
         logger.error(f"Backtest simulation returned an error: {result['error']}")
         print(f"\n❌ ERROR: Backtest simulation failed: {result['error']}")
         sys.exit(1)


    print("\n✅ Backtest simulation complete.")
    logger.info("Processing and displaying results...")

    output_dir = config.get("output_dir", "./output")
    ensure_dir_exists(output_dir) # Ensure output dir exists
    processor = ResultProcessor(output_dir=output_dir)

    # Print summary table to console
    print("\n" + "="*40)
    print("      Backtest Simulation Results")
    print("="*40)
    processor.print_summary_metrics(result["metrics"])
    print("="*40)


    # --- 5. Save Outputs (Plots, Logs) ---
    # Generate a base filename for outputs
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Sanitize pool address for filename
    safe_pool_addr = args.pool_address.replace('0x', '')[:12] # Shortened address
    token0 = result['parameters'].get('token0', 'T0')
    token1 = result['parameters'].get('token1', 'T1')
    # Use parameter values directly from result dict for consistency
    width_param = result['parameters'].get('initial_width_pct', 'na')
    rb_param = result['parameters'].get('rebalance_buffer_pct', 'na')
    co_param = result['parameters'].get('cutoff_buffer_pct', 'na')

    filename_base = (
        f"{args.exchange}_{token0}_{token1}_{safe_pool_addr}_"
        f"w{width_param}_rb{rb_param}_co{co_param}_" # Use exact params from result
        f"{timestamp_str}"
    ).replace('%', 'pct') # Basic filename generation


    # Generate plots if requested
    if args.plot:
        print("\n⏳ Generating plots...")
        logger.info("Generating plots...")
        try:
            plot_price_filename = f"{filename_base}_price_bounds.png"
            processor.plot_price_and_bounds(
                result["results_log"],
                result["parameters"], # Pass parameters for title context
                output_file=plot_price_filename
            )

            plot_perf_filename = f"{filename_base}_performance.png"
            processor.plot_performance_metrics(
                 result["results_log"],
                 result["parameters"], # Pass parameters for title context
                 output_file=plot_perf_filename
            )
            print(f"\n✅ Plots saved to directory: {os.path.abspath(output_dir)}")
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}", exc_info=True)
            print(f"\n⚠️ WARNING: Failed to generate plots: {e}")

    # Save detailed log if requested
    if args.save_log:
        print("\n⏳ Saving detailed results log...")
        log_filename = os.path.join(output_dir, f"{filename_base}_details.csv")
        try:
            # Ensure results_log is a DataFrame before saving
            if isinstance(result.get("results_log"), pd.DataFrame):
                result["results_log"].to_csv(log_filename, index=False)
                print(f"✅ Detailed results log saved to: {log_filename}")
            else:
                 logger.error("results_log is not a DataFrame, cannot save CSV.")
                 print("\n⚠️ WARNING: Could not save detailed log, results format unexpected.")
        except Exception as e:
            logger.error(f"Failed to save detailed log to {log_filename}: {e}", exc_info=True)
            print(f"\n⚠️ WARNING: Failed to save detailed log: {e}")

    print("\n" + "="*60)
    print("  Backtest completed successfully!")
    print("="*60)


def run_custom_mode(args=None):
    """Runs the tool in custom interactive mode."""
    args = SimpleNamespace()  # Create a namespace to mimic CLI arg structure

    # Load exchanges and pools
    print("\n\n=== Backtest Configuration ===")
    print("\nTip: Press ENTER to select default options shown in [brackets]")

    # Step 1: Choose Exchange
    exchanges = list(config["exchanges"].keys())
    print("\nAvailable Exchanges (enter the number):")
    for i, exchange in enumerate(exchanges, start=1):
        display_name = config["exchanges"][exchange].get("display", exchange.capitalize())
        print(f"{i}. {display_name}")

    # Get exchange selection
    while True:
        try:
            exchange_input = input("\nEnter exchange number (1-" + str(len(exchanges)) + "): ")
            exchange_idx = int(exchange_input) - 1
            if 0 <= exchange_idx < len(exchanges):
                args.exchange = exchanges[exchange_idx]
                break
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(exchanges)}.")
        except ValueError:
            print("Please enter a valid number.")

    # Step 2: Choose pool from available pools or enter manually
    pool_selection_complete = False
    manual_entry_attempts = 0
    selected_pool = None # Initialize selected_pool

    while not pool_selection_complete:
        pool_data = get_pools_for_exchange(args.exchange)

        if not pool_data or len(pool_data) == 0:
            print(f"\n⚠️ Could not fetch pools for {args.exchange}. This could be due to network issues or API limitations.")
            print("Options:")
            print("1. Try again")
            print("2. Enter pool address manually")
            print("3. Go back to exchange selection")

            while True:
                option = input("\nChoose an option (1-3): ")
                if option == "1":
                    # Continue the outer loop to try fetching pools again
                    break
                elif option == "2":
                    # Enter pool address manually
                    while True:
                        manual_pool = input("\nEnter pool address (0x...): ").strip()
                        if manual_pool.startswith("0x") and len(manual_pool) >= 42:
                            args.pool_address = manual_pool
                            selected_pool = {"id": args.pool_address, "token0": {"symbol": "?"}, "token1": {"symbol": "?"}} # Minimal placeholder
                            pool_selection_complete = True
                            break
                        else:
                            manual_entry_attempts += 1
                            print("Invalid pool address format. Address should start with 0x and be at least 42 characters.")
                            if manual_entry_attempts >= 3:
                                print("\n⚠️ Multiple failed attempts. Please verify the pool exists on the selected exchange.")
                                manual_entry_attempts = 0 # Reset attempts

                    if pool_selection_complete: break # Break the outer options loop
                elif option == "3":
                    # Go back to exchange selection (break inner loop, outer loop will restart)
                    print("\nReturning to exchange selection...")
                    # Need to break the option loop and the pool selection loop to re-prompt exchange
                    pool_selection_complete = True # Mark as complete to exit outer loop
                    args = None # Signal to re-run the whole custom mode? Or just re-prompt exchange?
                                # For simplicity, let's just exit and ask user to restart for now.
                    print("Please restart the custom mode to select a different exchange.")
                    sys.exit(0)
                else:
                    print("Invalid option. Please enter 1, 2, or 3.")
            if pool_selection_complete: break # Exit outer loop if manual address entered or going back
            else: continue # If user chose 'Try again', continue the outer loop

        else:
            # Display available pools
            print(f"\nAvailable {args.exchange.capitalize()} Pools (enter the number):")
            for i, pool in enumerate(pool_data, start=1):
                pool_name = f"{pool['token0']['symbol']}/{pool['token1']['symbol']}"
                print(f"{i}. {pool_name} (Fee: {pool.get('feeTier', '?')} bps)")

            # Get pool selection
            while True:
                try:
                    pool_input = input(f"\nEnter pool number (1-{len(pool_data)}) or 'a' to enter address manually: ")

                    if pool_input.lower() == 'a':
                        manual_pool_addr = input("\nEnter pool address (0x...): ").strip()
                        if manual_pool_addr.startswith("0x") and len(manual_pool_addr) >= 42:
                            args.pool_address = manual_pool_addr
                            selected_pool = {"id": args.pool_address, "token0": {"symbol": "?"}, "token1": {"symbol": "?"}} # Placeholder
                            pool_selection_complete = True
                            break
                        else:
                            print("Invalid pool address format. Address should start with 0x and be at least 42 characters.")
                            continue # Re-prompt for pool number or 'a'

                    pool_idx = int(pool_input) - 1
                    if 0 <= pool_idx < len(pool_data):
                        selected_pool = pool_data[pool_idx]
                        args.pool_address = selected_pool["id"]
                        pool_selection_complete = True
                        break
                    else:
                        print(f"Invalid selection. Please enter a number between 1 and {len(pool_data)} or 'a'.")
                except ValueError:
                     print("Invalid input. Please enter a number or 'a'.")

            break # Exit the main pool selection loop once a choice is made

    # Ensure selected_pool is set (either from list or manual entry)
    if selected_pool is None:
        print("Error: Pool selection failed unexpectedly.")
        sys.exit(1)

    print(f"\nSelected Pool: {selected_pool.get('token0',{}).get('symbol','?')}/{selected_pool.get('token1',{}).get('symbol','?')} ({args.pool_address})")


    # Step 3: Choose date range
    default_days_lookback = 90 # Default for custom mode
    default_end_date = datetime.now().date() - timedelta(days=1) # Yesterday
    default_start_date = default_end_date - timedelta(days=default_days_lookback)

    print("\nSelect Date Range:")
    date_format = "%Y-%m-%d"

    start_date_input = input(f"Start Date [{default_start_date.strftime(date_format)}]: ")
    if start_date_input:
        try:
            args.start_date = datetime.strptime(start_date_input, date_format).date()
        except ValueError:
            print(f"Invalid date format. Using default: {default_start_date.strftime(date_format)}")
            args.start_date = default_start_date
    else:
        args.start_date = default_start_date

    end_date_input = input(f"End Date [{default_end_date.strftime(date_format)}]: ")
    if end_date_input:
        try:
            args.end_date = datetime.strptime(end_date_input, date_format).date()
        except ValueError:
            print(f"Invalid date format. Using default: {default_end_date.strftime(date_format)}")
            args.end_date = default_end_date
    else:
        args.end_date = default_end_date

    # Validate dates
    if args.start_date >= args.end_date:
         print(f"\n❌ ERROR: Start date ({args.start_date}) must be before end date ({args.end_date}). Using defaults.")
         args.start_date = default_start_date
         args.end_date = default_end_date

    # Convert to datetime objects for the engine
    args.start_date = datetime.combine(args.start_date, datetime.min.time())
    args.end_date = datetime.combine(args.end_date, datetime.min.time())


    # Step 4: Strategy parameters
    print("\nStrategy Parameters:")

    # Position width (default 1.0%)
    while True:
        width_input = input("Position Width % [1.0]: ")
        try:
            args.width = float(width_input) if width_input else 1.0
            if args.width > 0: break
            else: print("Width must be greater than 0.")
        except ValueError: print("Invalid input. Please enter a number.")

    # Rebalance buffer (default 0.1 or 10% of width)
    while True:
        rebalance_input = input("Rebalance Buffer % (of width) [0.1]: ")
        try:
            args.rebalance_buffer = float(rebalance_input) if rebalance_input else 0.1
            if args.rebalance_buffer >= 0: break
            else: print("Rebalance buffer must be non-negative.")
        except ValueError: print("Invalid input. Please enter a number.")

    # Cutoff buffer (default 0.5 or 50% of width)
    while True:
        cutoff_input = input("Cutoff Buffer % (of width, 0 to disable) [0.5]: ")
        try:
            args.cutoff_buffer = float(cutoff_input) if cutoff_input else 0.5
            if args.cutoff_buffer == 0 or args.cutoff_buffer >= args.rebalance_buffer:
                break
            else:
                print(f"Cutoff buffer must be >= rebalance buffer ({args.rebalance_buffer}) or 0.")
        except ValueError: print("Invalid input. Please enter a number.")


    # Step 5: Investment amount
    while True:
        investment_input = input("Investment Amount USD [10000]: ")
        try:
            args.investment = float(investment_input) if investment_input else 10000.0
            if args.investment > 0: break
            else: print("Investment must be positive.")
        except ValueError: print("Invalid input. Please enter a number.")

    # Step 6: Transaction cost simulation
    tx_cost_input = input("Include Transaction Costs? (y/n) [y]: ")
    args.no_tx_costs = tx_cost_input.lower() in ['n', 'no']

    # If simulating tx costs, get gas and slippage defaults or overrides
    args.gas_cost = None
    args.slippage_pct = None
    if not args.no_tx_costs:
        # Get defaults from config for display
        default_gas = config.get("transaction_costs", {}).get("rebalance_gas_usd", 2.0)
        default_slippage = config.get("transaction_costs", {}).get("slippage_percentage", 0.05)

        while True:
             gas_input = input(f"Gas Cost USD (per tx) [{default_gas}]: ")
             try:
                  args.gas_cost = float(gas_input) if gas_input else None # None means use config default
                  if args.gas_cost is None or args.gas_cost >= 0: break
                  else: print("Gas cost cannot be negative.")
             except ValueError: print("Invalid input. Please enter a number.")

        while True:
             slippage_input = input(f"Slippage % (e.g., 0.1 for 0.1%) [{default_slippage}]: ")
             try:
                  args.slippage_pct = float(slippage_input) if slippage_input else None # None means use config default
                  if args.slippage_pct is None or args.slippage_pct >= 0: break
                  else: print("Slippage cannot be negative.")
             except ValueError: print("Invalid input. Please enter a number.")

    # Step 7: Output options
    plot_input = input("Generate Price and Performance Plots? (y/n) [y]: ")
    args.plot = not (plot_input.lower() in ['n', 'no'])

    save_log_input = input("Save Detailed Results Log CSV? (y/n) [n]: ")
    args.save_log = save_log_input.lower() in ['y', 'yes']

    # Add verbose option if needed
    args.verbose = False # Default for custom mode


    # Display configuration summary
    print("\n=== Backtest Configuration Summary ===")
    print(f"Exchange: {args.exchange.capitalize()}")
    pool_name_summary = f"{selected_pool['token0']['symbol']}/{selected_pool['token1']['symbol']}"
    print(f"Pool: {pool_name_summary} ({args.pool_address})")
    print(f"Date Range: {args.start_date.strftime(date_format)} to {args.end_date.strftime(date_format)}")
    print(f"Strategy: Width={args.width}%, RebalanceBuf={args.rebalance_buffer}%, CutoffBuf={args.cutoff_buffer}%")
    print(f"Investment: ${args.investment:,.2f}")

    tx_costs_status = "Included" if not args.no_tx_costs else "Excluded"
    tx_costs_details = ""
    if not args.no_tx_costs:
        gas_display_summary = args.gas_cost if args.gas_cost is not None else f"{default_gas} (default)"
        slippage_display_summary = args.slippage_pct if args.slippage_pct is not None else f"{default_slippage} (default)"
        tx_costs_details = f" (Gas=${gas_display_summary}, Slippage={slippage_display_summary}%)"

    print(f"Transaction Costs: {tx_costs_status}{tx_costs_details}")
    print(f"Generate Plots: {'Yes' if args.plot else 'No'}")
    print(f"Save Detailed Log: {'Yes' if args.save_log else 'No'}")

    # Confirm execution
    confirm = input("\nProceed with backtest? (y/n) [y]: ")
    if confirm.lower() in ['n', 'no']:
        print("Backtest cancelled.")
        return

    # Run the backtest with configured parameters
    run_single_backtest(args)


def run_quick_backtest(args):
    """
    Run a backtest with minimal parameters and sensible defaults.
    This is a streamlined version that requires minimal input.
    """
    # Calculate date ranges from days argument
    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=args.days)

    # Convert to datetime objects
    end_date_dt = datetime.combine(end_date, datetime.min.time())
    start_date_dt = datetime.combine(start_date, datetime.min.time())

    # Execute the backtest with quick options
    logger.info(f"Quick backtest for {args.exchange} pool {args.pool_address}")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({args.days} days)")
    logger.info(f"Strategy: Width {args.width}%, Rebalance Buffer {args.rebalance_buffer}%, Cutoff Buffer {args.cutoff_buffer}%")

    # Create a new args object with all the required properties for run_single_backtest
    class QuickArgs:
        pass

    quick_args = QuickArgs()
    quick_args.exchange = args.exchange
    quick_args.pool_address = args.pool_address
    quick_args.start_date = start_date_dt
    quick_args.end_date = end_date_dt
    quick_args.width = args.width
    quick_args.rebalance_buffer = args.rebalance_buffer
    quick_args.cutoff_buffer = args.cutoff_buffer
    quick_args.investment = args.investment
    quick_args.no_tx_costs = args.no_tx
    quick_args.gas_cost = None  # Use default from config
    quick_args.slippage_pct = None  # Use default from config
    quick_args.plot = args.plot
    quick_args.save_log = args.log
    quick_args.verbose = False

    # Run the backtest with our configured args
    run_single_backtest(quick_args)


# --- Main CLI Setup ---

def main():
    """Parse arguments and run the appropriate command."""
    parser = argparse.ArgumentParser(
        description="LP Optimizer CLI: Backtest vfat.io strategies for DEX pools using real historical data.",
        formatter_class=argparse.RawTextHelpFormatter # Use RawTextHelpFormatter for better multiline help
    )

    # --- Subparsers for commands ---
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)

    # --- Backtest Command Arguments ---
    backtest_parser = subparsers.add_parser("backtest",
                                            help="Run a single vfat strategy backtest with specific parameters.",
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    backtest_parser.add_argument("--exchange", type=str, required=True, choices=['aerodrome', 'shadow'],
                                 help="Exchange name ('aerodrome' on Base or 'shadow' on Sonic).")
    backtest_parser.add_argument("--pool-address", type=str, required=True,
                                 help="Address of the liquidity pool.")
    backtest_parser.add_argument("--width", type=float, required=True,
                                 help="Initial position width percentage (e.g., 1.0 for +/- 0.5%).")
    backtest_parser.add_argument("--rebalance-buffer", type=float, required=True,
                                 help="Rebalance trigger as % of width (e.g., 0.1 for 10%).")
    backtest_parser.add_argument("--cutoff-buffer", type=float, required=True,
                                 help="Rebalance prevention as % of width (e.g., 0.5 for 50%).\nMust be >= rebalance-buffer or 0 to disable cutoff.")
    backtest_parser.add_argument("--start-date", type=str, default=None,
                                 help="Start date (YYYY-MM-DD). Default: 90 days before end date.")
    backtest_parser.add_argument("--end-date", type=str, default=None,
                                 help="End date (YYYY-MM-DD). Default: yesterday.")
    backtest_parser.add_argument("--investment", type=float,
                                 default=config.get("backtest_defaults", {}).get("initial_investment", 10000.0),
                                 help="Initial investment amount in USD.")
    tx_group = backtest_parser.add_argument_group('Transaction Cost Overrides (Optional)')
    tx_group.add_argument("--no-tx-costs", action="store_true",
                          help="Disable simulation of gas and slippage costs.")
    tx_group.add_argument("--gas-cost", type=float, default=None,
                          help=f"Override gas cost USD per rebalance. Default: ${config.get('transaction_costs', {}).get('rebalance_gas_usd', 'N/A')}")
    tx_group.add_argument("--slippage-pct", type=float, default=None,
                          help=f"Override slippage percentage per rebalance. Default: {config.get('transaction_costs', {}).get('slippage_percentage', 'N/A')}%")
    output_group = backtest_parser.add_argument_group('Output Options')
    output_group.add_argument("--plot", action="store_true",
                              help="Generate and save result plots.")
    output_group.add_argument("--save-log", action="store_true",
                              help="Save detailed daily backtest log to CSV.")
    output_group.add_argument('-v', '--verbose', action='store_true', help="Enable DEBUG level logging.")
    backtest_parser.set_defaults(func=run_single_backtest)

    # --- Quick Command (Streamlined one-liner) ---
    quick_parser = subparsers.add_parser("quick",
                                        help="Run a backtest with minimal parameters and sensible defaults.",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    quick_parser.add_argument("exchange", type=str, choices=['aerodrome', 'shadow'], help="Exchange name.")
    quick_parser.add_argument("pool_address", type=str, help="Liquidity pool address.")
    quick_parser.add_argument("width", type=float, nargs='?', default=1.0, help="Position width percentage (default: 1.0).")
    quick_parser.add_argument("--rb", type=float, dest="rebalance_buffer", default=0.2, help="Rebalance buffer % of width (default: 0.2).")
    quick_parser.add_argument("--cb", type=float, dest="cutoff_buffer", default=0.5, help="Cutoff buffer % of width (default: 0.5). 0 disables.")
    quick_parser.add_argument("--investment", "-i", type=float, default=10000.0, help="Investment amount USD (default: 10000).")
    quick_parser.add_argument("--days", "-d", type=int, default=90, help="Days to backtest (default: 90).")
    quick_parser.add_argument("--plot", "-p", action="store_true", default=True, help="Generate plots (default: True).")
    quick_parser.add_argument("--log", "-l", action="store_true", default=True, help="Save detailed log (default: True).")
    quick_parser.add_argument("--no-tx", action="store_true", help="Disable transaction costs (default: enabled).")
    quick_parser.set_defaults(func=run_quick_backtest)

    # --- Custom Command (Interactive Mode) ---
    custom_parser = subparsers.add_parser("custom",
                                         help="Interactive custom mode for guided backtest setup.",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    custom_parser.add_argument('-v', '--verbose', action='store_true', help="Enable DEBUG level logging.")
    custom_parser.set_defaults(func=run_custom_mode)

    # --- Top USD Pairs Command ---
    top_parser = subparsers.add_parser("top-usd-pairs",
                                       help="Find top performing USD/USD pair pools (tokens contain 'usd') and strategies.",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    top_parser.add_argument("--exchange", type=str, required=True, choices=['aerodrome', 'shadow'],
                            help="Exchange name ('aerodrome' or 'shadow').")
    top_parser.add_argument("--days", "-d", type=int, default=90,
                            help="Number of past days to backtest (default: 90).")
    top_parser.add_argument("--investment", "-i", type=float, default=10000.0,
                            help="Initial investment for backtests (default: 10000).")
    top_parser.add_argument("--min-liquidity", type=float, default=10000,
                            help="Minimum liquidity threshold (USD) for considering pools (default: 10000).")
    top_parser.add_argument("--min-volume", type=float, default=1000,
                            help="Minimum daily volume threshold (USD) for considering pools (default: 1000).")
    top_parser.add_argument("--pool-limit", type=int, default=50,
                            help="Number of top pools (by TVL) to fetch and check (default: 50).")
    top_parser.add_argument("--max-workers", type=int, default=min(8, os.cpu_count() + 4),
                             help="Maximum number of parallel workers for backtesting (default: sensible max).")
    top_parser.add_argument("--plot", "-p", action="store_true", default=False,
                             help="Generate and save daily price change plots per pool and a final summary plot.")
    top_tx_group = top_parser.add_argument_group('Transaction Cost Overrides (Optional)')
    top_tx_group.add_argument("--no-tx-costs", action="store_true",
                          help="Disable simulation of gas and slippage costs.")
    top_tx_group.add_argument("--gas-cost", type=float, default=None,
                          help=f"Override gas cost USD per rebalance. Default: ${config.get('transaction_costs', {}).get('rebalance_gas_usd', 'N/A')}")
    top_tx_group.add_argument("--slippage-pct", type=float, default=None,
                           help=f"Override slippage percentage per rebalance. Default: {config.get('transaction_costs', {}).get('slippage_percentage', 'N/A')}%")
    top_parser.add_argument('-v', '--verbose', action='store_true', help="Enable DEBUG level logging.")
    top_parser.set_defaults(func=run_top_usd_pairs)


    # Parse arguments
    args = parser.parse_args()

    # --- Post-parsing Setup ---

    # Set logging level based on verbosity flag
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
             handler.setFormatter(logging.Formatter(log_format)) # Ensure format consistency
        logger.info("Verbose logging enabled (DEBUG level).")
    else:
        logging.getLogger().setLevel(logging.INFO)
        # Update formatter for existing handlers if needed
        for handler in logging.getLogger().handlers:
             handler.setFormatter(logging.Formatter(log_format))

    # Validate and parse dates for backtest command specifically
    if args.command == "backtest":
        try:
            if args.end_date is None:
                # Default to yesterday
                args.end_date = datetime.now().date() - timedelta(days=1)
            else:
                args.end_date = parse_date(args.end_date).date() # Use only the date part

            if args.start_date is None:
                default_days = config.get("backtest_defaults", {}).get("days", 90)
                args.start_date = args.end_date - timedelta(days=default_days)
            else:
                 args.start_date = parse_date(args.start_date).date() # Use only the date part

            # Ensure start_date is before end_date
            if args.start_date >= args.end_date:
                 parser.error(f"Start date ({args.start_date}) must be strictly before end date ({args.end_date}).")

            # Convert dates back to datetime objects for functions that need them
            args.start_date = datetime.combine(args.start_date, datetime.min.time())
            args.end_date = datetime.combine(args.end_date, datetime.min.time())

        except ValueError as e:
            parser.error(f"Invalid date format: {e}. Please use YYYY-MM-DD.")

        # Validate buffer percentages
        if hasattr(args, 'rebalance_buffer') and hasattr(args, 'cutoff_buffer'):
             if args.cutoff_buffer < args.rebalance_buffer and args.cutoff_buffer != 0:
                  parser.error(f"Cutoff buffer ({args.cutoff_buffer}%) must be >= rebalance buffer ({args.rebalance_buffer}%) or 0.")


    # Execute the function associated with the chosen command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        # Should not happen if command is required
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 