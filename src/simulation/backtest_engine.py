"""
Backtesting Engine Module

Simulates LP strategies based on historical data, focusing on the
vfat.io rebalancing approach.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

# Use absolute imports
from src.utils.helpers import safe_divide, calculate_volatility, format_percentage, format_currency
from src.simulation import config as cfg # Use config from simulation package

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Engine for backtesting LP strategies, specifically the vfat rebalancing model.
    Assumes input historical_data has daily frequency and necessary columns.
    """

    def __init__(
        self,
        pool_details: Dict[str, Any],
        historical_data: pd.DataFrame,
        config: Dict[str, Any] = None,
        gas_cost_usd: Optional[float] = None, # Allow overriding config via args
        slippage_pct: Optional[float] = None  # Allow overriding config via args
    ):
        """
        Initialize the backtesting engine.

        Args:
            pool_details (Dict): Dictionary containing pool metadata (id, feeTier, tokens, etc.).
                                 Must include 'feeTier'.
            historical_data (pd.DataFrame): DataFrame with historical price, volume, TVL, fees.
                                            Required columns: timestamp, price, volumeUSD, tvlUSD, feesUSD.
                                            Assumed to be sorted by timestamp and have daily frequency.
            config (Dict, optional): Configuration dictionary. Defaults to config from config module.
            gas_cost_usd (Optional[float]): Override for gas cost per rebalance.
            slippage_pct (Optional[float]): Override for slippage percentage per rebalance.
        """
        self.pool_details = pool_details
        self.historical_data = historical_data # Assumes already sorted and processed
        self.config = config or cfg.get_config()

        # Validate input data
        required_cols = ['timestamp', 'price', 'volumeUSD', 'tvlUSD', 'feesUSD']
        if not all(col in self.historical_data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.historical_data.columns]
            logger.error(f"Historical data is missing required columns: {missing}")
            raise ValueError(f"Historical data missing required columns: {missing}")
        if self.historical_data.empty:
            logger.error("Historical data provided to BacktestEngine is empty.")
            raise ValueError("Historical data cannot be empty.")
        if not isinstance(self.historical_data['timestamp'].iloc[0], pd.Timestamp):
             try:
                 # Attempt conversion if not already datetime objects
                 self.historical_data['timestamp'] = pd.to_datetime(self.historical_data['timestamp'])
                 logger.warning("Converted 'timestamp' column to datetime objects.")
             except Exception as e:
                  logger.error(f"Failed to convert 'timestamp' column to datetime objects: {e}")
                  raise ValueError("Timestamp column must contain datetime objects.")

        if not self.pool_details or 'feeTier' not in self.pool_details:
             logger.error("Pool details dictionary must be provided and include 'feeTier'.")
             raise ValueError("Pool details must include 'feeTier'.")

        # Extract key pool parameters
        try:
            # Fee tier from basis points (e.g., 500 = 0.05%) to decimal (0.0005)
            # Accepts float values for basis points now.
            fee_tier_str = str(self.pool_details.get('feeTier', '0')) # Ensure it's a string
            self.fee_tier_bps = float(fee_tier_str)
            if self.fee_tier_bps < 0:
                 logger.warning(f"Pool fee tier is negative ({self.fee_tier_bps}). Using 0. Fee calculations might be incorrect.")
                 self.fee_tier_bps = 0.0

            if self.fee_tier_bps == 0:
                 logger.warning(f"Pool fee tier is {self.fee_tier_bps}. Fee calculations might be incorrect.")
                 self.fee_tier_decimal = 0.0
            else:
                 # The formula remains the same: bps -> percentage -> decimal fee rate
                 # e.g., 5 bps -> 0.05% -> 0.0005 fee rate
                 # e.g., 0.5 bps -> 0.005% -> 0.00005 fee rate
                 self.fee_tier_decimal = self.fee_tier_bps / 10000 # Convert bps to decimal fee rate

            # Use f-string with formatting for potential float bps
            logger.info(f"Using fee tier: {self.fee_tier_bps:.2f} bps ({self.fee_tier_decimal:.6f})")
        except ValueError:
             # Catch error if the fee tier string cannot be converted to float
             logger.error(f"Invalid feeTier value in pool_details: {self.pool_details.get('feeTier')}")
             raise ValueError(f"Invalid feeTier value: '{self.pool_details.get('feeTier')}' cannot be converted to float.")


        # Transaction cost settings (use overrides if provided, else use config defaults)
        self.tx_costs_config = self.config.get("transaction_costs", {})
        self.gas_cost_usd = gas_cost_usd if gas_cost_usd is not None else self.tx_costs_config.get("rebalance_gas_usd", 2.0)
        # Slippage percentage is expected as e.g., 0.05 for 0.05%
        self.slippage_pct = slippage_pct if slippage_pct is not None else self.tx_costs_config.get("slippage_percentage", 0.05)
        self.slippage_decimal = self.slippage_pct / 100.0 # Convert percentage to decimal for calculations

        logger.info(f"Transaction cost settings: Gas=${self.gas_cost_usd:.2f}, Slippage={self.slippage_pct:.3f}% ({self.slippage_decimal:.5f})")


    def run_vfat_backtest(
        self,
        initial_width_pct: float,
        rebalance_buffer_pct: float,
        cutoff_buffer_pct: float,
        initial_investment: float = 10000.0,
        simulate_tx_costs: bool = True
    ) -> Dict[str, Any]:
        """
        Run a backtest simulating the vfat.io rebalancing strategy.

        Args:
            initial_width_pct (float): The initial width of the LP position relative to the starting price
                                       (e.g., 1.0 for +/- 0.5% around the price).
            rebalance_buffer_pct (float): The percentage *of the initial width* beyond the bounds
                                          that triggers a rebalance (e.g., 0.1 for 10%).
            cutoff_buffer_pct (float): The percentage *of the initial width* beyond the bounds
                                       where rebalancing stops (e.g., 0.5 for 50%). Set to 0 to disable cutoff.
            initial_investment (float): Initial investment amount in USD.
            simulate_tx_costs (bool): Whether to include transaction costs in the simulation.

        Returns:
            Dict[str, Any]: Dictionary containing backtest results, metrics, and parameters,
                            or an error dictionary if setup fails.
        """
        if self.historical_data.empty:
            logger.error("Cannot run backtest with empty historical data.")
            return {"error": "Empty historical data"}
        if not (initial_width_pct > 0 and rebalance_buffer_pct >= 0 and (cutoff_buffer_pct >= rebalance_buffer_pct or cutoff_buffer_pct == 0)):
             error_msg = "Invalid strategy parameters: width must be > 0, buffer >= 0, cutoff >= buffer or cutoff = 0."
             logger.error(error_msg + f" Got: width={initial_width_pct}, rebal={rebalance_buffer_pct}, cutoff={cutoff_buffer_pct}")
             return {"error": error_msg}


        logger.info(f"Starting vfat backtest: Width={initial_width_pct}%, RebalanceBuffer={rebalance_buffer_pct}%, CutoffBuffer={cutoff_buffer_pct}%")
        logger.info(f"Initial Investment: ${initial_investment:.2f}, Simulate Tx Costs: {simulate_tx_costs}")

        # --- Initialization ---
        results_log = []
        position_value = initial_investment # Tracks the value of the underlying assets
        cumulative_fees_earned = 0.0
        total_gas_costs = 0.0
        total_slippage_costs = 0.0
        rebalance_count = 0
        days_in_position = 0
        total_days = 0

        start_price = self.historical_data['price'].iloc[0]
        if start_price <= 0:
             logger.error(f"Invalid starting price: {start_price:.4f}. Cannot initialize position.")
             return {"error": "Invalid starting price (<= 0)"}

        # Calculate initial position bounds based on width percentage
        # Width is total range, so half_width is distance from center price
        half_width_multiplier = (initial_width_pct / 100.0) / 2.0
        lower_bound = start_price * (1 - half_width_multiplier)
        upper_bound = start_price * (1 + half_width_multiplier)
        current_width_abs = upper_bound - lower_bound # Absolute price width

        if lower_bound <= 0 or current_width_abs <= 0:
             logger.error(f"Invalid initial position bounds calculated: [{lower_bound:.4f}, {upper_bound:.4f}]. Check width and start price.")
             return {"error": "Invalid initial position bounds (lower <= 0 or width <= 0)"}

        logger.info(f"Initial Position: Price={start_price:.4f}, Lower={lower_bound:.4f}, Upper={upper_bound:.4f}, Width={current_width_abs:.4f} ({initial_width_pct}%)")

        # --- Simulation Loop ---
        for i, row in self.historical_data.iterrows():
            current_price = row['price']
            daily_volume = row['volumeUSD']
            daily_tvl = row['tvlUSD']
            daily_pool_fees = row['feesUSD']
            timestamp = row['timestamp']
            total_days += 1

            # Basic check for valid price data
            if pd.isna(current_price) or current_price <= 0:
                 logger.warning(f"Skipping day {timestamp.date()} due to invalid price: {current_price}")
                 # Log state without updates for this day? Or just skip? Skipping for now.
                 # Need to decide how to handle position value if price is invalid. Assume it holds previous value?
                 # Let's log the previous state but mark as invalid price day
                 if results_log:
                      last_log = results_log[-1].copy()
                      last_log['timestamp'] = timestamp
                      last_log['price'] = np.nan # Mark price as invalid
                      last_log['in_position'] = False # Assume out of position if price is unknown
                      last_log['daily_fees_earned'] = 0
                      last_log['rebalanced'] = False
                      last_log['rebalance_cutoff'] = False
                      last_log['gas_cost'] = 0
                      last_log['slippage_cost'] = 0
                      # Keep cumulative fees and position value from previous day
                      results_log.append(last_log)
                 continue # Skip to next day


            # Check if position is active (price within bounds)
            in_position = lower_bound <= current_price <= upper_bound

            daily_fees_earned_new = 0
            if in_position:
                days_in_position += 1
                # --- New Fee Calculation based on LP Share of Pool Fees ---
                if not pd.isna(daily_pool_fees) and not pd.isna(daily_tvl) and daily_tvl > 0:
                     lp_share = position_value / daily_tvl # Calculate LP's share of TVL
                     # Ensure lp_share is not negative or absurdly large (sanity check)
                     lp_share = max(0, min(lp_share, 1.0)) # Clamp share between 0 and 1
                     daily_fees_earned_new = lp_share * daily_pool_fees
                     cumulative_fees_earned += daily_fees_earned_new
                     # logger.debug(f"{timestamp.date()}: In Position. TVL={daily_tvl:.0f}, PosVal={position_value:.0f}, Share={lp_share:.4%}, PoolFees={daily_pool_fees:.2f}, Earned={daily_fees_earned_new:.2f}")
                else:
                    # logger.debug(f"{timestamp.date()}: In Position but missing TVL ({daily_tvl}) or Pool Fees ({daily_pool_fees}). Cannot calculate fees.")
                    pass # Cannot calculate fees if TVL or Pool Fees are missing/invalid
            # --- End New Fee Calculation ---
            else:
                 # logger.debug(f"{timestamp.date()}: Out of Position. Price={current_price:.4f}, Bounds=[{lower_bound:.4f}, {upper_bound:.4f}]")
                 pass

            # --- Rebalancing Logic ---
            rebalance_triggered = False
            rebalance_cutoff = False
            gas_cost_today = 0
            slippage_cost_today = 0

            # Calculate buffer thresholds in absolute price terms based on the *current* width
            # The vfat strategy resets to the *initial* width percentage, so maybe base buffers on initial width concept?
            # Let's recalculate the target width based on initial % and current price for buffer calc? No, use current bounds' width.
            rebalance_trigger_dist = current_width_abs * rebalance_buffer_pct
            cutoff_trigger_dist = current_width_abs * cutoff_buffer_pct if cutoff_buffer_pct > 0 else float('inf')

            # Check bounds breach
            lower_rebalance_price = lower_bound - rebalance_trigger_dist
            lower_cutoff_price = lower_bound - cutoff_trigger_dist
            upper_rebalance_price = upper_bound + rebalance_trigger_dist
            upper_cutoff_price = upper_bound + cutoff_trigger_dist

            if current_price < lower_rebalance_price:
                rebalance_triggered = True
                logger.debug(f"{timestamp.date()}: Price {current_price:.4f} < Lower Rebalance Threshold ({lower_rebalance_price:.4f})")
                if cutoff_buffer_pct > 0 and current_price < lower_cutoff_price:
                    rebalance_cutoff = True
                    logger.debug(f"{timestamp.date()}: Price {current_price:.4f} < Lower Cutoff Threshold ({lower_cutoff_price:.4f}) - Rebalance Prevented")

            elif current_price > upper_rebalance_price:
                rebalance_triggered = True
                logger.debug(f"{timestamp.date()}: Price {current_price:.4f} > Upper Rebalance Threshold ({upper_rebalance_price:.4f})")
                if cutoff_buffer_pct > 0 and current_price > upper_cutoff_price:
                    rebalance_cutoff = True
                    logger.debug(f"{timestamp.date()}: Price {current_price:.4f} > Upper Cutoff Threshold ({upper_cutoff_price:.4f}) - Rebalance Prevented")


            # Perform rebalance if triggered and not cut off
            if rebalance_buffer_pct > 0 and rebalance_triggered and not rebalance_cutoff:
                rebalance_count += 1
                logger.info(f"Rebalancing triggered on {timestamp.date()} at price {current_price:.4f}. Old bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")

                # Apply transaction costs BEFORE calculating new bounds based on reduced capital
                if simulate_tx_costs:
                    gas_cost_today = self.gas_cost_usd
                    # Slippage cost is % of the value being rebalanced (approximated by current position value)
                    slippage_cost_today = position_value * self.slippage_decimal
                    position_value -= (gas_cost_today + slippage_cost_today) # Deduct costs from principal
                    total_gas_costs += gas_cost_today
                    total_slippage_costs += slippage_cost_today
                    logger.debug(f"Applied Tx Costs: Gas=${gas_cost_today:.2f}, Slippage=${slippage_cost_today:.2f}. New Position Value: ${position_value:.2f}")

                    # Check if position value depleted
                    if position_value <= 0:
                         logger.warning(f"Position value depleted to {position_value:.2f} after transaction costs on {timestamp.date()}. Stopping simulation.")
                         # Log final state before breaking
                         results_log.append({
                             "timestamp": timestamp, "price": current_price, "lower_bound": lower_bound, "upper_bound": upper_bound,
                             "in_position": False, "daily_fees_earned": 0, "cumulative_fees": cumulative_fees_earned,
                             "rebalanced": True, "rebalance_cutoff": False, "gas_cost": gas_cost_today, "slippage_cost": slippage_cost_today,
                             "position_value": position_value
                         })
                         break # Stop simulation

                # Reset position around the current price using the *initial* width percentage
                center_price = current_price # New center price
                half_width_multiplier = (initial_width_pct / 100.0) / 2.0
                lower_bound = center_price * (1 - half_width_multiplier)
                upper_bound = center_price * (1 + half_width_multiplier)
                current_width_abs = upper_bound - lower_bound # Update current width based on new bounds

                # Validate new bounds
                if lower_bound <= 0 or current_width_abs <= 0:
                     logger.error(f"Invalid position bounds calculated after rebalance: [{lower_bound:.4f}, {upper_bound:.4f}]. Stopping simulation.")
                     # Log state before breaking
                     results_log.append({
                         "timestamp": timestamp, "price": current_price, "lower_bound": lower_bound, "upper_bound": upper_bound,
                         "in_position": False, "daily_fees_earned": 0, "cumulative_fees": cumulative_fees_earned,
                         "rebalanced": True, "rebalance_cutoff": False, "gas_cost": gas_cost_today, "slippage_cost": slippage_cost_today,
                         "position_value": position_value
                     })
                     break # Stop simulation

                logger.info(f"New Position after rebalance: Lower={lower_bound:.4f}, Upper={upper_bound:.4f}, Width={current_width_abs:.4f}")


            # Log daily state
            results_log.append({
                "timestamp": timestamp,
                "price": current_price,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "in_position": in_position,
                "daily_fees_earned": daily_fees_earned_new,
                "cumulative_fees": cumulative_fees_earned,
                "rebalanced": rebalance_triggered,
                "rebalance_cutoff": rebalance_cutoff,
                "gas_cost": gas_cost_today,
                "slippage_cost": slippage_cost_today,
                "position_value": position_value
            })

        # --- Final Calculations ---
        if not results_log:
             logger.error("No results were logged during the simulation.")
             return {"error": "Simulation produced no results."}

        results_df = pd.DataFrame(results_log)
        # Ensure timestamp column is datetime
        results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])


        final_value_assets = results_df['position_value'].iloc[-1] if not results_df.empty else initial_investment
        final_total_value = final_value_assets + cumulative_fees_earned # Total value including earned fees

        total_tx_costs = total_gas_costs + total_slippage_costs
        # Net profit considers starting capital, final asset value, fees earned, and tx costs
        net_profit = final_total_value - initial_investment - total_tx_costs
        # Gross profit considers value change + fees, but ignores tx costs
        gross_profit = final_total_value - initial_investment

        # Use total_days derived from loop, or calculate from DataFrame timestamps
        if len(results_df['timestamp']) > 1:
             duration_timedelta = results_df['timestamp'].iloc[-1] - results_df['timestamp'].iloc[0]
             # Add 1 day because the duration includes start and end day
             duration_days = duration_timedelta.days + 1
        elif len(results_df['timestamp']) == 1:
             duration_days = 1
        else:
             duration_days = 0

        if duration_days <= 0:
             logger.warning("Calculated duration is zero or negative. APRs will be zero.")
             duration_days = 1 # Avoid division by zero, but APRs will be misleading

        # APR Calculations (Annualized)
        # Gross APR (includes fees and asset value change, before tx costs)
        gross_apr = (gross_profit / initial_investment) * (365.0 / duration_days) * 100 if duration_days > 0 else 0
        # Net APR (includes fees, asset value change, after tx costs)
        net_apr = (net_profit / initial_investment) * (365.0 / duration_days) * 100 if duration_days > 0 else 0
        # Fees APR (only fees earned relative to initial investment, before tx costs)
        fees_apr = (cumulative_fees_earned / initial_investment) * (365.0 / duration_days) * 100 if duration_days > 0 else 0

        time_in_position_pct = safe_divide(days_in_position, duration_days) * 100 if duration_days > 0 else 0

        # Calculate price volatility over the actual period simulated
        period_prices = results_df['price'].dropna()
        vol = calculate_volatility(period_prices, window=len(period_prices), annualize=False) * 100 if len(period_prices) > 1 else 0.0 # Period volatility %

        metrics = {
            "initial_investment": initial_investment,
            "final_position_value": final_value_assets, # Value of assets at end
            "final_value_incl_fees": final_total_value, # Asset value + fees earned
            "total_fees_earned": cumulative_fees_earned,
            "total_gas_costs": total_gas_costs,
            "total_slippage_costs": total_slippage_costs,
            "total_tx_costs": total_tx_costs,
            "gross_profit": gross_profit,
            "net_profit": net_profit,
            "gross_apr": gross_apr, # %
            "net_apr": net_apr,     # %
            "fees_apr": fees_apr,   # %
            "rebalance_count": rebalance_count,
            "rebalance_frequency": safe_divide(rebalance_count, duration_days), # Rebalances per day
            "duration_days": duration_days,
            "time_in_position_days": days_in_position,
            "time_in_position_pct": time_in_position_pct, # %
            "volatility_pct": vol, # % (for the period)
        }

        parameters = {
            "pool_address": self.pool_details.get('id', 'N/A'),
            "exchange": self.pool_details.get('exchange', self.config.get('exchange_name', 'N/A')), # Add exchange name if available
            "token0": self.pool_details.get('token0', {}).get('symbol', 'T0'),
            "token1": self.pool_details.get('token1', {}).get('symbol', 'T1'),
            "fee_tier_bps": self.fee_tier_bps,
            "initial_width_pct": initial_width_pct,
            "rebalance_buffer_pct": rebalance_buffer_pct,
            "cutoff_buffer_pct": cutoff_buffer_pct,
            "simulate_tx_costs": simulate_tx_costs,
            "gas_cost_usd": self.gas_cost_usd if simulate_tx_costs else 0,
            "slippage_pct": self.slippage_pct if simulate_tx_costs else 0,
            "start_date": results_df['timestamp'].iloc[0].date() if not results_df.empty else 'N/A',
            "end_date": results_df['timestamp'].iloc[-1].date() if not results_df.empty else 'N/A',
        }

        logger.info(f"Backtest complete. Net APR: {metrics['net_apr']:.2f}%, Fees APR: {metrics['fees_apr']:.2f}%, Rebalances: {metrics['rebalance_count']}")

        return {
            "parameters": parameters,
            "metrics": metrics,
            "results_log": results_df # DataFrame of daily logs
        }

# Removed the __main__ block that used mock data.
# Testing should be done via dedicated test files or by running the main CLI script.