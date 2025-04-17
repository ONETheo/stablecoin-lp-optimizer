"""
Utility functions for generating plots related to backtesting results and market data.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any

# Attempt relative import first, fallback to absolute if run as script/different context
try:
    from .helpers import ensure_dir_exists, get_strategy_description
except ImportError:
    # This fallback might be needed if plotting.py is run standalone or imported differently
    from src.utils.helpers import ensure_dir_exists, get_strategy_description

import logging
logger = logging.getLogger(__name__)


def plot_daily_volatility(price_data: pd.DataFrame, pair_name: str, window: int, output_dir: str, filename: str):
    """Calculates and plots the absolute daily percentage price change (volatility)."""
    ensure_dir_exists(output_dir)
    output_path = os.path.join(output_dir, filename)

    if 'price' not in price_data.columns:
        logger.error(f"'price' column not found in data for {pair_name}. Skipping daily volatility plot.")
        return

    if price_data['price'].isnull().all():
        logger.warning(f"Price data for {pair_name} is all null. Skipping daily volatility plot.")
        return

    # Calculate absolute daily percentage change
    # Multiply by 100 to display as percentage
    daily_vol = price_data['price'].pct_change().abs() * 100

    # Handle potential NaN for the first entry
    daily_vol = daily_vol.fillna(0)

    plt.style.use('seaborn-v0_8-darkgrid') # Use a visually appealing style
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(price_data.index, daily_vol, label=f'Daily Volatility (%)', color='#FF7F0E', linewidth=1.5) # Use a distinct color

    # Add titles and labels
    ax.set_title(f'Daily Price Volatility for {pair_name}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Absolute Daily Price Change (%)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Format Y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}%'))

    # Improve date formatting on X-axis
    fig.autofmt_xdate()

    # Add text annotation for average daily volatility
    avg_vol = daily_vol.mean()
    ax.text(0.02, 0.95, f'Avg Daily Vol: {avg_vol:.3f}%', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    plt.tight_layout() # Adjust layout

    try:
        plt.savefig(output_path, dpi=150)
        logger.info(f"Daily volatility plot saved to: {output_path}")
        print(f"📈 Daily volatility plot saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save daily volatility plot {output_path}: {e}")
    finally:
        plt.close(fig) # Close the plot to free memory


def plot_backtest_summary(
    results: List[Dict[str, Any]],
    output_dir: str,
    filename: str,
    exchange_name: str = "Exchange",
    days: int = 90
):
    """
    Plots a summary bar chart of the best backtest results per pool.

    Args:
        results (List[Dict[str, Any]]): List of result dictionaries,
                                         sorted by performance (best first).
        output_dir (str): Directory to save the plot.
        filename (str): Name for the output image file (e.g., 'backtest_summary.png').
        exchange_name (str): Name of the exchange for the title.
        days (int): Number of days backtested for the title.
    """
    ensure_dir_exists(output_dir)
    output_path = os.path.join(output_dir, filename)

    if not results:
        logger.warning("No backtest results provided for summary plot.")
        print("⚠️ No backtest results to plot.")
        return

    # Prepare data for plotting
    pool_names = [r.get('pool_name', f"Pool_{i}") for i, r in enumerate(results)]
    net_aprs = [r.get('net_apr', 0.0) for r in results]
    strategies = [get_strategy_description(r.get('strategy', {})) for r in results]

    # Create figure and axes
    try:
        fig, ax = plt.subplots(figsize=(max(10, len(pool_names) * 1.5), 8)) # Dynamic width

        # Create bar chart
        colors = ['#4CAF50' if apr >= 0 else '#F44336' for apr in net_aprs] # Green/Red colors
        bars = ax.bar(pool_names, net_aprs, color=colors)

        # Add labels and title
        ax.set_ylabel('Net APR (%)')
        ax.set_title(f'{exchange_name.capitalize()} - Best Strategy Net APR per USD/USD Pool ({days}-Day Backtest)')
        ax.set_xticks(np.arange(len(pool_names))) # Use np.arange for consistent spacing
        ax.set_xticklabels(pool_names, rotation=45, ha='right', fontsize=9) # Rotate labels for readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add APR value labels and strategy info on top of bars
        for bar, strategy in zip(bars, strategies):
            height = bar.get_height()
            label_y_pos = height + (1 if height >=0 else -1) # Position label above/below bar
            va_align = 'bottom' if height >= 0 else 'top'
            ax.annotate(f'{height:.2f}%\n({strategy})', # Use newline for strategy
                        xy=(bar.get_x() + bar.get_width() / 2, label_y_pos),
                        xytext=(0, 0),  # No offset needed now
                        textcoords="offset points",
                        ha='center', va=va_align, fontsize=8, linespacing=0.9)

        # Adjust y-limits for better visualization
        min_apr = min(net_aprs) if net_aprs else 0
        max_apr = max(net_aprs) if net_aprs else 0
        padding = max(abs(min_apr * 0.1), abs(max_apr * 0.1), 5) # Add padding
        ax.set_ylim(min_apr - padding, max_apr + padding * 2) # Extra top padding for labels

        plt.tight_layout(rect=[0, 0.05, 1, 0.97]) # Adjust layout to prevent label cutoff
        plt.savefig(output_path, bbox_inches='tight') # Use bbox_inches='tight'
        plt.close() # Close plot
        logger.info(f"Backtest summary plot saved to: {output_path}")
        print(f"✅ Backtest summary plot saved to: {output_path}")
    except Exception as e:
         logger.error(f"Failed to generate or save backtest summary plot: {e}", exc_info=True)
         print(f"❌ Error generating backtest summary plot: {e}")

# Example usage (for testing purposes if run directly)
if __name__ == '__main__':
    # Create dummy data
    dummy_results = [
        {'pool_name': 'USDC.e/USDT', 'net_apr': 5.75, 'strategy': {'width': 0.05, 'rebalance_buffer': 0.1, 'cutoff_buffer': 0.2}},
        {'pool_name': 'scUSD/USDC.e', 'net_apr': -2.10, 'strategy': {'width': 0.1, 'rebalance_buffer': 0.2, 'cutoff_buffer': 0.4}},
        {'pool_name': 'axlUSDC/USDC.e', 'net_apr': 8.30, 'strategy': {'width': 0.02, 'rebalance_buffer': 0, 'cutoff_buffer': 0}},
        {'pool_name': 'USDV/USDC.e', 'net_apr': 1.50, 'strategy': {'width': 0.2, 'rebalance_buffer': 0.1, 'cutoff_buffer': 0.2}},
    ]
    dummy_output_dir = 'output'

    # Test backtest summary plot
    plot_backtest_summary(dummy_results, dummy_output_dir, 'dummy_backtest_summary.png', 'Shadow', 90)

    # Test volatility plot (requires a dummy price DataFrame)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = 1 + (np.random.randn(100) * 0.001).cumsum() # Simulate stablecoin price fluctuations
    dummy_price_df = pd.DataFrame({'price': prices}, index=dates)
    dummy_price_df['timestamp'] = dummy_price_df.index # Add timestamp column if needed

    plot_daily_volatility(dummy_price_df, 'DUMMY/PAIR', 30, dummy_output_dir, 'dummy_volatility.png')

    print("Dummy plots generated in 'output' directory (if matplotlib is installed).") 