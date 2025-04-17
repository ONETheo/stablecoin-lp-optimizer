"""
Helper Functions Module

This module provides utility functions for date handling, data processing,
and other common operations used across the USD* rewards backtesting engine.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Union, Tuple, Dict, Any, Optional
import pickle

# Set up logger
logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> datetime:
    """
    Parse a date string into a datetime object.
    
    Args:
        date_str (str): Date string in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
        
    Returns:
        datetime: Parsed datetime object
    """
    try:
        # Try different date formats
        if len(date_str) <= 10:  # Just date
            return datetime.strptime(date_str, '%Y-%m-%d')
        else:  # Date and time
            return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            # Try ISO format
            return datetime.fromisoformat(date_str)
        except ValueError as e:
            logger.error(f"Could not parse date string: {date_str}")
            raise ValueError(f"Invalid date format: {date_str}. Expected 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.") from e


def date_range(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    interval: str = '1h'
) -> List[datetime]:
    """
    Generate a list of datetimes between start_date and end_date at specified interval.
    
    Args:
        start_date (Union[str, datetime]): Start date
        end_date (Union[str, datetime]): End date
        interval (str, optional): Time interval ('1h', '1d', '15m', etc.). Defaults to '1h'.
        
    Returns:
        List[datetime]: List of datetime objects
    """
    # Parse dates if needed
    if isinstance(start_date, str):
        start_date = parse_date(start_date)
    if isinstance(end_date, str):
        end_date = parse_date(end_date)
        
    # Convert interval to timedelta
    if interval == '1h':
        delta = timedelta(hours=1)
    elif interval == '1d':
        delta = timedelta(days=1)
    elif interval == '15m':
        delta = timedelta(minutes=15)
    elif interval == '4h':
        delta = timedelta(hours=4)
    else:
        raise ValueError(f"Unsupported interval: {interval}. Supported: '15m', '1h', '4h', '1d'")
    
    # Generate timestamps
    timestamps = []
    current = start_date
    while current <= end_date:
        timestamps.append(current)
        current += delta
        
    return timestamps


def format_timestamp(timestamp: datetime, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Format a datetime object as a string.
    
    Args:
        timestamp (datetime): Datetime to format
        format_str (str, optional): Format string. Defaults to '%Y-%m-%d %H:%M:%S'.
        
    Returns:
        str: Formatted timestamp string
    """
    return timestamp.strftime(format_str)


def interpolate_missing_values(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    value_cols: List[str] = None,
    method: str = 'linear'
) -> pd.DataFrame:
    """
    Interpolate missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with possibly missing values
        timestamp_col (str, optional): Name of timestamp column. Defaults to 'timestamp'.
        value_cols (List[str], optional): Columns to interpolate. Defaults to all columns except timestamp_col.
        method (str, optional): Interpolation method ('linear', 'time', etc.). Defaults to 'linear'.
        
    Returns:
        pd.DataFrame: DataFrame with interpolated values
    """
    if df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Set index to timestamp for interpolation
    if timestamp_col in result_df.columns:
        result_df = result_df.set_index(timestamp_col)
    
    # Determine which columns to interpolate
    if value_cols is None:
        value_cols = [col for col in result_df.columns if col != timestamp_col]
    
    # Interpolate missing values
    for col in value_cols:
        if col in result_df.columns:
            result_df[col] = result_df[col].interpolate(method=method)
    
    # Reset index if we set it to timestamp
    if timestamp_col in df.columns:
        result_df = result_df.reset_index()
    
    return result_df


def moving_average(
    series: Union[List[float], np.ndarray, pd.Series],
    window: int = 7
) -> np.ndarray:
    """
    Calculate the moving average of a series.
    
    Args:
        series (Union[List[float], np.ndarray, pd.Series]): Data series
        window (int, optional): Window size. Defaults to 7.
        
    Returns:
        np.ndarray: Moving average values
    """
    if isinstance(series, list):
        series = np.array(series)
    elif isinstance(series, pd.Series):
        series = series.values
    
    if len(series) < window:
        # Not enough data, return original series
        return series
    
    # Calculate moving average
    weights = np.repeat(1.0, window) / window
    return np.convolve(series, weights, 'valid')


def calculate_volatility(
    prices: Union[List[float], np.ndarray, pd.Series],
    window: int = 30,
    annualize: bool = True,
    trading_periods: int = 365
) -> float:
    """
    Calculate price volatility (standard deviation of returns).
    
    Args:
        prices (Union[List[float], np.ndarray, pd.Series]): Price series
        window (int, optional): Window size for rolling volatility. Defaults to 30.
        annualize (bool, optional): Whether to annualize the volatility. Defaults to True.
        trading_periods (int, optional): Number of trading periods in a year. Defaults to 365.
        
    Returns:
        float: Volatility measure
    """
    if isinstance(prices, list):
        prices = np.array(prices)
    elif isinstance(prices, pd.Series):
        prices = prices.values
    
    if len(prices) < 2:
        return 0.0
    
    # Calculate returns
    returns = np.diff(prices) / prices[:-1]
    
    # Calculate volatility
    if len(returns) < window:
        volatility = np.std(returns)
    else:
        volatility = np.std(returns[-window:])
    
    # Annualize if requested
    if annualize:
        volatility = volatility * np.sqrt(trading_periods)
    
    return volatility


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide numbers, returning a default value if denominator is zero.
    
    Args:
        numerator (float): Numerator
        denominator (float): Denominator
        default (float, optional): Default value to return if denominator is zero. Defaults to 0.0.
        
    Returns:
        float: Division result or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except ZeroDivisionError:
        return default


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a number as a percentage string.
    
    Args:
        value (float): Number to format (e.g., 0.156 for 15.6%)
        decimals (int, optional): Number of decimal places. Defaults to 2.
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2, currency_symbol: str = '$') -> str:
    """
    Format a number as a currency string.
    
    Args:
        value (float): Number to format
        decimals (int, optional): Number of decimal places. Defaults to 2.
        currency_symbol (str, optional): Currency symbol. Defaults to '$'.
        
    Returns:
        str: Formatted currency string
    """
    return f"{currency_symbol}{value:,.{decimals}f}"


def ensure_dir_exists(directory_path: str):
    """Checks if a directory exists and creates it if not."""
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            logger.info(f"Created directory: {directory_path}")
        except OSError as e:
            logger.error(f"Failed to create directory {directory_path}: {e}")
            raise # Re-raise the exception if directory creation fails


def configure_logging(level=logging.INFO):
    """Basic logging configuration (placeholder)."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=log_format)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("python_graphql_client").setLevel(logging.WARNING)
    # Use the logger instance defined at the module level
    logger.info(f"Logging configured with level: {logging.getLevelName(level)}")


def get_strategy_description(strategy: Dict[str, Any]) -> str:
    """Generates a short description string for a strategy (placeholder)."""
    if not strategy:
        return "N/A"
    width = strategy.get('width', '?')
    rb = strategy.get('rebalance_buffer', '?')
    cb = strategy.get('cutoff_buffer', '?')
    return f"{width}%/{rb}%/{cb}%"


# --- Added Cache Functions START ---
def save_cache(data: Any, cache_file: str):
    """Saves data to a cache file using pickle."""
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        logger.debug(f"Data successfully cached to {cache_file}")
    except Exception as e:
        logger.error(f"Failed to save cache to {cache_file}: {e}")

def load_cache(cache_file: str) -> Optional[Any]:
    """Loads data from a cache file using pickle."""
    if not os.path.exists(cache_file):
        logger.debug(f"Cache file not found: {cache_file}")
        return None
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        logger.debug(f"Cache successfully loaded from {cache_file}")
        return data
    except Exception as e:
        logger.error(f"Failed to load cache from {cache_file}: {e}")
        return None
# --- Added Cache Functions END ---


if __name__ == "__main__":
    # Test the helper functions
    logging.basicConfig(level=logging.INFO)
    
    # Test parse_date
    print("Testing parse_date():")
    date1 = parse_date("2023-01-01")
    date2 = parse_date("2023-01-01 12:30:45")
    print(f"  Parsed date1: {date1}")
    print(f"  Parsed date2: {date2}")
    
    # Test date_range
    print("\nTesting date_range():")
    dates = date_range("2023-01-01", "2023-01-03", "1d")
    print(f"  Date range: {[format_timestamp(d, '%Y-%m-%d') for d in dates]}")
    
    # Test interpolate_missing_values
    print("\nTesting interpolate_missing_values():")
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='D'),
        'price': [1.0, np.nan, 3.0, np.nan, 5.0]
    })
    print("  Original DataFrame:")
    print(df)
    print("  Interpolated DataFrame:")
    print(interpolate_missing_values(df))
    
    # Test calculate_volatility
    print("\nTesting calculate_volatility():")
    prices = [1.0, 1.02, 0.99, 1.05, 1.03, 1.01, 1.04]
    volatility = calculate_volatility(prices, window=7, annualize=True)
    print(f"  Volatility: {format_percentage(volatility)}")
    
    print("\nAll helper function tests completed successfully!") 