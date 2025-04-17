"""
Data Module

This module provides data loading utilities for the
LP Optimizer backtesting engine, primarily fetching from subgraphs.
"""

from .data_loader import DataLoader, load_data

__all__ = [
    'DataLoader',
    'load_data'
] 