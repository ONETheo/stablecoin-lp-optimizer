"""
Simulation Module

Contains the core backtesting engine, result processing,
and configuration for LP strategy simulations.
"""
from .backtest_engine import BacktestEngine
from .result_processor import ResultProcessor
# Removed run_backtest_suite as optimization is deferred
# from .backtest_suite import run_backtest_suite

__all__ = [
    'BacktestEngine',
    'ResultProcessor',
    # 'run_backtest_suite'
]