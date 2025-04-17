"""
API Module

Handles interaction with external data sources, primarily TheGraph subgraphs.
"""
from .subgraph_client import get_client

__all__ = ['get_client']