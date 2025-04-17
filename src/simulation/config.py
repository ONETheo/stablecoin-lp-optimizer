"""
Configuration Module

Loads and provides access to application configuration settings.
"""

import os
import yaml
import logging
from typing import Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv() # Load .env file variables

# --- Default Configuration ---
DEFAULT_CONFIG = {
    "output_dir": "./output",
    "api_config": {
        "thegraph": {
            "api_key": os.getenv("SUBGRAPH_API_KEY"),
            "base": { # Network name for TheGraph endpoints
                "aerodrome": os.getenv("AERODROME_SUBGRAPH_ENDPOINT"),
                # Add other Base DEX subgraphs if needed (e.g., SushiSwap on Base)
            },
            # Add other networks like 'optimism', 'arbitrum' if needed
            # "optimism": { ... }
        },
        "shadow": { # Specific config for Shadow (assuming it uses TheGraph)
             # Provide the endpoint directly as a default if ENV var is missing
             "endpoint": os.getenv("SHADOW_SUBGRAPH_ENDPOINT", "https://gateway.thegraph.com/api/subgraphs/id/HGyx7TCqgbWieay5enLiRjshWve9TjHwiug3m66pmLGR"), 
             "backup_endpoint": os.getenv("SHADOW_BACKUP_SUBGRAPH_ENDPOINT")
        },
        # Add CoinGecko config if used as fallback (currently not implemented)
        # "coingecko": { ... }
    },
    "backtest_defaults": {
        "days": 90, # Default lookback period if start/end dates not specified
        "initial_investment": 10000.0,
    },
    "transaction_costs": {
        # Defaults, can be overridden by CLI args
        "rebalance_gas_usd": 2.0,  # Estimated gas cost per rebalance on Base L2
        "slippage_percentage": 1.0 # Estimated slippage per swap during rebalance (1.0%)
    },
    # Removed default pools - must be specified via CLI
    # "pools": { ... }
}

# --- Configuration Loading ---
_config = None

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file, falling back to defaults and environment variables.

    Args:
        config_path (str, optional): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The loaded configuration dictionary.
    """
    global _config
    if _config is not None:
        return _config

    config_data = DEFAULT_CONFIG.copy()

    # Load from YAML if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config:
                # Deep merge YAML config into defaults (simple update for now)
                # More sophisticated merging might be needed for nested dicts
                config_data.update(yaml_config)
                logger.info(f"Loaded configuration from {config_path}")
            else:
                 logger.warning(f"Configuration file {config_path} is empty. Using defaults.")
        except Exception as e:
            logger.error(f"Error loading configuration file {config_path}: {e}. Using defaults.")
    else:
        logger.info("No custom configuration file found or specified. Using default configuration and environment variables.")

    # Ensure critical API config is present, prioritizing .env over defaults if both exist
    # TheGraph API Key
    env_api_key = os.getenv("SUBGRAPH_API_KEY")
    if env_api_key:
        config_data["api_config"]["thegraph"]["api_key"] = env_api_key
    elif config_data["api_config"]["thegraph"]["api_key"] is None:
         logger.warning("SUBGRAPH_API_KEY not found in environment variables or config. Access to TheGraph Gateway might be limited.")

    # Aerodrome Endpoint
    env_aero_endpoint = os.getenv("AERODROME_SUBGRAPH_ENDPOINT")
    if env_aero_endpoint:
        config_data["api_config"]["thegraph"]["base"]["aerodrome"] = env_aero_endpoint
    elif config_data["api_config"]["thegraph"]["base"]["aerodrome"] is None:
        logger.error("AERODROME_SUBGRAPH_ENDPOINT not found in environment variables or config. Cannot query Aerodrome.")
        # Potentially raise an error here if Aerodrome is essential

    # Shadow Endpoint
    env_shadow_endpoint = os.getenv("SHADOW_SUBGRAPH_ENDPOINT")
    if env_shadow_endpoint:
        config_data["api_config"]["shadow"]["endpoint"] = env_shadow_endpoint
    elif config_data["api_config"]["shadow"]["endpoint"] is None:
        logger.warning("SHADOW_SUBGRAPH_ENDPOINT not found in environment variables or config. Shadow exchange queries will fail.")

    # Shadow Backup Endpoint
    env_shadow_backup_endpoint = os.getenv("SHADOW_BACKUP_SUBGRAPH_ENDPOINT")
    if env_shadow_backup_endpoint:
        config_data["api_config"]["shadow"]["backup_endpoint"] = env_shadow_backup_endpoint
    elif config_data["api_config"]["shadow"]["backup_endpoint"] is None:
        logger.info("SHADOW_BACKUP_SUBGRAPH_ENDPOINT not found in environment variables. Backup endpoint not available.")

    _config = config_data
    return _config

def get_config() -> Dict[str, Any]:
    """
    Returns the loaded configuration dictionary. Loads if not already loaded.
    """
    if _config is None:
        return load_config() # Load with default behavior
    return _config

# Example of accessing config values:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = get_config()
    print("Loaded Configuration Snippets:")
    print(f"Output Directory: {config.get('output_dir')}")
    print(f"TheGraph API Key Set: {'Yes' if config.get('api_config', {}).get('thegraph', {}).get('api_key') else 'No'}")
    print(f"Aerodrome Endpoint: {config.get('api_config', {}).get('thegraph', {}).get('base', {}).get('aerodrome')}")
    print(f"Shadow Endpoint: {config.get('api_config', {}).get('shadow', {}).get('endpoint')}")
    print(f"Shadow Backup Endpoint: {config.get('api_config', {}).get('shadow', {}).get('backup_endpoint')}")
    print(f"Default Backtest Days: {config.get('backtest_defaults', {}).get('days')}")
    print(f"Default Rebalance Gas Cost: ${config.get('transaction_costs', {}).get('rebalance_gas_usd')}")
    print(f"Default Slippage Percent: {config.get('transaction_costs', {}).get('slippage_percentage')}%")