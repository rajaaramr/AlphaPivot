# utils/kite_session.py
"""
Kite Connect Session Manager for the AlphaPivot Trading System.

This module provides a centralized function for loading an authenticated
KiteConnect session from the main `config.ini` file.
"""
from __future__ import annotations

from .configs import get_config_parser
from kiteconnect import KiteConnect

def load_kite() -> KiteConnect:
    """
    Loads an authenticated KiteConnect session from the config file.

    This function reads the `[kite]` section of the `config.ini` file to
    get the API key and access token, and then returns an authenticated
    KiteConnect object.

    Returns:
        An authenticated KiteConnect session object.

    Raises:
        KeyError: If the `api_key` or `access_token` is missing from the
                  `[kite]` section of the configuration file.
    """
    config = get_config_parser()
    if "kite" not in config:
        raise KeyError("Missing [kite] section in config.ini")

    try:
        api_key = config["kite"]["api_key"]
        access_token = config["kite"]["access_token"]
    except KeyError as e:
        raise KeyError(f"Missing required key in [kite] section: {e}")

    if not api_key or not access_token or "your_" in api_key:
        raise ValueError("API key or access token not configured in config.ini")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite