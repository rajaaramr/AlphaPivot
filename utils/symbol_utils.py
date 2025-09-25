# utils/symbol_utils.py
"""
Symbol-related utility functions for the AlphaPivot trading system.

This module provides a set of helper functions for working with trading
symbols, including formatting, validation, and generating instrument-specific
symbols.
"""
from __future__ import annotations
import os
from functools import lru_cache
from typing import Optional

from .configs import get_config_parser

# ---------- INI helpers ----------

@lru_cache(maxsize=1)
def _get_config():
    """Cached accessor for the main configuration."""
    return get_config_parser()

def get_futures_expiry_suffix() -> str:
    """
    Reads the futures expiry suffix (e.g., '25AUGFUT') from the [settings]
    section of the config.ini file.
    """
    config = _get_config()
    if "settings" in config and "fut_expiry_suffix" in config["settings"]:
        return config.get("settings", "fut_expiry_suffix").upper().strip()
    return ""

# ---------- Public API (DB-free) ----------

def get_spot_tradingsymbol(symbol: str) -> str:
    """
    Returns the canonical NSE spot symbol for a given symbol.
    """
    return (symbol or "").upper().strip()

def get_futures_tradingsymbol(symbol: str) -> Optional[str]:
    """
    Builds a futures tradingsymbol using the suffix from config.ini.

    Example: 'INFY' + '25AUGFUT' -> 'INFY25AUGFUT'
    """
    base = get_spot_tradingsymbol(symbol)
    suf = get_futures_expiry_suffix()
    if not base or not suf:
        return None
    return f"{base}{suf}"

def get_option_symbol_base(symbol: str) -> str:
    """
    Generates the base symbol for an option contract.
    """
    fut = get_futures_tradingsymbol(symbol)
    return fut or ""

def get_lot_size(symbol: str) -> int:
    """
    Placeholder for returning the lot size of a symbol.
    """
    return 1

# ---------- Convenience ----------

def is_index(symbol: str) -> bool:
    """Checks if a symbol is an index."""
    return get_spot_tradingsymbol(symbol) in {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}

def is_valid_tradingsymbol(symbol: str) -> bool:
    """Checks if a symbol is a valid futures tradingsymbol."""
    s = (symbol or "").upper().strip()
    return s.endswith("FUT") and len(s) > 10