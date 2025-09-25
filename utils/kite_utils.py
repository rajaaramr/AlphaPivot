# utils/kite_utils.py
"""
Kite Connect Utilities for the AlphaPivot Trading System.

This module provides a collection of helper functions for interacting with the
Kite Connect API, including authentication, instrument resolution, and data
fetching.
"""
from __future__ import annotations

import configparser
from datetime import date
from typing import Optional, Dict, List

from kiteconnect import KiteConnect

from .configs import get_config_parser
from .kite_session import load_kite

# ------------------- Local cache -------------------
_INSTR_CACHE: dict[str, Optional[List[dict]]] = {"NFO": None, "NSE": None}

def _load_instruments(kite: KiteConnect, exchange: str, force: bool = False) -> List[dict]:
    """Caches instrument lists to avoid repeated downloads."""
    exchange = exchange.upper()
    global _INSTR_CACHE
    if force or _INSTR_CACHE.get(exchange) is None:
        _INSTR_CACHE[exchange] = kite.instruments(exchange)
    return _INSTR_CACHE[exchange] or []

# ------------------- Kite Auth & Token Flow -------------------
def generate_login_url() -> str:
    """Generates a login URL for Kite Connect authentication."""
    config = get_config_parser()
    api_key = config.get("kite", "api_key")
    kite = KiteConnect(api_key=api_key)
    return kite.login_url()

def exchange_and_store_token(request_token: str) -> str:
    """
    Exchanges a request token for an access token and stores it in the config file.
    """
    config = get_config_parser()
    api_key = config.get("kite", "api_key")
    api_secret = config.get("kite", "api_secret")

    kite = KiteConnect(api_key=api_key)
    session = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session["access_token"]

    config.set("kite", "access_token", access_token)
    with open("config.ini", "w") as f:
        config.write(f)

    return access_token

# ------------------- Futures Instrument Resolver -------------------
def get_futures_instrument(symbol: str, kite: KiteConnect) -> Optional[Dict]:
    """
    Resolves the futures instrument for a given symbol.
    """
    try:
        nfo = _load_instruments(kite, "NFO")
    except Exception as e:
        print(f"❌ Error loading NFO instruments: {e}")
        return None

    config = get_config_parser()
    suffix = config.get("settings", "fut_expiry_suffix", fallback="")

    # Try config-specified suffix first
    if suffix:
        tsym = f"{symbol.upper()}{suffix}"
        for ins in nfo:
            if str(ins.get("tradingsymbol", "")).upper() == tsym:
                return {"tradingsymbol": ins["tradingsymbol"], "instrument_token": ins["instrument_token"]}

    # Fallback to nearest expiry
    today = date.today()
    expiries = sorted({
        ins.get("expiry").date() for ins in nfo
        if ins.get("name", "").upper() == symbol.upper() and ins.get("segment") == "NFO-FUT"
        and ins.get("expiry") and ins["expiry"].date() >= today
    })
    if not expiries:
        return None

    nearest_expiry = expiries[0]
    for ins in nfo:
        if (ins.get("name", "").upper() == symbol.upper() and
            ins.get("segment") == "NFO-FUT" and
            ins.get("expiry") and ins["expiry"].date() == nearest_expiry):
            return {"tradingsymbol": ins["tradingsymbol"], "instrument_token": ins["instrument_token"]}

    return None

# ------------------- Fetch Futures Data -------------------
def fetch_futures_data(symbol: str, kite: KiteConnect) -> Optional[Dict]:
    """
    Fetches the latest quote for a futures instrument.
    """
    instrument = get_futures_instrument(symbol, kite)
    if not instrument:
        return None

    try:
        key = f"NFO:{instrument['tradingsymbol']}"
        quote = (kite.quote([key]) or {}).get(key, {})

        last_price = quote.get("last_price", (quote.get("ohlc") or {}).get("close"))
        volume = quote.get("volume", quote.get("volume_traded", 0))

        if last_price is None:
            return None

        return {
            "tradingsymbol": instrument["tradingsymbol"],
            "last_price": float(last_price),
            "volume": int(volume),
            "instrument_token": instrument["instrument_token"],
        }
    except Exception as e:
        print(f"❌ Error fetching quote for {symbol}: {e}")
        return None