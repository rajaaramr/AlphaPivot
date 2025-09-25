# utils/indicators.py
"""
Config-driven indicator calculation module.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .configs import get_config_parser

# Optional TA helpers (robust fallbacks)
try:
    from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator, ROCIndicator
    from ta.volume import MFIIndicator
    from ta.volatility import AverageTrueRange
except ImportError:
    EMAIndicator = SMAIndicator = MACD = ADXIndicator = None
    RSIIndicator = ROCIndicator = None
    MFIIndicator = None
    AverageTrueRange = None

# =========================
# Config parsing
# =========================

@dataclass
class IndicatorConfig:
    enabled: Dict[str, bool]
    periods: Dict[str, Any]

def load_indicator_config() -> IndicatorConfig:
    """Loads indicator configuration from the central config.ini file."""
    cp = get_config_parser()

    enabled = {
        "rsi": cp.getboolean("indicators", "rsi", fallback=True),
        "rmi": cp.getboolean("indicators", "rmi", fallback=False),
        "ema": cp.getboolean("indicators", "ema", fallback=True),
        "macd": cp.getboolean("indicators", "macd", fallback=True),
        "adx": cp.getboolean("indicators", "adx", fallback=True),
        "roc": cp.getboolean("indicators", "roc", fallback=True),
        "atr": cp.getboolean("indicators", "atr", fallback=True),
        "mfi": cp.getboolean("indicators", "mfi", fallback=True),
        "sma": cp.getboolean("indicators", "sma", fallback=False),
    }

    periods = {
        "rsi": cp.getint("periods", "rsi", fallback=14),
        "ema": [int(p) for p in cp.get("periods", "ema", fallback="5,10,20,50").split(",")],
        "macd": tuple(int(p) for p in cp.get("periods", "macd", fallback="12,26,9").split(",")),
        "adx": cp.getint("periods", "adx", fallback=14),
        "roc": cp.getint("periods", "roc", fallback=14),
        "atr": cp.getint("periods", "atr", fallback=14),
        "mfi": cp.getint("periods", "mfi", fallback=14),
        "sma": [int(p) for p in cp.get("periods", "sma", fallback="5,10,20,50").split(",")],
    }

    return IndicatorConfig(enabled=enabled, periods=periods)

# =========================
# Helpers (math + DF)
# =========================

def _safe(x, default=0.0):
    """Safely convert a value to a float, handling NaN and other errors."""
    try:
        v = float(x)
        return default if not np.isfinite(v) else v
    except (ValueError, TypeError):
        return default

def _last(series, default=0.0):
    """Safely get the last value of a pandas Series."""
    try:
        return _safe(series.iloc[-1], default)
    except IndexError:
        return default

def _normalize_ohlcv(src: pd.DataFrame) -> pd.DataFrame:
    """Normalizes a DataFrame to ensure it has standard OHLCV columns."""
    df = src.copy()
    out = pd.DataFrame(index=df.index)

    def pick(cols):
        for c in cols:
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    out["open"] = pick(["open", "open_price"])
    out["high"] = pick(["high", "high_price"])
    out["low"] = pick(["low", "low_price"])
    out["close"] = pick(["close", "close_price"])
    out["volume"] = pick(["volume"]).fillna(0.0)

    if not out.empty:
        last = out.iloc[-1]
        for c in ("open", "high", "low"):
            if pd.isna(last[c]):
                out.loc[out.index[-1], c] = _safe(last["close"], 0.0)
    return out

# =========================
# Core calculator (single TF snapshot)
# =========================

def compute_indicators(df: pd.DataFrame, cfg: Optional[IndicatorConfig] = None) -> dict:
    """
    Compute indicator snapshot for a single timeframe.
    """
    if cfg is None:
        cfg = load_indicator_config()

    res: Dict[str, float] = {}

    try:
        d = _normalize_ohlcv(df)
        close, high, low, vol = d["close"], d["high"], d["low"], d["volume"]

        if cfg.enabled.get("ema", False):
            for p in cfg.periods.get("ema", []):
                res[f"ema_{p}"] = _last(close.ewm(span=p, adjust=False).mean())

        if cfg.enabled.get("sma", False):
            for p in cfg.periods.get("sma", []):
                res[f"sma_{p}"] = _last(close.rolling(p).mean())

        if cfg.enabled.get("rsi", False):
            p = cfg.periods.get("rsi", 14)
            delta = close.diff()
            gain = delta.where(delta > 0, 0).ewm(alpha=1/p, adjust=False).mean()
            loss = -delta.where(delta < 0, 0).ewm(alpha=1/p, adjust=False).mean()
            rs = gain / loss.replace(0, np.nan)
            res["rsi"] = _last(100 - (100 / (1 + rs)))

        if cfg.enabled.get("macd", False):
            fast, slow, signal = cfg.periods.get("macd", (12, 26, 9))
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            res["macd"] = _last(macd_line)
            res["macd_signal"] = _last(signal_line)
            res["macd_diff"] = _last(macd_line - signal_line)

    except Exception as e:
        print(f"[❌ indicators.py] Error computing indicators: {e}")
        return {}

    return res