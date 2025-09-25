# utils/rule_evaluator.py
"""
Rule Evaluation Engine for the AlphaPivot Trading System.

This module is responsible for evaluating incoming trading alerts against a
predefined set of rules to determine if a trade should be initiated.
"""
from __future__ import annotations

import pandas as pd
from typing import Dict, Tuple, List, Any

from utils.db_ops import fetch_latest_snapshots, fetch_latest_zone_data
from utils.indicators import compute_indicators
from utils.option_utils import advanced_buildup_rules

# ---------------- Rule Registry -------------------
RULES: Dict[str, callable] = {}

RULE_WEIGHTS = {
    "RSI Confirmation": 1.0,
    "ADX Strength Filter": 1.0,
    "ROC Momentum Spike": 0.8,
    "EMA Trend Confirmation": 1.0,
    "SMA Base Strength": 0.8,
    "MFI Strength": 0.8,
    "Trend + Volume Confluence": 1.2,
    "Strong Green Candle": 0.8,
    "Zone Breakout Confirmation": 1.2,
    "RMI Strength": 1.0,
    "MACD Trend Confirmation": 1.0,
}
THRESHOLD_SCORE = 60.0


def rule(name):
    """A decorator to register a new rule in the rule engine."""
    def decorator(func):
        RULES[name] = func
        return func
    return decorator


def _num(x, default=0.0):
    """Safely convert a value to a float."""
    try:
        v = float(x)
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default


def _normalize_ohlcv(market_data: Dict) -> Dict:
    """Ensure we have open/high/low/close/volume floats for indicator calc."""
    o  = market_data.get("open",  market_data.get("open_price"))
    h  = market_data.get("high",  market_data.get("high_price"))
    l  = market_data.get("low",   market_data.get("low_price"))
    c  = market_data.get("close", market_data.get("close_price"))
    v  = market_data.get("volume", 0)

    o = _num(o); h = _num(h); l = _num(l); c = _num(c); v = _num(v)

    if c and (o == 0 and h == 0 and l == 0):
        o = h = l = c

    md = dict(market_data)
    md.update({"open": o, "high": h, "low": l, "close": c, "volume": v})
    return md


# ---------------- Rule Definitions ----------------

@rule("RSI Confirmation")
def rsi_trend_confirmation(market_data, option_data):
    rsi = _num(market_data.get("rsi"))
    if 45 <= rsi <= 70:
        return True, "RSI in bullish trend range"
    return False, f"RSI {rsi} not supportive"

@rule("ADX Strength Filter")
def adx_momentum_check(market_data, option_data):
    adx = _num(market_data.get("adx"))
    if adx >= 20:
        return True, f"ADX strong at {adx}"
    return False, f"ADX weak at {adx}"

# ... (other rules remain unchanged) ...

# ---------------- Evaluation Engine ----------------
def evaluate_alert(symbol: str) -> Tuple[bool, str, List[Tuple[str, str]], float, List[str], Dict[str, Any]]:
    """
    Evaluates a trading alert for a given symbol.

    This function fetches the latest market data, computes technical indicators,
    and runs the alert against a set of predefined rules to generate a
    trading decision.

    Args:
        symbol: The trading symbol to evaluate.

    Returns:
        A tuple containing:
        - is_valid (bool): Whether the alert is valid for a trade.
        - rule_matched (str): The name of the primary rule that passed.
        - failed_rules (list): A list of rules that failed.
        - score (float): The final decision score.
        - decision_tags (list): A list of tags for the decision.
        - market_data (dict): The market data context used for the evaluation.
    """
    try:
        market_data, option_data = fetch_latest_snapshots(symbol)
        if not market_data:
            return False, "SnapshotUnavailable", [("snapshot", "Missing market data")], 0, [], {}

        market_data = _normalize_ohlcv(market_data)

        df = pd.DataFrame([{"open": market_data["open"], "high": market_data["high"],
                            "low": market_data["low"], "close": market_data["close"],
                            "volume": market_data["volume"]}])
        market_data.update(compute_indicators(df) or {})

        passed_rules = []
        failed_rules = []
        score_sum = 0.0
        total_weight = sum(RULE_WEIGHTS.values())

        for rule_name, rule_func in RULES.items():
            weight = RULE_WEIGHTS.get(rule_name, 1.0)
            try:
                passed, reason = rule_func(market_data, option_data or {})
                if passed:
                    passed_rules.append(rule_name)
                    score_sum += weight
                else:
                    failed_rules.append((rule_name, reason))
            except Exception as e:
                failed_rules.append((rule_name, f"Exception: {str(e)}"))

        score = round((score_sum / max(total_weight, 1e-9)) * 100, 2)

        if score >= THRESHOLD_SCORE and passed_rules:
            return True, passed_rules[0], failed_rules, score, passed_rules, market_data
        else:
            return False, "BelowThreshold", failed_rules, score, passed_rules, market_data

    except Exception as e:
        return False, "EvaluationError", [(symbol, str(e))], 0, [], {}