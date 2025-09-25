# utils/exit_rule_evaluator.py
"""
Exit Rule Evaluation Engine for the AlphaPivot Trading System.

This module is responsible for evaluating open trades against a predefined
set of exit rules to determine if a position should be closed.
"""
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
from datetime import datetime, timezone

from .db import get_db_connection
from .configs import get_config_parser

TZ = timezone.utc

@dataclass
class ExitSignal:
    """Represents the result of an exit evaluation."""
    should_exit: bool
    reason: str
    score: float  # 0..1

def _load_config() -> Dict:
    """Loads the exit rules configuration."""
    cp = get_config_parser()
    if "exit_rules" not in cp:
        raise ValueError("[exit_rules] section not found in config.ini")

    cfg = cp["exit_rules"]
    return {
        "interval": cfg.get("interval", "5m"),
        "atr_len": cfg.getint("atr_len", 14),
        "atr_mult": cfg.getfloat("atr_mult", 1.0),
        "fallback_sl_pct": cfg.getfloat("fallback_sl_pct", 0.008),
        "oi_lookback_min": cfg.getint("oi_lookback_min", 3),
        "zone_trail_weight": cfg.getfloat("zone_trail_weight", 1.0),
        "oi_weight": cfg.getfloat("oi_weight", 1.0),
        "sl_weight": cfg.getfloat("sl_weight", 1.0),
        "min_exit_score": cfg.getfloat("min_exit_score", 0.6),
    }

# ======== Public API ========
def evaluate_exit(symbol: str) -> Tuple[bool, str, float]:
    """
    Decides whether to exit an open trade for the given symbol.
    """
    cfg = _load_config()
    trade = _get_open_trade(symbol)
    if not trade:
        return False, "no_open_trade", 0.0

    side, entry_price = trade["side"], float(trade["entry_price_fut"] or 0.0)

    fut_px = _get_last_futures_close(symbol, cfg["interval"])
    if fut_px is None or entry_price <= 0:
        return False, "context_missing", 0.0

    trail_hit, trail_score = _footprint_trail_exit(symbol, side, fut_px, cfg)
    oi_exit, oi_score = _detect_oi_unwind(symbol, side, cfg)
    sl_hit, sl_score, sl_level = _stoploss_exit(side, entry_price, fut_px, symbol, cfg)

    votes = []
    if trail_hit: votes.append(cfg["zone_trail_weight"])
    if oi_exit:   votes.append(cfg["oi_weight"])
    if sl_hit:    votes.append(cfg["sl_weight"])

    raw = sum(votes)
    denom = (cfg["zone_trail_weight"] + cfg["oi_weight"] + cfg["sl_weight"]) or 1.0
    score01 = raw / denom

    reason = "hold"
    if sl_hit:
        reason = f"stoploss_hit({sl_level:.2f})"
    elif oi_exit:
        reason = "oi_unwind" if side == "LONG" else "short_cover"
    elif trail_hit:
        reason = "footprint_trail"

    should = score01 >= cfg["min_exit_score"]
    return should, reason, round(score01 * 100.0, 2)


# ======== Components ========
def _get_open_trade(symbol: str) -> Optional[dict]:
    # ... (implementation remains the same)
    pass

def _get_last_futures_close(symbol: str, interval: str) -> Optional[float]:
    # ... (implementation remains the same)
    pass

def _get_atr(symbol: str, interval: str, length: int) -> Optional[float]:
    # ... (implementation remains the same)
    pass

def _stoploss_exit(side: str, entry: float, last: float, symbol: str, cfg: Dict) -> Tuple[bool, float, float]:
    """Calculates if the stop-loss has been hit."""
    atr = _get_atr(symbol, cfg["interval"], cfg["atr_len"])
    if atr and atr > 0:
        sl = entry - cfg["atr_mult"] * atr if side == "LONG" else entry + cfg["atr_mult"] * atr
        hit = last <= sl if side == "LONG" else last >= sl
        dist = abs((sl - last) / (cfg["atr_mult"] * atr)) if cfg["atr_mult"] * atr else 1.0
        score = min(1.0, max(0.0, dist))
        return hit, score, sl
    else:
        sl = entry * (1.0 - cfg["fallback_sl_pct"]) if side == "LONG" else entry * (1.0 + cfg["fallback_sl_pct"])
        hit = last <= sl if side == "LONG" else last >= sl
        base = entry * cfg["fallback_sl_pct"]
        dist = abs(sl - last) / base if base else 1.0
        score = min(1.0, max(0.0, dist))
        return hit, score, sl

def _latest_zone(symbol: str, interval: str) -> Optional[dict]:
    # ... (implementation remains the same)
    pass

def _footprint_trail_exit(symbol: str, side: str, last: float, cfg: Dict) -> Tuple[bool, float]:
    """Determines if the footprint trail stop has been hit."""
    z = _latest_zone(symbol, cfg["interval"])
    if not z:
        return (False, 0.0)

    if side == "LONG":
        val = float(z["val"] or 0.0)
        if val and last < val and z["sb"]:
            rng = abs(float(z["vah"] or 0.0) - val) or (0.006 * last)
            depth = (val - last) / rng if rng else 0.0
            return True, min(1.0, max(0.0, depth))
    else:  # SHORT
        vah = float(z["vah"] or 0.0)
        if vah and last > vah and z["rb"]:
            rng = abs(vah - float(z["val"] or vah)) or (0.006 * last)
            depth = (last - vah) / rng if rng else 0.0
            return True, min(1.0, max(0.0, depth))
    return (False, 0.0)

def _detect_oi_unwind(symbol: str, side: str, cfg: Dict) -> Tuple[bool, float]:
    """Detects trade exits based on OI unwinding."""
    table = "options.ce_snapshot" if side == "LONG" else "options.pe_snapshot"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT ltp, oi, oi_change FROM {table} WHERE symbol=%s ORDER BY ts DESC LIMIT %s;",
            (symbol, cfg["oi_lookback_min"])
        )
        rows = cur.fetchall()

    if not rows or len(rows) < 2:
        return (False, 0.0)

    ltp_dir = 1 if rows[0][0] > rows[-1][0] else -1 if rows[0][0] < rows[-1][0] else 0
    oi_dir = 1 if rows[0][1] > rows[-1][1] else -1 if rows[0][1] < rows[-1][1] else 0

    if side == "LONG" and ltp_dir < 0 and oi_dir < 0:
        neg_oi_chg = [abs(r[2]) for r in rows if (r[2] or 0) < 0]
        conf = min(1.0, (sum(neg_oi_chg) / (sum(abs(r[2] or 0) for r in rows) or 1.0)))
        return True, conf
    elif side == "SHORT" and ltp_dir > 0 and oi_dir < 0:
        neg_oi_chg = [abs(r[2]) for r in rows if (r[2] or 0) < 0]
        conf = min(1.0, (sum(neg_oi_chg) / (sum(abs(r[2] or 0) for r in rows) or 1.0)))
        return True, conf

    return (False, 0.0)