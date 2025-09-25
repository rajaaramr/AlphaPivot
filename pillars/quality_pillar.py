# pillars/quality_pillar.py
from __future__ import annotations
import json, math, configparser
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

from .common import (  # provided by your codebase
    ema, atr, adx, resample, write_values, last_metric, clamp,
    TZ, DEFAULT_INI, BaseCfg, rsi, eval_expr
)
from pillars.common import min_bars_for_tf, ensure_min_bars, maybe_trim_last_bar


# -----------------------------
# Config
# -----------------------------
def _cfg(path=DEFAULT_INI) -> dict:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(path)
    return {
        # core knobs
        "atr_win":  cp.getint("quality", "atr_win",  fallback=14),
        "gap_sigma":cp.getfloat("quality", "gap_sigma", fallback=3.0),
        "vahval_break_vol_mult": cp.getfloat("quality", "vahval_break_vol_mult", fallback=2.0),
        "atr_veto_pctile": cp.getfloat("quality", "atr_veto_pctile", fallback=0.90),
        "vol_cv_good": cp.getfloat("quality", "vol_cv_good", fallback=0.50),
        "vol_cv_bad":  cp.getfloat("quality", "vol_cv_bad",  fallback=1.50),
        "near_vahval_atr": cp.getfloat("quality", "near_vahval_atr", fallback=0.25),

        # scenario engine
        "rules_mode": cp.get("quality", "rules_mode", fallback="additive").strip().lower(),
        "clamp_low": cp.getfloat("quality", "clamp_low", fallback=0.0),
        "clamp_high": cp.getfloat("quality", "clamp_high", fallback=100.0),
        "write_scenarios_debug": cp.getboolean("quality", "write_scenarios_debug", fallback=False),
        "min_bars": cp.getint("quality", "min_bars", fallback=120),

        # shared thresholds
        "bb_strong": cp.getfloat("thresholds", "bb_strong", fallback=6.5),

        # scenario list (can be empty)
        "scenarios_list": [s.strip() for s in cp.get("quality_scenarios", "list", fallback="").replace("\n", " ").split(",") if s.strip()],
        "cp": cp,  # stash parser to read each scenario section dynamically
    }


# -----------------------------
# Helpers
# -----------------------------
def _wick_body_eval(last_row: pd.Series) -> Tuple[float, bool, float]:
    """Return (points, wick_bad_bool, wick_ratio). For internal features; scoring is INI-based."""
    body = float(abs(last_row["close"] - last_row["open"]))
    upper = float(last_row["high"] - max(last_row["close"], last_row["open"]))
    lower = float(min(last_row["close"], last_row["open"]) - last_row["low"])
    wick = upper + lower
    ratio = wick / (body if body > 0 else 1e-9)
    pts = -10.0 if ratio > 2.0 else (5.0 if ratio < 1.0 else 0.0)
    return float(pts), bool(ratio > 2.0), float(ratio)

def _near_level(price: float, level: Optional[float], atr_last: float, near_atr: float) -> bool:
    if level is None or atr_last <= 0:
        return False
    return abs(price - float(level)) <= (near_atr * atr_last)

def _higher_tf(tf: str) -> str:
    tf = tf.lower()
    if tf in {"25m","25"}:   return "65m"
    if tf in {"65m","65"}:   return "125m"
    if tf in {"125m","125"}: return "1d"
    return "1d"


# -----------------------------
# Feature builder (what scenarios consume)
# -----------------------------
def _build_features(dtf: pd.DataFrame, vp: dict, cfg: dict, tf: str) -> Dict[str, Any]:
    # basic series
    o = dtf["open"]; c = dtf["close"]; h = dtf["high"]; l = dtf["low"]; v = dtf["volume"]
    last = dtf.iloc[-1]; prev = dtf.iloc[-2] if len(dtf) > 1 else dtf.iloc[-1]

    # ATR & ATR%
    ATR = atr(h, l, c, cfg["atr_win"])
    atr_last = float(ATR.iloc[-1] or 0.0)
    atr_pct_series = (ATR / c.replace(0, np.nan)) * 100.0
    atr_pct = float(atr_pct_series.iloc[-1] or 0.0)
    atr_pct_avg_20 = float(atr_pct_series.rolling(20).mean().iloc[-1]) if len(atr_pct_series.dropna()) else float(atr_pct)

    # ADX
    a14, dip, dim = adx(h, l, c, 14)
    adx14 = float(a14.iloc[-1] or 0.0)
    adx14_prev = float(a14.iloc[-2] or adx14) if len(a14) > 1 else adx14

    # Returns sigma (60)
    rets = c.pct_change().dropna()
    sigma60 = float(rets.rolling(60).std(ddof=1).iloc[-1]) if len(rets) >= 60 else float(rets.std(ddof=1) or 0.0)

    # Gap ratio
    prev_close = float(prev["close"])
    gap_ratio = abs(float(last["close"]) - prev_close) / (prev_close if prev_close else 1e-9)

    # Volume stats
    vol_mean20 = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else float(v.mean() or 0.0)
    vol_sd20   = float(v.rolling(20).std(ddof=1).iloc[-1]) if len(v) >= 20 else float(v.std(ddof=1) or 0.0)
    vol_spike_2sd = bool(float(v.iloc[-1]) > (vol_mean20 + 2.0 * (vol_sd20 if np.isfinite(vol_sd20) else 0.0)))
    vol_cv20 = float((v.rolling(20).std(ddof=1) / v.rolling(20).mean()).iloc[-1] or 0.0) if len(v) >= 20 else 1.0

    # Wick/body + ratio
    _, wick_bad, wick_ratio = _wick_body_eval(last)

    # VP anchors
    poc = vp.get("poc"); vah = vp.get("vah"); val = vp.get("val")
    price = float(last["close"])
    near_poc = bool(poc is not None and abs(price - float(poc)) / (atr_last if atr_last else 1e-9) <= 0.5)
    near_vah = _near_level(price, vah, atr_last, cfg["near_vahval_atr"])
    near_val = _near_level(price, val, atr_last, cfg["near_vahval_atr"])
    vol_ok_vahval = bool(float(v.iloc[-1]) >= (cfg["vahval_break_vol_mult"] * vol_mean20)) if vol_mean20 > 0 else True
    inside_va = bool((val is not None and vah is not None) and (float(val) <= price <= float(vah)))
    bb_score = float(vp.get("bb", 0.0) or 0.0)  # Block Builder score if you have it

    # Bollinger Bands (20, 2.0)
    n_bb, k_bb = 20, 2.0
    mavg = c.rolling(n_bb, min_periods=n_bb//2).mean()
    mstd = c.rolling(n_bb, min_periods=n_bb//2).std(ddof=1)
    bb_up = mavg + k_bb*mstd
    bb_lo = mavg - k_bb*mstd
    bb_width = (bb_up - bb_lo)
    bb_width_pct = float((bb_width.iloc[-1] / (c.iloc[-1] if c.iloc[-1] else 1e-9)) * 100.0) if pd.notna(bb_width.iloc[-1]) else 0.0

    # Distance outside bands (normalized by ATR): >0 only if outside
    up_val = float(bb_up.iloc[-1]) if pd.notna(bb_up.iloc[-1]) else float(c.iloc[-1])
    lo_val = float(bb_lo.iloc[-1]) if pd.notna(bb_lo.iloc[-1]) else float(c.iloc[-1])
    above = max(0.0, float(c.iloc[-1] - up_val))
    below = max(0.0, float(lo_val - c.iloc[-1]))
    outside_px = above if above > 0 else below
    close_dist_bb = (outside_px / (atr_last if atr_last else 1e-9)) if outside_px > 0 else 0.0

    # Simple candle patterns
    body = abs(float(last["close"] - last["open"]))
    rng = max(1e-9, float(last["high"] - last["low"]))
    prev_open = float(prev["open"]); prev_close2 = float(prev["close"])
    engulfing_bull = bool(
        (last["close"] > last["open"]) and (prev_close2 < prev_open) and
        (float(last["close"]) >= max(prev_open, prev_close2)) and
        (float(last["open"])  <= min(prev_open, prev_close2))
    )
    doji_candle = bool((body / rng) < 0.1)  # body <10% of range
    upper_wick = float(last["high"] - max(last["close"], last["open"]))
    lower_wick = float(min(last["close"], last["open"]) - last["low"])
    hammer_bullish = bool((lower_wick >= 2.0*body) and (upper_wick <= 0.3*body) and (last["close"] >= last["open"]))

    # RSI bullish divergence vs N bars back
    rsi14 = rsi(c, 14)
    N = 5
    rsi_bull_div = bool((float(c.iloc[-1]) < float(c.iloc[-N])) and (float(rsi14.iloc[-1]) > float(rsi14.iloc[-N])))

    # assemble features consumed by scenarios
    feats: Dict[str, Any] = {
        # raw OHLCV (last)
        "open": float(last["open"]), "close": float(last["close"]),
        "high": float(last["high"]), "low": float(last["low"]),
        "volume": float(last["volume"]),

        # anchors
        "poc": float(poc) if poc is not None else float("nan"),
        "vah": float(vah) if vah is not None else float("nan"),
        "val": float(val) if val is not None else float("nan"),
        "inside_va": inside_va,
        "bb_score": float(bb_score),

        # proximity & volume gating
        "near_poc": near_poc, "near_vah": near_vah, "near_val": near_val,
        "vol_ok_vahval": vol_ok_vahval,

        # ADX
        "adx14": adx14, "adx14_prev": adx14_prev,

        # ATR%
        "atr_pct": atr_pct, "atr_pct_avg_20": atr_pct_avg_20,

        # volatility / gaps
        "sigma60": sigma60, "gap_ratio": gap_ratio,

        # volume stats
        "volume_avg_20": vol_mean20, "vol_cv20": vol_cv20,
        "vol_spike_2sd": vol_spike_2sd,

        # wick & shape
        "wick_ratio": float(wick_ratio),

        # BB info
        "bb_upper_band": up_val,
        "bb_lower_band": lo_val,
        "bb_width_pct": bb_width_pct,
        "close_dist_bb": close_dist_bb,

        # patterns
        "engulfing_bull": engulfing_bull,
        "doji_candle": doji_candle,
        "hammer_bullish": hammer_bullish,

        # RSI divergence
        "rsi_bull_div": rsi_bull_div,

        # placeholders filled in score_quality (HTF)
        "vah_htf": float("nan"),
    }
    return feats


# -----------------------------
# Scenario engine
# -----------------------------
def _run_scenarios(features: Dict[str, Any], cfg: dict) -> Tuple[float, bool, bool, Dict[str, float], list]:
    cp = cfg["cp"]
    total = 0.0
    veto = False
    reversal = False
    parts: Dict[str, float] = {}
    fired: list[tuple[str, float]] = []

    # expose thresholds to expressions
    features = dict(features)
    features["bb_strong"] = float(cfg["bb_strong"])

    for name in cfg["scenarios_list"]:
        sec = f"quality_scenario.{name}"
        if not cp.has_section(sec):
            continue
        when = cp.get(sec, "when", fallback="").strip()
        if not when:
            continue
        ok = False
        try:
            ok = eval_expr(when, features)
        except Exception:
            ok = False

        if not ok:
            continue

        sc = float(cp.get(sec, "score", fallback="0").strip() or 0.0)
        total = (sc if cfg["rules_mode"] == "override" else (total + sc))
        parts[name] = sc
        fired.append((name, sc))

        # optional bonus
        if cp.has_option(sec, "bonus_when"):
            try:
                if eval_expr(cp.get(sec, "bonus_when"), features):
                    bonus_val = float(cp.get(sec, "bonus", fallback="0").strip() or 0.0)
                    total = (bonus_val if cfg["rules_mode"] == "override" else (total + bonus_val))
                    parts[name + ".bonus"] = bonus_val
                    fired.append((name + ".bonus", bonus_val))
            except Exception:
                pass

        # flags
        if cp.get(sec, "set_veto", fallback="false").strip().lower() in {"1","true","yes","on"}:
            veto = True
        if cp.get(sec, "set_reversal", fallback="false").strip().lower() in {"1","true","yes","on"}:
            reversal = True

    return total, veto, reversal, parts, fired


# -----------------------------
# Public API
# -----------------------------
def score_quality(symbol: str, kind: str, tf: str, df5: pd.DataFrame, base: BaseCfg, ini_path=DEFAULT_INI):
    """
    Scenario-driven Quality pillar.
    - Resamples 5m to tf
    - Builds rich feature map
    - Applies rules from INI [quality_scenarios] + [quality_scenario.*]
    - Writes QUAL.score + flags + optional scenario debug metrics
    """
    dftf = resample(df5, tf)
    dftf = maybe_trim_last_bar(dftf)
    if not ensure_min_bars(dftf, tf):
        return None

    cfg = _cfg(ini_path)

    # Volume profile / Block Builder anchors (from indicators store)
    vp = {
        "poc": last_metric(symbol, kind, tf, "VP.POC"),
        "vah": last_metric(symbol, kind, tf, "VP.VAH"),
        "val": last_metric(symbol, kind, tf, "VP.VAL"),
        "bb":  last_metric(symbol, kind, tf, "BB.score"),
    }

    feats = _build_features(dftf, vp, cfg, tf)

    # Pull HTF VAH if available and inject into features
    htf = _higher_tf(tf)
    vah_htf = last_metric(symbol, kind, htf, "VP.VAH")
    if vah_htf is not None and not (isinstance(vah_htf, float) and math.isnan(vah_htf)):
        feats["vah_htf"] = float(vah_htf)

    # Add a few convenience aliases scenarios might like
    feats.update({
        "price": feats["close"],
    })

    # Scenario engine
    total, veto, reversal, parts, fired = _run_scenarios(feats, cfg)

    # Clamp & time
    score = float(clamp(total, cfg["clamp_low"], cfg["clamp_high"]))
    ts = dftf.index[-1].to_pydatetime().replace(tzinfo=TZ)

    # Write
    rows = [
        (symbol, kind, tf, ts, "QUAL.score", float(score), json.dumps({}), base.run_id, base.source),
        (symbol, kind, tf, ts, "QUAL.veto_flag", 1.0 if veto else 0.0, "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "QUAL.reversal_flag", 1.0 if reversal else 0.0, "{}", base.run_id, base.source),
    ]

    # Debug: key features that help tuning (minimal set)
    debug_ctx = {
        "adx14": feats["adx14"],
        "atr_pct": feats["atr_pct"],
        "bb_width_pct": feats["bb_width_pct"],
        "wick_ratio": feats["wick_ratio"],
        "vol_cv20": feats["vol_cv20"],
        "near_vah": feats["near_vah"],
        "near_val": feats["near_val"],
        "near_poc": feats["near_poc"],
        "bb_score": feats["bb_score"],
    }
    rows.append((symbol, kind, tf, ts, "QUAL.debug_ctx", 0.0, json.dumps(debug_ctx), base.run_id, base.source))

    # Optional: write each fired scenario as its own metric
    if cfg["write_scenarios_debug"]:
        for name, sc in fired:
            rows.append((symbol, kind, tf, ts, f"QUAL.scenario.{name}", float(sc), "{}", base.run_id, base.source))

    write_values(rows)
    return (ts, score, veto, reversal)