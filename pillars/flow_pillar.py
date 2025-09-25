# pillars/flow_pillar.py
from __future__ import annotations
import json, math, configparser
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

from .common import (  # your shared utils
    ema, atr, adx, obv_series, resample, last_metric, write_values, clamp,
    TZ, DEFAULT_INI, BaseCfg
)
from pillars.common import min_bars_for_tf, ensure_min_bars, maybe_trim_last_bar


# -----------------------------
# Config
# -----------------------------
def _cfg(path=DEFAULT_INI) -> dict:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(path)

    # fallbacks for old [liquidity] keys if present
    spread_bps_veto = cp.getfloat(
        "flow", "spread_bps_veto",
        fallback=cp.getfloat("liquidity", "max_spread_bps", fallback=40.0)
    )
    min_turnover = cp.getfloat(
        "flow", "min_turnover",
        fallback=cp.getfloat("liquidity", "min_turnover", fallback=0.0)
    )

    return {
        # core knobs
        "mfi_len":         cp.getint("flow", "mfi_len", fallback=14),
        "rvol_strong":     cp.getfloat("flow", "rvol_strong", fallback=1.5),
        "rvol_extreme":    cp.getfloat("flow", "rvol_extreme", fallback=2.0),
        "vol_cv_good":     cp.getfloat("flow", "vol_cv_good", fallback=0.50),
        "vol_cv_bad":      cp.getfloat("flow", "vol_cv_bad",  fallback=1.50),
        "voi_scale":       cp.getfloat("flow", "voi_scale",    fallback=10.0),
        "roll_look":       cp.getint("flow",  "roll_look",     fallback=5),
        "roll_drop_pct":   cp.getfloat("flow","roll_drop_pct", fallback=0.35),
        "spread_bps_veto": spread_bps_veto,
        "min_rvol_veto":   cp.getfloat("flow", "min_rvol_veto",  fallback=0.50),
        "vol_rank_floor":  cp.getfloat("flow", "vol_rank_floor", fallback=0.20),
        "min_turnover":    min_turnover,

        # scenario engine
        "rules_mode": cp.get("flow", "rules_mode", fallback="additive").strip().lower(),
        "clamp_low":  cp.getfloat("flow", "clamp_low",  fallback=0.0),
        "clamp_high": cp.getfloat("flow", "clamp_high", fallback=100.0),
        "write_scenarios_debug": cp.getboolean("flow", "write_scenarios_debug", fallback=False),
        "min_bars": cp.getint("flow", "min_bars", fallback=120),

        # scenario list
        "scenarios_list": [s.strip() for s in cp.get("flow_scenarios", "list", fallback="").replace("\n"," ").split(",") if s.strip()],
        "cp": cp,
    }


# -----------------------------
# Helpers
# -----------------------------
def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series, n:int=14)->pd.Series:
    tp = (high + low + close) / 3.0
    rmf = tp * vol
    pos = (tp > tp.shift(1)).astype(float) * rmf
    neg = (tp < tp.shift(1)).astype(float) * rmf
    pos_n = pos.rolling(n, min_periods=max(5, n//2)).sum()
    neg_n = neg.rolling(n, min_periods=max(5, n//2)).sum().replace(0, np.nan)
    mr = pos_n / neg_n
    return 100 - (100 / (1 + mr))

def _ist(ts: pd.Timestamp) -> pd.Timestamp:
    try:
        return ts.tz_convert("Asia/Kolkata")
    except Exception:
        return ts.tz_localize("UTC").tz_convert("Asia/Kolkata")

def _hh_hl(series: pd.Series, look:int=20) -> bool:
    if len(series) < look + 2:
        return False
    hh = float(series.iloc[-1]) >= float(series.rolling(look).max().iloc[-2])
    slope = float(series.iloc[-1] - series.iloc[-look]) > 0
    return bool(hh and slope)

def _safe_eval(expr: str, scope: Dict[str, Any]) -> bool:
    if not expr or not expr.strip():
        return False
    return bool(eval(expr, {"__builtins__": {}}, scope))


# -----------------------------
# Feature builder (what scenarios consume)
# -----------------------------
def _build_features(dtf: pd.DataFrame, symbol: str, kind: str, tf: str, cfg: dict) -> Dict[str, Any]:
    # session flags
    ts_ist = _ist(dtf.index[-1])
    minutes = ts_ist.hour * 60 + ts_ist.minute
    open_min, close_min = 9*60 + 15, 15*60 + 30
    in_session = (open_min <= minutes <= close_min)
    last_vol_zero = (float(dtf["volume"].iloc[-1]) == 0.0)

    # ignore last bar if off-session & vol=0 (don’t poison RVOL/OBV/MFI)
    use = dtf.iloc[:-1] if ((not in_session) and last_vol_zero and len(dtf) > 1) else dtf
    o = use["open"]; c = use["close"]; h = use["high"]; l = use["low"]; v = use["volume"]

    # --- MFI ---
    mfi = _mfi(h, l, c, v, n=cfg["mfi_len"])
    mfi_val = float(mfi.iloc[-1] if len(mfi) else 50.0)
    mfi_slope = float(mfi.diff().iloc[-1]) if len(mfi) > 1 else 0.0
    mfi_up = (mfi_slope > 0)
    # recent previous high of MFI (exclude last bar)
    if len(mfi) >= 10:
        mfi_prev_high = float(mfi.iloc[-10:-1].max())
    else:
        mfi_prev_high = float(mfi_val)

    # --- OBV + structure ---
    obv = obv_series(c, v)
    obv_ema20 = obv.ewm(span=20, adjust=False).mean() if len(obv) else obv
    obv_above_ema = bool(len(obv) and len(obv_ema20) and (float(obv.iloc[-1]) > float(obv_ema20.iloc[-1])))

    def _hh(series: pd.Series, look:int=20) -> bool:
        if len(series) < look + 1: return False
        return bool(float(series.iloc[-1]) >= float(series.rolling(look).max().iloc[-2]))

    def _ll(series: pd.Series, look:int=20) -> bool:
        if len(series) < look + 1: return False
        return bool(float(series.iloc[-1]) <= float(series.rolling(look).min().iloc[-2]))

    price_hh = _hh(h, look=20)            # “higher high” using highs
    price_higher_high = price_hh          # alias for your INI name
    price_lower_low = _ll(l, look=20)     # “lower low” using lows

    # OBV higher-low proxy: recent 5-bar min > prior 5-bar min
    if len(obv) >= 12:
        recent_min = float(obv.tail(5).min())
        prior_min  = float(obv.shift(5).tail(5).min())
        obv_higher_low = bool(recent_min > prior_min)
    else:
        obv_higher_low = False

    # OBV higher-high proxy (kept from earlier logic)
    obv_hh = _hh(obv, look=20)

    # --- RVOL & stability ---
    v_avg = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else float(v.mean() or 1.0)
    rvol_now = float(v.iloc[-1] / max(1e-9, v_avg))
    rvol_strong = (rvol_now >= cfg["rvol_strong"])
    rvol_extreme = (rvol_now >= cfg["rvol_extreme"])
    vol_cv20 = float((v.rolling(20).std(ddof=1) / v.rolling(20).mean()).iloc[-1] or 0.0) if len(v) >= 20 else 1.0

    if len(v) >= 60:
        vol_rank60 = float((v.tail(60).rank(pct=True).iloc[-1]))
    else:
        vol_rank60 = 0.5

    # --- VAH/VAL from Volume Profile (for breakout/fail logic) ---
    vah = last_metric(symbol, kind, tf, "VP.VAH")
    val = last_metric(symbol, kind, tf, "VP.VAL")

    prev_close = float(c.iloc[-2]) if len(c) > 1 else float(c.iloc[-1])
    last_close = float(c.iloc[-1])
    last_high  = float(h.iloc[-1]); last_low = float(l.iloc[-1])

    if vah is not None:
        vah = float(vah)
        near_vah_break = bool((prev_close <= vah) and (last_close > vah))
        close_vah_break_fail = bool((last_high > vah) and (last_close <= vah))
    else:
        near_vah_break = False
        close_vah_break_fail = False

    if val is not None:
        val = float(val)
        near_val_break = bool((prev_close >= val) and (last_close < val))
        close_val_break_fail = bool((last_low < val) and (last_close >= val))
    else:
        near_val_break = False
        close_val_break_fail = False

    # --- Candle anatomy for wick_ratio & reversal ---
    body = abs(last_close - float(o.iloc[-1]))
    rng = max(1e-9, last_high - last_low)
    upper_wick = last_high - max(last_close, float(o.iloc[-1]))
    lower_wick = min(last_close, float(o.iloc[-1])) - last_low
    wick_ratio = float((upper_wick + lower_wick) / (body if body > 0 else 1e-9))

    # simple reversal candle: long tail against body
    bullish_rev = (last_close > float(o.iloc[-1])) and (lower_wick >= 2.0 * body)
    bearish_rev = (last_close < float(o.iloc[-1])) and (upper_wick >= 2.0 * body)
    price_reversal_candle = bool(bullish_rev or bearish_rev)

    # --- VOI & roll (futures only) ---
    voi_long_build = voi_short_build = voi_short_cover = voi_long_unwind = False
    voi_mag = 0.0
    roll_trap = False
    if "oi" in use.columns and kind == "futures":
        doi = use["oi"].diff()
        pc  = c.diff()
        last_doi = float(doi.iloc[-1]) if len(doi) else 0.0
        last_pc  = float(pc.iloc[-1])  if len(pc)  else 0.0

        if last_pc > 0 and last_doi > 0:   voi_long_build = True
        elif last_pc < 0 and last_doi > 0: voi_short_build = True
        elif last_pc > 0 and last_doi < 0: voi_short_cover = True
        elif last_pc < 0 and last_doi < 0: voi_long_unwind = True

        voi_mag = abs(last_doi) / max(1e-9, float(v.iloc[-1]))

        look = cfg["roll_look"]
        if len(use) > look + 1:
            oi0 = float(use["oi"].iloc[-look-1])
            oin = float(use["oi"].iloc[-1])
            if oi0 > 0:
                drop = (oi0 - oin) / oi0
                if drop >= cfg["roll_drop_pct"] and rvol_strong:
                    roll_trap = True

    # --- Liquidity evidence ---
    spread_bps = None
    for m in ("MKT.spread_bps", "LIQ.spread_bps", "MKT.spread_bps.mean"):
        x = last_metric(symbol, kind, tf, m)
        if x is not None:
            spread_bps = float(x); break

    turnover_est = float(dtf["close"].iloc[-1] * dtf["volume"].iloc[-1])
    min_turn = float(cfg["min_turnover"])
    illiquid = False
    if (spread_bps is not None) and (spread_bps >= cfg["spread_bps_veto"]):
        illiquid = True
    elif in_session and (not last_vol_zero) and (rvol_now < cfg["min_rvol_veto"]) and (vol_rank60 <= cfg["vol_rank_floor"]):
        illiquid = True
    elif (min_turn > 0.0) and in_session and (not last_vol_zero) and (turnover_est < min_turn):
        illiquid = True

    # --- Session flags ---
    near_open = in_session and (minutes - open_min <= 30)
    near_close = in_session and (close_min - minutes <= 30)
    mid_lunch = in_session and (12*60 <= minutes <= 13*60 + 30)

    # assemble features for scenarios
    feats: Dict[str, Any] = {
        "open": float(o.iloc[-1]), "close": last_close, "high": last_high, "low": last_low,
        "volume": float(v.iloc[-1]), "price": last_close,

        # MFI
        "mfi_val": float(mfi_val), "mfi_slope": float(mfi_slope), "mfi_up": bool(mfi_up),
        "mfi_prev_high": float(mfi_prev_high),

        # OBV structure
        "obv_above_ema": bool(obv_above_ema),
        "price_hh": bool(price_hh),
        "price_higher_high": bool(price_higher_high),
        "price_lower_low": bool(price_lower_low),
        "obv_hh": bool(obv_hh),
        "obv_higher_low": bool(obv_higher_low),

        # RVOL & stability
        "rvol_now": float(rvol_now),
        "rvol_strong": bool(rvol_strong),
        "rvol_extreme": bool(rvol_extreme),
        "vol_cv20": float(vol_cv20),
        "vol_rank60": float(vol_rank60),

        # VAH/VAL breakout/fail
        "near_vah_break": bool(near_vah_break),
        "near_val_break": bool(near_val_break),
        "close_vah_break_fail": bool(close_vah_break_fail),
        "close_val_break_fail": bool(close_val_break_fail),

        # candle anatomy / reversal
        "wick_ratio": float(wick_ratio),
        "price_reversal_candle": bool(price_reversal_candle),

        # VOI & roll
        "voi_long_build": bool(voi_long_build),
        "voi_short_build": bool(voi_short_build),
        "voi_short_cover": bool(voi_short_cover),
        "voi_long_unwind": bool(voi_long_unwind),
        "voi_mag": float(voi_mag),
        "roll_trap": bool(roll_trap),

        # Liquidity
        "spread_bps": float(spread_bps) if spread_bps is not None else float("nan"),
        "turnover_est": float(turnover_est),
        "illiquid_flag": bool(illiquid),

        # Session
        "in_session": bool(in_session),
        "near_open": bool(near_open),
        "near_close": bool(near_close),
        "mid_lunch": bool(mid_lunch),
        "last_vol_zero": bool(last_vol_zero),
    }
    return feats



# -----------------------------
# Scenario engine
# -----------------------------
def _run_scenarios(features: Dict[str, Any], cfg: dict) -> Tuple[float, bool, Dict[str, float], list]:
    cp = cfg["cp"]
    total = 0.0
    veto = False
    parts: Dict[str, float] = {}
    fired: list[tuple[str, float]] = []

    for name in cfg["scenarios_list"]:
        sec = f"flow_scenario.{name}"
        if not cp.has_section(sec): 
            continue
        when = cp.get(sec, "when", fallback="").strip()
        if not when:
            continue

        ok = False
        try:
            ok = _safe_eval(when, features)
        except Exception:
            ok = False
        if not ok:
            continue

        sc = float(cp.get(sec, "score", fallback="0").strip() or 0.0)
        total = (sc if cfg["rules_mode"] == "override" else (total + sc))
        parts[name] = sc
        fired.append((name, sc))

        if cp.has_option(sec, "bonus_when"):
            try:
                if _safe_eval(cp.get(sec, "bonus_when"), features):
                    bonus_val = float(cp.get(sec, "bonus", fallback="0").strip() or 0.0)
                    total = (bonus_val if cfg["rules_mode"] == "override" else (total + bonus_val))
                    parts[name + ".bonus"] = bonus_val
                    fired.append((name + ".bonus", bonus_val))
            except Exception:
                pass

        if cp.get(sec, "set_veto", fallback="false").strip().lower() in {"1","true","yes","on"}:
            veto = True

    return total, veto, parts, fired


# -----------------------------
# Public API
# -----------------------------
def score_flow(symbol: str, kind: str, tf: str, df5: pd.DataFrame, base: BaseCfg, ini_path=DEFAULT_INI):
    """
    Scenario-driven Flow pillar.
    - Resamples 5m to tf
    - Builds flow features (MFI/OBV/RVOL/VOI/Liquidity/Session)
    - Applies INI scenarios
    - Auto-veto if illiquid or roll trap (plus scenario veto)
    """
    dftf = resample(df5, tf)
    dftf = maybe_trim_last_bar(dftf)
    if not ensure_min_bars(dftf, tf):
        return None

    cfg = _cfg(ini_path)
    feats = _build_features(dftf, symbol, kind, tf, cfg)

    total, veto_scen, parts, fired = _run_scenarios(feats, cfg)

    # auto-veto on hard evidence
    auto_veto = bool(feats["illiquid_flag"] or feats["roll_trap"])
    score = float(clamp(total, cfg["clamp_low"], cfg["clamp_high"]))
    veto = bool(veto_scen or auto_veto)

    ts = dftf.index[-1].to_pydatetime().replace(tzinfo=TZ)

    rows = [
        (symbol, kind, tf, ts, "FLOW.score", float(score), json.dumps({}), base.run_id, base.source),
        (symbol, kind, tf, ts, "FLOW.veto_flag", 1.0 if veto else 0.0, "{}", base.run_id, base.source),
    ]

    # compact debug ctx to help tuning
    debug_ctx = {
        "mfi_val": feats["mfi_val"], "mfi_slope": feats["mfi_slope"],
        "obv_above_ema": feats["obv_above_ema"], "price_hh": feats["price_hh"], "obv_hh": feats["obv_hh"],
        "rvol": feats["rvol_now"], "vol_cv20": feats["vol_cv20"], "vol_rank60": feats["vol_rank60"],
        "voi_long_build": feats["voi_long_build"], "voi_short_build": feats["voi_short_build"],
        "voi_short_cover": feats["voi_short_cover"], "voi_long_unwind": feats["voi_long_unwind"], "voi_mag": feats["voi_mag"],
        "spread_bps": feats["spread_bps"], "turnover_est": feats["turnover_est"],
        "illiquid_flag": feats["illiquid_flag"], "roll_trap": feats["roll_trap"],
        "in_session": feats["in_session"], "near_open": feats["near_open"], "near_close": feats["near_close"],
    }
    rows.append((symbol, kind, tf, ts, "FLOW.debug_ctx", 0.0, json.dumps(debug_ctx), base.run_id, base.source))

    if cfg["write_scenarios_debug"]:
        for name, sc in fired:
            rows.append((symbol, kind, tf, ts, f"FLOW.scenario.{name}", float(sc), "{}", base.run_id, base.source))

    write_values(rows)
    return (ts, score, veto)
