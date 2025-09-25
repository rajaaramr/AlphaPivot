# pillars/trend.py
from __future__ import annotations
import json, math, configparser
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# Your shared helpers (unchanged contracts)
from .common import (  # expects these to exist exactly as before
    DEFAULT_INI, BaseCfg, TZ,
    ema, adx, atr, resample, last_metric, write_values, clamp,
    rsi, eval_expr
)
from pillars.common import min_bars_for_tf, ensure_min_bars, maybe_trim_last_bar


# -----------------------------
# Config (base + thresholds + scenario knobs)
# -----------------------------
def _load_cfg(path: str = DEFAULT_INI) -> dict:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(path)
    cfg = {
        # Core lookbacks
        "ema_fast": cp.getint("trend", "ema_fast", fallback=10),
        "ema_mid":  cp.getint("trend", "ema_mid",  fallback=20),
        "ema_slow": cp.getint("trend", "ema_slow", fallback=50),
        "adx_main": cp.getint("trend", "adx_main", fallback=14),
        "adx_fast": cp.getint("trend", "adx_fast", fallback=9),
        "roc_win":  cp.getint("trend", "roc_win",  fallback=5),
        "atr_win":  cp.getint("trend", "atr_win",  fallback=14),

        # Extra feature windows (for your new scenarios)
        "div_lookback": cp.getint("trend", "div_lookback", fallback=5),
        "bb_win": cp.getint("trend", "bb_win", fallback=20),
        "bb_k": cp.getfloat("trend", "bb_k", fallback=2.0),
        "kc_win": cp.getint("trend", "kc_win", fallback=20),
        "kc_mult": cp.getfloat("trend", "kc_mult", fallback=1.5),
        "vol_avg_win": cp.getint("trend", "vol_avg_win", fallback=20),

        # Penalty thresholds (ATR% of price)
        "atr_penalty_8":  cp.getfloat("trend", "atr_penalty_8",  fallback=8.0),
        "atr_penalty_10": cp.getfloat("trend", "atr_penalty_10", fallback=10.0),
        "atr_penalty_15": cp.getfloat("trend", "atr_penalty_15", fallback=15.0),

        # Misc / scenario engine controls
        "poc_align_bonus": cp.getfloat("trend", "poc_align_bonus", fallback=5.0),
        "rules_mode": cp.get("trend", "rules_mode", fallback="additive").lower(),  # additive | override
        "clamp_low": cp.getfloat("trend", "clamp_low", fallback=0.0),
        "clamp_high": cp.getfloat("trend", "clamp_high", fallback=100.0),
        "write_scenarios_debug": cp.getboolean("trend", "write_scenarios_debug", fallback=False),
        "min_bars": cp.getint("trend", "min_bars", fallback=120),
        "_ini_path": path,  # stash path so we can re-open for scenarios
    }
    return cfg


# -----------------------------
# Scenario loader (reads your [scenarios] list and scenario.* sections)
# -----------------------------
def _load_trend_scenarios(path: str) -> Tuple[str, List[dict]]:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(path)

    rules_mode = cp.get("trend", "rules_mode", fallback="additive").lower()

    names: List[str] = []
    if cp.has_section("scenarios"):
        raw = cp.get("scenarios", "list", fallback="")
        names = [n.strip() for n in raw.split(",") if n.strip()]

    scenarios: List[dict] = []
    for n in names:
        sec = f"scenario.{n}"
        if not cp.has_section(sec):
            continue
        scenarios.append({
            "name": n,
            "when": cp.get(sec, "when", fallback=""),
            "score": cp.getfloat(sec, "score", fallback=0.0),
            "bonus_when": cp.get(sec, "bonus_when", fallback=""),
            "bonus": cp.getfloat(sec, "bonus", fallback=0.0),
        })
    return rules_mode, scenarios


# -----------------------------
# Feature builder (exposes everything your INI references)
# -----------------------------
def _trend_features(dtf: pd.DataFrame, cfg: dict, vp: Dict[str, Optional[float]]) -> dict:
    c = dtf["close"]; h = dtf["high"]; l = dtf["low"]
    o = dtf["open"];  v = dtf.get("volume", pd.Series(index=dtf.index, dtype=float)).fillna(0)

    ema10 = ema(c, cfg["ema_fast"])
    ema20 = ema(c, cfg["ema_mid"])
    ema50 = ema(c, cfg["ema_slow"])

    adx14, dip, dim = adx(h, l, c, cfg["adx_main"])
    adx9,  _,   _   = adx(h, l, c, cfg["adx_fast"])

    macd_line = ema(c, 12) - ema(c, 26)
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    hist      = macd_line - macd_sig

    ATR = atr(h, l, c, cfg["atr_win"])
    atr_val = float(ATR.iloc[-1])
    atr_prev = float(ATR.iloc[-2]) if len(ATR) > 1 else atr_val

    roc = c.pct_change(cfg["roc_win"])

    # RSI + divergence helpers
    rsi_series = rsi(c, 14)
    div_n = int(cfg.get("div_lookback", 5))
    idx_n = max(1, min(div_n, len(c) - 1))

    # Bollinger + Keltner (TTM squeeze-style)
    bb_win = int(cfg.get("bb_win", 20)); bb_k = float(cfg.get("bb_k", 2.0))
    kc_win = int(cfg.get("kc_win", 20)); kc_mult = float(cfg.get("kc_mult", 1.5))

    bb_mid = c.rolling(bb_win).mean()
    bb_std = c.rolling(bb_win).std(ddof=0)
    bb_up  = bb_mid + bb_k * bb_std
    bb_lo  = bb_mid - bb_k * bb_std
    if len(c) >= bb_win:
        bb_w_now  = float((bb_up.iloc[-1] - bb_lo.iloc[-1]) / max(1e-9, c.iloc[-1]) * 100.0)
        bb_w_prev = float((bb_up.iloc[-2] - bb_lo.iloc[-2]) / max(1e-9, c.iloc[-2]) * 100.0) if len(c) > bb_win else bb_w_now
    else:
        bb_w_now = bb_w_prev = 0.0

    kc_mid = ema(c, kc_win)
    kc_up  = kc_mid + kc_mult * ATR
    kc_lo  = kc_mid - kc_mult * ATR
    squeeze = 1 if (bb_up.iloc[-1] - bb_lo.iloc[-1]) < (kc_up.iloc[-1] - kc_lo.iloc[-1]) else 0

    vol_avg_win = int(cfg.get("vol_avg_win", 20))
    v_avg = v.rolling(vol_avg_win).mean()

    close_prev    = float(c.iloc[-2]) if len(c) > 1 else float(c.iloc[-1])
    close_prev_n  = float(c.iloc[-idx_n]) if len(c) > idx_n else float(c.iloc[-1])
    rsi_now       = float(rsi_series.iloc[-1]) if len(rsi_series) else 50.0
    rsi_prev_n    = float(rsi_series.iloc[-idx_n]) if len(rsi_series) > idx_n else rsi_now
    atr_pct_now   = float((atr_val / max(1e-9, c.iloc[-1])) * 100.0)
    atr_pct_prev  = float((atr_prev / max(1e-9, c.iloc[-2])) * 100.0) if len(c) > 1 else atr_pct_now

    poc = vp.get("poc")
    poc_dist_atr = float(abs(float(c.iloc[-1]) - poc) / max(1e-9, atr_val)) if poc is not None else 0.0

    feat = {
        # Prices / EMAs
        "open": float(o.iloc[-1]), "close": float(c.iloc[-1]),
        "ema10": float(ema10.iloc[-1]), "ema20": float(ema20.iloc[-1]), "ema50": float(ema50.iloc[-1]),

        # ADX & DI
        "adx14": float(adx14.iloc[-1]),
        "adx14_prev": float(adx14.iloc[-2]) if len(adx14) > 1 else float(adx14.iloc[-1]),
        "adx9": float(adx9.iloc[-1]),
        "adx9_prev": float(adx9.iloc[-2]) if len(adx9) > 1 else float(adx9.iloc[-1]),
        "dip_gt_dim": bool(float(dip.iloc[-1]) > float(dim.iloc[-1])),

        # MACD
        "macd_line": float(macd_line.iloc[-1]),
        "macd_line_prev": float(macd_line.iloc[-2]) if len(macd_line) > 1 else float(macd_line.iloc[-1]),
        "macd_sig": float(macd_sig.iloc[-1]),
        "hist_diff": float(hist.diff().iloc[-1]) if len(hist) > 1 else 0.0,

        # ATR & ROC context
        "atr_pct": atr_pct_now,
        "atr_pct_prev": atr_pct_prev,
        "roc_abs_over_atr_ratio": float(abs(roc.iloc[-1]) / max(1e-9, atr_val / max(1e-9, float(c.iloc[-1])))),

        # Volume features
        "volume": float(v.iloc[-1]),
        "volume_avg_20": float(v_avg.iloc[-1]) if len(v_avg) else 0.0,

        # Squeeze & band widths
        "squeeze_flag": int(squeeze),
        "bb_width_pct": bb_w_now,
        "bb_width_pct_prev": bb_w_prev,

        # Divergence helpers
        "close_prev": close_prev,
        "close_prev_n": close_prev_n,
        "rsi_now": rsi_now,
        "rsi_prev_n": rsi_prev_n,

        # VP proximity
        "poc_dist_atr": poc_dist_atr,
    }
    return feat


# -----------------------------
# Core scorer (per TF)
# -----------------------------
def _trend_score(dtf: pd.DataFrame, vp: Dict[str, Optional[float]], cfg: dict) -> Tuple[float, dict]:
    if dtf is None or len(dtf) < max(cfg.get("min_bars", 120), cfg["ema_slow"] + 10):
        return 0.0, {
            "ema_pts": 0.0, "adx_pts": 0.0, "adx_acc": 0.0,
            "macd_pts": 0.0, "roc_pts": 0.0, "poc_bonus": 0.0, "atr_pct": 0.0
        }

    c = dtf["close"]; h = dtf["high"]; l = dtf["low"]

    # --- Backbone indicators
    ema10 = ema(c, cfg["ema_fast"])
    ema20 = ema(c, cfg["ema_mid"])
    ema50 = ema(c, cfg["ema_slow"])

    adx14, dip, dim = adx(h, l, c, cfg["adx_main"])
    adx9,  _,   _   = adx(h, l, c, cfg["adx_fast"])

    # EMA structure
    A = bool(ema10.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1])
    B = bool(c.iloc[-1] > ema20.iloc[-1])
    ema_pts = (
        40 if (A and B) else
        25 if sum([
            ema10.iloc[-1] > ema20.iloc[-1],
            ema20.iloc[-1] > ema50.iloc[-1],
            c.iloc[-1] > ema20.iloc[-1],
        ]) >= 2 else
        (15 if (ema10.iloc[-1] > ema50.iloc[-1] or c.iloc[-1] > ema20.iloc[-1]) else 0)
    )

    # Expansion/compression tweak
    g1 = (ema10.iloc[-1] - ema20.iloc[-1]) / max(1e-9, abs(ema20.iloc[-1]))
    g2 = (ema20.iloc[-1] - ema50.iloc[-1]) / max(1e-9, abs(ema50.iloc[-1]))
    if g1 > 0 and g2 > 0:
        ema_pts += clamp(5.0 * min(g1, 0.03) / 0.03, 0, 5)
    elif g1 < 0 and g2 < 0:
        ema_pts -= clamp(3.0 * min(abs(g1), 0.02) / 0.02, 0, 3)

    # ADX strength & direction bias
    a14 = float(adx14.iloc[-1]); a14_prev = float(adx14.iloc[-2]) if len(adx14) > 1 else a14
    adx_pts = 20 if (a14 >= 25 and a14 > a14_prev) else (10 if (20 <= a14 < 25 and abs(a14 - a14_prev) < 1.0) else 0)
    if float(dip.iloc[-1]) > float(dim.iloc[-1]):  # DI+
        adx_pts += 3
    else:
        adx_pts -= 3

    # ADX acceleration (fast over main + improving)
    a9 = float(adx9.iloc[-1]); a9_prev = float(adx9.iloc[-2]) if len(adx9) > 1 else a9
    adx_acc = 10 if (a9 > a14 and a9 > a9_prev and a14 > a14_prev) else (
        5 if (a9 > a14 and abs(a14 - a14_prev) < 0.5) else (
            -3 if (a9 > a14 and a9 < a9_prev and a14 < a14_prev) else 0
        )
    )

    # MACD regime
    macd_line = ema(c, 12) - ema(c, 26)
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    hist      = macd_line - macd_sig
    macd_pts = 0
    if macd_line.iloc[-1] > macd_sig.iloc[-1] and hist.diff().iloc[-1] > 0:
        macd_pts = 15
        if macd_line.iloc[-2] <= 0 <= macd_line.iloc[-1]:
            macd_pts = 20  # fresh zero-line cross bonus
    elif hist.diff().iloc[-1] >= 0:
        macd_pts = 10

    # ROC vs ATR confirmation
    roc = c.pct_change(cfg["roc_win"])
    ATR = atr(h, l, c, cfg["atr_win"])
    atr_pct = float((ATR.iloc[-1] / c.iloc[-1]) * 100.0)
    ratio   = abs(roc.iloc[-1]) / max(1e-9, (ATR.iloc[-1] / c.iloc[-1]))
    roc_pts = 5 if ratio >= 1.0 else (3 if ratio >= 0.7 else 0)

    # ATR stretch penalty
    pen = 15 if atr_pct > cfg["atr_penalty_15"] else (
        10 if atr_pct > cfg["atr_penalty_10"] else (
            5 if atr_pct > cfg["atr_penalty_8"] else 0
        )
    )

    # POC distance bonus (away from magnet during trend)
    poc_bonus = 0.0
    poc = vp.get("poc")
    if poc is not None:
        dist_atr = abs(c.iloc[-1] - poc) / max(1e-9, ATR.iloc[-1])
        if dist_atr >= 2.0:
            poc_bonus = cfg["poc_align_bonus"]

    base_total = ema_pts + adx_pts + adx_acc + macd_pts + roc_pts + poc_bonus - pen
    base_score = float(clamp(base_total, 0, 100))

    parts = {
        "ema_pts": float(ema_pts),
        "adx_pts": float(adx_pts),
        "adx_acc": float(adx_acc),
        "macd_pts": float(macd_pts),
        "roc_pts": float(roc_pts),
        "poc_bonus": float(poc_bonus),
        "atr_pct": float(atr_pct),
    }

    # ---- Scenario rules (from the same INI) ----
    score = base_score
    try:
        rules_mode, scenarios = _load_trend_scenarios(cfg["_ini_path"])
    except Exception:
        rules_mode, scenarios = (cfg.get("rules_mode", "additive"), [])

    if scenarios:
        F = _trend_features(dtf, cfg, vp)
        scen_total = 0.0
        scen_parts = {}
        for sc in scenarios:
            if eval_expr(sc["when"], F):
                scen_total += sc["score"]
                scen_parts[f"SCN.{sc['name']}"] = float(sc["score"])
                if sc.get("bonus_when") and eval_expr(sc["bonus_when"], F):
                    scen_total += sc["bonus"]
                    scen_parts[f"SCN.{sc['name']}.bonus"] = float(sc["bonus"])

        final_total = scen_total if rules_mode == "override" else (base_total + scen_total)
        score = float(clamp(final_total, cfg.get("clamp_low", 0.0), cfg.get("clamp_high", 100.0)))
        parts.update(scen_parts)

    return score, parts


# -----------------------------
# Public API
# -----------------------------
def score_trend(
    symbol: str,
    kind: str,                # "spot" | "futures"
    tf: str,                  # "15m", "65m", "1d", ...
    df5: pd.DataFrame,        # 5m base data for resampling
    base: BaseCfg,
    ini_path: str = DEFAULT_INI
) -> Optional[tuple]:
    """
    Computes Trend pillar for a given symbol/kind/timeframe from 5m base data.
    Writes:
      - TREND.score
      - TREND.veto_soft (ATR% > 15)
      - Debug components (EMA/ADX/MACD/ROC/POC/ATR%)
      - Optional: TREND.SCN.* (if write_scenarios_debug=true)
    """
    cfg = _load_cfg(ini_path)
    cfg["_ini_path"] = ini_path  # pass through to scenario loader

    dftf = resample(df5, tf)
    dftf = maybe_trim_last_bar(dftf)
    # keep your existing guard; add min_bars safety
    if not ensure_min_bars(dftf, tf) or len(dftf) < cfg.get("min_bars", 120):
        return None

    vp = {
        "poc": last_metric(symbol, kind, tf, "VP.POC"),
        "vah": last_metric(symbol, kind, tf, "VP.VAH"),
        "val": last_metric(symbol, kind, tf, "VP.VAL"),
    }

    score, parts = _trend_score(dtf, vp, cfg)

    # Soft veto: ATR% > 15
    veto_soft = parts["atr_pct"] > cfg["atr_penalty_15"]

    ts = dftf.index[-1].to_pydatetime().replace(tzinfo=TZ)
    ctx = {
        "ema": [cfg["ema_fast"], cfg["ema_mid"], cfg["ema_slow"]],
        "adx": [cfg["adx_main"], cfg["adx_fast"]],
        "rules_mode": cfg.get("rules_mode", "additive"),
    }

    rows = [
        (symbol, kind, tf, ts, "TREND.score", float(score), json.dumps(ctx), base.run_id, base.source),
        (symbol, kind, tf, ts, "TREND.veto_soft", 1.0 if veto_soft else 0.0, "{}", base.run_id, base.source),

        # debug components (always)
        (symbol, kind, tf, ts, "TREND.EMA_backbone", float(parts.get("ema_pts", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "TREND.ADX_persistence", float(parts.get("adx_pts", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "TREND.ADX_acceleration", float(parts.get("adx_acc", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "TREND.MACD_regime", float(parts.get("macd_pts", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "TREND.ROC_confirm", float(parts.get("roc_pts", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "TREND.poc_bonus", float(parts.get("poc_bonus", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "TREND.atr_pct", float(parts.get("atr_pct", 0.0)), "{}", base.run_id, base.source),
    ]

    # Optional: write scenario contributions as separate debug rows
    if cfg.get("write_scenarios_debug", False):
        for k, v in parts.items():
            if not k.startswith("SCN."):
                continue
            rows.append((symbol, kind, tf, ts, f"TREND.{k}", float(v), "{}", base.run_id, base.source))

    write_values(rows)
    return (ts, float(score))