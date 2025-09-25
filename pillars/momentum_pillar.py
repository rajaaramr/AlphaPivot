# pillars/momentum_pillar.py
from __future__ import annotations
import json, math, configparser
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd

# common helpers you already have:
# ema, atr, adx, obv_series, bb_width_pct, resample, write_values, clamp, TZ, DEFAULT_INI, BaseCfg
from .common import *
from pillars.common import min_bars_for_tf, ensure_min_bars, maybe_trim_last_bar


# -----------------------------
# Local helpers
# -----------------------------
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0); l = (-d).clip(lower=0)
    rs = g.ewm(alpha=1/n, adjust=False).mean() / (l.ewm(alpha=1/n, adjust=False).mean().replace(0, np.nan))
    return 100 - 100/(1+rs)

def _rmi(close: pd.Series, lb: int = 14, m: int = 5) -> pd.Series:
    diffm = close.diff(m)
    up = diffm.clip(lower=0); dn = (-diffm).clip(lower=0)
    ema_up = up.ewm(span=lb, adjust=False).mean()
    ema_dn = dn.ewm(span=lb, adjust=False).mean().replace(0, np.nan)
    return 100 - 100/(1 + (ema_up/ema_dn))

def _count_sign_flips(s: pd.Series, look: int = 12) -> int:
    x = np.sign(s.tail(look).fillna(0.0).values)
    return int(np.sum(np.abs(np.diff(x)) > 0))


# -----------------------------
# Config (base + scenario controls)
# -----------------------------
def _cfg(path: str = DEFAULT_INI) -> dict:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(path)
    return {
        # Core params
        "rsi_fast":  cp.getint("momentum", "rsi_fast",  fallback=5),
        "rsi_std":   cp.getint("momentum", "rsi_std",   fallback=14),
        "rmi_lb":    cp.getint("momentum", "rmi_lb",    fallback=14),
        "rmi_m":     cp.getint("momentum", "rmi_m",     fallback=5),
        "atr_win":   cp.getint("momentum", "atr_win",   fallback=14),

        # Regime thresholds
        "low_vol_thr":  cp.getfloat("momentum", "low_vol_thr",  fallback=3.0),
        "mid_vol_thr":  cp.getfloat("momentum", "mid_vol_thr",  fallback=6.0),
        "rvol_sigma_1": cp.getfloat("momentum", "rvol_sigma_1", fallback=1.0),
        "rvol_sigma_2": cp.getfloat("momentum", "rvol_sigma_2", fallback=2.0),

        # Scenario engine controls
        "rules_mode": cp.get("momentum", "rules_mode", fallback="additive").lower(),  # additive | override
        "clamp_low": cp.getfloat("momentum", "clamp_low", fallback=0.0),
        "clamp_high": cp.getfloat("momentum", "clamp_high", fallback=100.0),
        "write_scenarios_debug": cp.getboolean("momentum", "write_scenarios_debug", fallback=False),
        "min_bars": cp.getint("momentum", "min_bars", fallback=120),

        # Extra feature windows
        "vol_avg_win": cp.getint("momentum", "vol_avg_win", fallback=20),
        "bb_win": cp.getint("momentum", "bb_win", fallback=20),
        "bb_k": cp.getfloat("momentum", "bb_k", fallback=2.0),
        "div_lookback": cp.getint("momentum", "div_lookback", fallback=5),

        "_ini_path": path,
    }


# -----------------------------
# Scenario loader (namespaced to avoid Trend collisions)
# -----------------------------
def _load_mom_scenarios(path: str) -> Tuple[str, List[dict]]:
    """
    Looks for:
      [mom_scenarios]
      list = name1, name2, ...
      [mom_scenario.name1]
      when = <expr>
      score = +10
      bonus_when = <expr>  ; optional
      bonus = +5
    """
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(path)

    rules_mode = cp.get("momentum", "rules_mode", fallback="additive").lower()

    names: List[str] = []
    if cp.has_section("mom_scenarios"):
        raw = cp.get("mom_scenarios", "list", fallback="")
        names = [n.strip() for n in raw.split(",") if n.strip()]

    scenarios: List[dict] = []
    for n in names:
        sec = f"mom_scenario.{n}"
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
# Feature builder (covers all INI references you used)
# -----------------------------
def _mom_features(dtf: pd.DataFrame, cfg: dict) -> Dict[str, float | int | bool]:
    c = dtf["close"]; h = dtf["high"]; l = dtf["low"]
    o = dtf["open"];  v = dtf.get("volume", pd.Series(index=dtf.index, dtype=float)).fillna(0)

    # ATR & ATR%
    ATR = atr(h, l, c, cfg["atr_win"])
    atr_val = float(ATR.iloc[-1])
    atr_pct_now = float((atr_val / max(1e-9, c.iloc[-1])) * 100.0)
    atr_avg_20 = float(ATR.rolling(20).mean().iloc[-1]) if len(ATR) >= 20 else atr_val

    # RSI fast/std
    rsi_fast = _rsi(c, cfg["rsi_fast"])
    rsi_std  = _rsi(c, cfg["rsi_std"])

    # RMI
    rmi     = _rmi(c, lb=cfg["rmi_lb"], m=cfg["rmi_m"])

    # MACD / histogram
    macd_line = ema(c, 12) - ema(c, 26)
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    hist      = macd_line - macd_sig
    hist_ema  = hist.ewm(span=5, adjust=False).mean()
    hist_diff = float(hist.diff().iloc[-1]) if len(hist) > 1 else 0.0
    zero_cross_up = (macd_line.iloc[-2] <= 0 <= macd_line.iloc[-1]) if len(macd_line) > 1 else False
    zero_cross_down = (macd_line.iloc[-2] >= 0 >= macd_line.iloc[-1]) if len(macd_line) > 1 else False

    # OBV z-score
    obv  = obv_series(c, v)
    obv_d = obv.diff()
    mu = float(obv_d.ewm(span=5, adjust=False).mean().iloc[-1]) if len(obv_d.dropna()) else 0.0
    sd = float(obv_d.rolling(20).std(ddof=1).iloc[-1] or 0.0) if len(obv_d.dropna()) else 0.0
    z_obv = (float(obv_d.iloc[-1]) - mu) / (sd if sd > 0 else 1e9)

    # Relative volume
    vol_avg_win = int(cfg.get("vol_avg_win", 20))
    v_avg = v.rolling(vol_avg_win).mean()
    rvol_now = float(v.iloc[-1]) / max(1.0, float(v_avg.iloc[-1]) if len(v_avg.dropna()) else 1.0)

    # Simple MFI-ish up/down (consistent with base)
    tp = (h + l + c) / 3.0
    rmf = tp * v
    pos = (tp > tp.shift(1)).astype(float) * rmf
    neg = (tp < tp.shift(1)).astype(float) * rmf
    mfi_ratio = (pos.rolling(14, min_periods=7).sum() /
                 neg.replace(0, np.nan).rolling(14, min_periods=7).sum())
    mfi_now = float(100 - (100 / (1 + (mfi_ratio.iloc[-1] if not math.isnan(mfi_ratio.iloc[-1]) else 1.0))))
    mfi_up  = (mfi_ratio.diff().iloc[-1] or 0) > 0 if len(mfi_ratio.dropna()) else False

    # ROC vs ATR
    roc3 = c.pct_change(3)
    roc_atr_ratio = abs(float(roc3.iloc[-1])) / max(1e-9, atr_val / max(1e-9, float(c.iloc[-1])))

    # ADX/DI (expose DI values explicitly)
    a14, dip, dim = adx(h, l, c, 14)
    a9,  _,  _    = adx(h, l, c, 9)
    adx_rising = float(a14.diff().iloc[-1]) > 0 if len(a14) > 1 else False
    di_plus = float(dip.iloc[-1]) if len(dip) else 0.0
    di_minus = float(dim.iloc[-1]) if len(dim) else 0.0
    di_plus_gt = di_plus > di_minus

    # Bollinger width (% of price) + rank
    bw = bb_width_pct(c, n=cfg.get("bb_win", 20), k=cfg.get("bb_k", 2.0))
    bw_now = float(bw.iloc[-1]) if len(bw.dropna()) else 0.0
    bw_prev = float(bw.iloc[-2]) if len(bw.dropna()) > 1 else bw_now
    # rank (0-100) over last ~120 points
    tail = bw.tail(120).dropna()
    if len(tail) >= 20:
        bb_width_pct_rank = float((tail <= bw_now).mean() * 100.0)
    else:
        bb_width_pct_rank = 50.0
    # squeeze flag ~ bottom 20% of last 120
    squeeze_flag = int(bb_width_pct_rank <= 20.0)

    # Lookback N features for “_prev_n”
    n = max(1, int(cfg.get("div_lookback", 5)))
    close_prev_n = float(c.iloc[-n]) if len(c) > n else float(c.iloc[-1])
    low_prev_n   = float(l.iloc[-n]) if len(l) > n else float(l.iloc[-1])
    high_prev_n  = float(h.iloc[-n]) if len(h) > n else float(h.iloc[-1])
    rsi_prev5    = float(rsi_fast.iloc[-2]) if len(rsi_fast) > 1 else float(rsi_fast.iloc[-1])
    rsi5         = float(rsi_fast.iloc[-1]) if len(rsi_fast) else 50.0
    rsi_prev_std_n = float(rsi_std.iloc[-n]) if len(rsi_std) > n else float(rsi_std.iloc[-1])
    rmi_now      = float(rmi.iloc[-1]) if len(rmi) else 50.0
    rmi_prev_n   = float(rmi.iloc[-n]) if len(rmi) > n else rmi_now
    macd_hist_prev_n = float(hist.iloc[-n]) if len(hist) > n else float(hist.iloc[-1])
    bb_width_prev_n  = float(bw.iloc[-n]) if len(bw) > n else bw_now

    # EMA anchor
    ema50 = float(ema(c, 50).iloc[-1])

    # Whipsaw flips on hist (keep from base)
    flips_hist = _count_sign_flips(hist, look=12)

    feat: Dict[str, float | int | bool] = {
        # Price & volume
        "open": float(o.iloc[-1]), "close": float(c.iloc[-1]),
        "high": float(h.iloc[-1]), "low": float(l.iloc[-1]),
        "volume": float(v.iloc[-1]),
        "volume_avg_20": float(v_avg.iloc[-1]) if len(v_avg.dropna()) else 0.0,
        "rvol_now": float(rvol_now),

        # Vol & ATR
        "atr_pct": float(atr_pct_now),
        "atr_avg_20": float(atr_avg_20),

        # RSI/RMI
        "rsi_fast": float(rsi_fast.iloc[-1]) if len(rsi_fast) else 50.0,
        "rsi_std": float(rsi_std.iloc[-1]) if len(rsi_std) else 50.0,
        "rsi5": float(rsi5), "rsi_prev5": float(rsi_prev5),
        "rsi_prev_std_n": float(rsi_prev_std_n),
        "rmi": float(rmi_now), "rmi_now": float(rmi_now), "rmi_prev_n": float(rmi_prev_n),

        # MACD & histogram
        "macd_line": float(macd_line.iloc[-1]),
        "macd_sig": float(macd_sig.iloc[-1]),
        "hist": float(hist.iloc[-1]),
        "macd_hist": float(hist.iloc[-1]),
        "macd_hist_prev_n": float(macd_hist_prev_n),
        "hist_ema": float(hist_ema.iloc[-1]),
        "hist_diff": float(hist_diff),
        "zero_cross_up": bool(zero_cross_up),
        "zero_cross_down": bool(zero_cross_down),

        # OBV/MFI
        "z_obv": float(z_obv),
        "mfi_now": float(mfi_now),
        "mfi_up": bool(mfi_up),

        # ADX/DI
        "adx14": float(a14.iloc[-1]),
        "adx9": float(a9.iloc[-1]),
        "adx_rising": bool(adx_rising),
        "di_plus": float(di_plus),
        "di_minus": float(di_minus),
        "di_plus_gt": bool(di_plus_gt),

        # ROC/ATR
        "roc_atr_ratio": float(roc_atr_ratio),

        # Bands / squeeze / widths
        "bb_width_pct": float(bw_now),
        "bb_width": float(bw_now),               # alias for your INI
        "bb_width_pct_prev": float(bw_prev),
        "bb_width_prev_n": float(bb_width_prev_n),
        "bb_width_pct_rank": float(bb_width_pct_rank),
        "squeeze_flag": int(squeeze_flag),

        # Lookback comparators
        "close_prev_n": float(close_prev_n),
        "low_prev_n": float(low_prev_n),
        "high_prev_n": float(high_prev_n),

        # EMA anchor
        "ema50": float(ema50),

        # Whipsaw
        "whipsaw_flips": float(flips_hist),
    }
    return feat


def _eval_expr(expr: str, F: dict) -> bool:
    if not expr:
        return False
    return bool(eval(expr, {"__builtins__": None}, {k: F[k] for k in F}))


# -----------------------------
# Base momentum scorer (unchanged behavior)
# -----------------------------
def _momentum_score_base(dtf: pd.DataFrame, cfg: dict) -> Tuple[float, Dict[str, float], bool]:
    c = dtf["close"]; h = dtf["high"]; l = dtf["low"]; v = dtf["volume"]

    # Vol regime via ATR%
    ATR = atr(h, l, c, cfg["atr_win"])
    atr_pct = float((ATR.iloc[-1] / c.iloc[-1]) * 100.0)

    # Vol-aware RSI thresholds
    if atr_pct < cfg["low_vol_thr"]:
        rsi_thr = 60
    elif atr_pct < cfg["mid_vol_thr"]:
        rsi_thr = 65
    else:
        rsi_thr = 70

    rsi_fast = _rsi(c, cfg["rsi_fast"])
    rsi_std  = _rsi(c, cfg["rsi_std"])

    # RMI adaptive
    rmi_lb = 9 if atr_pct > cfg["mid_vol_thr"] else (21 if atr_pct < cfg["low_vol_thr"] else cfg["rmi_lb"])
    rmi_m  = 3 if atr_pct > cfg["mid_vol_thr"] else cfg["rmi_m"]
    rmi    = _rmi(c, lb=rmi_lb, m=rmi_m)

    # RMI vs RSI slope divergence (±10)
    rmi_slope = float(rmi.diff().iloc[-1]) if len(rmi) > 1 else 0.0
    rsi_slope = float(rsi_std.diff().iloc[-1]) if len(rsi_std) > 1 else 0.0
    rmi_pts = 10 if (rmi_slope > rsi_slope > 0) else (-10 if (rmi_slope < rsi_slope < 0) else 0)

    # RSI vol-aware score (0–15)
    rsi_val = float(rsi_std.iloc[-1]) if len(rsi_std) else 50.0
    if rsi_val >= rsi_thr:
        rsi_pts = 15
    elif rsi_val >= (rsi_thr - 5):
        rsi_pts = 8
    else:
        rsi_pts = 0

    # MACD hist vs EMA(hist) with volume-surge gating
    macd_line = ema(c, 12) - ema(c, 26)
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    hist      = macd_line - macd_sig
    hist_ema  = hist.ewm(span=5, adjust=False).mean()

    # Volume burst via OBVΔ z-score
    obv  = obv_series(c, v)
    obv_d = obv.diff()
    mu = float(obv_d.ewm(span=5, adjust=False).mean().iloc[-1]) if len(obv_d.dropna()) else 0.0
    sd = float(obv_d.rolling(20).std(ddof=1).iloc[-1] or 0.0) if len(obv_d.dropna()) else 0.0
    z  = (float(obv_d.iloc[-1]) - mu) / (sd if sd > 0 else 1e9)

    # MFI rising bonus
    tp = (h + l + c) / 3.0
    rmf = tp * v
    pos = (tp > tp.shift(1)).astype(float) * rmf
    neg = (tp < tp.shift(1)).astype(float) * rmf
    mfi_ratio = (pos.rolling(14, min_periods=7).sum() /
                 neg.replace(0, np.nan).rolling(14, min_periods=7).sum())
    mfi_up = (mfi_ratio.diff().iloc[-1] or 0) > 0 if len(mfi_ratio.dropna()) else False

    vol_pts = 15 if z >= 2.0 else (10 if z >= 1.0 else 0)
    if vol_pts > 0 and mfi_up:
        vol_pts = min(20, vol_pts + 5)

    # MACD points:
    zero_cross_now = (macd_line.iloc[-2] <= 0 <= macd_line.iloc[-1]) or \
                     (macd_line.iloc[-2] >= 0 >= macd_line.iloc[-1]) if len(macd_line) > 1 else False
    hist_above_ema = float(hist.iloc[-1]) > float(hist_ema.iloc[-1]) if len(hist) else False
    if zero_cross_now and z >= 2.0:
        macd_pts = 30
    elif hist_above_ema:
        macd_pts = 15
    else:
        macd_pts = 10 if zero_cross_now else 0

    # Whipsaw penalty / flag
    flips = _count_sign_flips(hist, look=12)
    whipsaw_flag = flips >= 4
    whipsaw_pen = -10 if whipsaw_flag else 0

    # ROC vs ATR adaptive (0–15)
    roc = c.pct_change(3)
    ratio = abs(float(roc.iloc[-1])) / max(1e-9, float(ATR.iloc[-1] / c.iloc[-1]))
    if ratio >= 1.0:
        roc_pts = 15
    elif ratio >= 0.7:
        roc_pts = 8
    else:
        roc_pts = 0

    # ADX slope + DI filter (0–15)
    a14, dip, dim = adx(h, l, c, 14)
    a9,  _,  _    = adx(h, l, c, 9)
    adx_ok = (float(a9.iloc[-1]) > float(a14.iloc[-1])) and \
             (float(a14.diff().iloc[-1]) > 0) and (float(dip.iloc[-1]) > float(dim.iloc[-1]))
    di_close = abs(float(dip.iloc[-1]) - float(dim.iloc[-1])) < 2.0
    adx_pts = 15 if adx_ok else (-5 if (float(a14.diff().iloc[-1]) > 0 and di_close) else 0)

    # Context bonus (post-squeeze breakout)
    bw = bb_width_pct(c, n=20, k=2.0)
    if len(bw.dropna()) > 40:
        p20 = float(np.nanpercentile(bw.tail(120).dropna(), 20)) if len(bw.dropna()) >= 50 else float(bw.dropna().quantile(0.2))
        squeeze = float(bw.iloc[-1]) <= p20
    else:
        squeeze = False
    ctx_bonus = 10 if (squeeze and hist_above_ema and rsi_pts > 0) else 0

    total = rmi_pts + rsi_pts + macd_pts + roc_pts + adx_pts + vol_pts + ctx_bonus + whipsaw_pen
    score = float(clamp(total, 0, 100))

    parts = {
        "rmi_pts": float(rmi_pts),
        "rsi_pts": float(rsi_pts),
        "macd_pts": float(macd_pts),
        "roc_pts": float(roc_pts),
        "adx_pts": float(adx_pts),
        "vol_pts": float(vol_pts),
        "ctx_bonus": float(ctx_bonus),
        "whipsaw_flips": float(flips),
        "atr_pct": float(atr_pct),
    }
    return score, parts, whipsaw_flag


# -----------------------------
# Public API (with scenario engine)
# -----------------------------
def score_momentum(symbol: str, kind: str, tf: str, df5: pd.DataFrame, base: BaseCfg, ini_path: str = DEFAULT_INI):
    cfg = _cfg(ini_path)
    dftf = resample(df5, tf)
    dftf = maybe_trim_last_bar(dftf)
    if not ensure_min_bars(dftf, tf) or len(dftf) < cfg.get("min_bars", 120):
        return None

    # Base model
    base_score, parts, whipsaw_flag = _momentum_score_base(dftf, cfg)
    total_score = base_score

    # Scenarios (optional)
    try:
        rules_mode, scenarios = _load_mom_scenarios(cfg["_ini_path"])
    except Exception:
        rules_mode, scenarios = (cfg.get("rules_mode", "additive"), [])

    if scenarios:
        F = _mom_features(dftf, cfg)
        scen_total = 0.0
        for sc in scenarios:
            if _eval_expr(sc["when"], F):
                scen_total += sc["score"]
                parts[f"SCN.{sc['name']}"] = float(sc["score"])
                if sc.get("bonus_when") and _eval_expr(sc["bonus_when"], F):
                    scen_total += sc["bonus"]
                    parts[f"SCN.{sc['name']}.bonus"] = float(sc["bonus"])

        total_score = scen_total if rules_mode == "override" else (base_score + scen_total)
        total_score = float(clamp(total_score, cfg.get("clamp_low", 0.0), cfg.get("clamp_high", 100.0)))

    ts = dftf.index[-1].to_pydatetime().replace(tzinfo=TZ)

    rows = [
        (symbol, kind, tf, ts, "MOM.score", float(total_score), json.dumps({
            "rules_mode": cfg.get("rules_mode", "additive")
        }), base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.whipsaw_flag", 1.0 if whipsaw_flag else 0.0, "{}", base.run_id, base.source),

        # debug parts (base)
        (symbol, kind, tf, ts, "MOM.RMI_adaptive", float(parts.get("rmi_pts", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.RSI_vol", float(parts.get("rsi_pts", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.MACD_hist_ema", float(parts.get("macd_pts", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.ROC_adaptive", float(parts.get("roc_pts", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.ADX_slope_DI", float(parts.get("adx_pts", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.Volume_burst", float(parts.get("vol_pts", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.Context_bonus", float(parts.get("ctx_bonus", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.whipsaw_flips", float(parts.get("whipsaw_flips", 0.0)), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "MOM.atr_pct", float(parts.get("atr_pct", 0.0)), "{}", base.run_id, base.source),
    ]

    # Optional: write scenario contributions
    if cfg.get("write_scenarios_debug", False):
        for k, v in parts.items():
            if k.startswith("SCN."):
                rows.append((symbol, kind, tf, ts, f"MOM.{k}", float(v), "{}", base.run_id, base.source))

    write_values(rows)
    return (ts, float(total_score))
