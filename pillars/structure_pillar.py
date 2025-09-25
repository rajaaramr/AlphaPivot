# pillars/structure_pillar.py
from __future__ import annotations

import os
import json
import configparser
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable
from datetime import datetime

import numpy as np
import pandas as pd

from .common import (
    TZ, DEFAULT_INI as COMMON_DEFAULT_INI,
    load_5m, resample, write_values, last_metric,
    clamp, now_ts, ema, atr, bb_width_pct,
    _as_list_csv as csv, _parse_weights as weights_map,
    ensure_min_bars, maybe_trim_last_bar
)
from utils.db import get_db_connection

# Use its own INI file by default
DEFAULT_INI = os.getenv("STRUCTURE_INI", "structure.ini")


# ========= Config =========

@dataclass
class StructCfg:
    section: str
    metric_prefix: str
    tfs: List[str]
    lookback_days: int
    mtf_weights: Dict[str, float]
    w_stop: float
    w_reward: float
    w_trigger: float
    w_path: float
    w_anchor: float
    near_anchor_atr: float
    min_stop_atr: float
    max_stop_atr: float
    rr_bins: Tuple[float, float, float]
    vol_surge_k: float
    squeeze_pct: float
    wick_clean_thr: float
    wick_dirty_thr: float
    veto_rr_floor: float
    run_id: str
    source: str

def load_cfg(ini_path: str = DEFAULT_INI, section: str = "structure") -> StructCfg:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(ini_path)

    sect = section
    tfs = csv(cp.get(sect, "tfs", fallback="25m,65m,125m"))
    mtf_weights = weights_map(cp.get(sect, "mtf_weights", fallback="25m:0.3,65m:0.3,125m:0.4"), tfs)

    metric_prefix = cp.get(sect, "metric_prefix", fallback="STRUCT")

    return StructCfg(
        section=sect,
        metric_prefix=metric_prefix,
        tfs=tfs,
        lookback_days=cp.getint(sect, "lookback_days", fallback=120),
        mtf_weights=mtf_weights or {"25m":0.3,"65m":0.3,"125m":0.4},
        w_stop=cp.getfloat(sect, "w_stop", fallback=0.25),
        w_reward=cp.getfloat(sect, "w_reward", fallback=0.25),
        w_trigger=cp.getfloat(sect, "w_trigger", fallback=0.20),
        w_path=cp.getfloat(sect, "w_path", fallback=0.10),
        w_anchor=cp.getfloat(sect, "w_anchor", fallback=0.20),
        near_anchor_atr=cp.getfloat(sect, "near_anchor_atr", fallback=0.75),
        min_stop_atr=cp.getfloat(sect, "min_stop_atr", fallback=0.5),
        max_stop_atr=cp.getfloat(sect, "max_stop_atr", fallback=3.0),
        rr_bins=tuple(float(x) for x in csv(cp.get(sect, "rr_bins", fallback="1.0,1.5,2.0"))[:3]) or (1.0,1.5,2.0),
        vol_surge_k=cp.getfloat(sect, "vol_surge_k", fallback=2.0),
        squeeze_pct=cp.getfloat(sect, "squeeze_pct", fallback=25.0),
        wick_clean_thr=cp.getfloat(sect, "wick_clean_thr", fallback=1.0),
        wick_dirty_thr=cp.getfloat(sect, "wick_dirty_thr", fallback=2.0),
        veto_rr_floor=cp.getfloat(sect, "veto_rr_floor", fallback=0.9),
        run_id=cp.get(sect, "run_id", fallback=os.getenv("RUN_ID","struct_run")),
        source=cp.get(sect, "source", fallback=os.getenv("SRC","structure")),
    )

# ========= Core Structure scoring =========

def _anchor_dict(symbol: str, kind: str, tf: str) -> Dict[str, Optional[float]]:
    return {
        "poc": last_metric(symbol, kind, tf, "VP.POC"),
        "val": last_metric(symbol, kind, tf, "VP.VAL"),
        "vah": last_metric(symbol, kind, tf, "VP.VAH"),
        "bb_score": last_metric(symbol, kind, tf, "BB.score"),
    }

def _wick_body_ratio(o: float, h: float, l: float, c: float) -> float:
    body = abs(c - o); full = (h - l)
    if full <= 0: return 0.0
    upper = max(0.0, h - max(c,o))
    lower = max(0.0, min(c,o) - l)
    wick = upper + lower
    return float(wick / max(1e-9, body))

def _dir_bias(close: pd.Series) -> int:
    # quick bias using EMA stack
    ema10, ema20, ema50 = ema(close,10), ema(close,20), ema(close,50)
    if len(close) < 50: return 0
    if float(ema10.iloc[-1]) > float(ema20.iloc[-1]) > float(ema50.iloc[-1]): return +1
    if float(ema10.iloc[-1]) < float(ema20.iloc[-1]) < float(ema50.iloc[-1]): return -1
    return 0

def _stop_and_target(close: float, atr_val: float, anchors: Dict[str, Optional[float]],
                     direction: int, cfg: StructCfg) -> Tuple[Optional[float], Optional[float]]:
    poc, val, vah = anchors.get("poc"), anchors.get("val"), anchors.get("vah")
    if direction > 0:
        # stop below nearest support
        candidates = [x for x in [val, poc] if x is not None and x < close]
        stop = max(candidates) if candidates else close - cfg.min_stop_atr*atr_val
        # target: next resistance
        target = vah if (vah is not None and vah > close) else (close + 2.0*atr_val)
    elif direction < 0:
        candidates = [x for x in [vah, poc] if x is not None and x > close]
        stop = min(candidates) if candidates else close + cfg.min_stop_atr*atr_val
        target = val if (val is not None and val < close) else (close - 2.0*atr_val)
    else:
        return None, None
    return float(stop), float(target)

def _rr_score(rr: float, bins: Tuple[float,float,float]) -> float:
    a,b,c = bins
    if rr >= c: return 25.0
    if rr >= b: return 18.0
    if rr >= a: return 10.0
    return 0.0

def _trigger_quality(d: pd.DataFrame, atr_val: float, direction: int,
                     anchors: Dict[str, Optional[float]], cfg: StructCfg) -> float:
    if len(d) < 30 or direction == 0: return 8.0
    close = d["close"]; vol = d["volume"]
    vah, val = anchors.get("vah"), anchors.get("val")
    vol_avg = float(vol.rolling(20, min_periods=10).mean().iloc[-1])
    v = float(vol.iloc[-1])
    last = float(close.iloc[-1]); prev = float(close.iloc[-2]) if len(close)>1 else last
    bonus = 0.0
    # breakout check with vol surge
    if direction > 0 and vah is not None and last > float(vah) >= prev and v > cfg.vol_surge_k * max(1e-9, vol_avg):
        bonus = 20.0
    elif direction < 0 and val is not None and last < float(val) <= prev and v > cfg.vol_surge_k * max(1e-9, vol_avg):
        bonus = 20.0
    elif v > 1.2 * max(1e-9, vol_avg):
        bonus = 12.0
    else:
        bonus = 8.0
    # compression breakout add-on
    bb = bb_width_pct(close, 20, 2.0)
    if len(bb.dropna()) >= 40:
        pct = float(pd.Series(bb).rank(pct=True).iloc[-1])
        if pct <= cfg.squeeze_pct/100.0:
            bonus += 5.0
    return clamp(bonus, 0, 100)

def _path_cleanliness(o: float, h: float, l: float, c: float, cfg: StructCfg) -> float:
    r = _wick_body_ratio(o,h,l,c)
    if r <= cfg.wick_clean_thr: return 10.0
    if r >= cfg.wick_dirty_thr: return 0.0
    # linear fade between thresholds
    t0, t1 = cfg.wick_clean_thr, cfg.wick_dirty_thr
    return float(10.0 * (1.0 - (r - t0) / max(1e-9, (t1 - t0))))

def _anchor_confluence_score(close: float, atr_val: float, anchors: Dict[str, Optional[float]], cfg: StructCfg) -> float:
    near = 0
    for k in ("poc","val","vah"):
        v = anchors.get(k)
        if v is None: continue
        if abs(close - float(v)) <= cfg.near_anchor_atr * atr_val:
            near += 1
    if near >= 2: return 20.0
    if near == 1: return 12.0
    return 6.0

def _breakout_obstruction_penalty(close: float, stop: float, target: float, anchors: Dict[str, Optional[float]]) -> float:
    # simple obstruction: POC lying between entry and target
    poc = anchors.get("poc")
    if poc is None: return 0.0
    lo, hi = sorted([close, target])
    return -5.0 if (lo < float(poc) < hi) else 0.0

def _compute_structure_for_direction(d: pd.DataFrame, anchors: Dict[str, Optional[float]],
                                     direction: int, cfg: StructCfg) -> Tuple[float, Dict[str,float], bool]:
    close = float(d["close"].iloc[-1]); o = float(d["open"].iloc[-1])
    h = float(d["high"].iloc[-1]); l = float(d["low"].iloc[-1])
    atr_val = float(atr(d["high"], d["low"], d["close"], 14).iloc[-1])
    if atr_val <= 0 or direction == 0:
        return 50.0, {"stop":10,"reward":10,"trigger":10,"path":10,"anchor":10,"rr":1.0,"stop_atr":1.0}, False

    stop, target = _stop_and_target(close, atr_val, anchors, direction, cfg)
    veto = False
    if stop is None or target is None:
        return 40.0, {"stop":5,"reward":5,"trigger":10,"path":10,"anchor":10,"rr":0.0,"stop_atr":0.0}, True

    risk = abs(close - stop)
    move = abs(target - close)
    stop_atr = risk / atr_val if atr_val > 0 else 0.0
    rr = (move / risk) if risk > 0 else 0.0

    # stop viability (0–25)
    if cfg.min_stop_atr <= stop_atr <= cfg.max_stop_atr:
        stop_score = 25.0
    else:
        if stop_atr < cfg.min_stop_atr:
            stop_score = max(0.0, 25.0 * (stop_atr / max(cfg.min_stop_atr, 1e-9)))
        else:
            stop_score = max(0.0, 25.0 * (cfg.max_stop_atr / max(stop_atr, 1e-9)))

    # reward (0–25)
    reward_score = _rr_score(rr, cfg.rr_bins)

    # trigger (0–20)
    trigger_score = _trigger_quality(d, atr_val, direction, anchors, cfg)

    # path (0–10) + obstruction up to −5
    path_score = _path_cleanliness(o,h,l,close, cfg) + _breakout_obstruction_penalty(close, stop, target, anchors)
    path_score = max(0.0, min(10.0, path_score))  # keep within 0–10 after obstruction

    # anchor confluence (0–20)
    anchor_score = _anchor_confluence_score(close, atr_val, anchors, cfg)

    # hard vetoes (spec)
    if rr < cfg.veto_rr_floor:
        veto = True
    if stop_atr < 0.2 or stop_atr > 5.0:
        veto = True

    total = (
        cfg.w_stop   * stop_score +
        cfg.w_reward * reward_score +
        cfg.w_trigger* trigger_score +
        cfg.w_path   * path_score +
        cfg.w_anchor * anchor_score
    )
    comp = {
        "stop": stop_score, "reward": reward_score, "trigger": trigger_score,
        "path": path_score, "anchor": anchor_score, "rr": rr, "stop_atr": stop_atr
    }
    return clamp(total, 0, 100), comp, veto

def _blend_scores(scores: Dict[str,float], weights: Dict[str,float]) -> float:
    if not scores: return 50.0
    s, w = 0.0, 0.0
    for tf, val in scores.items():
        wt = float(weights.get(tf, 0.0))
        s += wt * val
        w += wt
    return float(s / w) if w > 0 else float(np.mean(list(scores.values())))

# ========= Public driver =========

def process_symbol(symbol: str, *, kind: str, cfg: Optional[StructCfg] = None) -> int:
    cfg = cfg or load_cfg()
    df5 = load_5m(symbol, kind, cfg.lookback_days)
    if df5.empty:
        print(f"⚠️ {kind}:{symbol} no 5m data for {cfg.metric_prefix}")
        return 0

    per_tf_scores: Dict[str,float] = {}
    per_tf_vetos: Dict[str,bool] = {}
    rows: List[tuple] = []

    for tf in cfg.tfs:
        dftf = resample(df5, tf)
        dftf = maybe_trim_last_bar(dftf)
        if not ensure_min_bars(dftf, tf):
            continue

        anchors = _anchor_dict(symbol, kind, tf)
        bias = _dir_bias(dftf["close"])

        s_up, comp_up, veto_up = _compute_structure_for_direction(dftf, anchors, +1, cfg)
        s_dn, comp_dn, veto_dn = _compute_structure_for_direction(dftf, anchors, -1, cfg)

        if bias > 0: direction, score, comp, veto = (+1, s_up, comp_up, veto_up)
        elif bias < 0: direction, score, comp, veto = (-1, s_dn, comp_dn, veto_dn)
        else:
            direction, score, comp, veto = ((+1, s_up, comp_up, veto_up) if s_up >= s_dn else (-1, s_dn, comp_dn, veto_dn))

        ts = dftf.index[-1].to_pydatetime().replace(tzinfo=TZ)
        ctx = {
            "dir": int(direction),
            "anchors": {k: float(v) if v is not None else None for k,v in anchors.items()},
            "rr": float(comp.get("rr", 0.0)),
            "stop_atr": float(comp.get("stop_atr", 0.0)),
            "variant": cfg.metric_prefix,
        }

        per_tf_scores[tf] = score
        per_tf_vetos[tf]  = bool(veto)

        P = cfg.metric_prefix
        rows += [
            (symbol, kind, tf, ts, f"{P}.stop",        float(comp["stop"]),   json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, f"{P}.reward",      float(comp["reward"]), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, f"{P}.trigger",     float(comp["trigger"]),json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, f"{P}.path",        float(comp["path"]),   json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, f"{P}.anchor",      float(comp["anchor"]), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, f"{P}.rr",          float(comp["rr"]),     json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, f"{P}.stop_atr",    float(comp["stop_atr"]),json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, f"{P}.score",       float(score),          json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, f"{P}.dir",         float(direction),      json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, f"{P}.veto_flag",   1.0 if veto else 0.0,  "{}",            cfg.run_id, cfg.source),
        ]

    if not per_tf_scores:
        print(f"ℹ️ {kind}:{symbol} {cfg.metric_prefix}: no TF rows")
        return 0

    # MTF blend (weighted average, and OR the veto flags)
    mtf_score = _blend_scores(per_tf_scores, cfg.mtf_weights)
    mtf_veto  = any(per_tf_vetos.get(tf, False) for tf in per_tf_scores.keys())
    ts_latest = now_ts()
    ctx_mtf = {"weights": cfg.mtf_weights, "tfs": list(per_tf_scores.keys()), "variant": cfg.metric_prefix}

    P = cfg.metric_prefix
    rows += [
        (symbol, kind, "MTF", ts_latest, f"{P}.score",     float(mtf_score), json.dumps(ctx_mtf), cfg.run_id, cfg.source),
        (symbol, kind, "MTF", ts_latest, f"{P}.veto_flag", 1.0 if mtf_veto else 0.0, "{}",       cfg.run_id, cfg.source),
    ]

    write_values(rows)
    print(f"✅ {cfg.metric_prefix} {kind}:{symbol} → mtf={mtf_score:.1f}, tf={ {k: round(v,1) for k,v in per_tf_scores.items()} } veto_mtf={mtf_veto}")
    return 1

def run(symbols: Optional[List[str]] = None, kinds: Iterable[str] = ("futures","spot"), ini_path: str = DEFAULT_INI) -> int:
    # discover symbols from active webhooks
    if symbols is None:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT symbol
                  FROM webhooks.webhook_alerts
                 WHERE status IN ('INDICATOR_PROCESS','SIGNAL_PROCESS','DATA_PROCESSING')
            """)
            rows = cur.fetchall()
        symbols = [r[0] for r in rows or []]

    # NEW: allow multiple variants from INI
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(ini_path)
    if cp.has_section("structure_runner"):
        variants = csv(cp.get("structure_runner", "variants", fallback="structure"))
    else:
        variants = ["structure"]  # default: old behavior

    total = 0
    for s in symbols:
        for k in kinds:
            for sect in variants:
                try:
                    cfg = load_cfg(ini_path, section=sect)
                    total += process_symbol(s, kind=k, cfg=cfg)
                except Exception as e:
                    print(f"❌ {sect} {k}:{s} → {e}")
    print(f"🎯 structure_pillar wrote rows for {total} operations across {len(symbols)} symbol(s)")
    return total

if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:]]
    syms = [a for a in args if not a.startswith("--")]
    run(syms or None)