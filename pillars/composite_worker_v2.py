# composite_worker_v2.py
from __future__ import annotations

import os, json, math, configparser
from typing import Dict, List, Optional, Iterable, Tuple
from datetime import datetime, timezone

import numpy as np
import psycopg2.extras as pgx

from utils.db import get_db_connection
from pillars.common import (
    TZ, DEFAULT_INI, load_base_cfg, load_5m, resample, write_values,
    last_metric, clamp, now_ts, maybe_trim_last_bar, ensure_min_bars
)
from pillars.trend_pillar import score_trend
from pillars.momentum_pillar import score_momentum
from pillars.quality_pillar import score_quality
from pillars.flow_pillar import score_flow
from pillars.risk_pillar import score_risk
from pillars.structure_pillar import process_symbol as run_structure_for_symbol


DEFAULT_COMP_INI = os.getenv("COMPOSITE_V2_INI", "composite_v2.ini")


# ---------- DB helpers ----------
def _insert_composite(rows: List[tuple]) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO indicators.composite_v2
            (symbol, market_type, interval, ts,
             trend, momentum, quality, flow, structure, risk,
             quality_veto, flow_veto, structure_veto, risk_veto,
             final_score, final_prob, context, run_id, source)
        VALUES %s
        ON CONFLICT (symbol, market_type, interval, ts) DO UPDATE
           SET trend=EXCLUDED.trend,
               momentum=EXCLUDED.momentum,
               quality=EXCLUDED.quality,
               flow=EXCLUDED.flow,
               structure=EXCLUDED.structure,
               risk=EXCLUDED.risk,
               quality_veto=EXCLUDED.quality_veto,
               flow_veto=EXCLUDED.flow_veto,
               structure_veto=EXCLUDED.structure_veto,
               risk_veto=EXCLUDED.risk_veto,
               final_score=EXCLUDED.final_score,
               final_prob=EXCLUDED.final_prob,
               context=EXCLUDED.context,
               run_id=EXCLUDED.run_id,
               source=EXCLUDED.source
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(cur, sql, rows, page_size=1000)
        conn.commit()
        return len(rows)


def fetch_latest_nl(symbol: str, kind: str = "futures") -> Optional[float]:
    """
    Returns latest Non-Linear MTF score (0..100) for symbol/kind, or None.
    Metric: indicators.values: interval='MTF', metric='CONF_NL.score.mtf'
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT val
              FROM indicators.values
             WHERE symbol=%s
               AND market_type=%s
               AND interval='MTF'
               AND metric='CONF_NL.score.mtf'
             ORDER BY ts DESC
             LIMIT 1
        """, (symbol, kind))
        row = cur.fetchone()
    return float(row[0]) if row else None


# ---------- INI loader ----------
def _load_comp_cfg(path: str = DEFAULT_COMP_INI) -> dict:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";","#"), interpolation=None, strict=False)
    cp.read(path)

    # Pillar weights (normalize among pillars only)
    w = {
        "trend":     cp.getfloat("composite","w_trend",     fallback=0.22),
        "momentum":  cp.getfloat("composite","w_momentum",  fallback=0.22),
        "quality":   cp.getfloat("composite","w_quality",   fallback=0.18),
        "flow":      cp.getfloat("composite","w_flow",      fallback=0.14),
        "structure": cp.getfloat("composite","w_structure", fallback=0.14),
        "risk":      cp.getfloat("composite","w_risk",      fallback=0.10),
    }
    s = sum(w.values()) or 1.0
    w = {k: v/s for k,v in w.items()}

    # Nonlinear contribution (kept separate so you can toggle without changing pillar sum)
    w_nl = cp.getfloat("composite", "w_nl", fallback=0.12)

    # MTF TF weights
    tfw_raw = cp.get("composite","tf_weights",fallback="15m:0.30,30m:0.30,60m:0.25,120m:0.10,240m:0.05")
    tfw: Dict[str,float] = {}
    for part in tfw_raw.split(","):
        if ":" in part:
            k,v = part.split(":",1)
            try:
                tfw[k.strip()] = float(v.strip())
            except:
                pass
    tfw_sum = sum(tfw.values()) or 1.0
    tfw = {k: v/tfw_sum for k,v in tfw.items()}

    # Also write a 5m composite row that mirrors the MTF blend
    write_5m = cp.getboolean("composite", "write_5m", fallback=True)

    return {"pillar_w": w, "w_nl": w_nl, "tf_w": tfw, "write_5m": write_5m}


# ---------- helpers ----------
def _score_to_prob(s: float) -> float:
    # map 0..100 score to a squashed 0..1 probability-like value
    z = (s - 50.0) / 12.0
    p = 1.0 / (1.0 + math.exp(-z))
    return float(clamp(p, 0.0, 1.0))


def _blend_mtf(values: Dict[str, float], weights: Dict[str,float]) -> Optional[float]:
    # weighted average over present TFs only
    num = den = 0.0
    for tf, v in values.items():
        if v is None:
            continue
        w = float(weights.get(tf, 0.0))
        if w <= 0:
            continue
        num += w * float(v)
        den += w
    return (num/den) if den > 0 else None


# ---------- core ----------
def process_symbol(symbol: str, *, kind: str,
                   pillars_ini: str = DEFAULT_INI,
                   comp_ini: str = DEFAULT_COMP_INI) -> int:
    base = load_base_cfg(pillars_ini)
    cfg  = _load_comp_cfg(comp_ini)

    # 1) Load 5m once
    df5 = load_5m(symbol, kind, base.lookback_days)
    if df5.empty:
        print(f"⚠️ {kind}:{symbol} no candles")
        return 0

    # 2) STRUCTURE pillar writes its own per-TF & MTF metrics
    try:
        run_structure_for_symbol(symbol, kind=kind)
    except Exception as e:
        print(f"⚠️ STRUCT compute failed for {kind}:{symbol}: {e}")

    # 3) Per-TF scoring for TREND/MOM/QUAL/FLOW/RISK
    per_tf = {"TREND": {}, "MOMENTUM": {}, "QUALITY": {}, "FLOW": {}, "RISK": {}}
    veto_tf = {"QUALITY": {}, "FLOW": {}, "RISK": {}}

    for tf in base.tfs:
        dftf = resample(df5, tf)
        dftf = maybe_trim_last_bar(dftf)
        if not ensure_min_bars(dftf, tf):
            continue

        # TREND
        try:
            t_res = score_trend(symbol, kind, tf, df5, base)
            if t_res:
                per_tf["TREND"][tf] = float(t_res[1])
        except Exception as e:
            print(f"⚠️ TREND {kind}:{symbol}:{tf} → {e}")

        # MOMENTUM
        try:
            m_res = score_momentum(symbol, kind, tf, df5, base)
            if m_res:
                per_tf["MOMENTUM"][tf] = float(m_res[1])
        except Exception as e:
            print(f"⚠️ MOMENTUM {kind}:{symbol}:{tf} → {e}")

        # QUALITY
        try:
            q_res = score_quality(symbol, kind, tf, df5, base)
            if q_res:
                per_tf["QUALITY"][tf] = float(q_res[1])
                veto_tf["QUALITY"][tf] = bool(q_res[2])
        except Exception as e:
            print(f"⚠️ QUALITY {kind}:{symbol}:{tf} → {e}")

        # FLOW
        try:
            f_res = score_flow(symbol, kind, tf, df5, base)
            if f_res:
                per_tf["FLOW"][tf] = float(f_res[1])
                veto_tf["FLOW"][tf] = bool(f_res[2])
        except Exception as e:
            print(f"⚠️ FLOW {kind}:{symbol}:{tf} → {e}")

        # RISK
        try:
            r_res = score_risk(symbol, kind, tf, df5, base)
            if r_res:
                per_tf["RISK"][tf] = float(r_res[1])
                veto_tf["RISK"][tf] = bool(r_res[2])
        except Exception as e:
            print(f"⚠️ RISK {kind}:{symbol}:{tf} → {e}")

    # 4) MTF blends for pillars
    Wtf = cfg["tf_w"]
    mtf_trend    = _blend_mtf(per_tf["TREND"],    Wtf)
    mtf_momentum = _blend_mtf(per_tf["MOMENTUM"], Wtf)
    mtf_quality  = _blend_mtf(per_tf["QUALITY"],  Wtf)
    mtf_flow     = _blend_mtf(per_tf["FLOW"],     Wtf)
    mtf_risk     = _blend_mtf(per_tf["RISK"],     Wtf)

    # STRUCTURE MTF (already persisted by structure pillar)
    struct_mtf  = last_metric(symbol, kind, "MTF", "STRUCT.score")
    struct_veto = last_metric(symbol, kind, "MTF", "STRUCT.veto_flag")
    mtf_structure  = float(struct_mtf) if struct_mtf is not None else None
    veto_structure = bool((struct_veto or 0.0) >= 0.5)

    # Veto OR across TFs
    veto_quality = any(veto_tf["QUALITY"].values()) if veto_tf["QUALITY"] else False
    veto_flow    = any(veto_tf["FLOW"].values())    if veto_tf["FLOW"]    else False
    veto_risk    = any(veto_tf["RISK"].values())    if veto_tf["RISK"]    else False

    # 5) Non-Linear (MTF) score (0..100); neutral to 50 if missing
    nl_score = fetch_latest_nl(symbol, kind) or 50.0

    nz = lambda x: 50.0 if x is None else float(x)
    w  = cfg["pillar_w"]
    w_nl = float(cfg["w_nl"])

    base_score = (
        w["trend"]    * nz(mtf_trend) +
        w["momentum"] * nz(mtf_momentum) +
        w["quality"]  * nz(mtf_quality) +
        w["flow"]     * nz(mtf_flow) +
        w["structure"]* nz(mtf_structure) +
        w["risk"]     * nz(mtf_risk)
    )
    fused_score = (base_score + w_nl * float(nl_score)) / (1.0 + w_nl)
    fused_score = clamp(fused_score, 0, 100)

    hard_veto   = bool(veto_quality or veto_flow or veto_structure or veto_risk)
    final_score = 0.0 if hard_veto else fused_score
    final_prob  = _score_to_prob(final_score)

    ts_mtf = now_ts()
    ts_5m  = df5.index[-1].to_pydatetime().replace(tzinfo=TZ)  # align to latest 5m bar

    ctx = {
        "pillar_w": w,
        "tf_w": Wtf,
        "nonlinear": {"score": nl_score, "w_nl": w_nl},
        "per_tf": per_tf,
        "vetos": {
            "quality": bool(veto_quality),
            "flow": bool(veto_flow),
            "structure": bool(veto_structure),
            "risk": bool(veto_risk),
        }
    }

    rows: List[tuple] = []

    # Row A: MTF (diagnostics/history)
    rows.append((
        symbol, kind, "MTF", ts_mtf,
        float(nz(mtf_trend)), float(nz(mtf_momentum)), float(nz(mtf_quality)),
        float(nz(mtf_flow)), float(nz(mtf_structure)), float(nz(mtf_risk)),
        bool(veto_quality), bool(veto_flow), bool(veto_structure), bool(veto_risk),
        float(final_score), float(final_prob), json.dumps(ctx),
        os.getenv("RUN_ID","comp_v2"), os.getenv("SRC","composite_v2")
    ))

    # Row B: 5m mirror for downstream JOINs
    if cfg["write_5m"]:
        rows.append((
            symbol, kind, "5m", ts_5m,
            float(nz(mtf_trend)), float(nz(mtf_momentum)), float(nz(mtf_quality)),
            float(nz(mtf_flow)), float(nz(mtf_structure)), float(nz(mtf_risk)),
            bool(veto_quality), bool(veto_flow), bool(veto_structure), bool(veto_risk),
            float(final_score), float(final_prob), json.dumps(ctx | {"mirrored_from":"MTF"}),
            os.getenv("RUN_ID","comp_v2"), os.getenv("SRC","composite_v2")
        ))

    n = _insert_composite(rows) if rows else 0
    print(f"✅ COMP2 {kind}:{symbol} → score={final_score:.1f} prob={final_prob:.2f} veto={hard_veto} (NL={nl_score:.1f}, w_nl={w_nl})")
    return int(n)


# ---------- runner ----------
def run(symbols: Optional[List[str]] = None, kinds: Iterable[str] = ("futures","spot")) -> int:
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

    total = 0
    for s in symbols:
        for k in kinds:
            try:
                total += process_symbol(s, kind=k)
            except Exception as e:
                print(f"❌ COMP2 {k}:{s} → {e}")
    print(f"🎯 COMP2 wrote {total} row(s) across {len(symbols)} symbol(s)")
    return total


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:]]
    syms = [a for a in args if not a.startswith("--")]
    ks = [a.split("=",1)[1] for a in args if a.startswith("--kinds=")]
    kinds = tuple(ks[0].split(",")) if ks else ("futures","spot")
    run(syms or None, kinds=kinds)
