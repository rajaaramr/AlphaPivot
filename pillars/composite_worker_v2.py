# pillars/composite_worker_v2.py
"""
Composite Scorer for the AlphaPivot Trading System.

This module is responsible for:
- Orchestrating the execution of all the individual trading pillars.
- Blending the scores from the pillars into a single, composite score.
- Incorporating a non-linear model to refine the score.
- Applying vetoes from the pillars to generate a final trading signal.
- Writing the composite scores to the database for consumption by the executor.
"""
from __future__ import annotations

import os
import json
import math
import configparser
from typing import Dict, List, Optional, Iterable, Tuple
from datetime import datetime

import numpy as np
import psycopg2.extras as pgx

from utils.db import get_db_connection
from utils.configs import get_config_parser
from pillars.common import (
    TZ, load_5m, resample, write_values,
    last_metric, clamp, now_ts, maybe_trim_last_bar, ensure_min_bars, BaseCfg
)
from pillars.trend_pillar import score_trend
from pillars.momentum_pillar import score_momentum
from pillars.quality_pillar import score_quality
from pillars.flow_pillar import score_flow
from pillars.risk_pillar import score_risk
from pillars.structure_pillar import process_symbol as run_structure_for_symbol


# ---------- DB helpers ----------
def _insert_composite(rows: List[tuple]) -> int:
    """Inserts or updates composite score records in the database."""
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
    Returns the latest Non-Linear MTF score (0..100) for the given symbol.
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


# ---------- helpers ----------
def _score_to_prob(s: float) -> float:
    """Maps a 0-100 score to a 0-1 probability-like value."""
    z = (s - 50.0) / 12.0
    p = 1.0 / (1.0 + math.exp(-z))
    return float(clamp(p, 0.0, 1.0))


def _blend_mtf(values: Dict[str, float], weights: Dict[str,float]) -> Optional[float]:
    """Calculates a weighted average over the available timeframes."""
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
def process_symbol(symbol: str, *, kind: str, config: configparser.ConfigParser) -> int:
    """
    Processes a single symbol to generate and store its composite score.

    Args:
        symbol: The symbol to process.
        kind: The market kind ('futures' or 'spot').
        config: The application configuration object.

    Returns:
        The number of rows inserted into the database.
    """
    # --- Load Configuration ---
    core_cfg = config["core"]
    comp_cfg = config["composite"]

    base = BaseCfg(
        tfs=core_cfg.get("tfs", "25m,65m,125m").split(","),
        lookback_days=core_cfg.getint("lookback_days", 120),
        run_id=os.getenv("RUN_ID", "pillars_v2_run"),
        source=os.getenv("SRC", "pillars_v2")
    )

    pillar_weights = {
        "trend": comp_cfg.getfloat("w_trend", 0.22),
        "momentum": comp_cfg.getfloat("w_momentum", 0.22),
        "quality": comp_cfg.getfloat("w_quality", 0.18),
        "flow": comp_cfg.getfloat("w_flow", 0.14),
        "structure": comp_cfg.getfloat("w_structure", 0.14),
        "risk": comp_cfg.getfloat("w_risk", 0.10),
    }
    w_sum = sum(pillar_weights.values()) or 1.0
    pillar_weights = {k: v / w_sum for k, v in pillar_weights.items()}

    w_nl = comp_cfg.getfloat("w_nl", 0.12)
    write_5m = comp_cfg.getboolean("write_5m", True)

    tf_weights_raw = comp_cfg.get("tf_weights", "15m:0.30,30m:0.30,60m:0.25,120m:0.10,240m:0.05")
    tf_weights = {k.strip(): float(v) for k, v in (p.split(":") for p in tf_weights_raw.split(","))}
    tf_w_sum = sum(tf_weights.values()) or 1.0
    tf_weights = {k: v / tf_w_sum for k, v in tf_weights.items()}

    # 1) Load 5m data
    df5 = load_5m(symbol, kind, base.lookback_days)
    if df5.empty:
        print(f"⚠️ {kind}:{symbol} no candles")
        return 0

    # 2) Run STRUCTURE pillar
    try:
        run_structure_for_symbol(symbol, kind=kind)
    except Exception as e:
        print(f"⚠️ STRUCT compute failed for {kind}:{symbol}: {e}")

    # 3) Score all other pillars per timeframe
    per_tf = {"TREND": {}, "MOMENTUM": {}, "QUALITY": {}, "FLOW": {}, "RISK": {}}
    veto_tf = {"QUALITY": {}, "FLOW": {}, "RISK": {}}

    for tf in base.tfs:
        dftf = resample(df5, tf)
        if not ensure_min_bars(dftf, tf):
            continue

        for pillar_name, pillar_func in [
            ("TREND", score_trend), ("MOMENTUM", score_momentum),
            ("QUALITY", score_quality), ("FLOW", score_flow), ("RISK", score_risk)
        ]:
            try:
                res = pillar_func(symbol, kind, tf, df5, base)
                if res:
                    per_tf[pillar_name][tf] = float(res[1])
                    if pillar_name in veto_tf:
                        veto_tf[pillar_name][tf] = bool(res[2])
            except Exception as e:
                print(f"⚠️ {pillar_name} {kind}:{symbol}:{tf} → {e}")

    # 4) Blend scores and vetoes
    mtf_scores = {p: _blend_mtf(per_tf[p], tf_weights) for p in per_tf}
    mtf_vetoes = {p: any(veto_tf[p].values()) for p in veto_tf if veto_tf[p]}

    struct_mtf = last_metric(symbol, kind, "MTF", "STRUCT.score")
    struct_veto = last_metric(symbol, kind, "MTF", "STRUCT.veto_flag")
    mtf_scores["STRUCTURE"] = float(struct_mtf) if struct_mtf is not None else None
    mtf_vetoes["STRUCTURE"] = bool((struct_veto or 0.0) >= 0.5)

    # 5) Calculate final fused score
    nl_score = fetch_latest_nl(symbol, kind) or 50.0
    nz = lambda x: 50.0 if x is None else float(x)

    base_score = sum(pillar_weights[p.lower()] * nz(mtf_scores.get(p)) for p in pillar_weights)
    fused_score = (base_score + w_nl * nl_score) / (1.0 + w_nl)
    fused_score = clamp(fused_score, 0, 100)

    hard_veto = any(mtf_vetoes.values())
    final_score = 0.0 if hard_veto else fused_score
    final_prob = _score_to_prob(final_score)

    # 6) Prepare and insert records
    ts_mtf = now_ts()
    ts_5m = df5.index[-1].to_pydatetime().replace(tzinfo=TZ)

    ctx = {
        "pillar_w": pillar_weights, "tf_w": tf_weights,
        "nonlinear": {"score": nl_score, "w_nl": w_nl},
        "per_tf": per_tf, "vetos": mtf_vetoes
    }

    rows: List[tuple] = [(
        symbol, kind, "MTF", ts_mtf,
        nz(mtf_scores.get("TREND")), nz(mtf_scores.get("MOMENTUM")), nz(mtf_scores.get("QUALITY")),
        nz(mtf_scores.get("FLOW")), nz(mtf_scores.get("STRUCTURE")), nz(mtf_scores.get("RISK")),
        mtf_vetoes.get("QUALITY", False), mtf_vetoes.get("FLOW", False),
        mtf_vetoes.get("STRUCTURE", False), mtf_vetoes.get("RISK", False),
        final_score, final_prob, json.dumps(ctx),
        os.getenv("RUN_ID", "comp_v2"), os.getenv("SRC", "composite_v2")
    )]

    if write_5m:
        rows.append((
            symbol, kind, "5m", ts_5m,
            nz(mtf_scores.get("TREND")), nz(mtf_scores.get("MOMENTUM")), nz(mtf_scores.get("QUALITY")),
            nz(mtf_scores.get("FLOW")), nz(mtf_scores.get("STRUCTURE")), nz(mtf_scores.get("RISK")),
            mtf_vetoes.get("QUALITY", False), mtf_vetoes.get("FLOW", False),
            mtf_vetoes.get("STRUCTURE", False), mtf_vetoes.get("RISK", False),
            final_score, final_prob, json.dumps(ctx | {"mirrored_from": "MTF"}),
            os.getenv("RUN_ID", "comp_v2"), os.getenv("SRC", "composite_v2")
        ))

    n = _insert_composite(rows)
    print(f"✅ COMP2 {kind}:{symbol} → score={final_score:.1f} prob={final_prob:.2f} veto={hard_veto} (NL={nl_score:.1f}, w_nl={w_nl})")
    return n

# ---------- runner ----------
def run(symbols: Optional[List[str]] = None, kinds: Iterable[str] = ("futures","spot")) -> int:
    """
    Main runner function to process symbols.

    Args:
        symbols: A list of symbols to process. If None, discovers from webhooks.
        kinds: The market kinds to process.

    Returns:
        The total number of rows written to the database.
    """
    config = get_config_parser()

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
                total += process_symbol(s, kind=k, config=config)
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