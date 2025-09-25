# scheduler/create_trade_setups.py
"""
Creates trade setups in the analytics.decisions_live table based on the
latest composite scores.

This script is the final link in the signal generation pipeline, translating
the composite scores into actionable trade setups that can be executed by the
trade executor.
"""
from __future__ import annotations

import psycopg2
import psycopg2.extras
from typing import Dict, List, Optional

from utils.db import get_db_connection
from utils.configs import get_config_parser


def fetch_latest_composite_scores(cur) -> List[Dict]:
    """
    Fetches the latest composite scores for all symbols.

    Args:
        cur: A database cursor object.

    Returns:
        A list of dictionaries, where each dictionary represents the latest
        composite score for a symbol.
    """
    cur.execute("""
        SELECT DISTINCT ON (symbol, market_type, interval)
               symbol, market_type, interval, ts, final_score, final_prob,
               quality_veto, flow_veto, structure_veto, risk_veto
          FROM indicators.composite_v2
         WHERE interval = '5m'
      ORDER BY symbol, market_type, interval, ts DESC
    """)
    return cur.fetchall()

def fetch_trade_params(cur, symbol: str, market_type: str, ref_tf: str, entry_tf: str) -> Dict[str, Optional[float]]:
    """
    Fetches the necessary parameters for calculating stop-loss and target price.

    Args:
        cur: A database cursor object.
        symbol: The trading symbol.
        market_type: The market type ('futures' or 'spot').
        ref_tf: The reference timeframe for structure metrics (e.g., '30m').
        entry_tf: The timeframe for the entry price (e.g., '5m').

    Returns:
        A dictionary containing the required parameters.
    """
    cur.execute("""
        SELECT DISTINCT ON (metric, interval)
               metric, interval, val
          FROM indicators.values
         WHERE symbol = %s
           AND market_type = %s
           AND (
                 (interval = %s AND metric IN ('STRUCT.rr', 'STRUCT.stop_atr', 'ATR')) OR
                 (interval = %s AND metric = 'CLOSE')
               )
         ORDER BY metric, interval, ts DESC
    """, (symbol, market_type, ref_tf, entry_tf))

    params = {}
    for row in cur.fetchall():
        params[row['metric']] = row['val']
    return params

def insert_decision(cur, setup: Dict):
    """
    Inserts a new trade decision into the analytics.decisions_live table.

    Args:
        cur: A database cursor object.
        setup: A dictionary containing the details of the trade setup.
    """
    cur.execute("""
        INSERT INTO analytics.decisions_live
            (symbol, ts, bias, fut_close, stop_px, target_1r,
             pcr, composite, status, instrument)
        VALUES
            (%(symbol)s, %(ts)s, %(bias)s, %(entry_px)s, %(stop_px)s, %(target_px)s,
             %(pcr)s, %(composite)s, 'OPEN_SETUP', %(instrument)s)
        ON CONFLICT (symbol, ts) DO NOTHING;
    """, setup)


def run():
    """
    Main function to create trade setups.
    """
    config = get_config_parser()
    decisions_cfg = config["decisions"]
    long_threshold = decisions_cfg.getfloat("long_setup_threshold", 70)
    short_threshold = decisions_cfg.getfloat("short_setup_threshold", 30)
    ref_tf = decisions_cfg.get("reference_tf", "30m")
    entry_tf = "5m"

    with get_db_connection() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        scores = fetch_latest_composite_scores(cur)

        for s in scores:
            hard_veto = (
                s["quality_veto"] or s["flow_veto"] or
                s["structure_veto"] or s["risk_veto"]
            )
            if hard_veto:
                continue

            bias = None
            if s["final_score"] >= long_threshold:
                bias = "LONG_SETUP"
            elif s["final_score"] <= short_threshold:
                bias = "SHORT_SETUP"

            if bias:
                params = fetch_trade_params(cur, s["symbol"], s["market_type"], ref_tf, entry_tf)

                entry_px = params.get("CLOSE")
                rr = params.get("STRUCT.rr")
                stop_atr = params.get("STRUCT.stop_atr")
                atr = params.get("ATR")

                if not all([entry_px, rr, stop_atr, atr]):
                    print(f"⚠️ Skipping {s['symbol']}: Missing required structure metrics for trade calculation.")
                    continue

                risk_in_points = float(stop_atr) * float(atr)

                if bias == "LONG_SETUP":
                    stop_px = float(entry_px) - risk_in_points
                    target_px = float(entry_px) + (risk_in_points * float(rr))
                else: # SHORT_SETUP
                    stop_px = float(entry_px) + risk_in_points
                    target_px = float(entry_px) - (risk_in_points * float(rr))

                setup = {
                    "symbol": s["symbol"],
                    "ts": s["ts"],
                    "bias": bias,
                    "entry_px": entry_px,
                    "stop_px": stop_px,
                    "target_px": target_px,
                    "pcr": s.get("pcr"),
                    "composite": s["final_score"],
                    "instrument": "FUTURES"
                }
                insert_decision(cur, setup)
                print(f"✅ Created {bias} for {s['symbol']} at score {s['final_score']:.2f}")

        conn.commit()


if __name__ == "__main__":
    run()