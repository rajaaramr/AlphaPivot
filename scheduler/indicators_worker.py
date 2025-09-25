# scheduler/indicators_worker.py
"""
Indicator Calculation Worker for the AlphaPivot Trading System.

This module is responsible for calculating a suite of technical indicators
for all symbols in the universe. It is designed to be idempotent and safe
for concurrent execution. It fetches data since the last calculation to
avoid redundant processing.
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Optional, Iterable, List, Tuple, Dict, TypedDict
from datetime import datetime, timezone, timedelta

import pandas as pd
import psycopg2.extras as pgx

from utils.db import get_db_connection
from utils.configs import get_config_parser
from scheduler import nonlinear_features as nlf, update_vp_bb as vpbb, track_zone_breaks as tzb, update_indicators_multi_tf as classic

# =========================
# Globals / Config
# =========================
TZ = timezone.utc
CFG = get_config_parser()

# ---- runtime switches ----
SOURCE = os.getenv("IND_SOURCE", "universe").lower()
UNIVERSE_NAME = os.getenv("UNIVERSE_NAME", CFG.get("universe", "name", fallback="largecaps_v1"))
BASE_INTERVAL = os.getenv("BASE_INTERVAL", CFG.get("live", "interval", fallback="15m"))
IND_LOOKBACK_DAYS = CFG.getint("indicators", "indicator_lookback_days", fallback=10)
WORKER_RUN_ID = os.getenv("RUN_ID", datetime.now(TZ).strftime("%Y%m%dT%H%M%SZ_ind"))

def _parse_flags(argv: list[str]) -> None:
    """Parses command-line flags to override configuration."""
    global SOURCE, UNIVERSE_NAME, BASE_INTERVAL
    it = iter(argv)
    for tok in it:
        if tok == "--source":
            SOURCE = next(it, SOURCE).lower()
        elif tok == "--universe":
            UNIVERSE_NAME = next(it, UNIVERSE_NAME)
        elif tok == "--base-interval":
            BASE_INTERVAL = next(it, BASE_INTERVAL)

class WorkItem(TypedDict):
    unique_id: Optional[str]
    symbol: str

# =========================
# DB helpers & status
# =========================
def _exec(sql: str, params):
    """Executes a SQL statement."""
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        conn.commit()

def set_status(unique_id, *, status=None, sub_status=None, error=None):
    """Sets the status of a webhook alert."""
    if SOURCE != "webhooks" or not unique_id:
        return
    sets, vals = [], []
    if status: sets.append("status=%s"); vals.append(status)
    if sub_status: sets.append("sub_status=%s"); vals.append(sub_status)
    if error: sets.append("last_error=%s"); vals.append(error)
    if not sets: return
    vals.append(unique_id)
    _exec(f"UPDATE webhooks.webhook_alerts SET {', '.join(sets)}, last_checked_at=NOW() WHERE unique_id=%s", vals)

def load_intra_from_db(symbol: str, kind: str) -> pd.DataFrame:
    """
    Loads intraday data for a given symbol and market kind from the database.
    It fetches data since the last indicator calculation timestamp, with a
    fallback to a configured lookback period.
    """
    tbl = "market.spot_candles" if kind == "spot" else "market.futures_candles"
    intervals_to_try = [BASE_INTERVAL] + ([] if BASE_INTERVAL == "15m" else ["15m"])

    cutoff = None
    with get_db_connection() as conn, conn.cursor() as cur:
        last_ts_col = f"last_ind_{kind}_at"
        cur.execute(f"SELECT {last_ts_col} FROM reference.symbol_universe WHERE symbol = %s", (symbol,))
        row = cur.fetchone()
        if row and row[0]:
            cutoff = row[0] - timedelta(days=2)

    if cutoff is None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=IND_LOOKBACK_DAYS)

    with get_db_connection() as conn, conn.cursor() as cur:
        for iv in intervals_to_try:
            cur.execute(
                f"""
                SELECT ts, (open)::float8 AS open, (high)::float8 AS high,
                       (low)::float8 AS low, (close)::float8 AS close,
                       COALESCE(volume,0)::float8 AS volume
                  FROM {tbl}
                 WHERE symbol=%s AND interval=%s AND ts >= %s
                 ORDER BY ts ASC
                """,
                (symbol, iv, cutoff)
            )
            rows = cur.fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
                df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                return df.dropna(subset=["ts"]).set_index("ts").sort_index()
    return pd.DataFrame()

def get_last_ts_from_db(symbol: str, kind: str, tf: str, metric: str) -> Optional[pd.Timestamp]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT ts FROM indicators.values
             WHERE symbol=%s AND market_type=%s AND interval=%s AND metric=%s
             ORDER BY ts DESC LIMIT 1
        """, (symbol, kind, tf, metric))
        row = cur.fetchone()
    return pd.to_datetime(row[0], utc=True) if row else None

def upsert_indicator_rows(rows: List[tuple]) -> int:
    if not rows:
        return 0
    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(cur, """
            INSERT INTO indicators.values
                (symbol, market_type, interval, ts, metric, val, context, run_id, source)
            VALUES %s
            ON CONFLICT (symbol, market_type, interval, ts, metric) DO NOTHING
        """, rows, page_size=1000)
        conn.commit()
        return len(rows)

def _update_universe_last_ind(symbol: str, kind: str, run_id: str = WORKER_RUN_ID) -> None:
    # ... (implementation remains the same)
    pass

def _call_classic(symbol: str, *, kind: str) -> Tuple[int, int]:
    """
    Builds indicator rows via the classic module and writes them.
    """
    P = classic.load_cfg()
    out = classic.update_indicators_multi_tf(
        symbols=[symbol],
        kinds=(kind,),
        load_5m=load_intra_from_db,
        get_last_ts=get_last_ts_from_db,
        P=P
    )
    rows: List[tuple] = out.get("rows", [])
    inserted = upsert_indicator_rows(rows)
    return len(rows), inserted

def run_once(limit: int = 50, kinds: Iterable[str] = ("spot", "futures")):
    # ... (implementation remains the same)
    pass

if __name__ == "__main__":
    _parse_flags(sys.argv[1:])
    run_once()