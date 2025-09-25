# utils/db_ops.py
"""
Centralized database operations for the AlphaPivot trading system.

This module provides a collection of helper functions for interacting with the
PostgreSQL/TimescaleDB database. It handles common tasks such as inserting
market data, logging run statuses, and fetching data for dashboards.

Key features:
- Safe UPSERTs using ON CONFLICT or row-existence checks.
- UTC-aware timestamp handling for all database operations.
- Explicit commits on all writer functions to ensure data integrity.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Iterable, Tuple, Literal
from datetime import datetime, timezone
from utils.db import get_db_connection
import psycopg2.extras as pgx

# ---------------------------------
# Constants
# ---------------------------------
TZ = timezone.utc
DEFAULT_INTERVAL = "5m"

# ---------------------------------
# Helpers
# ---------------------------------
def _as_aware_utc(ts_in) -> datetime:
    """
    Converts a timestamp input into a timezone-aware UTC datetime object.

    Handles integers, floats, strings, and naive datetime objects.

    Args:
        ts_in: The timestamp to convert.

    Returns:
        A timezone-aware datetime object in UTC.
    """
    try:
        if isinstance(ts_in, datetime):
            dt = ts_in
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=TZ)
            return dt.astimezone(TZ)
        if isinstance(ts_in, (int, float)):
            return datetime.fromtimestamp(float(ts_in), tz=TZ)
        if isinstance(ts_in, str):
            s = ts_in.strip().replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=TZ)
                return dt.astimezone(TZ)
            except Exception:
                return datetime.fromtimestamp(float(s), tz=TZ)
        return datetime.now(TZ)
    except Exception:
        return datetime.now(TZ)

def json_dumps(obj) -> str:
    """
    Safely serializes a Python object to a JSON string.

    Args:
        obj: The object to serialize.

    Returns:
        A JSON string representation of the object, or an empty JSON object
        string ("{}") if serialization fails.
    """
    try:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return "{}"

def log_run_status(*, run_id: str, job: str, symbol: str | None,
                   phase: str, status: str, error_code: str | None = None,
                   info: dict | None = None) -> None:
    """
    Logs the status of a job run to the journal.run_status table.

    Args:
        run_id: The unique identifier for the run.
        job: The name of the job being run.
        symbol: The symbol being processed, if applicable.
        phase: The current phase of the job (e.g., "START", "FINISH").
        status: The status of the job (e.g., "SUCCESS", "FAIL").
        error_code: An optional error code if the job failed.
        info: An optional dictionary of additional information.
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO journal.run_status
              (run_id, job, symbol, phase, status, error_code, info)
            VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb)
            """,
            (run_id, job, symbol, phase, status, error_code, json_dumps(info or {}))
        )
        conn.commit()

def _row_exists(cur, table: str, symbol: str, interval: str, ts: datetime) -> bool:
    """Checks if a specific row exists in a given table."""
    cur.execute(
        f"SELECT 1 FROM {table} WHERE symbol=%s AND interval=%s AND ts=%s LIMIT 1",
        (symbol, interval, ts),
    )
    return cur.fetchone() is not None

# ---------------------------------
# INSERTS / UPSERTS
# ---------------------------------
def insert_futures_price(
    *,
    ts: datetime | str,
    symbol: str,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    oi: int = 0,
    interval: str = DEFAULT_INTERVAL,
    source: Optional[str] = None,
) -> None:
    """
    Inserts or updates a futures price candle in the database.

    This function performs an "upsert" operation: if a candle for the given
    symbol, interval, and timestamp already exists, it will be updated.
    Otherwise, a new record will be inserted.

    Args:
        ts: The timestamp of the candle.
        symbol: The futures symbol.
        open: The opening price.
        high: The highest price.
        low: The lowest price.
        close: The closing price.
        volume: The trading volume.
        oi: The open interest.
        interval: The candle interval (e.g., "5m", "15m").
        source: The data source.
    """
    ts = _as_aware_utc(ts)
    volume = float(volume or 0.0)
    oi = int(oi or 0)

    with get_db_connection() as conn, conn.cursor() as cur:
        if _row_exists(cur, "market.futures_candles", symbol, interval, ts):
            cur.execute(
                """
                UPDATE market.futures_candles
                   SET open=%s, high=%s, low=%s, close=%s,
                       volume=GREATEST(%s, volume),
                       oi=GREATEST(%s, oi),
                       source=COALESCE(%s, source)
                 WHERE symbol=%s AND interval=%s AND ts=%s
                """,
                (open, high, low, close, volume, oi, source, symbol, interval, ts),
            )
        else:
            cur.execute(
                """
                INSERT INTO market.futures_candles
                  (symbol, interval, ts, open, high, low, close, volume, oi, source)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (symbol, interval, ts, open, high, low, close, volume, oi, source),
            )
        conn.commit()

def insert_spot_price(
    *,
    ts: datetime | str,
    symbol: str,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    interval: str = DEFAULT_INTERVAL,
    source: Optional[str] = None,
) -> None:
    """
    Inserts or updates a spot price candle in the database.

    This function performs an "upsert" operation for spot price data.

    Args:
        ts: The timestamp of the candle.
        symbol: The spot symbol.
        open: The opening price.
        high: The highest price.
        low: The lowest price.
        close: The closing price.
        volume: The trading volume.
        interval: The candle interval.
        source: The data source.
    """
    ts = _as_aware_utc(ts)
    volume = float(volume or 0.0)

    with get_db_connection() as conn, conn.cursor() as cur:
        if _row_exists(cur, "market.spot_candles", symbol, interval, ts):
            cur.execute(
                """
                UPDATE market.spot_candles
                   SET open=%s, high=%s, low=%s, close=%s,
                       volume=GREATEST(%s, volume),
                       source=COALESCE(%s, source)
                 WHERE symbol=%s AND interval=%s AND ts=%s
                """,
                (open, high, low, close, volume, source, symbol, interval, ts),
            )
        else:
            cur.execute(
                """
                INSERT INTO market.spot_candles
                  (symbol, interval, ts, open, high, low, close, volume, source)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (symbol, interval, ts, open, high, low, close, volume, source),
            )
        conn.commit()

# ---------------------------------
# Adapters (compat with older code)
# ---------------------------------
def insert_webhook_alert(
    *, symbol: str, strategy_name: str, payload_json: dict | str,
    timeframe: Optional[str] = None, source: str = "TradingView",
    status: str = "PENDING", signal_type: Optional[str] = None,
    strategy_version: Optional[str] = None, rule_version: Optional[str] = None,
) -> str:
    """
    Inserts a new webhook alert into the database.

    This function is designed to handle alerts from external sources like
    TradingView and store them for processing.

    Args:
        symbol: The symbol associated with the alert.
        strategy_name: The name of the strategy that generated the alert.
        payload_json: The raw JSON payload of the alert.
        timeframe: The timeframe of the alert.
        source: The source of the alert.
        status: The initial status of the alert.
        signal_type: The type of signal.
        strategy_version: The version of the strategy.
        rule_version: The version of the rule.

    Returns:
        The unique ID of the inserted alert.
    """
    # embed extras into payload for audit
    if isinstance(payload_json, dict):
        pj = dict(payload_json)
        if signal_type is not None:        pj.setdefault("signal_type", signal_type)
        if strategy_version is not None:   pj.setdefault("strategy_version", strategy_version)
        if rule_version is not None:       pj.setdefault("rule_version", rule_version)
        payload = json_dumps(pj)
    else:
        payload = str(payload_json)

    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO webhooks.webhook_alerts
                (received_at, source, strategy, symbol, timeframe, payload, status, last_checked_at)
            VALUES (NOW(), %s, %s, %s, %s, %s::jsonb, %s, NOW())
            RETURNING unique_id;
            """,
            (source, strategy_name, symbol, timeframe, payload, status),
        )
        uid = cur.fetchone()[0]
        conn.commit()
        return str(uid)

def get_dashboard_rows(limit_alerts: int = 20, limit_trades: int = 20) -> Dict[str, Any]:
    """
    Fetches data for the main dashboard.

    This function retrieves a summary of the system's status, including
    counts of open trades and recent alerts.

    Args:
        limit_alerts: The maximum number of recent alerts to fetch.
        limit_trades: The maximum number of recent trades to fetch.

    Returns:
        A dictionary containing dashboard data.
    """
    out: Dict[str, Any] = {
        "counts": {"open_trades": 0, "pending_alerts": 0, "rejected_today": 0},
        "recent_alerts": [],
        "recent_trades": []
    }

    with get_db_connection() as conn, conn.cursor() as cur:
        try:
            cur.execute("SELECT COUNT(*) FROM journal.trading_journal WHERE status='OPEN';")
            out["counts"]["open_trades"] = int(cur.fetchone()[0])
        except Exception:
            pass

        try:
            cur.execute("SELECT COUNT(*) FROM webhooks.webhook_alerts WHERE status='PENDING';")
            out["counts"]["pending_alerts"] = int(cur.fetchone()[0])
        except Exception:
            pass

        try:
            cur.execute("""
                SELECT COUNT(*)
                  FROM webhooks.webhook_alerts
                 WHERE status='REJECTED'
                   AND DATE(received_at AT TIME ZONE 'UTC') = DATE(now() AT TIME ZONE 'UTC');
            """)
            out["counts"]["rejected_today"] = int(cur.fetchone()[0])
        except Exception:
            pass

        try:
            cur.execute("""
                SELECT symbol, status, COALESCE(received_at, last_checked_at) AS ts
                  FROM webhooks.webhook_alerts
                 ORDER BY ts DESC
                 LIMIT %s;
            """, (limit_alerts,))
            out["recent_alerts"] = [
                {"symbol": r[0], "status": r[1], "ts": r[2].isoformat() if r[2] else None}
                for r in cur.fetchall()
            ]
        except Exception:
            pass

        try:
            cur.execute("""
                SELECT symbol, side, status, entry_ts, exit_ts, decision_score
                  FROM journal.trading_journal
                 ORDER BY COALESCE(exit_ts, entry_ts) DESC
                 LIMIT %s;
            """, (limit_trades,))
            out["recent_trades"] = [
                {
                    "symbol": r[0], "side": r[1], "status": r[2],
                    "entry_ts": r[3].isoformat() if r[3] else None,
                    "exit_ts":  r[4].isoformat() if r[4] else None,
                    "score": float(r[5]) if r[5] is not None else None
                }
                for r in cur.fetchall()
            ]
        except Exception:
            pass

    return out


def insert_futures_bar(*, symbol, interval, ts, open_price, high_price, low_price, close_price, volume, oi):
    """
    Adapter function to insert a futures bar with keyword arguments matching
    a different convention.
    """
    insert_futures_price(
        ts=ts, symbol=symbol,
        open=open_price, high=high_price, low=low_price, close=close_price,
        volume=volume, oi=oi, interval=interval
    )

def insert_spot_bar(*, symbol, interval, ts, open_price, high_price, low_price, close_price, volume):
    """
    Adapter function to insert a spot bar with keyword arguments matching
    a different convention.
    """
    insert_spot_price(
        ts=ts, symbol=symbol,
        open=open_price, high=high_price, low=low_price, close=close_price,
        volume=volume, interval=interval
    )