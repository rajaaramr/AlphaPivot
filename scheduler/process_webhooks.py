# scheduler/process_webhooks.py
"""
Webhook Signal Processor for the AlphaPivot Trading System.

This module is responsible for the initial processing of incoming trading
alerts from webhooks. It acts as a gatekeeper, performing initial validation
and then routing valid alerts to the next stage of the data pipeline for
ingestion and further processing.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from utils.db import get_db_connection
from utils.configs import get_config_parser

# ---------------- Status constants ----------------
class SignalStatus:
    SIGNAL_PROCESS   = "SIGNAL_PROCESS"
    DATA_PROCESSING  = "DATA_PROCESSING"
    REJECTED         = "REJECTED"
    ERROR            = "ERROR"

class SubStatus:
    SIG_PENDING         = "SIG_PENDING"
    SIG_EVALUATING      = "SIG_EVALUATING"
    SIG_OK              = "SIG_OK"
    SIG_RETRY_WAIT      = "SIG_RETRY_WAIT"
    SIG_RETRY_DUE       = "SIG_RETRY_DUE"
    SIG_REJECTED        = "SIG_REJECTED"
    SIG_ERROR           = "SIG_ERROR"
    SIG_WAIT_MANUAL     = "SIG_WAIT_MANUAL"
    IND_OK              = "IND_OK"
    ZON_OK              = "ZON_OK"
    OI_OK               = "OI_OK"
    SIG_READY           = "SIG_READY"
    INGESTION_PENDING   = "INGESTION_PENDING"

# ---------------- Config / constants ----------------
BATCH_SIZE = 100
MAX_SIG_RETRIES = 3
RETRY_BACKOFF_MIN = [5, 15, 30]
TZ = timezone.utc

# ---------------- Types ----------------
@dataclass
class Alert:
    """Represents a trading alert fetched from the database."""
    unique_id: str
    symbol: str
    strategy: str
    payload: Optional[dict]
    received_at: datetime
    signal_type: Optional[str]

# ---------------- Helpers ----------------
def _utcnow() -> datetime:
    """Returns the current time in UTC."""
    return datetime.now(tz=TZ)

def _side_from_signal(signal_type: Optional[str]) -> Optional[str]:
    """Determines the trade side from a signal string."""
    if not signal_type:
        return None
    s = signal_type.strip().upper()
    if s in ("BUY", "LONG"):
        return "LONG"
    if s in ("SELL", "SHORT"):
        return "SHORT"
    return None

# ---------------- DB helpers ----------------
def _claim_signal_alerts(cur) -> List[Alert]:
    """Atomically claims a batch of pending alerts for processing."""
    sql = """
    UPDATE webhooks.webhook_alerts AS w
       SET sub_status = %s,
           last_checked_at = now()
     WHERE w.unique_id IN (
           SELECT unique_id
             FROM webhooks.webhook_alerts
            WHERE status = %s
              AND (
                    sub_status IN (%s,%s,%s,%s,%s)
                 OR (sub_status = %s AND now() >= COALESCE(next_retry_at, now()))
              )
              AND COALESCE(retry_count, 0) < %s
            ORDER BY COALESCE(next_retry_at, received_at) ASC, received_at ASC
            LIMIT %s
            FOR UPDATE SKIP LOCKED
     )
    RETURNING w.unique_id, w.symbol, COALESCE(w.strategy,'UNKNOWN') AS strategy,
              w.payload, w.received_at, w.signal_type;
    """
    cur.execute(
        sql,
        (
            SubStatus.SIG_EVALUATING, SignalStatus.SIGNAL_PROCESS,
            SubStatus.SIG_PENDING, SubStatus.IND_OK, SubStatus.ZON_OK,
            SubStatus.OI_OK, SubStatus.SIG_READY, SubStatus.SIG_RETRY_WAIT,
            MAX_SIG_RETRIES, BATCH_SIZE,
        )
    )
    rows = cur.fetchall()
    return [Alert(str(uid), sym, strat, pld, rcvd, st) for uid, sym, strat, pld, rcvd, st in rows]

# ... (other DB helpers remain unchanged) ...

# ---------------- Main runner ----------------
def process_signal_stage(single_run: bool = True) -> None:
    """
    Processes the initial signal stage of the webhook pipeline.
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        routed = rejected = errs = 0
        alerts = _claim_signal_alerts(cur)
        if not alerts:
            print("🔍 No signal alerts ready.")
            return

        print(f"🚦 Processing {len(alerts)} signal alert(s)...")
        for a in alerts:
            try:
                side = _side_from_signal(a.signal_type)
                if side not in ("LONG", "SHORT"):
                    _finalize_alert(cur, a.unique_id, status=SignalStatus.REJECTED, rejection_reason="Unsupported/missing side")
                    rejected += 1
                    continue

                if not _has_manual_override(cur, a.symbol, a.strategy):
                    _schedule_retry(cur, a.unique_id, _next_attempt_number(cur, a.unique_id), "No manual override found")
                    rejected += 1
                    continue

                _finalize_alert(cur, a.unique_id, status=SignalStatus.DATA_PROCESSING)
                _set_sub(cur, a.unique_id, SubStatus.INGESTION_PENDING)
                _clear_retry_fields_on_success(cur, a.unique_id)
                routed += 1

            except Exception as e:
                _finalize_alert(cur, a.unique_id, status=SignalStatus.ERROR, rejection_reason=str(e)[:500])
                errs += 1

        print(f"📝 Done. Routed to ingestion={routed}, Rejected={rejected}, Errors={errs}")

if __name__ == "__main__":
    process_signal_stage(single_run=True)