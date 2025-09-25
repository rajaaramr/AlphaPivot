# scheduler/recheck_rejected.py
"""
Re-evaluation Engine for Rejected Alerts.

This module provides a mechanism for periodically re-evaluating previously
rejected trading alerts. It safely claims a batch of rejected alerts, runs
them through the rule engine again, and if they are now valid, creates a
new trade entry.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence

from utils.db import get_db_connection
from utils.rule_evaluator import evaluate_alert
from utils.configs import get_config_parser

TZ = timezone.utc

# --- Table Names ---
ALERTS_TABLE = "webhooks.webhook_alerts"
TRADES_TABLE = "journal.trading_journal"

def _utcnow_str() -> str:
    """Returns the current time in UTC as a string."""
    return datetime.now(tz=TZ).strftime("%Y-%m-%d %H:%M:%S")

def load_config_flag() -> bool:
    """Checks the config file to see if re-checking is enabled."""
    config = get_config_parser()
    return config.getboolean("recheck", "retry_rejected_alerts", fallback=False)

def _claim_rejected(conn, batch_size: int = 200) -> Sequence[tuple]:
    """Atomically claims a batch of rejected alerts for re-checking."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            WITH cte AS (
              SELECT id
              FROM {ALERTS_TABLE}
              WHERE status = 'REJECTED'
              ORDER BY COALESCE(alert_time, timestamp) ASC
              LIMIT %s
              FOR UPDATE SKIP LOCKED
            )
            UPDATE {ALERTS_TABLE} wa
            SET status = 'RECHECKING',
                last_checked_at = %s,
                last_message = %s
            FROM cte
            WHERE wa.id = cte.id
            RETURNING wa.id, wa.symbol, wa.strategy_name, wa.strategy_version, wa.signal_type
            """,
            (batch_size, _utcnow_str(), "retry"),
        )
        return cur.fetchall()

def _insert_trading_journal(cursor, alert_id: int, symbol: str, strategy: str,
                            version: str, signal_type: str, rule_matched: str,
                            score: float, decision_tags: Optional[Iterable[str]] = None):
    """Inserts a new trade into the trading journal."""
    cursor.execute(
        f"""
        INSERT INTO {TRADES_TABLE} (
            alert_id, timestamp, symbol, strategy_name, strategy_version,
            signal_type, rule_matched, confidence_score, rule_engine_version,
            decision_tags, status
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'OPEN')
        """,
        (alert_id, _utcnow_str(), symbol, strategy or "UNKNOWN", version or "v1.0",
         (signal_type or "UNKNOWN").upper(), rule_matched, float(score or 0),
         "retry", ",".join(list(decision_tags or [])))
    )

def reprocess_rejected_alerts(batch_size: int = 200) -> None:
    """
    Re-evaluates rejected alerts and creates new trades if they are now valid.
    """
    with get_db_connection() as conn:
        claimed = _claim_rejected(conn, batch_size=batch_size)
        if not claimed:
            print("🔁 No rejected alerts to reprocess")
            return

        print(f"🔁 Reprocessing {len(claimed)} rejected alerts…")

        with conn.cursor() as cursor:
            for alert_id, symbol, strategy, version, signal in claimed:
                try:
                    result = evaluate_alert(symbol)
                    is_valid, rule_matched, failed_reasons, score, decision_tags, _ = result

                    if is_valid:
                        _insert_trading_journal(
                            cursor, alert_id, symbol, strategy, version or "v1.0",
                            signal or "LONG", rule_matched, score, decision_tags
                        )
                        cursor.execute(
                            f"UPDATE {ALERTS_TABLE} SET status = 'ACCEPTED', last_checked_at = %s, "
                            f"last_message = %s WHERE id = %s",
                            (_utcnow_str(), (rule_matched or "")[:255], alert_id)
                        )
                    else:
                        msg = "; ".join(f"{r}:{rsn}" for r, rsn in (failed_reasons or []))[:255]
                        cursor.execute(
                            f"UPDATE {ALERTS_TABLE} SET status = 'REJECTED', last_checked_at = %s, "
                            f"last_message = %s WHERE id = %s",
                            (_utcnow_str(), msg, alert_id)
                        )
                except Exception as e:
                    cursor.execute(
                        f"UPDATE {ALERTS_TABLE} SET status = 'ERROR', last_checked_at = %s, "
                        f"last_message = %s WHERE id = %s",
                        (_utcnow_str(), str(e)[:255], alert_id)
                    )
        conn.commit()
        print("✅ Recheck complete.")

if __name__ == "__main__":
    if load_config_flag():
        reprocess_rejected_alerts()
    else:
        print("⚠️ Retry of rejected alerts is disabled in config.ini. Exiting.")