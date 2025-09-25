# scheduler/fetch_and_process_alerts.py
"""
Webhook Alert Processor for the AlphaPivot Trading System.

This module is responsible for processing pending webhook alerts, evaluating
them against a set of rules, and creating trade entries in the journal and
the decisions_live table.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from kiteconnect import KiteConnect

from utils.db import get_db_connection
from utils.kite_utils import fetch_futures_data
from utils.rule_evaluator import evaluate_alert
from utils.exit_rule_evaluator import evaluate_exit
from utils.option_strikes import pick_nearest_option
from utils.db_ops import fetch_latest_zone_data
from utils.kite_session import load_kite

# ---------------- Config / constants ----------------
BATCH_SIZE = 100
TZ = timezone.utc

# ------------- Types --------------------
@dataclass
class Alert:
    """Represents a trading alert fetched from the database."""
    unique_id: str
    symbol: str
    strategy: str
    payload: Optional[dict]
    received_at: datetime

# ------------- Utils --------------------
def _utcnow() -> datetime:
    """Returns the current time in UTC."""
    return datetime.now(tz=TZ)

def _side_from_signal(signal_type: Optional[str]) -> Optional[str]:
    """Determines the trade side from a signal string."""
    if not signal_type:
        return None
    s = signal_type.strip().lower()
    if s in ("buy", "long"):
        return "LONG"
    if s in ("sell", "short"):
        return "SHORT"
    if s in ("close", "exit"):
        return "CLOSE"
    return None

# --------- DB helpers (Timescale) --------
def _claim_pending_alerts(cur) -> List[Alert]:
    """Atomically claims a batch of pending alerts for processing."""
    sql = """
    UPDATE webhooks.webhook_alerts AS w
       SET status = 'PROCESSING',
           last_checked_at = now()
     WHERE w.unique_id IN (
           SELECT unique_id
             FROM webhooks.webhook_alerts
            WHERE status = 'PENDING'
            ORDER BY received_at ASC
            LIMIT %s
            FOR UPDATE SKIP LOCKED
     )
    RETURNING w.unique_id, w.symbol, w.strategy, w.payload, w.received_at;
    """
    cur.execute(sql, (BATCH_SIZE,))
    rows = cur.fetchall()
    alerts: List[Alert] = []
    for uid, sym, strat, payload, rcvd in rows:
        payload_dict = payload if isinstance(payload, dict) else None
        alerts.append(Alert(str(uid), sym, strat or "UNKNOWN", payload_dict, rcvd))
    return alerts

def _next_attempt_number(cur, unique_id: str) -> int:
    cur.execute(
        "SELECT COALESCE(MAX(attempt_number),0) FROM journal.rejections_log WHERE unique_id=%s;",
        (unique_id,)
    )
    return int(cur.fetchone()[0] or 0) + 1

def _log_attempt(cur, unique_id: str, attempt_number: int, status: str,
                 rejection_reason: Optional[str] = None, trigger: Optional[str] = None, notes: Optional[str] = None):
    cur.execute(
        """
        INSERT INTO journal.rejections_log
            (unique_id, attempt_number, processed_at, status, rejection_reason, re_run_trigger, notes)
        VALUES (%s,%s, now(), %s, %s, %s, %s);
        """,
        (unique_id, attempt_number, status, rejection_reason, trigger, notes)
    )

def _finalize_alert(cur, unique_id: str, status: str, rejection_reason: Optional[str] = None):
    cur.execute(
        """
        UPDATE webhooks.webhook_alerts
           SET status = %s,
               last_checked_at = now(),
               rejection_reason = COALESCE(%s, rejection_reason)
         WHERE unique_id = %s;
        """,
        (status, rejection_reason, unique_id)
    )

def _insert_entry(cur,
                  unique_id: str,
                  symbol: str,
                  side: Optional[str],
                  entry_price_fut: Optional[float],
                  rule_matched: Optional[str],
                  decision_score: Optional[float],
                  option_type: Optional[str],
                  option_strike: Optional[float],
                  option_symbol: Optional[str],
                  market_data: Dict):
    """
    Inserts a new trade entry into the journal and creates a decision record.
    """
    cur.execute(
        """
        INSERT INTO journal.trading_journal
            (unique_id, symbol, side, entry_ts,
             entry_price_fut, rule_matched, decision_score,
             option_type, option_strike, option_symbol,
             status)
        VALUES (%s,%s,%s, now(), %s,%s,%s, %s,%s,%s, 'OPEN');
        """,
        (unique_id, symbol, side, entry_price_fut, rule_matched,
         float(decision_score or 0), option_type, option_strike, option_symbol)
    )

    if entry_price_fut:
        stop_px = entry_price_fut * 0.99 if side == "LONG" else entry_price_fut * 1.01
        target_px = entry_price_fut * 1.02 if side == "LONG" else entry_price_fut * 0.98

        cur.execute(
            """
            INSERT INTO analytics.decisions_live
                (symbol, ts, bias, fut_close, stop_px, target_1r, composite, status, instrument)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'OPEN_SETUP', %s)
            ON CONFLICT (symbol, ts) DO NOTHING;
            """,
            (symbol, _utcnow(), f"{side}_SETUP", entry_price_fut, stop_px, target_px,
             decision_score, "FUTURES")
        )

def _fetch_latest_open_trade(cur, symbol: str):
    cur.execute(
        """
        SELECT trade_id, side, entry_price_fut
          FROM journal.trading_journal
         WHERE symbol=%s AND status='OPEN'
         ORDER BY entry_ts DESC
         LIMIT 1;
        """,
        (symbol,)
    )
    return cur.fetchone()

def _close_trade(cur, trade_id: int, exit_price_fut: float, reason: str,
                 score: float, exit_direction: str, entry_side: Optional[str], entry_price_fut: Optional[float]):
    side = (entry_side or "LONG").upper()
    ep = float(entry_price_fut or 0.0)
    pnl_raw = (exit_price_fut - ep) if side == "LONG" else (ep - exit_price_fut)
    pnl_pct = (pnl_raw / ep * 100.0) if ep else 0.0

    cur.execute(
        """
        UPDATE journal.trading_journal
           SET exit_ts = now(), exit_price_fut= %s, exit_reason = %s,
               decision_score = %s, status = 'CLOSED', pnl = %s,
               pnl_pct = %s, exit_direction = %s
         WHERE trade_id = %s;
        """,
        (exit_price_fut, reason, float(score or 0), pnl_raw, pnl_pct, exit_direction, trade_id)
    )

def _preprocess_alert(symbol: str, kite: KiteConnect):
    fut = fetch_futures_data(symbol, kite)
    if not fut:
        return None, "Futures data unavailable"
    zone = fetch_latest_zone_data(symbol) or {}
    return {
        "future_price": float(fut.get("last_price") or 0),
        "volume": int(fut.get("volume") or 0),
        "support_zone": zone.get("val"),
        "resistance_zone": zone.get("vah"),
        "zone_break_type": zone.get("zone_break_type"),
        "zone_conf_score": zone.get("zone_confidence_score"),
    }, None

def process_webhook_alerts() -> None:
    kite = load_kite()
    with get_db_connection() as conn, conn.cursor() as cur:
        alerts = _claim_pending_alerts(cur)
        if not alerts:
            print("🔍 No pending alerts.")
            return

        print(f"🚦 Processing {len(alerts)} alert(s)...")
        symbol_ctx_cache: Dict[str, Tuple[Optional[Dict], Optional[str]]] = {}

        for a in alerts:
            try:
                payload = a.payload or {}
                signal_type = payload.get("signal_type") or payload.get("signal")
                side = _side_from_signal(signal_type)

                if a.symbol not in symbol_ctx_cache:
                    symbol_ctx_cache[a.symbol] = _preprocess_alert(a.symbol, kite)
                ctx, ctx_err = symbol_ctx_cache[a.symbol]

                attempt = _next_attempt_number(cur, a.unique_id)

                if ctx_err or not ctx:
                    reason = ctx_err or "context missing"
                    _log_attempt(cur, a.unique_id, attempt, status="rejected", rejection_reason=reason)
                    _finalize_alert(cur, a.unique_id, "REJECTED", rejection_reason=reason)
                    continue

                if side == "CLOSE":
                    should_close, reason, score = evaluate_exit(a.symbol)
                    if should_close:
                        row = _fetch_latest_open_trade(cur, a.symbol)
                        if row:
                            trade_id, entry_side, entry_price_fut = row
                            _close_trade(cur, trade_id, float(ctx["future_price"]), reason, score, "CLOSE", entry_side, entry_price_fut)
                            _log_attempt(cur, a.unique_id, attempt, status="accepted", notes=f"exit: {reason}")
                            _finalize_alert(cur, a.unique_id, "ACCEPTED")
                        else:
                            _log_attempt(cur, a.unique_id, attempt, status="rejected", rejection_reason="No OPEN trade found")
                            _finalize_alert(cur, a.unique_id, "REJECTED", rejection_reason="No OPEN trade found")
                    else:
                        _log_attempt(cur, a.unique_id, attempt, status="rejected", rejection_reason=reason)
                        _finalize_alert(cur, a.unique_id, "REJECTED", rejection_reason=reason)
                    continue

                is_valid, rule_matched, failed_rules, score, decision_tags, market_data = evaluate_alert(a.symbol)

                if is_valid:
                    entry_side = side or "LONG"
                    fut_price = float(ctx["future_price"])
                    opt_type = "CE" if entry_side == "LONG" else "PE"

                    option_info = pick_nearest_option(a.symbol, kite, target_price=fut_price, option_type=opt_type)
                    expiry, strike, tsym = (option_info.expiry, option_info.strike, option_info.tradingsymbol) if option_info else (None, None, None)

                    _insert_entry(cur, a.unique_id, a.symbol, entry_side, fut_price, rule_matched, score, opt_type, strike, tsym, market_data)
                    _log_attempt(cur, a.unique_id, attempt, status="accepted", notes=f"rule={rule_matched}, score={score}, opt={opt_type}@{strike} {tsym or ''}".strip())
                    _finalize_alert(cur, a.unique_id, "ACCEPTED")
                else:
                    reason_text = "; ".join(f"{r}: {rsn}" for (r, rsn) in (failed_rules or [])) or "Rules not met"
                    _log_attempt(cur, a.unique_id, attempt, status="rejected", rejection_reason=reason_text)
                    _finalize_alert(cur, a.unique_id, "REJECTED", rejection_reason=reason_text)
            except Exception as row_err:
                print(f"⚠️ [{a.symbol}] Insert fail: {row_err}")

        print("📝 Processing complete.")

if __name__ == "__main__":
    process_webhook_alerts()