# executor.py
"""
The trade execution engine for the AlphaPivot trading system.

This module is responsible for:
- Fetching open trading setups from the database.
- Calculating trade sizes based on risk parameters.
- Opening trades for futures and options instruments.
- Updating the status of trades in the database.
"""
from __future__ import annotations
import math
from datetime import datetime, timezone
import psycopg2, psycopg2.extras

from utils.db import get_db_connection
from utils.configs import get_config_parser

def fetch_open_setups(cur):
    """
    Fetches open trading setups from the analytics.decisions_live table.

    Args:
        cur: A database cursor object.

    Returns:
        A list of dictionaries, where each dictionary represents an open setup.
    """
    cur.execute("""
      SELECT d.symbol, d.ts, d.bias, d.instrument, d.expiry_date, d.strike,
             d.fut_close AS entry_px, d.stop_px, d.target_1r, d.pcr, d.composite
      FROM analytics.decisions_live d
      WHERE d.status = 'OPEN_SETUP'
        AND d.bias IN ('LONG_SETUP','SHORT_SETUP')
        AND d.instrument IS NOT NULL
    """)
    return cur.fetchall()

def latest_option_ltp(cur, symbol, expiry, strike, side):
    """
    Fetches the latest traded price (LTP) for a given option contract.

    Args:
        cur: A database cursor object.
        symbol: The underlying symbol of the option.
        expiry: The expiry date of the option.
        strike: The strike price of the option.
        side: The trade side ('LONG' for CE, 'SHORT' for PE).

    Returns:
        The latest traded price as a float, or None if not found.
    """
    cur.execute("""
      SELECT ltp
      FROM market.options_chain_snapshots
      WHERE symbol=%s AND expiry_date=%s AND strike=%s AND type=%s
      ORDER BY snapshot_ts DESC
      LIMIT 1
    """, (symbol, expiry, strike, 'CE' if side=='LONG' else 'PE'))
    row = cur.fetchone()
    return float(row[0]) if row and row[0] is not None else None

def open_trade(cur, symbol, side, instrument, entry_px, stop_px, target_px,
               decision_ts, meta: dict, qty: float, risk_per_trade: float):
    """
    Opens a new trade by inserting records into the paper_trades and trade_journal tables.

    Args:
        cur: A database cursor object.
        symbol: The trading symbol.
        side: The trade side ('LONG' or 'SHORT').
        instrument: The instrument type ('FUTURES', 'OPTION_CE', 'OPTION_PE').
        entry_px: The entry price of the trade.
        stop_px: The stop-loss price.
        target_px: The target price.
        decision_ts: The timestamp of the trading decision.
        meta: A dictionary containing metadata about the trade.
        qty: The quantity of the trade.
        risk_per_trade: The total risk amount for the trade.

    Returns:
        The trade_id of the newly created trade.
    """
    # paper_trades
    cur.execute("""
      INSERT INTO analytics.paper_trades
        (symbol, opened_at, side, entry_px, stop_px, target_px,
         size_units, risk_per_trade, status, decision_ts,
         pcr_at_entry, composite_at_entry, instrument, expiry_date, strike)
      VALUES
        (%(symbol)s, NOW(), %(side)s, %(entry)s, %(stop)s, %(target)s,
         %(qty)s, %(risk)s, 'OPEN', %(decision_ts)s,
         %(pcr)s, %(comp)s, %(instrument)s, %(expiry)s, %(strike)s)
      RETURNING trade_id
    """, {
        "symbol": symbol, "side": side,
        "entry": entry_px, "stop": stop_px, "target": target_px,
        "qty": qty, "risk": risk_per_trade,
        "decision_ts": decision_ts, "pcr": meta.get("pcr"), "comp": meta.get("composite"),
        "instrument": instrument, "expiry": meta.get("expiry"), "strike": meta.get("strike"),
    })
    trade_id = cur.fetchone()[0]
    # journal
    cur.execute("""
      INSERT INTO analytics.trade_journal (trade_id, ts, event, note, px)
      VALUES (%s, NOW(), 'OPENED', %s, %s)
    """, (trade_id, f"{instrument}", entry_px))
    return trade_id

def run():
    """
    The main execution loop of the trade executor.

    This function connects to the database, fetches open setups, calculates
    trade parameters, and opens trades for both futures and options.
    """
    # --- Load Configuration ---
    config = get_config_parser()
    executor_cfg = config["executor"]
    CAPITAL = executor_cfg.getfloat("capital", 1000000)
    RISK_PCT = executor_cfg.getfloat("risk_pct", 0.0075)
    DUAL_LEG = executor_cfg.getboolean("dual_leg", False)
    OPTION_STOP_PCT = executor_cfg.getfloat("option_stop_pct", 0.35)
    OPTION_TGT_PCT = executor_cfg.getfloat("option_tgt_pct", 0.50)

    with get_db_connection() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        setups = fetch_open_setups(cur)
        for s in setups:
            symbol = s["symbol"]
            side = 'LONG' if s["bias"] == 'LONG_SETUP' else 'SHORT'
            instrument = s["instrument"]
            entry = float(s["entry_px"])
            stop  = float(s["stop_px"])
            target = float(s["target_1r"])
            risk_amount = CAPITAL * RISK_PCT
            per_unit_risk = abs(entry - stop)
            if per_unit_risk <= 0:
                continue

            # FUTURES leg (always when instrument is FUTURES; also when DUAL_LEG=True)
            fut_qty = math.floor(risk_amount / per_unit_risk)
            opened_any = False
            if instrument == 'FUTURES' or DUAL_LEG:
                if fut_qty >= 1:
                    open_trade(cur, symbol, side, 'FUTURES', entry, stop, target,
                               s["ts"], {"pcr": s["pcr"], "composite": s["composite"]},
                               fut_qty, risk_amount)
                    opened_any = True

            # OPTIONS leg (when instrument is OPTION_* or DUAL_LEG)
            if instrument in ('OPTION_CE','OPTION_PE') or DUAL_LEG:
                ltp = latest_option_ltp(cur, symbol, s["expiry_date"], s["strike"], side)
                if ltp and ltp > 0:
                    opt_stop_px = ltp * (1.0 - OPTION_STOP_PCT if side=='LONG' else 1.0 + OPTION_STOP_PCT)
                    # Risk per lot = |ltp - opt_stop_px|
                    per_lot_risk = abs(ltp - opt_stop_px)
                    opt_qty = math.floor(risk_amount / per_lot_risk) if per_lot_risk > 0 else 0
                    if opt_qty >= 1:
                        # Target premium = +50% default (can tune)
                        opt_tgt_px = ltp * (1.0 + OPTION_TGT_PCT if side=='LONG' else 1.0 - OPTION_TGT_PCT)
                        open_trade(cur, symbol, side,
                                   'OPTION_CE' if side=='LONG' else 'OPTION_PE',
                                   ltp, opt_stop_px, opt_tgt_px, s["ts"],
                                   {"pcr": s["pcr"], "composite": s["composite"],
                                    "expiry": s["expiry_date"], "strike": s["strike"]},
                                   opt_qty, risk_amount)
                        opened_any = True

            if opened_any:
                cur.execute("""
                  UPDATE analytics.decisions_live
                  SET status='OPEN'
                  WHERE symbol=%s AND ts=%s
                """, (symbol, s["ts"]))
        conn.commit()

if __name__ == "__main__":
    run()