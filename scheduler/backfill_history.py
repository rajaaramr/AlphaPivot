# scheduler/backfill_history.py
# Seed & maintain 15m spot/futures history in Timescale (Zerodha Kite).

from __future__ import annotations

import os
import time
import configparser
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Dict, Any, Iterable, Literal

import psycopg2.extras as pgx

from utils.db import get_db_connection
from utils.kite_utils import load_kite
from utils.db_ops import insert_spot_price, insert_futures_price

# ---------------- Globals ----------------
TZ = timezone.utc
IS_SYNC = False                   # set in __main__ based on --mode
universe_name = "largecaps_v1"     # set in __main__ (or INI) before run

# Map our interval strings to Kite's (we force 15m for backfill/sync)
_KITE_TF = {
    "1m": "minute", "3m": "3minute", "5m": "5minute", "10m": "10minute",
    "15m": "15minute", "30m": "30minute", "60m": "60minute", "1d": "day",
}

# ---------------- INI helper ----------------
def _read_ini(path: str = "configs/data.ini") -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    if os.path.exists(path):
        cfg.read(path)
    return cfg

# ---------------- Discovery helpers ----------------
def discover_symbols(source: str = "webhooks",
                     lookback_days: int = 14,
                     universe_name: str | None = None) -> list[str]:
    syms: list[str] = []
    with get_db_connection() as conn, conn.cursor() as cur:
        if source == "webhooks":
            cur.execute("""
                SELECT DISTINCT symbol
                  FROM webhooks.webhook_alerts
                 WHERE received_at >= now() - INTERVAL %s
                 ORDER BY 1
            """, (f"{lookback_days} days",))
            rows = cur.fetchall() or []
            syms = [r[0] for r in rows]
        elif source == "universe":
            if not universe_name:
                return []
            cur.execute("""
                SELECT symbol
                  FROM reference.symbol_universe
                 WHERE universe_name=%s
                 ORDER BY 1
            """, (universe_name,))
            rows = cur.fetchall() or []
            syms = [r[0] for r in rows]
    return [s.upper() for s in syms if s]

def snapshot_universe(symbols: list[str],
                      universe_name: str = "largecaps_v1",
                      source: str = "webhooks") -> int:
    if not symbols:
        return 0
    symbols = sorted(set(s.upper() for s in symbols))
    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(cur, """
            INSERT INTO reference.symbol_universe (universe_name, symbol, source)
            VALUES %s
            ON CONFLICT (universe_name, symbol) DO NOTHING
        """, [(universe_name, s, source) for s in symbols])
        conn.commit()
    return len(symbols)

# ---------------- Zerodha helpers ----------------
def _historical_data_with_retry(kite, token: int, start: datetime, end: datetime,
                                tf: str, with_oi: bool, *, max_retries=4) -> List[Dict[str, Any]]:
    attempt = 0
    pause = 0.75
    while True:
        try:
            data = kite.historical_data(token, start, end, interval=tf,
                                        continuous=False, oi=with_oi) or []
            out: List[Dict[str, Any]] = []
            for c in data:
                ts = c["date"]
                if getattr(ts, "tzinfo", None) is None:
                    ts = ts.replace(tzinfo=TZ)
                else:
                    ts = ts.astimezone(TZ)
                out.append({
                    "ts": ts,
                    "open": float(c["open"]), "high": float(c["high"]),
                    "low": float(c["low"]),  "close": float(c["close"]),
                    "volume": int(c.get("volume") or 0),
                    "oi": int(c.get("oi") or 0) if with_oi else 0,
                })
            return out
        except Exception as e:
            msg = str(e)
            attempt += 1
            if attempt <= max_retries and ("429" in msg or "Too many" in msg or "timeout" in msg or "500" in msg):
                time.sleep(pause)
                pause *= 2.0
                continue
            raise

def _instrument_maps(kite) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build tradingsymbol -> instrument_token maps for NSE (spot) and NFO (futures).
    We do this once per run to avoid calling kite.instruments() repeatedly.
    """
    spot_map: Dict[str, int] = {}
    fut_map: Dict[str, int] = {}
    for ins in (kite.instruments("NSE") or []):
        ts = str(ins.get("tradingsymbol", "")).upper()
        tok = ins.get("instrument_token")
        if ts and tok:
            spot_map[ts] = int(tok)
    for ins in (kite.instruments("NFO") or []):
        ts = str(ins.get("tradingsymbol", "")).upper()
        tok = ins.get("instrument_token")
        if ts and tok:
            fut_map[ts] = int(tok)
    return spot_map, fut_map

def _guess_spot_tsym(symbol: str) -> str:
    # Swap for your mapping util if needed.
    return symbol.upper()

# ---------------- Core writers ----------------
def _write_spot_rows(symbol: str, rows: List[Dict[str, Any]], *, interval: str) -> int:
    wrote = 0
    for r in rows:
        insert_spot_price(
            ts=r["ts"], symbol=symbol,
            open=r["open"], high=r["high"], low=r["low"], close=r["close"],
            volume=r["volume"], interval=interval, source="engine",
        )
        wrote += 1
    return wrote

def _write_fut_rows(symbol: str, rows: List[Dict[str, Any]], *, interval: str) -> int:
    wrote = 0
    for r in rows:
        insert_futures_price(
            ts=r["ts"], symbol=symbol,
            open=r["open"], high=r["high"], low=r["low"], close=r["close"],
            volume=r["volume"], oi=r["oi"], interval=interval, source="engine",
        )
        wrote += 1
    return wrote

# ---------------- Cursor helpers: FUT ----------------
def _get_fut_cursor(conn, symbol: str, universe_name: str) -> tuple[datetime | None, datetime | None, str]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT fut_15m_next_from_ts, fut_15m_target_until_ts, COALESCE(fut_15m_status,'NOT_STARTED')
              FROM reference.symbol_universe
             WHERE symbol = %s AND universe_name = %s
        """, (symbol, universe_name))
        row = cur.fetchone()
        return (row[0], row[1], row[2]) if row else (None, None, "NOT_STARTED")

def _update_fut_cursor(conn, symbol: str, universe_name: str, *,
                       status: str | None = None,
                       next_from_ts: datetime | None = None,
                       last_ingested_ts: datetime | None = None,
                       error: str | None = None) -> None:
    sets: List[str] = []
    params: List[Any] = []
    if status is not None:
        sets.append("fut_15m_status = %s");           params.append(status)
    if next_from_ts is not None:
        sets.append("fut_15m_next_from_ts = %s");     params.append(next_from_ts)
    if last_ingested_ts is not None:
        sets.append("fut_15m_last_ingested_ts = %s"); params.append(last_ingested_ts)
    if error is not None:
        sets.append("fut_last_error = %s");           params.append(error)
    if not sets:
        return
    sql = f"UPDATE reference.symbol_universe SET {', '.join(sets)} WHERE symbol = %s AND universe_name = %s"
    params.extend([symbol, universe_name])
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()

# ---------------- Cursor helpers: SPOT ----------------
def _get_spot_cursor(conn, symbol: str, universe_name: str) -> tuple[datetime | None, datetime | None, str]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT spot_15m_next_from_ts, spot_15m_target_until_ts, COALESCE(spot_15m_status,'NOT_STARTED')
              FROM reference.symbol_universe
             WHERE symbol = %s AND universe_name = %s
        """, (symbol, universe_name))
        row = cur.fetchone()
        return (row[0], row[1], row[2]) if row else (None, None, "NOT_STARTED")

def _update_spot_cursor(conn, symbol: str, universe_name: str, *,
                        status: str | None = None,
                        next_from_ts: datetime | None = None,
                        last_ingested_ts: datetime | None = None,
                        error: str | None = None) -> None:
    sets: List[str] = []
    params: List[Any] = []
    if status is not None:
        sets.append("spot_15m_status = %s");           params.append(status)
    if next_from_ts is not None:
        sets.append("spot_15m_next_from_ts = %s");     params.append(next_from_ts)
    if last_ingested_ts is not None:
        sets.append("spot_15m_last_ingested_ts = %s"); params.append(last_ingested_ts)
    if error is not None:
        sets.append("spot_last_error = %s");           params.append(error)
    if not sets:
        return
    sql = f"UPDATE reference.symbol_universe SET {', '.join(sets)} WHERE symbol = %s AND universe_name = %s"
    params.extend([symbol, universe_name])
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()

# ---------------- Public API ----------------
def backfill_symbols(symbols: Iterable[str], *,
                     kinds: Iterable[Literal["spot", "futures"]] = ("futures",),
                     interval: str = "15m",
                     lookback_days: int = 60,
                     chunk_days: int = 1,
                     pace_sleep: float = 0.4) -> Dict[str, Tuple[int, int]]:
    """
    Resume-safe 15m backfill for selected kinds.
    Futures path unchanged.
    Spot path mirrors futures with its own cursor/state.
    Returns {symbol: (spot_rows_written, fut_rows_written)}.
    """
    is_sync = IS_SYNC
    kite = load_kite()
    tf = _KITE_TF.get("15m", "15minute")
    now_utc = datetime.now(tz=TZ)

    spot_map, fut_map = _instrument_maps(kite)
    results: Dict[str, Tuple[int, int]] = {}

    # Optional quick diagnostics for SPOT: show missing tokens up front
    if "spot" in kinds:
        missing = [s for s in (sym.upper() for sym in symbols) if _guess_spot_tsym(s) not in spot_map]
        if missing:
            preview = ", ".join(missing[:15])
            more = f" …(+{len(missing)-15} more)" if len(missing) > 15 else ""
            print(f"⚠️ Missing NSE tokens for: {preview}{more}")

    with get_db_connection() as conn:
        for sym in (s.upper() for s in symbols):
            wrote_s, wrote_f = 0, 0

            # ---------- SPOT ----------
            if "spot" in kinds:
                start_ts, target_ts, status = _get_spot_cursor(conn, sym, universe_name)
                if start_ts is None:
                    start_ts = now_utc - timedelta(days=lookback_days)
                if target_ts is None:
                    target_ts = now_utc

                spot_tsym = _guess_spot_tsym(sym)
                spot_token = spot_map.get(spot_tsym)
                print(f"[SPOT] {sym} start={start_ts} target={target_ts} status={status} token={'YES' if spot_token else 'NO'}", flush=True)
                if not spot_token:
                    _update_spot_cursor(conn, sym, universe_name, status="ERROR",
                                        error=f"NSE instrument token not found for '{spot_tsym}'")
                    print(f"⚠️ SPOT {sym}: missing NSE token → status=ERROR (no cursor advance)")
                else:
                    cur_start = start_ts
                    while cur_start < target_ts:
                        cur_end = min(cur_start + timedelta(days=chunk_days), target_ts)
                        try:
                            data = _historical_data_with_retry(
                                kite, spot_token, cur_start, cur_end, tf, with_oi=False
                            )
                            wrote = len(data)
                            if wrote:
                                wrote_s += _write_spot_rows(sym, data, interval=interval)

                            _update_spot_cursor(
                                conn, sym, universe_name,
                                status=("INTRADAY_SYNCING" if is_sync else "HISTORY_BACKFILLING"),
                                next_from_ts=cur_end,
                                last_ingested_ts=(cur_end - timedelta(minutes=15)),
                            )
                            print(f"SPOT {sym}: {cur_start:%Y-%m-%d}→{cur_end:%Y-%m-%d} rows={wrote}")
                            time.sleep(pace_sleep)
                        except Exception as e:
                            _update_spot_cursor(conn, sym, universe_name, status="ERROR", error=str(e))
                            print(f"⚠️ SPOT backfill {sym} {cur_start:%Y-%m-%d}->{cur_end:%Y-%m-%d}: {e}")
                            break
                        cur_start = cur_end

                    new_next, new_target, _ = _get_spot_cursor(conn, sym, universe_name)
                    if new_next and new_target and new_next >= new_target:
                        _update_spot_cursor(conn, sym, universe_name,
                                            status=("INTRADAY_OK" if is_sync else "HISTORY_DONE"))

            # ---------- FUTURES ----------
            if "futures" in kinds:
                start_ts, target_ts, status = _get_fut_cursor(conn, sym, universe_name)
                if start_ts is None:
                    start_ts = now_utc - timedelta(days=lookback_days)
                if target_ts is None:
                    target_ts = now_utc

                # Heuristic token: first NFO symbol starting with {sym} and containing FUT
                fut_token: Optional[int] = None
                for ts, tok in fut_map.items():
                    if ts.startswith(sym) and "FUT" in ts:
                        fut_token = tok
                        break

                if not fut_token:
                    _update_fut_cursor(conn, sym, universe_name, status="ERROR",
                                       error=f"NFO instrument token not found for prefix '{sym}*FUT*'")
                    print(f"⚠️ FUT {sym}: missing NFO token → status=ERROR (no cursor advance)")
                else:
                    cur_start = start_ts
                    while cur_start < target_ts:
                        cur_end = min(cur_start + timedelta(days=chunk_days), target_ts)
                        try:
                            data = _historical_data_with_retry(
                                kite, fut_token, cur_start, cur_end, tf, with_oi=True
                            )
                            wrote = len(data)
                            if wrote:
                                wrote_f += _write_fut_rows(sym, data, interval=interval)

                            _update_fut_cursor(
                                conn, sym, universe_name,
                                status=("INTRADAY_SYNCING" if is_sync else "HISTORY_BACKFILLING"),
                                next_from_ts=cur_end,
                                last_ingested_ts=(cur_end - timedelta(minutes=15)),
                            )
                            print(f"FUT  {sym}: {cur_start:%Y-%m-%d}→{cur_end:%Y-%m-%d} rows={wrote}")
                            time.sleep(pace_sleep)
                        except Exception as e:
                            _update_fut_cursor(conn, sym, universe_name, status="ERROR", error=str(e))
                            print(f"⚠️ FUT backfill {sym} {cur_start:%Y-%m-%d}->{cur_end:%Y-%m-%d}: {e}")
                            break
                        cur_start = cur_end

                    new_next, new_target, _ = _get_fut_cursor(conn, sym, universe_name)
                    if new_next and new_target and new_next >= new_target:
                        _update_fut_cursor(conn, sym, universe_name,
                                           status=("INTRADAY_OK" if is_sync else "HISTORY_DONE"))

            results[sym] = (wrote_s, wrote_f)
            print(f"✅ {sym}: spot_15m={wrote_s} rows, fut_15m={wrote_f} rows")

    return results

def sync_today(symbols: Iterable[str], *,
               kinds: Iterable[Literal["spot","futures"]] = ("spot", "futures"),
               interval: str = "15m",
               lookback_days_today: int = 2,
               chunk_days: int = 1,
               pace_sleep: float = 0.4) -> Dict[str, Tuple[int, int]]:
    """
    Nightly catch-up (e.g., schedule after market close IST). Pull last ~1–2 days by
    reusing backfill with a small lookback.
    """
    return backfill_symbols(
        symbols,
        kinds=kinds,
        interval=interval,
        lookback_days=lookback_days_today,
        chunk_days=chunk_days,
        pace_sleep=pace_sleep,
    )

# ---------------- CLI ----------------
if __name__ == "__main__":
    import sys
    from typing import Optional

    # Robust option parser: strips leading `--`, supports "--k=v" and "--k v"
    argv = sys.argv[1:]
    opts: Dict[str, str] = {}
    args: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            keyval = tok[2:]
            if "=" in keyval:
                k, v = keyval.split("=", 1)
                opts[k.lower()] = v
            else:
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    opts[keyval.lower()] = argv[i + 1]
                    i += 1
                else:
                    opts[keyval.lower()] = "1"
        else:
            args.append(tok)
        i += 1

    def _bool(x: str | None) -> bool:
        return str(x).lower() in {"1", "true", "yes", "on"}

    def _parse_kinds(raw: Optional[str], flags: Dict[str, str]) -> tuple[str, ...]:
        order = ["spot", "futures"]
        chosen: list[str] = []
        if _bool(flags.get("spot")): chosen.append("spot")
        if _bool(flags.get("futures")) or _bool(flags.get("future")) or _bool(flags.get("fut")):
            chosen.append("futures")
        if raw:
            for part in raw.replace(";", ",").split(","):
                p = part.strip().lower()
                if p in {"spot", "s", "cash"}: chosen.append("spot")
                elif p in {"futures", "future", "fut", "f"}: chosen.append("futures")
                elif p in {"both", "all", "*"}: chosen += order
        if not chosen:
            chosen = order[:]
        out: list[str] = []
        for k in order:
            if k in chosen and k not in out:
                out.append(k)
        return tuple(out)

    # Read INI
    ini = _read_ini()
    ini_uni   = ini.get("universe", "name",          fallback="largecaps_v1")
    ini_tf    = ini.get("live",     "interval",      fallback="15m")
    ini_look  = ini.getint("ingest","lookback_days", fallback=60)

    # CLI config (flags override INI)
    mode = opts.get("mode", "seed").lower()              # seed | sync
    kinds = _parse_kinds(opts.get("kinds"), opts)
    interval = opts.get("interval", ini_tf).lower()
    lookback = int(opts.get("lookback_days", str(ini_look if mode == "seed" else 2)))
    chunk = int(opts.get("chunk_days", "1"))
    discover = opts.get("discover", "universe").lower()  # webhooks | universe | none
    lookback_discovery = int(opts.get("discover_lookback_days", "14"))

    # Set globals
    IS_SYNC = (mode == "sync")
    universe_name = opts.get("universe", ini_uni)

    print(f"[CFG] mode={mode} kinds={kinds} lookback={lookback}d chunk={chunk}d "
          f"discover={discover} uni={universe_name}", flush=True)

    # Symbols: explicit args > discovery > defaults
    symbols = [s.upper() for s in args]
    if not symbols:
        if discover == "webhooks":
            symbols = discover_symbols("webhooks", lookback_days=lookback_discovery)
        elif discover == "universe":
            symbols = discover_symbols("universe", universe_name=universe_name)
        else:
            symbols = ["RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK"]

    # Optional: snapshot universe
    snapshot = _bool(opts.get("snapshot", "false"))
    if snapshot:
        n = snapshot_universe(symbols, universe_name=universe_name, source=discover)
        print(f"📌 Universe snapshot '{universe_name}': {n} symbols recorded")

    # Run
    if mode == "seed":
        out = backfill_symbols(symbols, kinds=kinds, interval=interval,
                               lookback_days=lookback, chunk_days=chunk)
    else:
        out = sync_today(symbols, kinds=kinds, interval=interval,
                         lookback_days_today=lookback, chunk_days=chunk)

    total_s = sum(v[0] for v in out.values())
    total_f = sum(v[1] for v in out.values())
    print(f"TOTAL: spot={total_s}, fut={total_f} across {len(symbols)} symbols")
