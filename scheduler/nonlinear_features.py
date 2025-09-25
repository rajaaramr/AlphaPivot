# scheduler/nonlinear_features.py
from __future__ import annotations

import ast
import math
import json
import configparser
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import psycopg2.extras as pgx

from utils.db import get_db_connection

DEFAULT_INI = "nonlinear.ini"
TZ = pd.Timestamp.utcnow().tz

# ---------------------- Config ----------------------

@dataclass
class NLConfig:
    tfs: List[str]                      # e.g. ["25m","65m","125m"]
    lookback_n: int                     # how many past points to zscore per metric
    score_method: str                   # "logistic" (default) or "tanh"
    score_k: float                      # sharpness of mapping
    mtf_weights: Dict[str, float]       # {"25m":0.3,"65m":0.3,"125m":0.4}
    interactions: Dict[str, str]        # {"RSIxADX":"RSI * ADX", "MACDxATR":"MACD.hist * ATR"}

def _csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def _weights_map(s: str) -> Dict[str, float]:
    # "25m:0.3,65m:0.3,125m:0.4"
    out: Dict[str,float] = {}
    for p in _csv(s):
        if ":" in p:
            k,v = p.split(":",1)
            try: out[k.strip()] = float(v.strip())
            except: pass
    return out

def _load_ini(path: str = DEFAULT_INI) -> NLConfig:
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";","#"), interpolation=None, strict=False)
    cfg.read(path)

    tfs = _csv(cfg.get("nonlinear","tfs",fallback="25m,65m,125m"))
    lookback_n = cfg.getint("nonlinear","lookback_n",fallback=120)
    score_method = cfg.get("nonlinear","score_method",fallback="logistic").strip().lower()
    score_k = cfg.getfloat("nonlinear","score_k",fallback=1.0)
    mtf_weights = _weights_map(cfg.get("nonlinear","mtf_weights",fallback="25m:0.3,65m:0.3,125m:0.4"))

    interactions: Dict[str,str] = {}
    if cfg.has_section("interactions"):
        for name, expr in cfg.items("interactions"):
            interactions[name.strip()] = expr.strip()

    # sane defaults if empty
    if not interactions:
        interactions = {
            "RSIxADX": "RSI * ADX",
            "MACDxATR": "MACD.hist * ATR",
            "MFIxROC": "MFI * ROC",
        }

    return NLConfig(
        tfs=tfs,
        lookback_n=lookback_n,
        score_method=score_method,
        score_k=score_k,
        mtf_weights=mtf_weights or {"25m":0.3,"65m":0.3,"125m":0.4},
        interactions=interactions
    )

# ---------------------- DB helpers ----------------------

def _fetch_series(symbol: str, kind: str, tf: str, metric: str, n: int) -> pd.Series:
    """
    Pull last N values of a metric from indicators.values
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT ts, val
              FROM indicators.values
             WHERE symbol=%s AND market_type=%s AND interval=%s AND metric=%s
             ORDER BY ts DESC
             LIMIT %s
            """,
            (symbol, kind, tf, metric, n),
        )
        rows = cur.fetchall()
    if not rows: return pd.Series(dtype="float64")
    df = pd.DataFrame(rows, columns=["ts","val"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    return pd.to_numeric(df["val"], errors="coerce").dropna()

def _write_rows(rows: List[tuple]) -> int:
    if not rows: return 0
    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(
            cur,
            """
            INSERT INTO indicators.values
                (symbol, market_type, interval, ts, metric, val, context, run_id, source)
            VALUES %s
            ON CONFLICT (symbol, market_type, interval, ts, metric)
            DO UPDATE SET
                val=EXCLUDED.val,
                context=EXCLUDED.context,
                run_id=EXCLUDED.run_id,
                source=EXCLUDED.source
            """,
            rows,
            page_size=1000
        )
        conn.commit()
        return len(rows)

# ---------------------- Safe expression eval ----------------------

_ALLOWED_OPS = {
    ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.USub, ast.UAdd, ast.Pow, ast.Mod,
    ast.FloorDiv,  # just in case
}
def _safe_eval(expr: str, vars_map: Dict[str, float]) -> float:
    """
    Evaluate arithmetic expression over allowed metric names.
    Names may include dots (e.g. MACD.hist) -> map to 'MACD_hist' internal var.
    """
    # replace dots in names with underscores for AST Names
    repl_map = {k: k.replace(".","_") for k in vars_map.keys()}
    inv_map = {v:k for k,v in repl_map.items()}
    expr2 = expr
    # sort by length to avoid partial replacements
    for k in sorted(vars_map.keys(), key=len, reverse=True):
        expr2 = expr2.replace(k, repl_map[k])

    def _eval(node) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Num):
            return float(node.n)
        if isinstance(node, ast.Constant) and isinstance(node.value,(int,float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
            return -_eval(node.operand) if isinstance(node.op, ast.USub) else +_eval(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            a = _eval(node.left); b = _eval(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b if b != 0 else 0.0
            if isinstance(node.op, ast.Mod): return a % b if b != 0 else 0.0
            if isinstance(node.op, ast.Pow): return a ** b
            if isinstance(node.op, ast.FloorDiv): return a // b if b != 0 else 0.0
        if isinstance(node, ast.Name):
            name = inv_map.get(node.id, node.id)
            return float(vars_map.get(name, 0.0))
        raise ValueError("Disallowed expression")
    try:
        tree = ast.parse(expr2, mode="eval")
        return float(_eval(tree))
    except Exception:
        return 0.0

# ---------------------- Z-score & scoring ----------------------

def _zscore(s: pd.Series) -> float:
    if s is None or s.empty: return 0.0
    m = float(s.tail(len(s)).mean())
    st = float(s.tail(len(s)).std(ddof=1)) if len(s) > 2 else 0.0
    if not np.isfinite(st) or st == 0.0:
        return 0.0
    x = float(s.iloc[-1])
    return (x - m) / st

def _score(x: float, method: str, k: float) -> float:
    if method == "tanh":
        # map to [0,1]
        return 0.5 * (math.tanh(k * x) + 1.0)
    # logistic default
    return 1.0 / (1.0 + math.exp(-k * x))

def _blend_logits(p_map: Dict[str,float], weights: Dict[str,float]) -> float:
    if not p_map: return 0.5
    z = 0.0; wsum = 0.0
    for tf, p in p_map.items():
        w = float(weights.get(tf, 0.0))
        if w <= 0: continue
        p = min(1-1e-6, max(1e-6, float(p)))
        z += w * math.log(p/(1-p))
        wsum += w
    if wsum <= 0: return 0.5
    z /= wsum
    return 1.0/(1.0+math.exp(-z))

# ---------------------- Core ----------------------

_BASE_METRICS = {
    "RSI": "RSI",
    "ADX": "ADX",
    "ROC": "ROC",
    "ATR": "ATR",            # in indicators.values it’s plain "ATR"
    "MFI": "MFI",            # your writer uses "MFI"
    "DI+": "DI+",
    "DI-": "DI-",
    "MACD.hist": "MACD.hist",
    "MACD.line": "MACD.line",
    "MACD.signal": "MACD.signal",
}

def _component_series(symbol: str, kind: str, tf: str, metric_key: str, n: int) -> pd.Series:
    # metric_key is like "MACD.hist" or "RSI"
    return _fetch_series(symbol, kind, tf, _BASE_METRICS.get(metric_key, metric_key), n)

def _vars_for_tf(symbol: str, kind: str, tf: str, comp_keys: List[str], n: int) -> Dict[str,float]:
    # produce z-scored latest values for all components
    out: Dict[str,float] = {}
    for key in comp_keys:
        s = _component_series(symbol, kind, tf, key, n)
        out[key] = float(_zscore(s))
    return out

def _components_in_expr(expr: str) -> List[str]:
    # pull tokens that look like metric names (letters, digits, . and _)
    # We assume your metrics keys appear exactly in the expression.
    tokens = set()
    buf = ""
    for ch in expr:
        if ch.isalnum() or ch in "._":
            buf += ch
        else:
            if buf: tokens.add(buf); buf = ""
    if buf: tokens.add(buf)
    # remove pure numbers
    return [t for t in tokens if not all(c.isdigit() or c=="." for c in t)]

def _now_utc():
    return pd.Timestamp.utcnow().to_pydatetime()

def process_symbol(symbol: str, *, kind: str = "futures", cfg: Optional[NLConfig] = None) -> int:
    cfg = cfg or _load_ini()
    rows: List[tuple] = []

    # build per-TF probabilities then MTF blend
    p_by_tf: Dict[str,float] = {}

    for tf in cfg.tfs:
        # union of all components needed across all interactions
        needed: set[str] = set()
        for expr in cfg.interactions.values():
            for comp in _components_in_expr(expr):
                needed.add(comp)
        vars_map = _vars_for_tf(symbol, kind, tf, sorted(needed), cfg.lookback_n)

        # per-interaction values (z-space)
        inter_vals: Dict[str,float] = {}
        zsum = 0.0
        for name, expr in cfg.interactions.items():
            z = _safe_eval(expr, vars_map)
            inter_vals[name] = float(z)
            zsum += z

        # map combined zsum -> probability
        p = _score(zsum, cfg.score_method, cfg.score_k)
        p_by_tf[tf] = p

        ts = _now_utc()
        ctx_tf = json.dumps({"tf": tf, "interactions": list(cfg.interactions.keys())})

        # write each interaction raw value as NL.<name>
        for name, z in inter_vals.items():
            rows.append((
                symbol, kind, tf, ts, f"NL.{name}", float(z), ctx_tf, "nl_run", "nonlinear"
            ))
        # write probability and score for this TF
        rows += [
            (symbol, kind, tf, ts, f"CONF_NL.prob.{tf}", float(p), ctx_tf, "nl_run", "nonlinear"),
            (symbol, kind, tf, ts, f"CONF_NL.score.{tf}", float(round(100.0*p,2)), ctx_tf, "nl_run", "nonlinear"),
        ]

    # MTF blend
    if p_by_tf:
        p_mtf = _blend_logits(p_by_tf, cfg.mtf_weights)
        ts = _now_utc()
        ctx_mtf = json.dumps({"weights": cfg.mtf_weights})
        rows += [
            (symbol, kind, "MTF", ts, "CONF_NL.prob.mtf", float(p_mtf), ctx_mtf, "nl_run", "nonlinear"),
            (symbol, kind, "MTF", ts, "CONF_NL.score.mtf", float(round(100.0*p_mtf,2)), ctx_mtf, "nl_run", "nonlinear"),
        ]

    return _write_rows(rows)

def run(symbols: Optional[List[str]] = None, kinds: Tuple[str,...] = ("futures","spot"), ini_path: str = DEFAULT_INI) -> int:
    cfg = _load_ini(ini_path)
    if symbols is None:
        # discover from webhooks in active stages
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
                total += process_symbol(s, kind=k, cfg=cfg)
            except Exception as e:
                print(f"❌ NL {k}:{s} → {e}")
    print(f"🎯 nonlinear_features wrote rows for {total} operations across {len(symbols)} symbol(s)")
    return total

if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:]]
    syms = [a for a in args if not a.startswith("--")]
    run(syms or None)
