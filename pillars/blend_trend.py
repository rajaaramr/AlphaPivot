# pillars/blend_trend.py
"""
Blends pillar scores across multiple timeframes to generate a final
multi-timeframe (MTF) score.
"""
import json
from .common import BaseCfg, last_metric, write_values, now_ts, clamp
from utils.configs import get_config_parser

def _parse_csv_list(x: str) -> list[str]:
    """Parses a comma-separated string into a list of strings."""
    return [s.strip() for s in (x or "").split(",") if s.strip()]

def _parse_weights_map(s: str, tfs: list[str]) -> dict[str, float]:
    """Parses a comma-separated string of weights into a dictionary."""
    raw = {}
    for part in _parse_csv_list(s):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                raw[k.strip()] = float(v.strip())
            except (ValueError, TypeError):
                pass

    if not tfs:
        return {}

    filtered_raw = {k: v for k, v in raw.items() if k in tfs}
    total = sum(filtered_raw.values()) or 1.0
    return {k: v / total for k, v in filtered_raw.items()}

def _blend_pillar_mtf(symbol: str, kind: str, base: BaseCfg, pillar: str, score_key: str, veto_key: str | None, weights: dict):
    """
    Generic blender for any pillar.

    This function blends scores across multiple timeframes for a given pillar
    and writes the final MTF score back to the database.
    """
    W = weights
    num = den = 0.0
    used_tfs = []

    for tf, w in W.items():
        if tf not in base.tfs:
            continue
        s = last_metric(symbol, kind, tf, f"{pillar}.{score_key}")
        if s is None:
            continue
        num += float(w) * float(s)
        den += float(w)
        used_tfs.append(tf)

    if den == 0.0:
        return

    score_mtf = clamp(num / den, 0, 100)
    ts = now_ts()

    rows = [
        (symbol, kind, "MTF", ts, f"{pillar}.{score_key}", float(score_mtf),
         json.dumps({"scope": "MTF", "used_tfs": used_tfs}), base.run_id, base.source)
    ]

    if veto_key:
        tfs_for_veto = used_tfs if used_tfs else base.tfs
        veto = any(((last_metric(symbol, kind, tf, f"{pillar}.{veto_key}") or 0.0) > 0.5) for tf in tfs_for_veto)
        rows.append(
            (symbol, kind, "MTF", ts, f"{pillar}.{veto_key}", 1.0 if veto else 0.0, "{}", base.run_id, base.source)
        )

    write_values(rows)

def blend_trend_mtf(symbol: str, kind: str):
    """
    Blends TREND and RISK scores across multiple timeframes.
    """
    config = get_config_parser()
    core_cfg = config["core"]
    blend_cfg = config["blend"]

    tfs = _parse_csv_list(core_cfg.get("tfs", "25m,65m,125m"))
    base = BaseCfg(
        tfs=tfs,
        lookback_days=core_cfg.getint("lookback_days", 120),
        run_id="blend_run",
        source="blend_trend"
    )

    weights = _parse_weights_map(blend_cfg.get("mtf_weights", ""), tfs)

    _blend_pillar_mtf(
        symbol=symbol, kind=kind, base=base,
        pillar="TREND", score_key="score", veto_key="veto_soft",
        weights=weights
    )

    _blend_pillar_mtf(
        symbol=symbol, kind=kind, base=base,
        pillar="RISK", score_key="score", veto_key="veto_flag",
        weights=weights
    )