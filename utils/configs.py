# utils/configs.py
from __future__ import annotations
import os, configparser

_DEF_PATH = os.getenv("RAJ_CFG_PATH", "configs/data.ini")

def load_ini(path: str | None = None) -> configparser.ConfigParser:
    """Load INI with safe defaults. Env var RAJ_CFG_PATH can override path."""
    cfg = configparser.ConfigParser()
    cfg.read_dict({
        "backfill": {"lookback_days": "60", "gap_minutes": "1440"},
        "live":     {"interval": "15m", "fetch_options": "false", "fetch_pcr": "false"},
        "options":  {"strike_range": "3"},
    })
    cfg.read(path or _DEF_PATH)
    return cfg

def get_bool(cfg: configparser.ConfigParser, sect: str, key: str) -> bool:
    return str(cfg.get(sect, key, fallback="false")).lower() in {"1","true","yes","on"}
