# File: utils/compose_signals.py
"""
Signal Composition Module for the AlphaPivot Trading System.

This module is responsible for composing the final trading signal by combining
various data sources, including technical indicators and market buildup data.
"""
from __future__ import annotations
from typing import Dict

import pandas as pd

from .indicators import compute_indicators
from .buildups import compute_futures_buildup, compute_optionchain_buildup
from .configs import get_config_parser

def compose_blob(dfs_by_tf: Dict[str, pd.DataFrame], symbol: str) -> Dict:
    """
    Composes a comprehensive signal blob for a given symbol.

    This function combines technical indicators, futures buildup, and option
    chain buildup data into a single dictionary, which can be used for
    generating trading decisions.

    Args:
        dfs_by_tf: A dictionary of pandas DataFrames, keyed by timeframe.
        symbol: The trading symbol.

    Returns:
        A dictionary containing the composed signal blob.
    """
    config = get_config_parser()

    # We can compute indicators on the fly for the required timeframes
    blob = {}
    for tf, df in dfs_by_tf.items():
        blob[tf] = compute_indicators(df)

    fut = compute_futures_buildup(symbol)
    oc = compute_optionchain_buildup(symbol)

    blob["fut_buildup"] = fut
    blob["oc_buildup"] = oc

    # The final score composition logic that was here previously
    # has been moved to the pillars/composite_worker_v2.py module.
    # This module now focuses solely on composing the raw data blob.

    return blob