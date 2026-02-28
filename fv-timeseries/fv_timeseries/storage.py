"""Parquet read/write helpers for FV timeseries data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _parquet_path(output_dir: str, event_ticker: str, T: float) -> Path:
    T_str = f"{T:.4f}".rstrip("0").rstrip(".")
    return Path(output_dir) / f"{event_ticker}_T{T_str}.parquet"


def save_timeseries(df: pd.DataFrame, output_dir: str, event_ticker: str, T: float) -> Path:
    """Write timeseries DataFrame to parquet. Overwrites any existing file."""
    path = _parquet_path(output_dir, event_ticker, T)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_timeseries(event_ticker: str, T: float = 1.0, output_dir: str = "data/") -> pd.DataFrame:
    """Read timeseries parquet for a given event ticker and temperature."""
    path = _parquet_path(output_dir, event_ticker, T)
    if not path.exists():
        raise FileNotFoundError(f"No timeseries found at {path}")
    return pd.read_parquet(path)
