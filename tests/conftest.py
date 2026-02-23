"""Shared fixtures for backtesting tests."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

# -- Backtesting fixtures --


# ---------------------------------------------------------------------------
# Single-market dataset helpers (used by validity tests)
# ---------------------------------------------------------------------------


def _make_single_market_df(ticker: str, result: str, volume: int = 100) -> pd.DataFrame:
    """Return a single-row markets DataFrame for a controlled test market."""
    base = pd.Timestamp("2024-01-15 09:00:00")
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "event_ticker": "TEST-EVENT",
                "market_type": "binary",
                "title": f"Test {ticker}",
                "yes_sub_title": "Yes",
                "no_sub_title": "No",
                "status": "finalized",
                "yes_bid": None,
                "yes_ask": None,
                "no_bid": None,
                "no_ask": None,
                "last_price": None,
                "volume": volume,
                "volume_24h": volume // 10,
                "open_interest": 0,
                "result": result,
                "created_time": base,
                "open_time": base,
                "close_time": base + pd.Timedelta(days=1),
            }
        ]
    )


def _make_single_market_trades_df(
    ticker: str,
    trades: list[tuple[int, str, int]],
) -> pd.DataFrame:
    """Return a trades DataFrame.

    Args:
        ticker: Market ticker.
        trades: List of (yes_price_cents, taker_side, count) tuples.
    """
    base = pd.Timestamp("2024-01-15 10:00:00")
    rows = []
    for i, (yes_price, taker_side, count) in enumerate(trades):
        rows.append(
            {
                "trade_id": f"v-{i}",
                "ticker": ticker,
                "count": count,
                "yes_price": yes_price,
                "no_price": 100 - yes_price,
                "taker_side": taker_side,
                "created_time": base + pd.Timedelta(hours=i),
                "_fetched_at": base,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def make_dataset(
    tmp_path: Path,
) -> Callable[[str, str, list[tuple[int, str, int]]], tuple[Path, Path]]:
    """Factory fixture: create a minimal Kalshi dataset in a temp directory.

    Returns a callable ``_make(ticker, result, trades)`` that writes parquet
    files and returns ``(trades_dir, markets_dir)``.

    Example::

        def test_something(make_dataset):
            td, md = make_dataset("MKT-X", "yes", [
                (35, "yes", 5),   # yes_price_cents=35, taker_side="yes", count=5
                (20, "no", 10),
            ])
    """
    _counter = [0]

    def _make(
        ticker: str,
        result: str,
        trades: list[tuple[int, str, int]],
        volume: int = 100,
    ) -> tuple[Path, Path]:
        _counter[0] += 1
        n = _counter[0]
        trades_dir = tmp_path / f"v{n}_trades"
        markets_dir = tmp_path / f"v{n}_markets"
        trades_dir.mkdir()
        markets_dir.mkdir()
        _make_single_market_trades_df(ticker, trades).to_parquet(trades_dir / "trades.parquet")
        _make_single_market_df(ticker, result, volume).to_parquet(markets_dir / "markets.parquet")
        return trades_dir, markets_dir

    return _make


def _make_bt_kalshi_trades() -> pd.DataFrame:
    """Kalshi trades for backtesting: 3 markets, chronological, varied prices."""
    rows = []
    base = pd.Timestamp("2024-01-15 10:00:00")
    trade_data = [
        ("BT-MKT-A", 25, "yes", 5),
        ("BT-MKT-A", 30, "no", 3),
        ("BT-MKT-B", 80, "yes", 2),
        ("BT-MKT-A", 15, "yes", 10),
        ("BT-MKT-C", 50, "yes", 4),
        ("BT-MKT-B", 85, "no", 6),
        ("BT-MKT-A", 20, "no", 8),
        ("BT-MKT-C", 45, "yes", 3),
        ("BT-MKT-B", 75, "yes", 5),
        ("BT-MKT-C", 55, "no", 7),
    ]
    for i, (ticker, yes_price, taker_side, count) in enumerate(trade_data):
        rows.append(
            {
                "trade_id": f"bt-{i}",
                "ticker": ticker,
                "count": count,
                "yes_price": yes_price,
                "no_price": 100 - yes_price,
                "taker_side": taker_side,
                "created_time": base + pd.Timedelta(hours=i),
                "_fetched_at": base,
            }
        )
    return pd.DataFrame(rows)


def _make_bt_kalshi_markets() -> pd.DataFrame:
    """Kalshi markets for backtesting: 3 markets with open/close times and results."""
    base = pd.Timestamp("2024-01-15 09:00:00")
    return pd.DataFrame(
        [
            {
                "ticker": "BT-MKT-A",
                "event_ticker": "BT-EVENT",
                "market_type": "binary",
                "title": "Test Market A",
                "yes_sub_title": "Yes",
                "no_sub_title": "No",
                "status": "finalized",
                "yes_bid": None,
                "yes_ask": None,
                "no_bid": None,
                "no_ask": None,
                "last_price": None,
                "volume": 500,
                "volume_24h": 100,
                "open_interest": 0,
                "result": "yes",
                "created_time": base,
                "open_time": base,
                "close_time": base + pd.Timedelta(days=1),
            },
            {
                "ticker": "BT-MKT-B",
                "event_ticker": "BT-EVENT",
                "market_type": "binary",
                "title": "Test Market B",
                "yes_sub_title": "Yes",
                "no_sub_title": "No",
                "status": "finalized",
                "yes_bid": None,
                "yes_ask": None,
                "no_bid": None,
                "no_ask": None,
                "last_price": None,
                "volume": 300,
                "volume_24h": 50,
                "open_interest": 0,
                "result": "no",
                "created_time": base,
                "open_time": base,
                "close_time": base + pd.Timedelta(days=1),
            },
            {
                "ticker": "BT-MKT-C",
                "event_ticker": "BT-EVENT",
                "market_type": "binary",
                "title": "Test Market C",
                "yes_sub_title": "Yes",
                "no_sub_title": "No",
                "status": "finalized",
                "yes_bid": None,
                "yes_ask": None,
                "no_bid": None,
                "no_ask": None,
                "last_price": None,
                "volume": 200,
                "volume_24h": 30,
                "open_interest": 0,
                "result": "yes",
                "created_time": base,
                "open_time": base,
                "close_time": base + pd.Timedelta(days=1),
            },
        ]
    )


@pytest.fixture(scope="session")
def bt_kalshi_trades_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("bt_kalshi_trades")
    _make_bt_kalshi_trades().to_parquet(d / "trades.parquet")
    return d


@pytest.fixture(scope="session")
def bt_kalshi_markets_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("bt_kalshi_markets")
    _make_bt_kalshi_markets().to_parquet(d / "markets.parquet")
    return d
