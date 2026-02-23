"""First-time setup smoke tests.

These tests verify that the project is correctly installed and ready to use.
They are deliberately simple: if any of them fail, the user knows exactly
which part of the setup documented in the README is broken.

Failure modes caught here:
  * Rust engine not compiled  (HAS_RUST is False / ImportError)
  * Missing Python dependency  (ImportError on runtime packages)
  * Broken module structure    (ImportError on internal modules)
  * Strategy auto-discovery broken
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------


class TestImports:
    """All public modules must be importable after a fresh `uv sync` + `make build-rust`."""

    def test_rust_module_importable(self) -> None:
        """The compiled Rust extension can be imported.

        Failure means the engine was not built.
        Fix: cd crates/backtesting_engine && maturin develop --release
        """
        from backtesting_engine import RustEngine  # noqa: F401

    def test_rust_engine_flag_is_true(self) -> None:
        """HAS_RUST is True — confirms the extension loaded at import time."""
        from src.backtesting.rust_engine import HAS_RUST

        assert HAS_RUST, (
            "Rust engine not available.\n"
            "Fix: cd crates/backtesting_engine && maturin develop --release"
        )

    def test_engine_class_importable(self) -> None:
        from src.backtesting.rust_engine import Engine  # noqa: F401

    def test_models_importable(self) -> None:
        from src.backtesting.models import (  # noqa: F401
            BacktestResult,
            Fill,
            MarketInfo,
            MarketStatus,
            Order,
            OrderAction,
            OrderStatus,
            Platform,
            PortfolioSnapshot,
            Position,
            Side,
            TradeEvent,
        )

    def test_strategy_base_importable(self) -> None:
        from src.backtesting.strategy import Strategy  # noqa: F401

    def test_kalshi_feed_importable(self) -> None:
        from src.backtesting.feeds.kalshi import KalshiFeed  # noqa: F401

    def test_polymarket_feed_importable(self) -> None:
        from src.backtesting.feeds.polymarket import PolymarketFeed  # noqa: F401

    def test_metrics_importable(self) -> None:
        from src.backtesting.metrics import compute_metrics  # noqa: F401

    def test_plotting_importable(self) -> None:
        import src.backtesting.plotting  # noqa: F401


# ---------------------------------------------------------------------------
# Runtime dependencies
# ---------------------------------------------------------------------------


class TestRuntimeDependencies:
    """All packages listed in pyproject.toml's [project.dependencies] must be importable."""

    @pytest.mark.parametrize(
        "package",
        [
            "bokeh",
            "duckdb",
            "numpy",
            "pandas",
            "pyarrow",
            "tqdm",
            "websockets",
        ],
    )
    def test_dependency_importable(self, package: str) -> None:
        try:
            importlib.import_module(package)
        except ImportError:
            pytest.fail(
                f"Required dependency '{package}' is not installed.\n"
                "Fix: uv sync"
            )


# ---------------------------------------------------------------------------
# Engine smoke test
# ---------------------------------------------------------------------------


class TestEngineSmoke:
    """Engine initializes and runs without errors with minimal valid data."""

    def test_engine_runs_to_completion(
        self, bt_kalshi_trades_dir: Path, bt_kalshi_markets_dir: Path
    ) -> None:
        from src.backtesting.feeds.kalshi import KalshiFeed
        from src.backtesting.models import TradeEvent
        from src.backtesting.rust_engine import Engine
        from src.backtesting.strategy import Strategy

        class Minimal(Strategy):
            def __init__(self) -> None:
                super().__init__(name="minimal_smoke", description="smoke", initial_cash=1000.0)

            def on_trade(self, trade: TradeEvent) -> None:
                pass

        feed = KalshiFeed(trades_dir=bt_kalshi_trades_dir, markets_dir=bt_kalshi_markets_dir)
        result = Engine(feed=feed, strategy=Minimal(), progress=False).run()

        assert result is not None
        assert result.strategy_name == "minimal_smoke"
        assert result.initial_cash == 1000.0
        assert result.final_equity == pytest.approx(1000.0)

    def test_engine_returns_backtest_result_type(
        self, bt_kalshi_trades_dir: Path, bt_kalshi_markets_dir: Path
    ) -> None:
        from src.backtesting.feeds.kalshi import KalshiFeed
        from src.backtesting.models import BacktestResult, TradeEvent
        from src.backtesting.rust_engine import Engine
        from src.backtesting.strategy import Strategy

        class Minimal(Strategy):
            def __init__(self) -> None:
                super().__init__(name="result_type", description="smoke", initial_cash=1000.0)

            def on_trade(self, trade: TradeEvent) -> None:
                pass

        feed = KalshiFeed(trades_dir=bt_kalshi_trades_dir, markets_dir=bt_kalshi_markets_dir)
        result = Engine(feed=feed, strategy=Minimal(), progress=False).run()
        assert isinstance(result, BacktestResult)

    def test_strategy_autodiscovery(self) -> None:
        """Strategy.load() finds bundled example strategies without errors."""
        from src.backtesting.strategy import Strategy

        strategies = Strategy.load()
        assert isinstance(strategies, list)
        assert len(strategies) >= 1
        names = {cls().name for cls in strategies}  # type: ignore[call-arg]
        assert "gambling_addiction" in names
