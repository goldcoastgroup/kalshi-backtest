"""Unit tests for src.backtesting.metrics.compute_metrics.

These tests exercise the metrics module directly with synthetic equity curves
and fill records, without running the full engine.  The goal is to catch regressions
in the metric formulas themselves (total_return, max_drawdown, Sharpe, etc.).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from src.backtesting.metrics import compute_metrics
from src.backtesting.models import Fill, OrderAction, PortfolioSnapshot, Side

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T0 = datetime(2024, 1, 1)


def _snap(days: float, equity: float, cash: float | None = None) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp=T0 + timedelta(days=days),
        cash=cash if cash is not None else equity,
        total_equity=equity,
        unrealized_pnl=0.0,
        num_positions=0,
    )


def _fill(
    market_id: str,
    action: OrderAction = OrderAction.BUY,
    price: float = 0.50,
    quantity: float = 10.0,
    commission: float = 0.0,
) -> Fill:
    return Fill(
        order_id="test",
        market_id=market_id,
        action=action,
        side=Side.YES,
        price=price,
        quantity=quantity,
        timestamp=T0,
        commission=commission,
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_equity_curve_returns_all_zeros(self) -> None:
        m = compute_metrics([], [], 10_000.0)
        assert m["total_return"] == 0.0
        assert m["sharpe_ratio"] == 0.0
        assert m["sortino_ratio"] == 0.0
        assert m["max_drawdown"] == 0.0
        assert m["num_fills"] == 0.0

    def test_single_snapshot_no_return(self) -> None:
        m = compute_metrics([_snap(0, 10_000.0)], [], 10_000.0)
        assert m["total_return"] == pytest.approx(0.0)
        assert m["sharpe_ratio"] == pytest.approx(0.0)
        assert m["max_drawdown"] == pytest.approx(0.0)

    def test_all_metric_keys_present(self) -> None:
        expected_keys = {
            "total_return",
            "final_equity",
            "annualized_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "max_drawdown_duration_days",
            "num_fills",
            "total_commission",
            "num_market_trades",
            "win_rate",
            "avg_trade_pnl",
            "avg_win",
            "avg_loss",
            "profit_factor",
            "total_realized_pnl",
        }
        m = compute_metrics([_snap(0, 10_000.0), _snap(180, 11_000.0)], [], 10_000.0)
        assert expected_keys.issubset(m.keys())


# ---------------------------------------------------------------------------
# Total return
# ---------------------------------------------------------------------------


class TestTotalReturn:
    def test_positive_return(self) -> None:
        m = compute_metrics([_snap(0, 10_000.0), _snap(180, 11_000.0)], [], 10_000.0)
        assert m["total_return"] == pytest.approx(0.10)
        assert m["final_equity"] == pytest.approx(11_000.0)

    def test_negative_return(self) -> None:
        m = compute_metrics([_snap(0, 10_000.0), _snap(180, 9_000.0)], [], 10_000.0)
        assert m["total_return"] == pytest.approx(-0.10)

    def test_zero_return(self) -> None:
        m = compute_metrics([_snap(0, 10_000.0), _snap(180, 10_000.0)], [], 10_000.0)
        assert m["total_return"] == pytest.approx(0.0)

    def test_formula_matches_definition(self) -> None:
        """total_return = (final - initial) / initial."""
        initial, final = 7_500.0, 9_000.0
        m = compute_metrics([_snap(0, initial), _snap(365, final)], [], initial)
        assert m["total_return"] == pytest.approx((final - initial) / initial)


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_monotone_rise_gives_zero_drawdown(self) -> None:
        curve = [_snap(float(i), 10_000.0 + i * 200.0) for i in range(6)]
        m = compute_metrics(curve, [], 10_000.0)
        assert m["max_drawdown"] == pytest.approx(0.0)

    def test_known_drawdown(self) -> None:
        # Peak = 12_000, trough = 9_000 → dd = (12_000 - 9_000) / 12_000 = 0.25
        curve = [
            _snap(0, 10_000.0),
            _snap(1, 12_000.0),  # new peak
            _snap(2, 9_000.0),  # trough
            _snap(3, 11_000.0),
        ]
        m = compute_metrics(curve, [], 10_000.0)
        assert m["max_drawdown"] == pytest.approx(0.25)

    def test_drawdown_bounded_zero_to_one(self) -> None:
        curve = [
            _snap(0, 10_000.0),
            _snap(1, 8_000.0),
            _snap(2, 12_000.0),
        ]
        m = compute_metrics(curve, [], 10_000.0)
        assert 0.0 <= m["max_drawdown"] <= 1.0

    def test_drawdown_from_initial_peak(self) -> None:
        # The very first point is the peak; equity drops immediately
        curve = [
            _snap(0, 10_000.0),
            _snap(1, 6_000.0),  # dd = (10_000 - 6_000) / 10_000 = 0.40
        ]
        m = compute_metrics(curve, [], 10_000.0)
        assert m["max_drawdown"] == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# Fill / commission metrics
# ---------------------------------------------------------------------------


class TestFillMetrics:
    def test_num_fills_counted_correctly(self) -> None:
        fills = [_fill("A"), _fill("B"), _fill("A")]
        curve = [_snap(0, 10_000.0), _snap(180, 9_500.0)]
        m = compute_metrics(curve, fills, 10_000.0)
        assert m["num_fills"] == pytest.approx(3.0)

    def test_commission_summed_across_fills(self) -> None:
        fills = [
            _fill("A", commission=0.05),
            _fill("B", commission=0.025),
            _fill("A", commission=0.10),
        ]
        curve = [_snap(0, 10_000.0), _snap(180, 9_500.0)]
        m = compute_metrics(curve, fills, 10_000.0)
        assert m["total_commission"] == pytest.approx(0.175)

    def test_no_fills_trade_metrics_are_zero(self) -> None:
        curve = [_snap(0, 10_000.0), _snap(180, 9_500.0)]
        m = compute_metrics(curve, [], 10_000.0)
        assert m["num_fills"] == 0.0
        assert m["total_commission"] == 0.0
        assert m["win_rate"] == 0.0
        assert m["num_market_trades"] == 0.0
        assert m["profit_factor"] == 0.0

    def test_win_rate_bounded_zero_to_one(self) -> None:
        fills = [_fill("A", price=0.30), _fill("B", price=0.80)]
        curve = [_snap(0, 10_000.0), _snap(180, 9_000.0)]
        m = compute_metrics(curve, fills, 10_000.0)
        assert 0.0 <= m["win_rate"] <= 1.0

    def test_profit_factor_inf_when_no_losses(self) -> None:
        """profit_factor = gross_profit / gross_loss; if gross_loss = 0 → inf."""
        # market_pnls with all positive: one market, bought at 0.30, no sell recorded
        # (fill-only P&L: buy cost is negative pnl for that market)
        # Provide market_pnls directly to control the outcome
        fills = [_fill("A", OrderAction.BUY, price=0.10, quantity=10.0)]
        curve = [_snap(0, 10_000.0), _snap(180, 11_000.0)]
        market_pnls = {"A": 5.0}  # positive P&L on the only traded market
        m = compute_metrics(curve, fills, 10_000.0, market_pnls=market_pnls)
        assert m["profit_factor"] == float("inf")


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------


class TestSharpeRatio:
    def test_constant_equity_gives_zero_sharpe(self) -> None:
        """Zero return volatility → Sharpe = 0."""
        curve = [_snap(float(i), 10_000.0) for i in range(10)]
        m = compute_metrics(curve, [], 10_000.0)
        assert m["sharpe_ratio"] == pytest.approx(0.0)

    def test_steady_gains_positive_sharpe(self) -> None:
        """Monotone upward equity → all period returns are positive → Sharpe > 0."""
        curve = [_snap(float(i), 10_000.0 + i * 100.0) for i in range(20)]
        m = compute_metrics(curve, [], 10_000.0)
        assert m["sharpe_ratio"] > 0.0

    def test_sharpe_is_always_finite(self) -> None:
        """Sharpe should never be NaN or Inf for well-formed equity curves."""
        curve = [
            _snap(0, 10_000.0),
            _snap(1, 10_500.0),
            _snap(2, 9_800.0),
            _snap(3, 11_200.0),
        ]
        m = compute_metrics(curve, [], 10_000.0)
        assert math.isfinite(m["sharpe_ratio"])
        assert math.isfinite(m["sortino_ratio"])
