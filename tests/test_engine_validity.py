"""Engine validity tests — fill semantics, P&L math, cash conservation, slippage.

These tests use controlled single-market datasets to verify that the engine's
core logic is correct independent of strategy complexity.  Each test class
targets one specific property of the simulation:

  * TestCLOBFillSemantics  — pure price-condition fill semantics
  * TestResolutionPnL      — settlement payouts (YES/NO × long YES/NO)
  * TestCashAccounting     — commission deduction, no-leverage enforcement
  * TestSlippage           — slippage parameter behaviour (not applied to maker fills)
  * TestOrderCancellation  — cancelled orders never fill
  * TestEquityCurveIntegrity — equity curve values are finite and consistent
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.backtesting.feeds.kalshi import KalshiFeed
from src.backtesting.models import TradeEvent
from src.backtesting.rust_engine import Engine
from src.backtesting.strategy import Strategy

# ---------------------------------------------------------------------------
# Reusable test strategies
# ---------------------------------------------------------------------------


class _BuyYesOnFirstTrade(Strategy):
    """Place a single buy YES limit order on the first market seen."""

    def __init__(self, limit_price: float = 0.30, quantity: float = 10.0, initial_cash: float = 1000.0) -> None:
        super().__init__(name="buy_yes_limit", description="test", initial_cash=initial_cash)
        self._limit_price = limit_price
        self._quantity = quantity
        self._placed = False

    def on_trade(self, trade: TradeEvent) -> None:
        if not self._placed:
            self.buy_yes(trade.market_id, price=self._limit_price, quantity=self._quantity)
            self._placed = True


class _BuyNoOnFirstTrade(Strategy):
    """Place a single buy NO limit order on the first market seen."""

    def __init__(self, limit_price: float = 0.30, quantity: float = 10.0, initial_cash: float = 1000.0) -> None:
        super().__init__(name="buy_no_limit", description="test", initial_cash=initial_cash)
        self._limit_price = limit_price
        self._quantity = quantity
        self._placed = False

    def on_trade(self, trade: TradeEvent) -> None:
        if not self._placed:
            self.buy_no(trade.market_id, price=self._limit_price, quantity=self._quantity)
            self._placed = True


class _SellYesOnFirstTrade(Strategy):
    """Place a single sell YES limit order on the first market seen."""

    def __init__(self, limit_price: float = 0.30, quantity: float = 10.0, initial_cash: float = 1000.0) -> None:
        super().__init__(name="sell_yes_limit", description="test", initial_cash=initial_cash)
        self._limit_price = limit_price
        self._quantity = quantity
        self._placed = False

    def on_trade(self, trade: TradeEvent) -> None:
        if not self._placed:
            self.sell_yes(trade.market_id, price=self._limit_price, quantity=self._quantity)
            self._placed = True


class _CancelOnSecondTrade(Strategy):
    """Place a buy YES order on trade 0, cancel it on trade 1."""

    def __init__(self) -> None:
        super().__init__(name="cancel_test", description="test", initial_cash=1000.0)
        self._order = None
        self._cancelled = False

    def on_trade(self, trade: TradeEvent) -> None:
        if self._order is None:
            self._order = self.buy_yes(trade.market_id, price=0.30, quantity=10.0)
        elif not self._cancelled:
            self.cancel_order(self._order.order_id)
            self._cancelled = True


# ---------------------------------------------------------------------------
# Shared engine constructor with zero fees for exact P&L math
# ---------------------------------------------------------------------------


def _engine(
    feed: KalshiFeed,
    strategy: Strategy,
    *,
    slippage: float = 0.0,
    commission_rate: float = 0.0,
    snapshot_interval: int = 1,
) -> Engine:
    return Engine(
        feed=feed,
        strategy=strategy,
        commission_rate=commission_rate,
        slippage=slippage,
        snapshot_interval=snapshot_interval,
        progress=False,
    )


# ---------------------------------------------------------------------------
# CLOB fill semantics
# ---------------------------------------------------------------------------


class TestCLOBFillSemantics:
    """Pure price-condition fill semantics.

    The engine uses pure price-condition matching — no taker-side filter.
    on_trade fires before check_fills, so an order placed in on_trade fills
    against the same trade that triggered it ("buy at the price you see").

    A resting limit order fills whenever the trade price satisfies the limit,
    regardless of which side initiated the trade:
      BUY  YES fills when trade.yes_price <= order.price
      SELL YES fills when trade.yes_price >= order.price
      BUY  NO  fills when trade.no_price  <= order.price
      SELL NO  fills when trade.no_price  >= order.price
    """

    def test_buy_yes_fills_on_no_taker(self, make_dataset) -> None:
        """Buy YES limit fills when price condition is satisfied (taker=no)."""
        td, md = make_dataset(
            "MKT-X",
            "yes",
            [
                (50, "yes", 5),  # trade 0 — strategy places buy YES @ 0.30; yes=0.50 > 0.30, no fill
                (20, "no", 10),  # trade 1 — yes=0.20 ≤ 0.30 → FILL
            ],
        )
        result = _engine(KalshiFeed(td, md), _BuyYesOnFirstTrade(limit_price=0.30)).run()
        assert len(result.fills) == 1

    def test_buy_yes_fills_regardless_of_taker_side(self, make_dataset) -> None:
        """Buy YES limit fills on price condition alone — taker side is irrelevant."""
        td, md = make_dataset(
            "MKT-X",
            "yes",
            [
                (50, "yes", 5),  # trade 0 — strategy places buy YES @ 0.30; yes=0.50 > 0.30, no fill
                (20, "yes", 10),  # trade 1 — yes=0.20 ≤ 0.30, taker=yes → FILLS (no taker filter)
                (20, "yes", 10),  # trade 2 — order already filled
            ],
        )
        result = _engine(KalshiFeed(td, md), _BuyYesOnFirstTrade(limit_price=0.30)).run()
        assert len(result.fills) == 1

    def test_buy_yes_no_fill_when_price_above_limit(self, make_dataset) -> None:
        """Buy YES limit does NOT fill when the market trades above the limit price."""
        td, md = make_dataset(
            "MKT-X",
            "yes",
            [
                (50, "yes", 5),  # trade 0 — strategy places buy YES @ 0.30
                (40, "no", 10),  # trade 1 — yes=0.40 > 0.30 → NO fill
                (35, "no", 10),  # trade 2 — yes=0.35 > 0.30 → NO fill
            ],
        )
        result = _engine(KalshiFeed(td, md), _BuyYesOnFirstTrade(limit_price=0.30)).run()
        assert len(result.fills) == 0

    def test_buy_no_fills_on_price_condition(self, make_dataset) -> None:
        """Buy NO limit fills when no_price <= limit, regardless of taker side."""
        # Trade 0: yes=80, no=20; strategy places buy NO @ 0.25; no_price=0.20 ≤ 0.25 → fills immediately.
        td, md = make_dataset(
            "MKT-X",
            "no",
            [
                (80, "no", 5),  # trade 0 — strategy places buy NO @ 0.25; no=0.20 ≤ 0.25 → FILL
                (75, "yes", 10),  # trade 1 — order already filled
            ],
        )
        result = _engine(KalshiFeed(td, md), _BuyNoOnFirstTrade(limit_price=0.25)).run()
        assert len(result.fills) == 1

    def test_buy_no_fills_regardless_of_taker_side(self, make_dataset) -> None:
        """Buy NO limit fills on price condition alone — taker=no does not block the fill."""
        td, md = make_dataset(
            "MKT-X",
            "no",
            [
                (80, "no", 5),  # trade 0 — strategy places buy NO @ 0.25; no=0.20 ≤ 0.25 → FILL
                (75, "no", 10),  # trade 1 — order already filled
            ],
        )
        result = _engine(KalshiFeed(td, md), _BuyNoOnFirstTrade(limit_price=0.25)).run()
        assert len(result.fills) == 1

    def test_sell_yes_fills_on_yes_taker(self, make_dataset) -> None:
        """Sell YES limit fills when yes_price >= limit."""
        td, md = make_dataset(
            "MKT-X",
            "yes",
            [
                (25, "no", 5),  # trade 0 — strategy places sell YES @ 0.30; yes=0.25 < 0.30, no fill
                (35, "yes", 10),  # trade 1 — yes=0.35 ≥ 0.30 → FILL
            ],
        )
        result = _engine(KalshiFeed(td, md), _SellYesOnFirstTrade(limit_price=0.30)).run()
        assert len(result.fills) == 1

    def test_sell_yes_fills_regardless_of_taker_side(self, make_dataset) -> None:
        """Sell YES limit fills on price condition alone — taker=no does not block the fill."""
        td, md = make_dataset(
            "MKT-X",
            "yes",
            [
                (25, "no", 5),  # trade 0 — strategy places sell YES @ 0.30; yes=0.25 < 0.30, no fill
                (35, "no", 10),  # trade 1 — yes=0.35 ≥ 0.30, taker=no → FILLS (no taker filter)
            ],
        )
        result = _engine(KalshiFeed(td, md), _SellYesOnFirstTrade(limit_price=0.30)).run()
        assert len(result.fills) == 1

    def test_fill_price_equals_trade_price_not_limit(self, make_dataset) -> None:
        """Orders fill at the trade's market price, not the order's limit price.

        A limit buy acts as a price ceiling: the order fills if the market trades
        at or below the limit, and the actual fill price is the market trade price.
        """
        td, md = make_dataset(
            "MKT-X",
            "yes",
            [
                (50, "yes", 5),  # trade 0 — places buy YES @ 0.40 limit
                (20, "no", 10),  # trade 1 — trade at yes=0.20 triggers fill; price = 0.20
            ],
        )
        result = _engine(KalshiFeed(td, md), _BuyYesOnFirstTrade(limit_price=0.40)).run()
        assert len(result.fills) == 1
        assert result.fills[0].price == pytest.approx(0.20)

    def test_multiple_orders_fill_from_single_trade(self, make_dataset) -> None:
        """Two resting limit orders on the same market both fill against one eligible trade."""

        class TwoOrderStrategy(Strategy):
            def __init__(self) -> None:
                super().__init__(name="two_order", description="test", initial_cash=1000.0)
                self._count = 0

            def on_trade(self, trade: TradeEvent) -> None:
                if self._count < 2:
                    self.buy_yes(trade.market_id, price=0.40, quantity=5.0)
                self._count += 1

        td, md = make_dataset(
            "MKT-X",
            "yes",
            [
                (50, "yes", 5),  # trade 0 — places order #1
                (50, "yes", 5),  # trade 1 — places order #2 (order #1 doesn't fill: yes > limit)
                (30, "no", 100),  # trade 2 — yes=0.30 ≤ 0.40 → fills both
            ],
        )
        result = _engine(KalshiFeed(td, md), TwoOrderStrategy()).run()
        assert len(result.fills) == 2


# ---------------------------------------------------------------------------
# Resolution P&L
# ---------------------------------------------------------------------------


class TestResolutionPnL:
    """Settlement payouts are correct for all four long/resolution combinations.

    With slippage=0 and commission_rate=0 the math is exact:
      Long YES N contracts @ entry_price, resolves YES  → payout = N, cost = N * entry_price
      Long YES N contracts @ entry_price, resolves NO   → payout = 0
      Long NO  N contracts @ entry_price, resolves NO   → payout = N
      Long NO  N contracts @ entry_price, resolves YES  → payout = 0
    """

    def test_long_yes_wins_on_yes_resolution(self, make_dataset) -> None:
        """Buy YES fills at trade price 0.20, market resolves YES → final_equity = 1000 - 2.00 + 10.00."""
        td, md = make_dataset(
            "MKT-X",
            "yes",
            [
                (50, "yes", 5),  # trade 0 — places buy YES @ 0.25 limit
                (20, "no", 10),  # trade 1 — yes=0.20 ≤ 0.25, taker=no → fills at trade price 0.20
            ],
        )
        result = _engine(KalshiFeed(td, md), _BuyYesOnFirstTrade(limit_price=0.25, quantity=10.0)).run()
        assert len(result.fills) == 1
        assert result.fills[0].price == pytest.approx(0.20)
        assert result.final_equity == pytest.approx(1008.0)  # 1000 - 2.00 + 10.00

    def test_long_yes_loses_on_no_resolution(self, make_dataset) -> None:
        """Buy YES fills at trade price 0.20, market resolves NO → final_equity = 1000 - 2.00."""
        td, md = make_dataset(
            "MKT-X",
            "no",
            [
                (50, "yes", 5),
                (20, "no", 10),
            ],
        )
        result = _engine(KalshiFeed(td, md), _BuyYesOnFirstTrade(limit_price=0.25, quantity=10.0)).run()
        assert len(result.fills) == 1
        assert result.final_equity == pytest.approx(998.0)  # 1000 - 2.00 + 0.00

    def test_long_no_wins_on_no_resolution(self, make_dataset) -> None:
        """Buy NO fills at trade price 0.25, market resolves NO → final_equity = 1000 - 2.50 + 10.00."""
        # Trade 0: yes=70, no=30; no_price=0.30 > limit 0.25 → no immediate fill.
        # Trade 1: yes=75, no=25; no_price=0.25 ≤ 0.25 → fills at trade price 0.25.
        td, md = make_dataset(
            "MKT-X",
            "no",
            [
                (70, "no", 5),  # trade 0 — places buy NO @ 0.25; no=0.30 > 0.25, no fill
                (75, "yes", 10),  # trade 1 — no=0.25 ≤ 0.25 → fills at trade price 0.25
            ],
        )
        result = _engine(KalshiFeed(td, md), _BuyNoOnFirstTrade(limit_price=0.25, quantity=10.0)).run()
        assert len(result.fills) == 1
        assert result.fills[0].price == pytest.approx(0.25)
        assert result.final_equity == pytest.approx(1007.5)  # 1000 - 2.50 + 10.00

    def test_long_no_loses_on_yes_resolution(self, make_dataset) -> None:
        """Buy NO fills at trade price 0.25, market resolves YES → final_equity = 1000 - 2.50."""
        td, md = make_dataset(
            "MKT-X",
            "yes",  # resolves YES — NO holder gets nothing
            [
                (70, "no", 5),  # trade 0 — places buy NO @ 0.25; no=0.30 > 0.25, no fill
                (75, "yes", 10),  # trade 1 — no=0.25 ≤ 0.25 → fills at trade price 0.25
            ],
        )
        result = _engine(KalshiFeed(td, md), _BuyNoOnFirstTrade(limit_price=0.25, quantity=10.0)).run()
        assert len(result.fills) == 1
        assert result.final_equity == pytest.approx(997.5)  # 1000 - 2.50 + 0.00

    def test_no_fill_no_pnl(self, make_dataset) -> None:
        """If no order fills, final equity equals initial cash regardless of resolution."""
        td, md = make_dataset(
            "MKT-X",
            "yes",
            [
                (50, "yes", 5),  # trade 0 — places buy YES @ 0.20 (limit below market)
                (50, "yes", 10),  # trade 1 — yes=0.50 > limit 0.20 → no fill
            ],
        )
        result = _engine(KalshiFeed(td, md), _BuyYesOnFirstTrade(limit_price=0.20)).run()
        assert len(result.fills) == 0
        assert result.final_equity == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Cash accounting
# ---------------------------------------------------------------------------


class TestCashAccounting:
    """Commission is deducted from cash; buys cannot exceed available cash."""

    def test_commission_reduces_final_equity(self, make_dataset) -> None:
        """A run with commission_rate > 0 produces lower equity than one with 0%."""
        td, md = make_dataset(
            "MKT-X",
            "yes",
            [
                (50, "yes", 5),
                (30, "no", 10),  # yes=0.30 ≤ limit 0.40 → fill
            ],
        )
        feed_a = KalshiFeed(td, md)
        no_fee = _engine(feed_a, _BuyYesOnFirstTrade(limit_price=0.40), commission_rate=0.0).run()

        feed_b = KalshiFeed(td, md)
        with_fee = Engine(
            feed=feed_b,
            strategy=_BuyYesOnFirstTrade(limit_price=0.40),
            commission_rate=0.01,
            slippage=0.0,
            progress=False,
        ).run()

        assert len(no_fee.fills) == 1
        assert len(with_fee.fills) == 1
        assert with_fee.final_equity < no_fee.final_equity

    def test_commission_equals_rate_times_fill_cost(self, make_dataset) -> None:
        """commission = commission_rate × P × (1 − P) × qty  (Kalshi fee formula)."""
        td, md = make_dataset(
            "MKT-X",
            "yes",
            [
                (50, "yes", 5),
                (30, "no", 10),
            ],
        )
        rate = 0.07
        feed = KalshiFeed(td, md)
        result = Engine(
            feed=feed,
            strategy=_BuyYesOnFirstTrade(limit_price=0.40, quantity=10.0),
            commission_rate=rate,
            slippage=0.0,
            progress=False,
        ).run()
        assert len(result.fills) == 1
        f = result.fills[0]
        expected_commission = rate * f.price * (1 - f.price) * f.quantity
        assert f.commission == pytest.approx(expected_commission, rel=1e-6)

    def test_no_leverage_fill_capped_by_cash(self, make_dataset) -> None:
        """An oversized order is capped so the fill cost never exceeds available cash."""

        class OverbuyStrategy(Strategy):
            def __init__(self) -> None:
                super().__init__(name="overbuy", description="test", initial_cash=10.0)
                self._placed = False

            def on_trade(self, trade: TradeEvent) -> None:
                if not self._placed:
                    self.buy_yes(trade.market_id, price=0.50, quantity=10_000.0)
                    self._placed = True

        td, md = make_dataset(
            "MKT-X",
            "yes",
            [
                (60, "yes", 5),  # trade 0 — places oversized order
                (50, "no", 1000),  # trade 1 — yes=0.50 ≤ limit 0.50, large liquidity → partial fill
            ],
        )
        result = Engine(
            feed=KalshiFeed(td, md),
            strategy=OverbuyStrategy(),
            commission_rate=0.0,
            slippage=0.0,
            progress=False,
        ).run()

        assert len(result.fills) > 0
        total_cost = sum(f.price * f.quantity + f.commission for f in result.fills)
        assert total_cost <= 10.0 + 1e-9


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------


class TestSlippage:
    """Slippage parameter behaviour for resting limit (maker) fills.

    The engine's `apply_market_impact` function exists for potential future
    aggressive / market-order strategies but is NOT applied to resting limit
    orders.  Resting limit orders always fill at the trade price — the only
    transaction cost is the exchange commission.  Passing slippage > 0 to the
    Engine constructor therefore has no effect on maker fill prices.
    """

    def test_zero_slippage_fill_price_equals_trade_price(self, make_dataset) -> None:
        """With slippage=0, fill price equals the market trade price (no impact added)."""
        td, md = make_dataset(
            "MKT-X",
            "yes",
            [
                (50, "yes", 5),
                (30, "no", 10),  # trade at yes=0.30; fill price = trade price 0.30
            ],
        )
        result = _engine(KalshiFeed(td, md), _BuyYesOnFirstTrade(limit_price=0.40), slippage=0.0).run()
        assert len(result.fills) == 1
        assert result.fills[0].price == pytest.approx(0.30)

    def test_nonzero_slippage_has_no_effect_on_maker_fills(self, make_dataset) -> None:
        """Slippage is not applied to resting limit orders — fill price equals trade price."""
        td, md = make_dataset(
            "MKT-X",
            "yes",
            [
                (60, "yes", 5),
                (30, "no", 10),  # triggers fill for both runs
            ],
        )
        feed_a = KalshiFeed(td, md)
        no_slip = _engine(feed_a, _BuyYesOnFirstTrade(limit_price=0.50), slippage=0.0).run()

        feed_b = KalshiFeed(td, md)
        with_slip = Engine(
            feed=feed_b,
            strategy=_BuyYesOnFirstTrade(limit_price=0.50),
            slippage=0.05,
            commission_rate=0.0,
            progress=False,
        ).run()

        assert len(no_slip.fills) == 1
        assert len(with_slip.fills) == 1
        # Resting limit orders fill at trade price regardless of the slippage setting.
        assert with_slip.fills[0].price == pytest.approx(no_slip.fills[0].price)

    def test_slippage_setting_does_not_affect_fill_price_at_any_odds(self, make_dataset) -> None:
        """Slippage has no effect near 50% or near 5% — maker fills always equal trade price."""
        slippage = 0.005

        td_mid, md_mid = make_dataset("MKT-MID", "yes", [(60, "yes", 5), (45, "no", 10)])
        td_mid_0, md_mid_0 = make_dataset("MKT-MID-0", "yes", [(60, "yes", 5), (45, "no", 10)])
        td_ext, md_ext = make_dataset("MKT-EXT", "yes", [(10, "yes", 5), (4, "no", 10)])
        td_ext_0, md_ext_0 = make_dataset("MKT-EXT-0", "yes", [(10, "yes", 5), (4, "no", 10)])

        res_mid = Engine(
            feed=KalshiFeed(td_mid, md_mid),
            strategy=_BuyYesOnFirstTrade(limit_price=0.50),
            slippage=slippage,
            commission_rate=0.0,
            progress=False,
        ).run()
        res_mid_0 = Engine(
            feed=KalshiFeed(td_mid_0, md_mid_0),
            strategy=_BuyYesOnFirstTrade(limit_price=0.50),
            slippage=0.0,
            commission_rate=0.0,
            progress=False,
        ).run()
        res_ext = Engine(
            feed=KalshiFeed(td_ext, md_ext),
            strategy=_BuyYesOnFirstTrade(limit_price=0.05),
            slippage=slippage,
            commission_rate=0.0,
            progress=False,
        ).run()
        res_ext_0 = Engine(
            feed=KalshiFeed(td_ext_0, md_ext_0),
            strategy=_BuyYesOnFirstTrade(limit_price=0.05),
            slippage=0.0,
            commission_rate=0.0,
            progress=False,
        ).run()

        assert len(res_mid.fills) == 1
        assert len(res_ext.fills) == 1

        # Slippage has no effect on maker fills at any price level.
        assert res_mid.fills[0].price == pytest.approx(res_mid_0.fills[0].price)
        assert res_ext.fills[0].price == pytest.approx(res_ext_0.fills[0].price)


# ---------------------------------------------------------------------------
# Order cancellation
# ---------------------------------------------------------------------------


class TestOrderCancellation:
    """Cancelled orders are removed from the book and never fill."""

    def test_cancelled_order_does_not_fill(self, make_dataset) -> None:
        """Order placed on trade 0, cancelled on trade 1, eligible trade on trade 2 → 0 fills."""
        td, md = make_dataset(
            "MKT-X",
            "yes",
            [
                (50, "yes", 5),  # trade 0 — strategy places buy YES @ 0.30
                (50, "yes", 5),  # trade 1 — strategy cancels; yes > limit anyway
                (20, "no", 10),  # trade 2 — would fill but order was cancelled
            ],
        )
        result = _engine(KalshiFeed(td, md), _CancelOnSecondTrade()).run()
        assert len(result.fills) == 0

    def test_unfilled_order_cancels_cleanly(self, make_dataset) -> None:
        """cancel_all removes all pending orders; subsequent eligible trades produce no fills."""

        class CancelAllStrategy(Strategy):
            def __init__(self) -> None:
                super().__init__(name="cancel_all", description="test", initial_cash=1000.0)
                self._step = 0

            def on_trade(self, trade: TradeEvent) -> None:
                if self._step == 0:
                    self.buy_yes(trade.market_id, price=0.40, quantity=5.0)
                    self.buy_yes(trade.market_id, price=0.40, quantity=5.0)
                elif self._step == 1:
                    self.cancel_all()
                self._step += 1

        td, md = make_dataset(
            "MKT-X",
            "yes",
            [
                (50, "yes", 5),  # trade 0 — places 2 orders
                (50, "yes", 5),  # trade 1 — cancels all
                (20, "no", 100),  # trade 2 — would fill both, but all cancelled
            ],
        )
        result = _engine(KalshiFeed(td, md), CancelAllStrategy()).run()
        assert len(result.fills) == 0


# ---------------------------------------------------------------------------
# Equity curve integrity
# ---------------------------------------------------------------------------


class TestEquityCurveIntegrity:
    """Equity curve values are finite, non-negative, and consistent with final_equity."""

    def test_equity_curve_all_finite(self, bt_kalshi_trades_dir: Path, bt_kalshi_markets_dir: Path) -> None:
        """No NaN or Inf values appear anywhere in the equity curve."""

        class NoOp(Strategy):
            def __init__(self) -> None:
                super().__init__(name="noop_finite", description="test", initial_cash=1000.0)

            def on_trade(self, trade: TradeEvent) -> None:
                pass

        feed = KalshiFeed(trades_dir=bt_kalshi_trades_dir, markets_dir=bt_kalshi_markets_dir)
        result = Engine(feed=feed, strategy=NoOp(), snapshot_interval=1, progress=False).run()

        for snap in result.equity_curve:
            assert math.isfinite(snap.total_equity), f"Non-finite equity at {snap.timestamp}"
            assert math.isfinite(snap.cash), f"Non-finite cash at {snap.timestamp}"
            assert math.isfinite(snap.unrealized_pnl), f"Non-finite unrealized_pnl at {snap.timestamp}"

    def test_noop_equity_curve_stays_constant(self, bt_kalshi_trades_dir: Path, bt_kalshi_markets_dir: Path) -> None:
        """A strategy that never trades should have a flat equity curve."""

        class NoOp(Strategy):
            def __init__(self) -> None:
                super().__init__(name="noop_flat", description="test", initial_cash=500.0)

            def on_trade(self, trade: TradeEvent) -> None:
                pass

        feed = KalshiFeed(trades_dir=bt_kalshi_trades_dir, markets_dir=bt_kalshi_markets_dir)
        result = Engine(feed=feed, strategy=NoOp(), snapshot_interval=1, progress=False).run()

        for snap in result.equity_curve:
            assert snap.total_equity == pytest.approx(500.0)

    def test_equity_curve_populated_at_snapshot_interval_one(
        self, bt_kalshi_trades_dir: Path, bt_kalshi_markets_dir: Path
    ) -> None:
        """snapshot_interval=1 yields at least as many snapshots as there are trades."""

        class NoOp(Strategy):
            def __init__(self) -> None:
                super().__init__(name="noop_dense", description="test", initial_cash=1000.0)

            def on_trade(self, trade: TradeEvent) -> None:
                pass

        feed = KalshiFeed(trades_dir=bt_kalshi_trades_dir, markets_dir=bt_kalshi_markets_dir)
        result = Engine(feed=feed, strategy=NoOp(), snapshot_interval=1, progress=False).run()

        # The fixture has 10 trades
        assert len(result.equity_curve) >= 10

    def test_num_fills_metric_matches_fills_list(self, bt_kalshi_trades_dir: Path, bt_kalshi_markets_dir: Path) -> None:
        """result.metrics['num_fills'] must equal len(result.fills)."""

        class BuyOnSeen(Strategy):
            def __init__(self) -> None:
                super().__init__(name="buy_seen", description="test", initial_cash=5000.0)
                self._seen: set[str] = set()

            def on_trade(self, trade: TradeEvent) -> None:
                if trade.market_id not in self._seen:
                    self.buy_yes(trade.market_id, price=0.50, quantity=5.0)
                    self._seen.add(trade.market_id)

        feed = KalshiFeed(trades_dir=bt_kalshi_trades_dir, markets_dir=bt_kalshi_markets_dir)
        result = Engine(
            feed=feed,
            strategy=BuyOnSeen(),
            commission_rate=0.0,
            slippage=0.0,
            progress=False,
        ).run()

        assert result.metrics["num_fills"] == pytest.approx(float(len(result.fills)))
