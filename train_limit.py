"""
Kalshi backtest: Fair Value Limit Order strategy.

Simple passive market-making:
- On each FV update, place a resting BUY below FV and SELL above FV
- Edge parameter controls distance from fair value
- Cancel and replace when FV changes
- Fills come from queue position mechanics when trades cross our levels

Usage:
    uv run python train_limit.py
"""

from __future__ import annotations

import math
import time

import prepare
from engine import BacktestEngine, FairValueData, Fill, OrderSide, OrderStatus, TimeInForce
from engine.strategy import Strategy


# ── Config ──

EVENT_TICKERS = [
    "KXRT-BRI",
    "KXRT-HOP",
    "KXRT-REM",
]
STARTING_BALANCE = 10_000
ORDER_SIZE = 10          # contracts per side
EDGE = 0.03             # place orders 3 cents from FV
TICK = 0.01             # Kalshi tick size
MAX_POSITION = 50       # max abs position per instrument


# ── Strategy ──

def round_to_tick(price: float) -> float:
    """Round price to nearest Kalshi tick (0.01)."""
    return round(round(price / TICK) * TICK, 2)


class FairValueLimitOrder(Strategy):
    """
    Passive market-making around fair value.

    Places resting BUY at fv - edge and SELL at fv + edge.
    Cancels and replaces when FV updates.
    Respects position limits.
    """

    def __init__(self, instrument_id: str):
        super().__init__(instrument_id)
        self._buy_order_id: str | None = None
        self._sell_order_id: str | None = None
        self._last_fv: float | None = None

    def on_data(self, data: FairValueData) -> None:
        fv = data.fv

        # Skip degenerate FVs
        if fv <= 0.02 or fv >= 0.98:
            self._cancel_all()
            return

        # Only re-quote if FV moved at least 1 tick
        if self._last_fv is not None and abs(fv - self._last_fv) < TICK:
            return
        self._last_fv = fv

        self._requote(fv, data.timestamp_ns)

    def on_book_update(self, instrument_id: str, timestamp_ns: int = 0) -> None:
        pass  # Limit order strategy doesn't react to book updates

    def on_fill(self, fill: Fill) -> None:
        # Clear our tracking when an order fills
        if self._buy_order_id and fill.order_id == self._buy_order_id:
            self._buy_order_id = None
        if self._sell_order_id and fill.order_id == self._sell_order_id:
            self._sell_order_id = None

    def on_stop(self) -> None:
        self._cancel_all()

    def _cancel_all(self) -> None:
        if self._buy_order_id:
            self.cancel_order(self._buy_order_id)
            self._buy_order_id = None
        if self._sell_order_id:
            self.cancel_order(self._sell_order_id)
            self._sell_order_id = None

    def _requote(self, fv: float, timestamp_ns: int) -> None:
        self._cancel_all()

        pos = self.get_position()
        signed_qty = pos.signed_qty

        buy_price = round_to_tick(fv - EDGE)
        sell_price = round_to_tick(fv + EDGE)

        # Clamp to valid range
        buy_price = max(0.01, min(0.99, buy_price))
        sell_price = max(0.01, min(0.99, sell_price))

        # Don't cross: sell must be above buy
        if sell_price <= buy_price:
            return

        # Place buy if not max long
        if signed_qty < MAX_POSITION:
            buy_qty = min(ORDER_SIZE, MAX_POSITION - signed_qty)
            if buy_qty > 0:
                order, fill = self.submit_order(
                    OrderSide.Buy, buy_price, buy_qty,
                    time_in_force=TimeInForce.POST_ONLY, timestamp_ns=timestamp_ns,
                )
                if order.status == OrderStatus.Resting:
                    self._buy_order_id = order.id
                if fill:
                    self.on_fill(fill)

        # Place sell if not max short
        if signed_qty > -MAX_POSITION:
            sell_qty = min(ORDER_SIZE, MAX_POSITION + signed_qty)
            if sell_qty > 0:
                order, fill = self.submit_order(
                    OrderSide.Sell, sell_price, sell_qty,
                    time_in_force=TimeInForce.POST_ONLY, timestamp_ns=timestamp_ns,
                )
                if order.status == OrderStatus.Resting:
                    self._sell_order_id = order.id
                if fill:
                    self.on_fill(fill)


# ── Main ──

def main() -> None:
    print("Loading data...")
    t0 = time.time()
    data = prepare.load(EVENT_TICKERS)
    load_time = time.time() - t0
    print(f"Data loaded in {load_time:.1f}s\n")

    engine = BacktestEngine(data.instruments, STARTING_BALANCE)
    for inst in data.instruments:
        engine.add_strategy(FairValueLimitOrder(inst.id))

    print(f"Running backtest with {len(data.instruments)} instruments...")
    t0 = time.time()
    engine.run(data.fair_values, data.orderbook_deltas, data.trades)
    run_time = time.time() - t0
    print(f"Backtest completed in {run_time:.1f}s")

    engine.print_results()


if __name__ == "__main__":
    main()
