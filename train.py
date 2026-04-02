"""
Kalshi backtest strategy definition.

This is the ONLY file the training agent modifies.
It defines the strategy class, configuration constants, and calls
prepare.run_backtest() to execute.
"""

from __future__ import annotations

import math

import prepare
from engine import FairValueData, Fill, OrderSide, OrderStatus, TimeInForce
from engine.strategy import Strategy


# ── Config ──────────────────────────────────────────────────────────

HALF_SPREAD = 0.03          # quote 3c each side of FV
MAX_POSITION = 50           # max contracts long or short per instrument
ORDER_SIZE = 10             # contracts per quote
REQUOTE_INTERVAL_NS = 5 * 60 * 1_000_000_000   # requote every 5 min
REQUOTE_FV_DELTA = 0.02     # or if FV moved more than 2c


# ── Strategy ────────────────────────────────────────────────────────

class FVMarketMaker(Strategy):
    """
    Simple market maker: quote bid/ask around fair value.

    - On FV update, requote if enough time elapsed or FV moved significantly.
    - POST_ONLY bid at (fv - half_spread), ask at (fv + half_spread).
    - Skips the side where position limit would be breached.
    """

    def __init__(self, instrument_id: str):
        super().__init__(instrument_id)
        self._bid_order_id: str | None = None
        self._ask_order_id: str | None = None
        self._last_fv: float | None = None
        self._last_quote_fv: float | None = None
        self._last_quote_ts: int = 0

    def on_data(self, data: FairValueData) -> None:
        self._last_fv = data.fv

        # Only requote if enough time passed or FV moved meaningfully
        fv_moved = (
            self._last_quote_fv is None
            or abs(data.fv - self._last_quote_fv) >= REQUOTE_FV_DELTA
        )
        time_elapsed = data.timestamp_ns - self._last_quote_ts >= REQUOTE_INTERVAL_NS

        if fv_moved or time_elapsed:
            self._requote(data.timestamp_ns)

    def on_book_update(self, instrument_id: str, timestamp_ns: int = 0) -> None:
        pass

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_stop(self) -> None:
        pass

    def _requote(self, timestamp_ns: int) -> None:
        fv = self._last_fv
        if fv is None:
            return

        # Cancel existing quotes
        if self._bid_order_id is not None:
            self.cancel_order(self._bid_order_id)
            self._bid_order_id = None
        if self._ask_order_id is not None:
            self.cancel_order(self._ask_order_id)
            self._ask_order_id = None

        self._last_quote_fv = fv
        self._last_quote_ts = timestamp_ns

        pos = self.get_position()
        qty = pos.signed_qty

        # Compute quote prices (round to nearest cent)
        bid_price = math.floor((fv - HALF_SPREAD) * 100) / 100
        ask_price = math.ceil((fv + HALF_SPREAD) * 100) / 100

        # Clamp to valid range
        bid_price = max(0.01, min(0.99, bid_price))
        ask_price = max(0.01, min(0.99, ask_price))

        # Don't quote if spread collapses
        if bid_price >= ask_price:
            return

        # Quote bid (only if not max long)
        if qty < MAX_POSITION:
            bid_qty = min(ORDER_SIZE, MAX_POSITION - int(qty))
            if bid_qty >= 1:
                order, fill = self.submit_order(
                    OrderSide.Buy, bid_price, bid_qty,
                    time_in_force=TimeInForce.POST_ONLY,
                    timestamp_ns=timestamp_ns,
                )
                if order.status not in (OrderStatus.Rejected, OrderStatus.Canceled):
                    self._bid_order_id = order.id

        # Quote ask (only if not max short)
        if qty > -MAX_POSITION:
            ask_qty = min(ORDER_SIZE, MAX_POSITION + int(qty))
            if ask_qty >= 1:
                order, fill = self.submit_order(
                    OrderSide.Sell, ask_price, ask_qty,
                    time_in_force=TimeInForce.POST_ONLY,
                    timestamp_ns=timestamp_ns,
                )
                if order.status not in (OrderStatus.Rejected, OrderStatus.Canceled):
                    self._ask_order_id = order.id


# ── Run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prepare.run_backtest(strategy_factory=FVMarketMaker)
