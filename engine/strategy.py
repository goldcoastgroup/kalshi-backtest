"""Strategy base class for backtesting."""

from __future__ import annotations

from engine._engine import (
    EngineCore,
    FairValueData,
    Fill,
    Order,
    OrderSide,
    Position,
    TimeInForce,
)


class Strategy:
    """Base class for backtest strategies. Subclass in train.py."""

    def __init__(self, instrument_id: str):
        self.instrument_id = instrument_id
        self._core: EngineCore | None = None

    def _bind(self, core: EngineCore) -> None:
        """Called by BacktestEngine to connect this strategy to the engine core."""
        self._core = core

    # ── Callbacks (override in subclass) ──

    def on_start(self) -> None:
        """Called once before the event loop starts."""

    def on_data(self, data: FairValueData) -> None:
        """Called when a FairValueData event arrives for this instrument."""

    def on_book_update(self, instrument_id: str, timestamp_ns: int = 0) -> None:
        """Called after orderbook deltas are applied (on F_LAST)."""

    def on_fill(self, fill: Fill) -> None:
        """Called when one of this strategy's orders is filled."""

    def on_stop(self) -> None:
        """Called once after the event loop ends."""

    # ── Engine interaction ──

    def submit_order(
        self,
        side: OrderSide,
        price: float,
        quantity: float,
        time_in_force: TimeInForce | None = None,
        reduce_only: bool = False,
        timestamp_ns: int = 0,
    ) -> tuple[Order, Fill | None]:
        """Submit an order. Returns (Order, optional Fill if immediate)."""
        if time_in_force is None:
            time_in_force = TimeInForce.GTC
        return self._core.submit_order(
            self.instrument_id, side, price, quantity, time_in_force, reduce_only, timestamp_ns,
        )

    def modify_order(self, order_id: str, new_quantity: float) -> bool:
        """Modify quantity of a resting order. Returns True if found."""
        return self._core.modify_order(order_id, new_quantity)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a resting order. Returns True if found."""
        return self._core.cancel_order(order_id)

    def best_bid(self, instrument_id: str | None = None) -> tuple[float, float] | None:
        """Best bid (price, size) for an instrument."""
        return self._core.best_bid(instrument_id or self.instrument_id)

    def best_ask(self, instrument_id: str | None = None) -> tuple[float, float] | None:
        """Best ask (price, size) for an instrument."""
        return self._core.best_ask(instrument_id or self.instrument_id)

    def get_position(self, instrument_id: str | None = None) -> Position:
        """Get current position for an instrument."""
        return self._core.get_position(instrument_id or self.instrument_id)

    def get_balance(self) -> float:
        """Get current available cash balance."""
        return self._core.balance()

    def get_free_balance(self) -> float:
        """Get free balance (cash minus capital locked by resting orders)."""
        return self._core.free_balance()
