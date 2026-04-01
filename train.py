"""
Kalshi backtest: KxrtBaseline market-making strategy.

This is the file the training agent modifies. Running it executes
the backtest and prints results.

Usage:
    python train.py
"""

from __future__ import annotations

import math
import time

import prepare
from engine import BacktestEngine, FairValueData, Fill, OrderSide, OrderStatus
from engine.strategy import Strategy


# ── Config (agent modifies these) ──

EVENT_TICKERS = [
    "KXRT-BRI",
    "KXRT-HOP",
    "KXRT-REM",
]
MAX_SIZE = 500
SPREAD = 0.05
KELLY_FRACTION = 0.5
BANKROLL_CAP = 1000
STARTING_BALANCE = 10_000


# ── Strategy ──

class KxrtBaseline(Strategy):
    """
    FV-centered market maker for Kalshi KXRT binary contracts.

    Quotes bid/ask symmetrically around fair value with a fixed spread.
    Order size via half-Kelly. Post-only orders; modify in-place when
    only quantity changes. Holds positions to settlement.
    """

    def __init__(self, instrument_id: str):
        super().__init__(instrument_id)
        self._bid_order = None
        self._ask_order = None
        self._signed_qty = 0.0
        self._settled = False

    def on_data(self, data: FairValueData) -> None:
        if data.instrument_id != self.instrument_id:
            return
        if self._settled:
            return

        fv = data.fv
        bid_price = _round_cent(fv - SPREAD)
        ask_price = _round_cent(fv + SPREAD)

        best_ask = self.best_ask()
        best_bid = self.best_bid()
        best_ask_price = best_ask[0] if best_ask else None
        best_bid_price = best_bid[0] if best_bid else None

        # ── Bid side ──
        bid_ok = bid_price >= 0.01
        if bid_ok and best_ask_price is not None and bid_price >= best_ask_price:
            bid_ok = False
        if bid_ok:
            bid_qty = _kelly_size(fv, bid_price, OrderSide.Buy, self.get_balance())
            if bid_qty > 0 and self._signed_qty < MAX_SIZE:
                qty = min(bid_qty, MAX_SIZE - self._signed_qty)
                self._update_or_place(OrderSide.Buy, bid_price, qty, data.timestamp_ns)
            else:
                self._cancel_bid()
        else:
            self._cancel_bid()

        # ── Ask side ──
        ask_ok = ask_price <= 0.99
        if ask_ok and best_bid_price is not None and ask_price <= best_bid_price:
            ask_ok = False
        if ask_ok:
            ask_qty = _kelly_size(fv, ask_price, OrderSide.Sell, self.get_balance())
            if ask_qty > 0 and self._signed_qty > -MAX_SIZE:
                qty = min(ask_qty, MAX_SIZE + self._signed_qty)
                self._update_or_place(OrderSide.Sell, ask_price, qty, data.timestamp_ns)
            else:
                self._cancel_ask()
        else:
            self._cancel_ask()

    def on_book_update(self, instrument_id: str) -> None:
        if instrument_id != self.instrument_id:
            return
        if self._settled:
            return
        if self._signed_qty == 0.0:
            return

        best_bid = self.best_bid()
        best_ask = self.best_ask()

        settling_yes = best_bid is not None and best_bid[0] >= 0.99
        settling_no = best_ask is not None and best_ask[0] <= 0.01

        if not settling_yes and not settling_no:
            return

        self._settled = True
        self._cancel_all()

        # Close position at settlement price
        if self._signed_qty > 0:
            close_price = 0.99 if settling_yes else 0.01
            self.submit_order(
                OrderSide.Sell, close_price, abs(self._signed_qty),
                post_only=False, timestamp_ns=0,
            )
        elif self._signed_qty < 0:
            close_price = 0.99 if settling_yes else 0.01
            self.submit_order(
                OrderSide.Buy, close_price, abs(self._signed_qty),
                post_only=False, timestamp_ns=0,
            )

    def on_fill(self, fill: Fill) -> None:
        # Update signed_qty from account position
        pos = self.get_position()
        self._signed_qty = pos.signed_qty

        # Clean up order references
        if self._bid_order and fill.order_id == self._bid_order.id:
            self._bid_order = None
        if self._ask_order and fill.order_id == self._ask_order.id:
            self._ask_order = None

    def on_stop(self) -> None:
        self._cancel_all()

    def _update_or_place(
        self, side: OrderSide, price: float, qty: float, timestamp_ns: int,
    ) -> None:
        order = self._bid_order if side == OrderSide.Buy else self._ask_order

        if order and not order.is_closed:
            if abs(order.price - price) < 1e-9:
                # Same price — modify quantity only
                if abs(order.quantity - qty) > 1e-9:
                    self.modify_order(order.id, qty)
                return
            else:
                # Price changed — cancel and replace
                self.cancel_order(order.id)

        new_order, fill = self.submit_order(side, price, qty, post_only=True, timestamp_ns=timestamp_ns)
        if side == OrderSide.Buy:
            self._bid_order = new_order
        else:
            self._ask_order = new_order

        if fill:
            self.on_fill(fill)

    def _cancel_bid(self) -> None:
        if self._bid_order and not self._bid_order.is_closed:
            self.cancel_order(self._bid_order.id)
            self._bid_order = None

    def _cancel_ask(self) -> None:
        if self._ask_order and not self._ask_order.is_closed:
            self.cancel_order(self._ask_order.id)
            self._ask_order = None

    def _cancel_all(self) -> None:
        self._cancel_bid()
        self._cancel_ask()


# ── Helpers ──

def _round_cent(value: float) -> float:
    """Round to nearest cent."""
    return round(value * 100) / 100


def _kelly_size(fv: float, price: float, side: OrderSide, bankroll: float) -> float:
    """Compute order size using fractional Kelly criterion."""
    if side == OrderSide.Buy:
        if fv <= price or price >= 1.0:
            return 0.0
        kelly_f = (fv - price) / (1.0 - price)
        cost_per = price
    else:
        if fv >= price or price <= 0.0:
            return 0.0
        kelly_f = (price - fv) / price
        cost_per = 1.0 - price

    effective_bankroll = min(bankroll, BANKROLL_CAP)
    if effective_bankroll <= 0 or cost_per <= 0:
        return 0.0

    scaled = kelly_f * KELLY_FRACTION
    contracts = int(scaled * effective_bankroll / cost_per)
    return max(0.0, float(contracts))


# ── Main ──

def main() -> None:
    print("Loading data...")
    t0 = time.time()
    data = prepare.load(EVENT_TICKERS)
    load_time = time.time() - t0
    print(f"Data loaded in {load_time:.1f}s\n")

    engine = BacktestEngine(data.instruments, STARTING_BALANCE)
    for inst in data.instruments:
        engine.add_strategy(KxrtBaseline(inst.id))

    print(f"Running backtest with {len(data.instruments)} instruments...")
    t0 = time.time()
    engine.run(data.fair_values, data.orderbook_deltas)
    run_time = time.time() - t0
    print(f"Backtest completed in {run_time:.1f}s")

    engine.print_results()


if __name__ == "__main__":
    main()
