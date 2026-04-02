"""
Kalshi backtest: OrderBookImbalance strategy.

Emulates the nautilus OrderBookImbalance strategy:
- On book update, checks bid/ask size ratio
- When imbalance exceeds threshold, submits aggressive FOK-style order
- Buys at best ask when bid >> ask, sells at best bid when ask >> bid

Usage:
    python train.py
"""

from __future__ import annotations

import time

import prepare
from engine import BacktestEngine, FairValueData, Fill, OrderSide, OrderStatus, TimeInForce
from engine.strategy import Strategy


# ── Config (matches nautilus kxrt_baseline_backtest.py) ──

EVENT_TICKERS = [
    "KXRT-BRI",
    "KXRT-HOP",
    "KXRT-REM",
]
MAX_TRADE_SIZE = 100
TRIGGER_MIN_SIZE = 1.0
TRIGGER_IMBALANCE_RATIO = 0.20
MIN_SECONDS_BETWEEN_TRIGGERS = 60.0
STARTING_BALANCE = 10_000


# ── Strategy ──

class OrderBookImbalance(Strategy):
    """
    Order book imbalance strategy (aggressive taker).

    When bid size significantly exceeds ask size (ratio below threshold),
    buys at the ask. When ask size exceeds bid size, sells at the bid.
    """

    def __init__(self, instrument_id: str):
        super().__init__(instrument_id)
        self._last_trigger_ns: int | None = None

    def on_data(self, data: FairValueData) -> None:
        pass  # This strategy doesn't use FV data

    def on_book_update(self, instrument_id: str, timestamp_ns: int = 0) -> None:
        if instrument_id != self.instrument_id:
            return
        self._check_trigger(timestamp_ns)

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_stop(self) -> None:
        pass

    def _check_trigger(self, timestamp_ns: int) -> None:
        bid = self.best_bid()
        ask = self.best_ask()

        if bid is None or ask is None:
            return

        bid_price, bid_size = bid
        ask_price, ask_size = ask

        # Nautilus: `if not spread:` skips None and 0.0 (locked), but
        # allows negative spread (crossed book) through.
        if ask_price == bid_price:
            return

        if bid_size <= 0 or ask_size <= 0:
            return

        smaller = min(bid_size, ask_size)
        larger = max(bid_size, ask_size)
        ratio = smaller / larger

        if larger <= TRIGGER_MIN_SIZE:
            return
        if ratio > TRIGGER_IMBALANCE_RATIO:
            return

        # Cooldown check (60 seconds = 60e9 nanoseconds)
        cooldown_ns = int(MIN_SECONDS_BETWEEN_TRIGGERS * 1_000_000_000)
        if self._last_trigger_ns is not None:
            if timestamp_ns - self._last_trigger_ns < cooldown_ns:
                return

        if bid_size > ask_size:
            side = OrderSide.Buy
            price = ask_price
            level_size = ask_size
        else:
            side = OrderSide.Sell
            price = bid_price
            level_size = bid_size

        trade_qty = min(level_size, MAX_TRADE_SIZE)
        if trade_qty <= 0:
            return

        self._last_trigger_ns = timestamp_ns

        # Submit aggressive order (FOK)
        order, fill = self.submit_order(
            side, price, trade_qty, time_in_force=TimeInForce.FOK, timestamp_ns=timestamp_ns,
        )
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
        engine.add_strategy(OrderBookImbalance(inst.id))

    print(f"Running backtest with {len(data.instruments)} instruments...")
    t0 = time.time()
    engine.run(data.fair_values, data.orderbook_deltas, data.trades)
    run_time = time.time() - t0
    print(f"Backtest completed in {run_time:.1f}s")

    engine.print_results()


if __name__ == "__main__":
    main()
