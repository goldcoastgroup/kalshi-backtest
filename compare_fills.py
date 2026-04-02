"""Compare per-instrument fill counts between our engine and nautilus."""
from __future__ import annotations
import time
from collections import Counter

import prepare
from engine import BacktestEngine, OrderSide, OrderStatus, TimeInForce
from engine.strategy import Strategy

EVENT_TICKERS = ["KXRT-BRI", "KXRT-HOP", "KXRT-REM"]
MAX_TRADE_SIZE = 100
TRIGGER_MIN_SIZE = 1.0
TRIGGER_IMBALANCE_RATIO = 0.20
MIN_SECONDS_BETWEEN_TRIGGERS = 60.0
STARTING_BALANCE = 10_000


class OrderBookImbalance(Strategy):
    def __init__(self, instrument_id: str):
        super().__init__(instrument_id)
        self._last_trigger_ns: int | None = None

    def on_data(self, data) -> None:
        pass

    def on_book_update(self, instrument_id: str, timestamp_ns: int = 0) -> None:
        if instrument_id != self.instrument_id:
            return
        self._check_trigger(timestamp_ns)

    def on_fill(self, fill) -> None:
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
        order, fill = self.submit_order(side, price, trade_qty, time_in_force=TimeInForce.FOK, timestamp_ns=timestamp_ns)


data = prepare.load(EVENT_TICKERS)
engine = BacktestEngine(data.instruments, STARTING_BALANCE)
for inst in data.instruments:
    engine.add_strategy(OrderBookImbalance(inst.id))
engine.run(data.fair_values, data.orderbook_deltas, data.trades)

# Count fills per instrument
core = engine._core
all_fills = core.all_fills()
all_orders = core.all_orders()

fill_counts = Counter()
for f in all_fills:
    fill_counts[f.instrument_id] += 1

order_counts = Counter()
canceled_counts = Counter()
for o in all_orders:
    order_counts[o.instrument_id] += 1
    if o.status == OrderStatus.Canceled:
        canceled_counts[o.instrument_id] += 1

NAUTILUS_FILLS = {
    'KXRT-BRI-30': 3, 'KXRT-BRI-35': 2, 'KXRT-BRI-40': 2, 'KXRT-BRI-45': 713,
    'KXRT-BRI-50': 931, 'KXRT-BRI-52': 214, 'KXRT-BRI-55': 1398, 'KXRT-BRI-57': 925,
    'KXRT-BRI-60': 2116, 'KXRT-BRI-62': 796, 'KXRT-BRI-65': 1470, 'KXRT-BRI-70': 608,
    'KXRT-BRI-75': 725, 'KXRT-BRI-80': 338, 'KXRT-BRI-85': 209, 'KXRT-BRI-90': 35,
    'KXRT-HOP-75': 249, 'KXRT-HOP-85': 313, 'KXRT-HOP-90': 996,
    'KXRT-HOP-92': 1339, 'KXRT-HOP-93': 2036, 'KXRT-HOP-94': 1085,
    'KXRT-HOP-95': 1526, 'KXRT-HOP-96': 842, 'KXRT-HOP-97': 91,
    'KXRT-REM-15': 26, 'KXRT-REM-20': 98, 'KXRT-REM-25': 94,
    'KXRT-REM-30': 274, 'KXRT-REM-35': 359, 'KXRT-REM-40': 46,
    'KXRT-REM-45': 2351, 'KXRT-REM-50': 370, 'KXRT-REM-55': 1266,
    'KXRT-REM-60': 2640, 'KXRT-REM-65': 474, 'KXRT-REM-70': 91,
    'KXRT-REM-75': 334, 'KXRT-REM-90': 109,
}

print(f"{'Instrument':<18s}  {'Fills':>6s}  {'Naut':>6s}  {'Diff':>6s}  {'Orders':>7s}  {'Cancel':>6s}")
total_ours = 0
total_naut = 0
total_diff = 0
for iid in sorted(set(list(fill_counts.keys()) + list(NAUTILUS_FILLS.keys()))):
    ours = fill_counts.get(iid, 0)
    naut = NAUTILUS_FILLS.get(iid, 0)
    diff = ours - naut
    total_ours += ours
    total_naut += naut
    total_diff += diff
    flag = " <--" if diff != 0 else ""
    print(f"{iid:<18s}  {ours:6d}  {naut:6d}  {diff:+6d}  {order_counts.get(iid,0):7d}  {canceled_counts.get(iid,0):6d}{flag}")

print(f"\n{'TOTAL':<18s}  {total_ours:6d}  {total_naut:6d}  {total_diff:+6d}")
print(f"PnL: ${core.balance() - core.starting_balance():+,.2f}")
print(f"Target PnL: $-1,612.78")
