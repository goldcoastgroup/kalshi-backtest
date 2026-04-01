# Kalshi Backtest Engine — Design Spec

## Overview

A standalone backtesting engine for Kalshi binary option markets, decoupled from NautilusTrader. The project enables iterative strategy optimization via an autonomous training agent (like xgb-auto) but specialized for modifying and optimizing Kalshi backtest strategies.

The engine replays historical orderbook data and model-derived fair values chronologically, matching orders against a simulated L2 orderbook with Kalshi-specific fee and fill semantics.

## Motivation

NautilusTrader is a powerful but complex framework with many abstractions not needed for Kalshi-only backtesting. A stripped-down engine that the user can fully understand and modify makes it easier to:
- Debug strategy behavior
- Add Kalshi-specific features
- Let an autonomous agent optimize strategy parameters and logic

## Project Structure

```
kalshi-backtest/
├── engine/
│   ├── __init__.py        # Public API exports + BacktestEngine orchestrator
│   ├── types.py           # Data classes: Order, Fill, Position, FairValueData, etc.
│   ├── orderbook.py       # L2 OrderBook
│   ├── exchange.py        # SimulatedExchange — order matching, fill logic
│   ├── account.py         # CashAccount — balance, positions, PnL
│   └── strategy.py        # Strategy base class
├── prepare.py             # Data loading + parquet caching (FROZEN)
├── train.py               # Strategy definition + config + execution (AGENT MODIFIES)
├── program.md             # Training agent instructions
├── pyproject.toml
└── .env                   # MONGODB_URI
```

## Engine Architecture

### Event Replay Loop

All data (FairValueData + OrderBookDelta) is merged and sorted by timestamp. The engine processes events chronologically:

```
for event in sorted(all_events, key=timestamp):
    if OrderBookDelta:
        orderbook.apply(delta)
        exchange.check_resting_orders()
        for strategy in strategies_for(instrument):
            strategy.on_book_update(instrument_id)
    elif FairValueData:
        for strategy in strategies_for(instrument):
            strategy.on_data(data)
```

### types.py — Data Classes

**Instrument**
- `id: str` — e.g. "KXRT-BRI-50"
- `event_ticker: str` — e.g. "KXRT-BRI"
- `price_precision: int` — decimal places for prices (4 for Kalshi)
- `size_precision: int` — decimal places for quantities (2 for Kalshi)
- `expiration_ns: int` — Unix nanosecond expiration timestamp

**FairValueData**
- `timestamp_ns: int` — Unix nanoseconds
- `instrument_id: str`
- `fv: float` — model fair value [0, 1]
- `theta: float` — hourly time decay
- `gamma_pos: float` — sensitivity to positive review
- `gamma_neg: float` — sensitivity to negative review
- `new_review: bool`
- `hours_left: float`
- `cur_score: float`
- `total_reviews: int`

**OrderBookDelta**
- `instrument_id: str`
- `timestamp_ns: int`
- `action: str` — "CLEAR", "ADD", "UPDATE", "DELETE"
- `side: str` — "BUY" or "SELL"
- `price: float`
- `size: float`
- `flags: int` — bitmask: F_SNAPSHOT=1, F_LAST=2

**Order**
- `id: str` — unique order ID
- `instrument_id: str`
- `side: str` — "BUY" or "SELL"
- `price: Decimal`
- `quantity: Decimal`
- `filled_qty: Decimal`
- `post_only: bool`
- `status: str` — "SUBMITTED", "RESTING", "FILLED", "CANCELED", "REJECTED"
- `is_maker: bool | None` — set on fill
- `avg_fill_price: Decimal | None`
- `submit_timestamp_ns: int`
- `fill_timestamp_ns: int | None`

**Fill**
- `order_id: str`
- `instrument_id: str`
- `side: str`
- `price: Decimal`
- `quantity: Decimal`
- `fee: Decimal`
- `is_maker: bool`
- `timestamp_ns: int`

**Position**
- `instrument_id: str`
- `signed_qty: Decimal` — positive = long, negative = short
- `realized_pnl: Decimal`
- `cost_basis: Decimal` — total cost of current position
- `entry_count: int` — number of position entries (for stats)

### orderbook.py — L2 OrderBook

Maintains a per-instrument orderbook with bid and ask sides.

Internal state: `bids: SortedDict[float, float]` (price → size, descending), `asks: SortedDict[float, float]` (price → size, ascending).

Methods:
- `apply(delta: OrderBookDelta)` — process CLEAR/ADD/UPDATE/DELETE
  - CLEAR: wipe both sides
  - ADD: set price level to given size
  - UPDATE: update price level size
  - DELETE: remove price level
- `best_bid() -> tuple[float, float] | None` — (price, size)
- `best_ask() -> tuple[float, float] | None` — (price, size)
- `consume_ask(qty: float) -> float` — remove up to qty from best ask, return actual consumed
- `consume_bid(qty: float) -> float` — remove up to qty from best bid, return actual consumed

### exchange.py — SimulatedExchange

Manages resting orders and matching logic for all instruments.

**Order submission:**
1. Validate: price in [0.01, 0.99], quantity > 0, sufficient balance
2. If BUY and price >= best_ask:
   - If `post_only`: reject order
   - Else: fill at best_ask price, qty = min(order_qty, ask_depth). Taker fill.
3. If SELL and price <= best_bid:
   - If `post_only`: reject order
   - Else: fill at best_bid price, qty = min(order_qty, bid_depth). Taker fill.
4. If no cross: add to resting orders

**On book update (check_resting_orders):**
- For each resting BUY where price >= best_ask: fill at order's price (maker fill), qty = min(order_qty, ask_depth)
- For each resting SELL where price <= best_bid: fill at order's price (maker fill), qty = min(order_qty, bid_depth)

**Modify order:** change quantity of a resting order (price cannot change — cancel and re-submit for price change).

**Cancel order:** remove from resting orders, set status to CANCELED.

**Fee model (Kalshi proportional):**
- Maker fills: $0 fee
- Taker fills: `ceil(0.07 * qty * price * (1 - price) * 100) / 100`

### account.py — CashAccount

- `starting_balance: Decimal`
- `balance: Decimal` — current available cash
- `positions: dict[str, Position]` — per-instrument net positions
- On BUY fill: `balance -= fill_price * fill_qty + fee`
- On SELL fill: `balance += fill_price * fill_qty - fee`
- Position updated: signed_qty adjusted, realized_pnl computed when position reduces/closes
- `balance_free() -> Decimal` — current balance

Realized PnL calculation:
- When a fill reduces position size (e.g., selling when long), compute PnL as:
  `pnl = (fill_price - avg_entry_price) * fill_qty` for closing a long
  `pnl = (avg_entry_price - fill_price) * fill_qty` for closing a short

### strategy.py — Strategy Base Class

```python
class Strategy:
    """Base class for backtest strategies."""

    def __init__(self, instrument_id: str):
        self.instrument_id = instrument_id
        self._engine = None  # set by BacktestEngine

    # ── Callbacks (override in subclass) ──
    def on_start(self): ...
    def on_data(self, data: FairValueData): ...
    def on_book_update(self, instrument_id: str): ...
    def on_fill(self, fill: Fill): ...
    def on_stop(self): ...

    # ── Engine interaction (provided by base class) ──
    def submit_order(self, side, price, quantity, post_only=True) -> Order: ...
    def modify_order(self, order_id, new_quantity): ...
    def cancel_order(self, order_id): ...
    def get_book(self, instrument_id=None) -> OrderBook: ...
    def get_position(self, instrument_id=None) -> Position: ...
    def get_balance() -> Decimal: ...
```

### BacktestEngine (engine/__init__.py)

```python
class BacktestEngine:
    def __init__(self, instruments, starting_balance):
        self.account = CashAccount(starting_balance)
        self.exchange = SimulatedExchange(account, orderbooks)
        self.strategies = {}  # instrument_id -> Strategy

    def add_strategy(self, strategy: Strategy): ...

    def run(self, fair_values, orderbook_deltas):
        # Merge and sort all events by timestamp
        # Call on_start() for all strategies
        # Process events chronologically
        # Call on_stop() for all strategies

    def print_results(): ...
```

## prepare.py

Loads all data needed for backtesting with parquet caching.

**Cache location:** `~/.cache/kalshi-backtest/`

```
~/.cache/kalshi-backtest/
├── fair_values/
│   ├── KXRT-BRI.parquet     # FV data per event
│   └── KXRT-HOP.parquet
├── orderbooks/
│   ├── KXRT-BRI.parquet     # OB deltas per event
│   └── KXRT-HOP.parquet
├── instruments.parquet       # All instrument metadata
└── outcomes.json             # Settlement outcomes {instrument_id: 1.0|0.0}
```

**Public API:**

```python
@dataclass
class BacktestData:
    instruments: list[Instrument]
    fair_values: list[FairValueData]
    orderbook_deltas: list[OrderBookDelta]

def load(event_tickers: list[str], refresh=False) -> BacktestData:
    """Load all backtest data, using parquet cache when available."""
```

**Data loading pipeline:**
1. Query MongoDB `kxrt.events` for market tickers per event
2. Generate fair values via `kxrt_fv.backtest()` (cache per event)
3. Fetch instruments from Kalshi REST API `GET /trade-api/v2/markets/{ticker}` (cache all)
4. Load orderbook deltas from MongoDB `kxrt-training.orderbook-active` (cache per event)
5. Load settlement outcomes from MongoDB `kxrt.markets` (cache as JSON)
6. Build synthetic settlement deltas (book snap to 0.99/0.01)
7. Return `BacktestData` with all data merged

## train.py

The starter strategy is a direct port of nautilus_gct's KxrtBaseline:

**Config constants:**
```python
EVENT_TICKERS = ["KXRT-BRI", "KXRT-HOP", "KXRT-REM"]
MAX_SIZE = 500
SPREAD = 0.05
KELLY_FRACTION = 0.5
BANKROLL_CAP = 1000
STARTING_BALANCE = 10_000
```

**Strategy: KxrtBaseline**
- FV-centered market maker with fixed spread
- Half-Kelly order sizing
- Post-only limit orders
- Modify-in-place when only quantity changes (preserve queue priority)
- Cancel-and-replace when price changes
- Settlement detection via book snap to 0.99/0.01
- Position close at settlement

**Execution:**
```python
if __name__ == "__main__":
    data = prepare.load(EVENT_TICKERS)
    engine = BacktestEngine(data.instruments, STARTING_BALANCE)
    for inst in data.instruments:
        engine.add_strategy(KxrtBaseline(inst.id, ...))
    engine.run(data.fair_values, data.orderbook_deltas)
    engine.print_results()
```

**Output block** matches nautilus_gct's print_results():
- Order statistics (total, filled, canceled, rejected, maker/taker split)
- Account & PnL (starting/final balance, return %)
- Position statistics (wins, losses, win rate, profit factor)
- Per-instrument breakdown
- Per-event breakdown
- Capital efficiency (turnover, PnL/turnover)

## Dependencies

```toml
[project]
name = "kalshi-backtest"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas",
    "numpy",
    "pyarrow",
    "pymongo",
    "python-dotenv",
    "httpx",
    "sortedcontainers",
]
```

Plus `kxrt_fv` from the sandbox repo (via PYTHONPATH or editable install).

## Verification Plan

Run the baseline backtest in both nautilus_gct and kalshi-backtest with the same 3 events and compare:
- Final balance and total PnL
- Number of fills, maker/taker split
- Per-instrument PnL breakdown
- Position count and win rate

Results should match closely. Minor differences are acceptable if caused by fill-timing edge cases, but overall PnL direction and magnitude must align.
