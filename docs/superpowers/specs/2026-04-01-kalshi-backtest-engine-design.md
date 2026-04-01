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
├── crates/
│   └── engine/                # Rust crate — compiled to Python module via PyO3
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs         # PyO3 module entry point
│           ├── types.rs       # Data structs: Order, Fill, Position, FairValueData, etc.
│           ├── orderbook.rs   # L2 OrderBook
│           ├── exchange.rs    # SimulatedExchange — order matching, fill logic
│           └── account.rs     # CashAccount — balance, positions, PnL
├── engine/
│   ├── __init__.py            # Re-exports from Rust module + BacktestEngine orchestrator
│   └── strategy.py            # Strategy base class (pure Python — strategies subclass this)
├── prepare.py                 # Data loading + parquet caching (FROZEN)
├── train.py                   # Strategy definition + config + execution (AGENT MODIFIES)
├── program.md                 # Training agent instructions
├── pyproject.toml             # maturin build config
├── Cargo.toml                 # Workspace root
└── .env                       # MONGODB_URI
```

### Language Boundary

The performance-critical engine core (orderbook, matching, account) is written in Rust and compiled to a native Python extension via PyO3 + maturin. Strategies remain in pure Python.

| Layer | Language | Who modifies |
|-------|----------|--------------|
| `crates/engine/` (orderbook, matching, account) | Rust | User (rare) |
| `engine/strategy.py` (strategy base class) | Python | User (rare) |
| `train.py` (strategy impl, config, execution) | Python | Training agent |
| `prepare.py` (data loading, caching) | Python | Frozen |

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

### types.rs — Data Structs (exposed to Python via PyO3)

All Rust structs derive `#[pyclass]` for direct Python access.

**Instrument**
- `id: String` — e.g. "KXRT-BRI-50"
- `event_ticker: String` — e.g. "KXRT-BRI"
- `price_precision: u8` — decimal places for prices (4 for Kalshi)
- `size_precision: u8` — decimal places for quantities (2 for Kalshi)
- `expiration_ns: i64` — Unix nanosecond expiration timestamp

**FairValueData**
- `timestamp_ns: i64` — Unix nanoseconds
- `instrument_id: String`
- `fv: f64` — model fair value [0, 1]
- `theta: f64` — hourly time decay
- `gamma_pos: f64` — sensitivity to positive review
- `gamma_neg: f64` — sensitivity to negative review
- `new_review: bool`
- `hours_left: f64`
- `cur_score: f64`
- `total_reviews: i32`

**OrderBookDelta**
- `instrument_id: String`
- `timestamp_ns: i64`
- `action: BookAction` — enum: Clear, Add, Update, Delete
- `side: OrderSide` — enum: Buy, Sell
- `price: f64`
- `size: f64`
- `flags: u8` — bitmask: F_SNAPSHOT=1, F_LAST=2

**Order**
- `id: String` — unique order ID
- `instrument_id: String`
- `side: OrderSide`
- `price: f64` — stored as cents internally (i64) for exact arithmetic
- `quantity: f64`
- `filled_qty: f64`
- `post_only: bool`
- `status: OrderStatus` — enum: Submitted, Resting, Filled, Canceled, Rejected
- `is_maker: Option<bool>` — set on fill
- `avg_fill_price: Option<f64>`
- `submit_timestamp_ns: i64`
- `fill_timestamp_ns: Option<i64>`

**Fill**
- `order_id: String`
- `instrument_id: String`
- `side: OrderSide`
- `price: f64`
- `quantity: f64`
- `fee: f64`
- `is_maker: bool`
- `timestamp_ns: i64`

**Position**
- `instrument_id: String`
- `signed_qty: f64` — positive = long, negative = short
- `realized_pnl: f64`
- `avg_entry_price: f64`
- `entry_count: u32` — number of position entries (for stats)

### orderbook.rs — L2 OrderBook

Maintains a per-instrument orderbook with bid and ask sides.

Internal state: `bids: BTreeMap<i64, f64>` (price-in-cents → size, descending), `asks: BTreeMap<i64, f64>` (price-in-cents → size, ascending). Prices stored as integer cents for exact comparison.

Methods (exposed via `#[pymethods]`):
- `apply(delta: &OrderBookDelta)` — process Clear/Add/Update/Delete
  - Clear: wipe both sides
  - Add: set price level to given size
  - Update: update price level size
  - Delete: remove price level
- `best_bid() -> Option<(f64, f64)>` — (price, size)
- `best_ask() -> Option<(f64, f64)>` — (price, size)
- `consume_ask(qty: f64) -> f64` — remove up to qty from best ask, return actual consumed
- `consume_bid(qty: f64) -> f64` — remove up to qty from best bid, return actual consumed

### exchange.rs — SimulatedExchange

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

**Cancel order:** remove from resting orders, set status to Canceled.

**Fee model (Kalshi proportional, built into exchange):**
- Maker fills: $0 fee
- Taker fills: `ceil(0.07 * qty * price * (1 - price) * 100) / 100`

### account.rs — CashAccount

- `starting_balance: f64`
- `balance: f64` — current available cash
- `positions: HashMap<String, Position>` — per-instrument net positions
- On BUY fill: `balance -= fill_price * fill_qty + fee`
- On SELL fill: `balance += fill_price * fill_qty - fee`
- Position updated: signed_qty adjusted, realized_pnl computed when position reduces/closes
- `balance_free() -> f64` — current balance

Realized PnL calculation:
- When a fill reduces position size (e.g., selling when long), compute PnL as:
  `pnl = (fill_price - avg_entry_price) * fill_qty` for closing a long
  `pnl = (avg_entry_price - fill_price) * fill_qty` for closing a short

### strategy.py — Strategy Base Class (pure Python)

```python
class Strategy:
    """Base class for backtest strategies. Subclassed in train.py."""

    def __init__(self, instrument_id: str):
        self.instrument_id = instrument_id
        self._engine = None  # set by BacktestEngine

    # ── Callbacks (override in subclass) ──
    def on_start(self): ...
    def on_data(self, data: FairValueData): ...
    def on_book_update(self, instrument_id: str): ...
    def on_fill(self, fill: Fill): ...
    def on_stop(self): ...

    # ── Engine interaction (provided by base class, delegates to Rust) ──
    def submit_order(self, side, price, quantity, post_only=True) -> Order: ...
    def modify_order(self, order_id, new_quantity): ...
    def cancel_order(self, order_id): ...
    def get_book(self, instrument_id=None) -> OrderBook: ...
    def get_position(self, instrument_id=None) -> Position: ...
    def get_balance() -> float: ...
```

### BacktestEngine (engine/__init__.py — Python orchestrator calling Rust)

The BacktestEngine lives in Python but delegates hot-path operations to the Rust exchange/account. The event loop itself runs in Rust for maximum throughput — Python strategies are called back via PyO3.

```python
class BacktestEngine:
    def __init__(self, instruments, starting_balance):
        # Creates Rust-side exchange + account
        self._core = _engine.EngineCore(instruments, starting_balance)
        self.strategies = {}  # instrument_id -> Strategy

    def add_strategy(self, strategy: Strategy): ...

    def run(self, fair_values, orderbook_deltas):
        # Pass all events to Rust core
        # Rust sorts and iterates, calling back into Python strategies
        # via on_data() / on_book_update() / on_fill()

    def print_results(): ...
```

The Rust `EngineCore` exposes:
- `run(events, strategy_callbacks)` — the main event loop
- `submit_order(instrument_id, side, price, qty, post_only)` — called by strategy
- `modify_order(order_id, new_qty)`
- `cancel_order(order_id)`
- Query methods: `best_bid`, `best_ask`, `get_position`, `get_balance`
- Results accessors: `orders()`, `fills()`, `positions()`, `final_balance()`

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

## Build System

Uses **maturin** to compile the Rust crate into a Python extension module. The project is a mixed Rust+Python package.

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

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
]

[tool.maturin]
features = ["pyo3/extension-module"]
module-name = "engine._engine"
```

```toml
# Cargo.toml (workspace root)
[workspace]
members = ["crates/engine"]

# crates/engine/Cargo.toml
[package]
name = "kalshi-backtest-engine"
version = "0.1.0"
edition = "2021"

[lib]
name = "_engine"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
```

**Development workflow:**
- `maturin develop` — compile Rust and install into current venv
- `uv run train.py` — run backtest (after maturin develop)
- Rust changes require `maturin develop` rebuild; Python changes take effect immediately

Plus `kxrt_fv` from the sandbox repo (via PYTHONPATH or editable install).

## Verification Plan

Run the baseline backtest in both nautilus_gct and kalshi-backtest with the same 3 events and compare:
- Final balance and total PnL
- Number of fills, maker/taker split
- Per-instrument PnL breakdown
- Position count and win rate

Results should match closely. Minor differences are acceptable if caused by fill-timing edge cases, but overall PnL direction and magnitude must align.
