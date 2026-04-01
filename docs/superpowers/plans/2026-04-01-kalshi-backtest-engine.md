# Kalshi Backtest Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Kalshi backtesting engine in Rust+Python that reproduces the results of nautilus_gct's KxrtBaseline backtest.

**Architecture:** Performance-critical engine core (orderbook, matching, account) in Rust compiled to a Python extension via PyO3 + maturin. Event loop and strategy callbacks in Python. Data loading in Python with parquet caching.

**Tech Stack:** Rust (PyO3 0.23, BTreeMap), Python 3.12 (maturin, pandas, pyarrow, pymongo, httpx), kxrt_fv from sandbox repo.

**Spec:** `docs/superpowers/specs/2026-04-01-kalshi-backtest-engine-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `Cargo.toml` | Create | Workspace root |
| `pyproject.toml` | Create | maturin build config + Python deps |
| `crates/engine/Cargo.toml` | Create | Rust crate config |
| `crates/engine/src/lib.rs` | Create | PyO3 module entry point |
| `crates/engine/src/types.rs` | Create | Enums + data structs |
| `crates/engine/src/orderbook.rs` | Create | L2 OrderBook |
| `crates/engine/src/account.rs` | Create | CashAccount + Position tracking |
| `crates/engine/src/exchange.rs` | Create | EngineCore: matching, fills, order mgmt |
| `engine/__init__.py` | Create | BacktestEngine Python orchestrator |
| `engine/strategy.py` | Create | Strategy base class |
| `prepare.py` | Create | Data loading + parquet caching |
| `train.py` | Create | KxrtBaseline strategy + execution |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `Cargo.toml`
- Create: `crates/engine/Cargo.toml`
- Create: `crates/engine/src/lib.rs`
- Create: `pyproject.toml`
- Create: `engine/__init__.py`

- [ ] **Step 1: Create workspace Cargo.toml**

```toml
# Cargo.toml
[workspace]
members = ["crates/engine"]
resolver = "2"
```

- [ ] **Step 2: Create crate Cargo.toml**

```toml
# crates/engine/Cargo.toml
[package]
name = "kalshi-backtest-engine"
version = "0.1.0"
edition = "2021"

[lib]
name = "_engine"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.23", features = ["extension-module"] }
```

- [ ] **Step 3: Create minimal lib.rs**

```rust
// crates/engine/src/lib.rs
use pyo3::prelude::*;

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    Ok(())
}
```

- [ ] **Step 4: Create pyproject.toml**

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "kalshi-backtest"
version = "0.1.0"
requires-python = ">=3.12"
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
python-source = "."
```

- [ ] **Step 5: Create engine/__init__.py placeholder**

```python
# engine/__init__.py
from engine._engine import *  # noqa: F401,F403
```

- [ ] **Step 6: Set up venv and install dependencies**

```bash
cd /home/bgram/dev/kalshi-backtest
pip install maturin
pip install -e /home/bgram/dev/sandbox
```

- [ ] **Step 7: Build and verify**

Run: `cd /home/bgram/dev/kalshi-backtest && maturin develop`

Expected: Compiles successfully, installs `engine._engine` module.

Run: `python -c "from engine._engine import __version__; print(__version__)"`

Expected: `0.1.0`

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml crates/ pyproject.toml engine/__init__.py
git commit -m "feat: project scaffolding with Rust+PyO3+maturin"
```

---

### Task 2: Rust Types

**Files:**
- Create: `crates/engine/src/types.rs`
- Modify: `crates/engine/src/lib.rs`

- [ ] **Step 1: Write types.rs with all enums and data structs**

```rust
// crates/engine/src/types.rs
use pyo3::prelude::*;

// ── Enums ──

#[pyclass(eq, eq_int)]
#[derive(Clone, Debug, PartialEq)]
pub enum BookAction {
    Clear = 0,
    Add = 1,
    Update = 2,
    Delete = 3,
}

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OrderSide {
    Buy = 0,
    Sell = 1,
}

#[pyclass(eq, eq_int)]
#[derive(Clone, Debug, PartialEq)]
pub enum OrderStatus {
    Submitted = 0,
    Resting = 1,
    Filled = 2,
    Canceled = 3,
    Rejected = 4,
}

// ── Flag constants ──

pub const F_SNAPSHOT: u8 = 1;
pub const F_LAST: u8 = 2;

// ── Data Structs ──

#[pyclass]
#[derive(Clone, Debug)]
pub struct Instrument {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub event_ticker: String,
    #[pyo3(get)]
    pub price_precision: u8,
    #[pyo3(get)]
    pub size_precision: u8,
    #[pyo3(get)]
    pub expiration_ns: i64,
}

#[pymethods]
impl Instrument {
    #[new]
    pub fn new(
        id: String,
        event_ticker: String,
        price_precision: u8,
        size_precision: u8,
        expiration_ns: i64,
    ) -> Self {
        Self { id, event_ticker, price_precision, size_precision, expiration_ns }
    }

    fn __repr__(&self) -> String {
        format!("Instrument(id='{}', event='{}')", self.id, self.event_ticker)
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct FairValueData {
    #[pyo3(get)]
    pub timestamp_ns: i64,
    #[pyo3(get)]
    pub instrument_id: String,
    #[pyo3(get)]
    pub fv: f64,
    #[pyo3(get)]
    pub theta: f64,
    #[pyo3(get)]
    pub gamma_pos: f64,
    #[pyo3(get)]
    pub gamma_neg: f64,
    #[pyo3(get)]
    pub new_review: bool,
    #[pyo3(get)]
    pub hours_left: f64,
    #[pyo3(get)]
    pub cur_score: f64,
    #[pyo3(get)]
    pub total_reviews: i32,
}

#[pymethods]
impl FairValueData {
    #[new]
    #[pyo3(signature = (timestamp_ns, instrument_id, fv, theta=0.0, gamma_pos=0.0, gamma_neg=0.0, new_review=false, hours_left=0.0, cur_score=0.0, total_reviews=0))]
    pub fn new(
        timestamp_ns: i64,
        instrument_id: String,
        fv: f64,
        theta: f64,
        gamma_pos: f64,
        gamma_neg: f64,
        new_review: bool,
        hours_left: f64,
        cur_score: f64,
        total_reviews: i32,
    ) -> Self {
        Self {
            timestamp_ns, instrument_id, fv, theta,
            gamma_pos, gamma_neg, new_review, hours_left,
            cur_score, total_reviews,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct OrderBookDelta {
    #[pyo3(get)]
    pub instrument_id: String,
    #[pyo3(get)]
    pub timestamp_ns: i64,
    #[pyo3(get)]
    pub action: BookAction,
    #[pyo3(get)]
    pub side: OrderSide,
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub size: f64,
    #[pyo3(get)]
    pub flags: u8,
}

#[pymethods]
impl OrderBookDelta {
    #[new]
    pub fn new(
        instrument_id: String,
        timestamp_ns: i64,
        action: BookAction,
        side: OrderSide,
        price: f64,
        size: f64,
        flags: u8,
    ) -> Self {
        Self { instrument_id, timestamp_ns, action, side, price, size, flags }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct Order {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub instrument_id: String,
    #[pyo3(get)]
    pub side: OrderSide,
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub quantity: f64,
    #[pyo3(get)]
    pub filled_qty: f64,
    #[pyo3(get)]
    pub post_only: bool,
    #[pyo3(get)]
    pub status: OrderStatus,
    #[pyo3(get)]
    pub is_maker: Option<bool>,
    #[pyo3(get)]
    pub avg_fill_price: Option<f64>,
    #[pyo3(get)]
    pub submit_timestamp_ns: i64,
    #[pyo3(get)]
    pub fill_timestamp_ns: Option<i64>,
}

#[pymethods]
impl Order {
    fn __repr__(&self) -> String {
        format!(
            "Order(id='{}', {:?} {} @ {:.4} qty={}, status={:?})",
            self.id, self.side, self.instrument_id, self.price, self.quantity, self.status,
        )
    }

    #[getter]
    fn is_closed(&self) -> bool {
        matches!(self.status, OrderStatus::Filled | OrderStatus::Canceled | OrderStatus::Rejected)
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct Fill {
    #[pyo3(get)]
    pub order_id: String,
    #[pyo3(get)]
    pub instrument_id: String,
    #[pyo3(get)]
    pub side: OrderSide,
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub quantity: f64,
    #[pyo3(get)]
    pub fee: f64,
    #[pyo3(get)]
    pub is_maker: bool,
    #[pyo3(get)]
    pub timestamp_ns: i64,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct Position {
    #[pyo3(get)]
    pub instrument_id: String,
    #[pyo3(get)]
    pub signed_qty: f64,
    #[pyo3(get)]
    pub realized_pnl: f64,
    #[pyo3(get)]
    pub avg_entry_price: f64,
    #[pyo3(get)]
    pub entry_count: u32,
}

impl Position {
    pub fn new(instrument_id: String) -> Self {
        Self {
            instrument_id,
            signed_qty: 0.0,
            realized_pnl: 0.0,
            avg_entry_price: 0.0,
            entry_count: 0,
        }
    }
}

#[pymethods]
impl Position {
    fn __repr__(&self) -> String {
        format!(
            "Position('{}', qty={}, pnl={:.2}, avg_entry={:.4})",
            self.instrument_id, self.signed_qty, self.realized_pnl, self.avg_entry_price,
        )
    }
}
```

- [ ] **Step 2: Register types in lib.rs**

```rust
// crates/engine/src/lib.rs
use pyo3::prelude::*;

mod types;

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;

    // Enums
    m.add_class::<types::BookAction>()?;
    m.add_class::<types::OrderSide>()?;
    m.add_class::<types::OrderStatus>()?;

    // Constants
    m.add("F_SNAPSHOT", types::F_SNAPSHOT)?;
    m.add("F_LAST", types::F_LAST)?;

    // Data structs
    m.add_class::<types::Instrument>()?;
    m.add_class::<types::FairValueData>()?;
    m.add_class::<types::OrderBookDelta>()?;
    m.add_class::<types::Order>()?;
    m.add_class::<types::Fill>()?;
    m.add_class::<types::Position>()?;

    Ok(())
}
```

- [ ] **Step 3: Build and verify types from Python**

Run: `maturin develop`

Then: `python -c "from engine._engine import Instrument, OrderSide, BookAction; i = Instrument('KXRT-BRI-50', 'KXRT-BRI', 4, 2, 0); print(i)"`

Expected: `Instrument(id='KXRT-BRI-50', event='KXRT-BRI')`

- [ ] **Step 4: Commit**

```bash
git add crates/engine/src/types.rs crates/engine/src/lib.rs
git commit -m "feat: Rust types with PyO3 bindings (enums, data structs)"
```

---

### Task 3: Rust OrderBook

**Files:**
- Create: `crates/engine/src/orderbook.rs`
- Modify: `crates/engine/src/lib.rs`

- [ ] **Step 1: Write orderbook tests**

Add to bottom of `crates/engine/src/orderbook.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    #[test]
    fn test_empty_book() {
        let book = OrderBook::new();
        assert!(book.best_bid().is_none());
        assert!(book.best_ask().is_none());
    }

    #[test]
    fn test_add_levels() {
        let mut book = OrderBook::new();
        book.apply_raw(&BookAction::Add, &OrderSide::Buy, 0.50, 10.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Buy, 0.49, 5.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.55, 8.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.60, 3.0);

        let (bp, bs) = book.best_bid().unwrap();
        assert!((bp - 0.50).abs() < 1e-9);
        assert!((bs - 10.0).abs() < 1e-9);

        let (ap, as_) = book.best_ask().unwrap();
        assert!((ap - 0.55).abs() < 1e-9);
        assert!((as_ - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_clear() {
        let mut book = OrderBook::new();
        book.apply_raw(&BookAction::Add, &OrderSide::Buy, 0.50, 10.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.55, 8.0);
        book.apply_raw(&BookAction::Clear, &OrderSide::Buy, 0.0, 0.0);
        assert!(book.best_bid().is_none());
        assert!(book.best_ask().is_none());
    }

    #[test]
    fn test_update_and_delete() {
        let mut book = OrderBook::new();
        book.apply_raw(&BookAction::Add, &OrderSide::Buy, 0.50, 10.0);
        book.apply_raw(&BookAction::Update, &OrderSide::Buy, 0.50, 20.0);
        let (_, size) = book.best_bid().unwrap();
        assert!((size - 20.0).abs() < 1e-9);

        book.apply_raw(&BookAction::Delete, &OrderSide::Buy, 0.50, 0.0);
        assert!(book.best_bid().is_none());
    }

    #[test]
    fn test_consume_ask() {
        let mut book = OrderBook::new();
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.55, 10.0);
        let consumed = book.consume_ask(7.0);
        assert!((consumed - 7.0).abs() < 1e-9);
        let (_, remaining) = book.best_ask().unwrap();
        assert!((remaining - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_consume_bid() {
        let mut book = OrderBook::new();
        book.apply_raw(&BookAction::Add, &OrderSide::Buy, 0.50, 10.0);
        let consumed = book.consume_bid(15.0); // more than available
        assert!((consumed - 10.0).abs() < 1e-9);
        assert!(book.best_bid().is_none());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p kalshi-backtest-engine`

Expected: compilation error — `OrderBook` not defined.

- [ ] **Step 3: Write OrderBook implementation**

```rust
// crates/engine/src/orderbook.rs
use std::collections::BTreeMap;
use crate::types::*;

/// L2 orderbook for a single instrument.
///
/// Prices stored internally as integer ticks (price * 10_000) for exact
/// comparison. BTreeMap gives O(log n) insert/remove and O(1) best price.
pub struct OrderBook {
    /// price_ticks → size, max first (descending iteration via .iter().next_back())
    bids: BTreeMap<i64, f64>,
    /// price_ticks → size, min first (ascending iteration via .iter().next())
    asks: BTreeMap<i64, f64>,
}

const TICK_SCALE: f64 = 10_000.0;

#[inline]
fn to_ticks(price: f64) -> i64 {
    (price * TICK_SCALE).round() as i64
}

#[inline]
fn from_ticks(ticks: i64) -> f64 {
    ticks as f64 / TICK_SCALE
}

impl OrderBook {
    pub fn new() -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
        }
    }

    /// Apply a raw book action (used internally and in tests).
    pub fn apply_raw(&mut self, action: &BookAction, side: &OrderSide, price: f64, size: f64) {
        match action {
            BookAction::Clear => {
                self.bids.clear();
                self.asks.clear();
            }
            BookAction::Add | BookAction::Update => {
                let ticks = to_ticks(price);
                let book = match side {
                    OrderSide::Buy => &mut self.bids,
                    OrderSide::Sell => &mut self.asks,
                };
                if size > 0.0 {
                    book.insert(ticks, size);
                } else {
                    book.remove(&ticks);
                }
            }
            BookAction::Delete => {
                let ticks = to_ticks(price);
                let book = match side {
                    OrderSide::Buy => &mut self.bids,
                    OrderSide::Sell => &mut self.asks,
                };
                book.remove(&ticks);
            }
        }
    }

    /// Apply an OrderBookDelta.
    pub fn apply(&mut self, delta: &OrderBookDelta) {
        self.apply_raw(&delta.action, &delta.side, delta.price, delta.size);
    }

    /// Best bid: highest buy price. Returns (price, size).
    pub fn best_bid(&self) -> Option<(f64, f64)> {
        self.bids.iter().next_back().map(|(&k, &v)| (from_ticks(k), v))
    }

    /// Best ask: lowest sell price. Returns (price, size).
    pub fn best_ask(&self) -> Option<(f64, f64)> {
        self.asks.iter().next().map(|(&k, &v)| (from_ticks(k), v))
    }

    /// Remove up to `qty` contracts from the best ask level.
    /// Returns the quantity actually consumed.
    pub fn consume_ask(&mut self, qty: f64) -> f64 {
        if let Some((&ticks, &size)) = self.asks.iter().next() {
            let consumed = qty.min(size);
            let remaining = size - consumed;
            if remaining <= 0.0 {
                self.asks.remove(&ticks);
            } else {
                self.asks.insert(ticks, remaining);
            }
            consumed
        } else {
            0.0
        }
    }

    /// Remove up to `qty` contracts from the best bid level.
    /// Returns the quantity actually consumed.
    pub fn consume_bid(&mut self, qty: f64) -> f64 {
        if let Some((&ticks, &size)) = self.bids.iter().next_back() {
            let consumed = qty.min(size);
            let remaining = size - consumed;
            if remaining <= 0.0 {
                self.bids.remove(&ticks);
            } else {
                self.bids.insert(ticks, remaining);
            }
            consumed
        } else {
            0.0
        }
    }
}
```

- [ ] **Step 4: Add `mod orderbook;` to lib.rs**

Add after `mod types;`:
```rust
mod orderbook;
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p kalshi-backtest-engine`

Expected: all 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/engine/src/orderbook.rs crates/engine/src/lib.rs
git commit -m "feat: L2 OrderBook with BTreeMap, full test coverage"
```

---

### Task 4: Rust CashAccount

**Files:**
- Create: `crates/engine/src/account.rs`
- Modify: `crates/engine/src/lib.rs`

- [ ] **Step 1: Write account tests**

Add to bottom of `crates/engine/src/account.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OrderSide;

    #[test]
    fn test_initial_balance() {
        let acc = CashAccount::new(10_000.0);
        assert!((acc.balance - 10_000.0).abs() < 1e-9);
        assert!((acc.starting_balance - 10_000.0).abs() < 1e-9);
    }

    #[test]
    fn test_buy_fill_deducts_balance() {
        let mut acc = CashAccount::new(10_000.0);
        acc.process_fill("KXRT-BRI-50", &OrderSide::Buy, 0.50, 10.0, 0.175);
        // cost = 0.50 * 10 + 0.175 = 5.175
        assert!((acc.balance - 9994.825).abs() < 1e-6);
    }

    #[test]
    fn test_sell_fill_adds_balance() {
        let mut acc = CashAccount::new(10_000.0);
        // First buy to establish position
        acc.process_fill("KXRT-BRI-50", &OrderSide::Buy, 0.50, 10.0, 0.0);
        // Then sell at higher price
        acc.process_fill("KXRT-BRI-50", &OrderSide::Sell, 0.60, 10.0, 0.0);
        // balance = 10000 - 5.0 + 6.0 = 10001.0
        assert!((acc.balance - 10001.0).abs() < 1e-6);
    }

    #[test]
    fn test_position_tracking_long() {
        let mut acc = CashAccount::new(10_000.0);
        acc.process_fill("KXRT-BRI-50", &OrderSide::Buy, 0.50, 10.0, 0.0);
        let pos = acc.get_position("KXRT-BRI-50");
        assert!((pos.signed_qty - 10.0).abs() < 1e-9);
        assert!((pos.avg_entry_price - 0.50).abs() < 1e-9);
    }

    #[test]
    fn test_realized_pnl_on_close() {
        let mut acc = CashAccount::new(10_000.0);
        acc.process_fill("KXRT-BRI-50", &OrderSide::Buy, 0.40, 10.0, 0.0);
        acc.process_fill("KXRT-BRI-50", &OrderSide::Sell, 0.60, 10.0, 0.0);
        let pos = acc.get_position("KXRT-BRI-50");
        // pnl = (0.60 - 0.40) * 10 = 2.0
        assert!((pos.realized_pnl - 2.0).abs() < 1e-6);
        assert!((pos.signed_qty).abs() < 1e-9); // flat
    }

    #[test]
    fn test_partial_close_pnl() {
        let mut acc = CashAccount::new(10_000.0);
        acc.process_fill("KXRT-BRI-50", &OrderSide::Buy, 0.40, 10.0, 0.0);
        acc.process_fill("KXRT-BRI-50", &OrderSide::Sell, 0.60, 5.0, 0.0);
        let pos = acc.get_position("KXRT-BRI-50");
        assert!((pos.signed_qty - 5.0).abs() < 1e-9);
        assert!((pos.realized_pnl - 1.0).abs() < 1e-6); // (0.60-0.40)*5
        assert!((pos.avg_entry_price - 0.40).abs() < 1e-9); // unchanged
    }

    #[test]
    fn test_short_position() {
        let mut acc = CashAccount::new(10_000.0);
        acc.process_fill("KXRT-BRI-50", &OrderSide::Sell, 0.60, 10.0, 0.0);
        let pos = acc.get_position("KXRT-BRI-50");
        assert!((pos.signed_qty - (-10.0)).abs() < 1e-9);
        assert!((pos.avg_entry_price - 0.60).abs() < 1e-9);
    }

    #[test]
    fn test_close_short_pnl() {
        let mut acc = CashAccount::new(10_000.0);
        acc.process_fill("KXRT-BRI-50", &OrderSide::Sell, 0.60, 10.0, 0.0);
        acc.process_fill("KXRT-BRI-50", &OrderSide::Buy, 0.40, 10.0, 0.0);
        let pos = acc.get_position("KXRT-BRI-50");
        // pnl = (0.60 - 0.40) * 10 = 2.0
        assert!((pos.realized_pnl - 2.0).abs() < 1e-6);
        assert!((pos.signed_qty).abs() < 1e-9);
    }

    #[test]
    fn test_fee_deducted_from_balance() {
        let mut acc = CashAccount::new(10_000.0);
        // Kalshi fee for buy 10 @ 0.50: ceil(0.07 * 10 * 0.50 * 0.50 * 100) / 100 = 0.18
        acc.process_fill("X", &OrderSide::Buy, 0.50, 10.0, 0.18);
        assert!((acc.balance - (10_000.0 - 5.0 - 0.18)).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p kalshi-backtest-engine`

Expected: compilation error — `CashAccount` not defined.

- [ ] **Step 3: Write CashAccount implementation**

```rust
// crates/engine/src/account.rs
use std::collections::HashMap;
use crate::types::*;

/// Cash account with netting positions for binary option backtesting.
pub struct CashAccount {
    pub starting_balance: f64,
    pub balance: f64,
    positions: HashMap<String, Position>,
}

impl CashAccount {
    pub fn new(starting_balance: f64) -> Self {
        Self {
            starting_balance,
            balance: starting_balance,
            positions: HashMap::new(),
        }
    }

    /// Process a fill: update balance and position.
    pub fn process_fill(
        &mut self,
        instrument_id: &str,
        side: &OrderSide,
        price: f64,
        qty: f64,
        fee: f64,
    ) {
        // Update balance
        match side {
            OrderSide::Buy => self.balance -= price * qty + fee,
            OrderSide::Sell => self.balance += price * qty - fee,
        }

        // Update position
        let pos = self.positions
            .entry(instrument_id.to_string())
            .or_insert_with(|| Position::new(instrument_id.to_string()));

        let fill_signed = match side {
            OrderSide::Buy => qty,
            OrderSide::Sell => -qty,
        };

        let old_qty = pos.signed_qty;
        let new_qty = old_qty + fill_signed;

        // Check if this fill reduces the position (realize PnL)
        let is_reducing = (old_qty > 0.0 && fill_signed < 0.0)
            || (old_qty < 0.0 && fill_signed > 0.0);

        if is_reducing && old_qty.abs() > 1e-12 {
            let close_qty = qty.min(old_qty.abs());
            let pnl = if old_qty > 0.0 {
                // Closing long: profit = (sell_price - avg_entry) * qty
                (price - pos.avg_entry_price) * close_qty
            } else {
                // Closing short: profit = (avg_entry - buy_price) * qty
                (pos.avg_entry_price - price) * close_qty
            };
            pos.realized_pnl += pnl;

            // If we flipped sides (e.g., long 10 → sell 15 = short 5),
            // the new side's avg entry is the fill price
            if (new_qty > 0.0 && old_qty < 0.0) || (new_qty < 0.0 && old_qty > 0.0) {
                pos.avg_entry_price = price;
                pos.entry_count += 1;
            }
            // If fully closed or only partially closed, avg_entry stays
        } else if !is_reducing && qty > 0.0 {
            // Increasing or opening: update weighted avg entry price
            if old_qty.abs() < 1e-12 {
                pos.avg_entry_price = price;
                pos.entry_count += 1;
            } else {
                let total_cost = pos.avg_entry_price * old_qty.abs() + price * qty;
                pos.avg_entry_price = total_cost / (old_qty.abs() + qty);
            }
        }

        pos.signed_qty = new_qty;
    }

    /// Get position for an instrument (returns flat position if none exists).
    pub fn get_position(&self, instrument_id: &str) -> Position {
        self.positions
            .get(instrument_id)
            .cloned()
            .unwrap_or_else(|| Position::new(instrument_id.to_string()))
    }

    /// Get all positions that have had activity.
    pub fn all_positions(&self) -> Vec<Position> {
        self.positions.values().cloned().collect()
    }
}
```

- [ ] **Step 4: Add `mod account;` to lib.rs**

Add after `mod orderbook;`:
```rust
mod account;
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p kalshi-backtest-engine`

Expected: all account tests + orderbook tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/engine/src/account.rs crates/engine/src/lib.rs
git commit -m "feat: CashAccount with position netting and realized PnL"
```

---

### Task 5: Rust EngineCore (Exchange + PyO3 Module)

**Files:**
- Create: `crates/engine/src/exchange.rs`
- Modify: `crates/engine/src/lib.rs`

This is the largest task. EngineCore wraps orderbooks, resting orders, account, and fee calculation, exposed to Python.

- [ ] **Step 1: Write exchange tests**

Add to bottom of `crates/engine/src/exchange.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn make_engine() -> EngineCore {
        let instruments = vec![
            Instrument::new("X".into(), "EV".into(), 4, 2, 0),
        ];
        EngineCore::new(instruments, 10_000.0, 0.07)
    }

    fn add_ask(engine: &mut EngineCore, price: f64, size: f64) {
        let delta = OrderBookDelta::new(
            "X".into(), 0, BookAction::Add, OrderSide::Sell, price, size, F_LAST,
        );
        engine.apply_delta(&delta);
    }

    fn add_bid(engine: &mut EngineCore, price: f64, size: f64) {
        let delta = OrderBookDelta::new(
            "X".into(), 0, BookAction::Add, OrderSide::Buy, price, size, F_LAST,
        );
        engine.apply_delta(&delta);
    }

    #[test]
    fn test_post_only_resting() {
        let mut engine = make_engine();
        add_ask(&mut engine, 0.60, 100.0);
        add_bid(&mut engine, 0.40, 100.0);

        let (order, fill) = engine.submit_order("X", OrderSide::Buy, 0.45, 10.0, true, 100);
        assert_eq!(order.status, OrderStatus::Resting);
        assert!(fill.is_none());
    }

    #[test]
    fn test_post_only_reject_on_cross() {
        let mut engine = make_engine();
        add_ask(&mut engine, 0.50, 100.0);

        let (order, fill) = engine.submit_order("X", OrderSide::Buy, 0.50, 10.0, true, 100);
        assert_eq!(order.status, OrderStatus::Rejected);
        assert!(fill.is_none());
    }

    #[test]
    fn test_aggressive_fill() {
        let mut engine = make_engine();
        add_ask(&mut engine, 0.50, 100.0);
        add_bid(&mut engine, 0.40, 100.0);

        let (order, fill) = engine.submit_order("X", OrderSide::Buy, 0.50, 10.0, false, 100);
        assert_eq!(order.status, OrderStatus::Filled);
        let f = fill.unwrap();
        assert!((f.price - 0.50).abs() < 1e-9);
        assert!((f.quantity - 10.0).abs() < 1e-9);
        assert!(!f.is_maker);
    }

    #[test]
    fn test_resting_order_fills_on_book_move() {
        let mut engine = make_engine();
        add_ask(&mut engine, 0.60, 100.0);
        add_bid(&mut engine, 0.40, 100.0);

        // Place resting buy at 0.45
        engine.submit_order("X", OrderSide::Buy, 0.45, 10.0, true, 100);

        // Move ask down to 0.45 — should trigger fill
        let delta = OrderBookDelta::new(
            "X".into(), 200, BookAction::Add, OrderSide::Sell, 0.45, 50.0, F_LAST,
        );
        engine.apply_delta(&delta);
        let fills = engine.check_resting_orders("X", 200);

        assert_eq!(fills.len(), 1);
        assert!((fills[0].price - 0.45).abs() < 1e-9);
        assert!(fills[0].is_maker);
        assert!((fills[0].fee).abs() < 1e-9); // maker = 0 fee
    }

    #[test]
    fn test_cancel_order() {
        let mut engine = make_engine();
        add_ask(&mut engine, 0.60, 100.0);

        let (order, _) = engine.submit_order("X", OrderSide::Buy, 0.45, 10.0, true, 100);
        assert!(engine.cancel_order(&order.id));
        // Second cancel returns false
        assert!(!engine.cancel_order(&order.id));
    }

    #[test]
    fn test_modify_order() {
        let mut engine = make_engine();
        add_ask(&mut engine, 0.60, 100.0);

        let (order, _) = engine.submit_order("X", OrderSide::Buy, 0.45, 10.0, true, 100);
        assert!(engine.modify_order(&order.id, 20.0));
    }

    #[test]
    fn test_taker_fee_calculation() {
        let mut engine = make_engine();
        add_ask(&mut engine, 0.50, 100.0);
        add_bid(&mut engine, 0.40, 100.0);

        let (_, fill) = engine.submit_order("X", OrderSide::Buy, 0.50, 10.0, false, 100);
        let f = fill.unwrap();
        // fee = ceil(0.07 * 10 * 0.50 * 0.50 * 100) / 100 = ceil(17.5)/100 = 0.18
        assert!((f.fee - 0.18).abs() < 1e-9);
    }

    #[test]
    fn test_fill_limited_by_depth() {
        let mut engine = make_engine();
        add_ask(&mut engine, 0.50, 5.0); // only 5 available

        let (order, fill) = engine.submit_order("X", OrderSide::Buy, 0.50, 10.0, false, 100);
        let f = fill.unwrap();
        assert!((f.quantity - 5.0).abs() < 1e-9);
        // Remaining 5 should rest
        assert_eq!(order.status, OrderStatus::Filled); // partial fill still marks filled for consumed portion
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p kalshi-backtest-engine`

Expected: compilation error — `EngineCore` not defined.

- [ ] **Step 3: Write EngineCore implementation**

```rust
// crates/engine/src/exchange.rs
use std::collections::HashMap;
use crate::account::CashAccount;
use crate::orderbook::OrderBook;
use crate::types::*;

/// Core backtest engine: orderbooks + order matching + account.
///
/// Exposed to Python via PyO3. The Python BacktestEngine calls these methods
/// while iterating over events.
#[cfg_attr(not(test), pyo3::pyclass)]
pub struct EngineCore {
    books: HashMap<String, OrderBook>,
    resting_orders: Vec<Order>,
    pub account: CashAccount,
    instruments: HashMap<String, Instrument>,
    next_order_id: u64,
    fee_rate: f64,
    all_orders: Vec<Order>,
    all_fills: Vec<Fill>,
}

impl EngineCore {
    pub fn new(instruments: Vec<Instrument>, starting_balance: f64, fee_rate: f64) -> Self {
        let mut books = HashMap::new();
        let mut inst_map = HashMap::new();
        for inst in &instruments {
            books.insert(inst.id.clone(), OrderBook::new());
            inst_map.insert(inst.id.clone(), inst.clone());
        }
        Self {
            books,
            resting_orders: Vec::new(),
            account: CashAccount::new(starting_balance),
            instruments: inst_map,
            next_order_id: 0,
            fee_rate,
            all_orders: Vec::new(),
            all_fills: Vec::new(),
        }
    }

    /// Apply an orderbook delta to the appropriate book.
    pub fn apply_delta(&mut self, delta: &OrderBookDelta) {
        if let Some(book) = self.books.get_mut(&delta.instrument_id) {
            book.apply(delta);
        }
    }

    /// Best bid for an instrument.
    pub fn best_bid(&self, instrument_id: &str) -> Option<(f64, f64)> {
        self.books.get(instrument_id).and_then(|b| b.best_bid())
    }

    /// Best ask for an instrument.
    pub fn best_ask(&self, instrument_id: &str) -> Option<(f64, f64)> {
        self.books.get(instrument_id).and_then(|b| b.best_ask())
    }

    /// Submit a new order. Returns (Order, Option<Fill>).
    pub fn submit_order(
        &mut self,
        instrument_id: &str,
        side: OrderSide,
        price: f64,
        quantity: f64,
        post_only: bool,
        timestamp_ns: i64,
    ) -> (Order, Option<Fill>) {
        let order_id = format!("O-{}", self.next_order_id);
        self.next_order_id += 1;

        let mut order = Order {
            id: order_id,
            instrument_id: instrument_id.to_string(),
            side: side.clone(),
            price,
            quantity,
            filled_qty: 0.0,
            post_only,
            status: OrderStatus::Submitted,
            is_maker: None,
            avg_fill_price: None,
            submit_timestamp_ns: timestamp_ns,
            fill_timestamp_ns: None,
        };

        let book = match self.books.get_mut(instrument_id) {
            Some(b) => b,
            None => {
                order.status = OrderStatus::Rejected;
                self.all_orders.push(order.clone());
                return (order, None);
            }
        };

        // Check for crossing
        let crosses = match side {
            OrderSide::Buy => book.best_ask().map_or(false, |(ap, _)| price >= ap),
            OrderSide::Sell => book.best_bid().map_or(false, |(bp, _)| price <= bp),
        };

        if crosses {
            if post_only {
                order.status = OrderStatus::Rejected;
                self.all_orders.push(order.clone());
                return (order, None);
            }

            // Aggressive fill at book price
            let (fill_price, fill_qty) = match side {
                OrderSide::Buy => {
                    let (ap, _) = book.best_ask().unwrap();
                    let consumed = book.consume_ask(quantity);
                    (ap, consumed)
                }
                OrderSide::Sell => {
                    let (bp, _) = book.best_bid().unwrap();
                    let consumed = book.consume_bid(quantity);
                    (bp, consumed)
                }
            };

            if fill_qty > 0.0 {
                let fee = self.compute_taker_fee(fill_price, fill_qty);
                self.account.process_fill(instrument_id, &side, fill_price, fill_qty, fee);

                order.filled_qty = fill_qty;
                order.status = OrderStatus::Filled;
                order.is_maker = Some(false);
                order.avg_fill_price = Some(fill_price);
                order.fill_timestamp_ns = Some(timestamp_ns);

                let fill = Fill {
                    order_id: order.id.clone(),
                    instrument_id: instrument_id.to_string(),
                    side: side.clone(),
                    price: fill_price,
                    quantity: fill_qty,
                    fee,
                    is_maker: false,
                    timestamp_ns,
                };
                self.all_fills.push(fill.clone());
                self.all_orders.push(order.clone());
                return (order, Some(fill));
            }
        }

        // No cross or empty book — rest the order
        order.status = OrderStatus::Resting;
        self.resting_orders.push(order.clone());
        self.all_orders.push(order.clone());
        (order, None)
    }

    /// Check if any resting orders for this instrument can now fill.
    /// Called after applying orderbook deltas (on F_LAST).
    pub fn check_resting_orders(&mut self, instrument_id: &str, timestamp_ns: i64) -> Vec<Fill> {
        let mut fills = Vec::new();
        let mut filled_indices = Vec::new();

        for (i, order) in self.resting_orders.iter().enumerate() {
            if order.instrument_id != instrument_id {
                continue;
            }

            let should_fill = match order.side {
                OrderSide::Buy => {
                    self.books.get(&order.instrument_id)
                        .and_then(|b| b.best_ask())
                        .map_or(false, |(ap, _)| order.price >= ap)
                }
                OrderSide::Sell => {
                    self.books.get(&order.instrument_id)
                        .and_then(|b| b.best_bid())
                        .map_or(false, |(bp, _)| order.price <= bp)
                }
            };

            if should_fill {
                filled_indices.push(i);
            }
        }

        // Process fills in reverse to preserve indices during removal
        for &i in filled_indices.iter().rev() {
            let order = self.resting_orders.remove(i);
            let book = self.books.get_mut(&order.instrument_id).unwrap();

            let fill_qty = match order.side {
                OrderSide::Buy => book.consume_ask(order.quantity),
                OrderSide::Sell => book.consume_bid(order.quantity),
            };

            if fill_qty > 0.0 {
                // Maker fill — at order's resting price, 0 fee
                self.account.process_fill(
                    &order.instrument_id, &order.side, order.price, fill_qty, 0.0,
                );

                let fill = Fill {
                    order_id: order.id.clone(),
                    instrument_id: order.instrument_id.clone(),
                    side: order.side.clone(),
                    price: order.price,
                    quantity: fill_qty,
                    fee: 0.0,
                    is_maker: true,
                    timestamp_ns,
                };
                fills.push(fill.clone());
                self.all_fills.push(fill);

                // Update the order record in all_orders
                if let Some(o) = self.all_orders.iter_mut().find(|o| o.id == order.id) {
                    o.filled_qty = fill_qty;
                    o.status = OrderStatus::Filled;
                    o.is_maker = Some(true);
                    o.avg_fill_price = Some(order.price);
                    o.fill_timestamp_ns = Some(timestamp_ns);
                }
            }
        }

        fills
    }

    /// Cancel a resting order. Returns true if found and cancelled.
    pub fn cancel_order(&mut self, order_id: &str) -> bool {
        if let Some(pos) = self.resting_orders.iter().position(|o| o.id == order_id) {
            self.resting_orders.remove(pos);
            if let Some(o) = self.all_orders.iter_mut().find(|o| o.id == order_id) {
                o.status = OrderStatus::Canceled;
            }
            true
        } else {
            false
        }
    }

    /// Modify quantity of a resting order. Returns true if found.
    pub fn modify_order(&mut self, order_id: &str, new_quantity: f64) -> bool {
        if let Some(order) = self.resting_orders.iter_mut().find(|o| o.id == order_id) {
            order.quantity = new_quantity;
            if let Some(o) = self.all_orders.iter_mut().find(|o| o.id == order_id) {
                o.quantity = new_quantity;
            }
            true
        } else {
            false
        }
    }

    /// Kalshi taker fee: ceil(fee_rate * qty * price * (1 - price) * 100) / 100
    fn compute_taker_fee(&self, price: f64, qty: f64) -> f64 {
        if self.fee_rate <= 0.0 || price <= 0.0 || price >= 1.0 {
            return 0.0;
        }
        let raw = self.fee_rate * qty * price * (1.0 - price);
        (raw * 100.0).ceil() / 100.0
    }

    // ── Result accessors ──

    pub fn balance(&self) -> f64 {
        self.account.balance
    }

    pub fn starting_balance(&self) -> f64 {
        self.account.starting_balance
    }

    pub fn get_position(&self, instrument_id: &str) -> Position {
        self.account.get_position(instrument_id)
    }

    pub fn all_orders(&self) -> Vec<Order> {
        self.all_orders.clone()
    }

    pub fn all_fills(&self) -> Vec<Fill> {
        self.all_fills.clone()
    }

    pub fn all_positions(&self) -> Vec<Position> {
        self.account.all_positions()
    }
}
```

- [ ] **Step 4: Add PyO3 methods to EngineCore**

Add at the bottom of `exchange.rs`, before the `#[cfg(test)]` block:

```rust
#[cfg(not(test))]
#[pyo3::pymethods]
impl EngineCore {
    #[new]
    fn py_new(instruments: Vec<Instrument>, starting_balance: f64, fee_rate: f64) -> Self {
        Self::new(instruments, starting_balance, fee_rate)
    }

    #[pyo3(name = "apply_delta")]
    fn py_apply_delta(&mut self, delta: &OrderBookDelta) {
        self.apply_delta(delta);
    }

    #[pyo3(name = "best_bid")]
    fn py_best_bid(&self, instrument_id: &str) -> Option<(f64, f64)> {
        self.best_bid(instrument_id)
    }

    #[pyo3(name = "best_ask")]
    fn py_best_ask(&self, instrument_id: &str) -> Option<(f64, f64)> {
        self.best_ask(instrument_id)
    }

    #[pyo3(name = "submit_order")]
    fn py_submit_order(
        &mut self,
        instrument_id: &str,
        side: OrderSide,
        price: f64,
        quantity: f64,
        post_only: bool,
        timestamp_ns: i64,
    ) -> (Order, Option<Fill>) {
        self.submit_order(instrument_id, side, price, quantity, post_only, timestamp_ns)
    }

    #[pyo3(name = "modify_order")]
    fn py_modify_order(&mut self, order_id: &str, new_quantity: f64) -> bool {
        self.modify_order(order_id, new_quantity)
    }

    #[pyo3(name = "cancel_order")]
    fn py_cancel_order(&mut self, order_id: &str) -> bool {
        self.cancel_order(order_id)
    }

    #[pyo3(name = "check_resting_orders")]
    fn py_check_resting_orders(&mut self, instrument_id: &str, timestamp_ns: i64) -> Vec<Fill> {
        self.check_resting_orders(instrument_id, timestamp_ns)
    }

    #[pyo3(name = "balance")]
    fn py_balance(&self) -> f64 {
        self.balance()
    }

    #[pyo3(name = "starting_balance")]
    fn py_starting_balance(&self) -> f64 {
        self.starting_balance()
    }

    #[pyo3(name = "get_position")]
    fn py_get_position(&self, instrument_id: &str) -> Position {
        self.get_position(instrument_id)
    }

    #[pyo3(name = "all_orders")]
    fn py_all_orders(&self) -> Vec<Order> {
        self.all_orders()
    }

    #[pyo3(name = "all_fills")]
    fn py_all_fills(&self) -> Vec<Fill> {
        self.all_fills()
    }

    #[pyo3(name = "all_positions")]
    fn py_all_positions(&self) -> Vec<Position> {
        self.all_positions()
    }
}
```

- [ ] **Step 5: Register EngineCore in lib.rs**

Update `crates/engine/src/lib.rs`:

```rust
use pyo3::prelude::*;

mod types;
mod orderbook;
mod account;
mod exchange;

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;

    // Enums
    m.add_class::<types::BookAction>()?;
    m.add_class::<types::OrderSide>()?;
    m.add_class::<types::OrderStatus>()?;

    // Constants
    m.add("F_SNAPSHOT", types::F_SNAPSHOT)?;
    m.add("F_LAST", types::F_LAST)?;

    // Data structs
    m.add_class::<types::Instrument>()?;
    m.add_class::<types::FairValueData>()?;
    m.add_class::<types::OrderBookDelta>()?;
    m.add_class::<types::Order>()?;
    m.add_class::<types::Fill>()?;
    m.add_class::<types::Position>()?;

    // Engine
    m.add_class::<exchange::EngineCore>()?;

    Ok(())
}
```

- [ ] **Step 6: Run Rust tests**

Run: `cargo test -p kalshi-backtest-engine`

Expected: all tests pass (orderbook + account + exchange).

- [ ] **Step 7: Build Python module and verify**

Run: `maturin develop`

Then:
```python
python -c "
from engine._engine import EngineCore, Instrument, OrderSide, OrderBookDelta, BookAction, F_LAST

inst = Instrument('X', 'EV', 4, 2, 0)
core = EngineCore([inst], 10000.0, 0.07)
delta = OrderBookDelta('X', 0, BookAction.Add, OrderSide.Sell, 0.55, 100.0, F_LAST)
core.apply_delta(delta)
print('best_ask:', core.best_ask('X'))
print('balance:', core.balance())
"
```

Expected:
```
best_ask: (0.55, 100.0)
balance: 10000.0
```

- [ ] **Step 8: Commit**

```bash
git add crates/engine/src/exchange.rs crates/engine/src/lib.rs
git commit -m "feat: EngineCore with order matching, fills, fees, PyO3 bindings"
```

---

### Task 6: Python Engine Package

**Files:**
- Create: `engine/strategy.py`
- Modify: `engine/__init__.py`

- [ ] **Step 1: Write engine/strategy.py**

```python
# engine/strategy.py
"""Strategy base class for backtesting."""

from __future__ import annotations

from engine._engine import (
    EngineCore,
    FairValueData,
    Fill,
    Order,
    OrderSide,
    Position,
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

    def on_book_update(self, instrument_id: str) -> None:
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
        post_only: bool = True,
        timestamp_ns: int = 0,
    ) -> tuple[Order, Fill | None]:
        """Submit a limit order. Returns (Order, optional Fill if immediate)."""
        return self._core.submit_order(
            self.instrument_id, side, price, quantity, post_only, timestamp_ns,
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
```

- [ ] **Step 2: Write engine/__init__.py with BacktestEngine**

```python
# engine/__init__.py
"""Kalshi backtest engine — Rust core with Python orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine._engine import (  # noqa: F401
    BookAction,
    EngineCore,
    F_LAST,
    F_SNAPSHOT,
    FairValueData,
    Fill,
    Instrument,
    Order,
    OrderBookDelta,
    OrderSide,
    OrderStatus,
    Position,
)
from engine.strategy import Strategy


class BacktestEngine:
    """Orchestrates event replay, calling into the Rust EngineCore."""

    def __init__(
        self,
        instruments: list[Instrument],
        starting_balance: float,
        fee_rate: float = 0.07,
    ):
        self._core = EngineCore(instruments, starting_balance, fee_rate)
        self._instruments = {inst.id: inst for inst in instruments}
        self._strategies: dict[str, Strategy] = {}

    def add_strategy(self, strategy: Strategy) -> None:
        strategy._bind(self._core)
        self._strategies[strategy.instrument_id] = strategy

    def run(
        self,
        fair_values: list[FairValueData],
        orderbook_deltas: list[OrderBookDelta],
    ) -> None:
        """Run the backtest: merge events, replay chronologically."""
        # Build sorted event list: (timestamp_ns, type_priority, index, event)
        # type_priority: 0=OB delta (process book first), 1=FV data
        events: list[tuple[int, int, object]] = []
        for delta in orderbook_deltas:
            events.append((delta.timestamp_ns, 0, delta))
        for fv in fair_values:
            events.append((fv.timestamp_ns, 1, fv))

        events.sort(key=lambda e: (e[0], e[1]))

        # Start strategies
        for strategy in self._strategies.values():
            strategy.on_start()

        # Event loop
        for _, type_prio, event in events:
            if type_prio == 0:
                delta = event
                self._core.apply_delta(delta)
                if delta.flags & F_LAST:
                    fills = self._core.check_resting_orders(
                        delta.instrument_id, delta.timestamp_ns,
                    )
                    for fill in fills:
                        strat = self._strategies.get(fill.instrument_id)
                        if strat:
                            strat.on_fill(fill)
                    strat = self._strategies.get(delta.instrument_id)
                    if strat:
                        strat.on_book_update(delta.instrument_id)
            else:
                fv = event
                strat = self._strategies.get(fv.instrument_id)
                if strat:
                    strat.on_data(fv)

        # Stop strategies
        for strategy in self._strategies.values():
            strategy.on_stop()

    def print_results(self) -> None:
        """Print comprehensive backtest statistics."""
        starting = self._core.starting_balance()
        final = self._core.balance()
        pnl = final - starting

        all_orders = self._core.all_orders()
        all_fills = self._core.all_fills()
        positions = self._core.all_positions()

        filled = [o for o in all_orders if o.status == OrderStatus.Filled]
        canceled = [o for o in all_orders if o.status == OrderStatus.Canceled]
        rejected = [o for o in all_orders if o.status == OrderStatus.Rejected]

        buys = [o for o in filled if o.side == OrderSide.Buy]
        sells = [o for o in filled if o.side == OrderSide.Sell]
        maker_fills = [o for o in filled if o.is_maker is True]
        taker_fills = [o for o in filled if o.is_maker is False]

        fill_qtys = [o.filled_qty for o in filled]
        fill_prices = [o.avg_fill_price for o in filled if o.avg_fill_price is not None]

        print(f"\n{'=' * 70}")
        print("ORDER STATISTICS")
        print(f"{'=' * 70}")
        print(f"Total orders:      {len(all_orders):,}")
        if all_orders:
            print(f"Filled:            {len(filled):,}  ({100*len(filled)/len(all_orders):.1f}%)")
        print(f"Canceled:          {len(canceled):,}")
        print(f"Rejected:          {len(rejected):,}")
        print()
        print(f"Buy fills:         {len(buys):,}  ({sum(o.filled_qty for o in buys):,.0f} contracts)")
        print(f"Sell fills:        {len(sells):,}  ({sum(o.filled_qty for o in sells):,.0f} contracts)")
        if filled:
            print(f"Maker fills:       {len(maker_fills):,}  ({100*len(maker_fills)/len(filled):.1f}%)")
            print(f"Taker fills:       {len(taker_fills):,}  ({100*len(taker_fills)/len(filled):.1f}%)")
        print()
        if fill_qtys:
            print(f"Total fill qty:    {sum(fill_qtys):,.0f} contracts")
            print(f"Avg fill qty:      {np.mean(fill_qtys):,.1f}")
            print(f"Median fill qty:   {np.median(fill_qtys):,.1f}")
            print(f"Max fill qty:      {max(fill_qtys):,.0f}")
        if fill_prices:
            print(f"Avg fill price:    {np.mean(fill_prices):.4f}")
            print(f"Median fill price: {np.median(fill_prices):.4f}")

        # Account
        print(f"\n{'=' * 70}")
        print("ACCOUNT & PNL")
        print(f"{'=' * 70}")
        print(f"Starting balance:  ${starting:,.2f}")
        print(f"Final balance:     ${final:,.2f}")
        print(f"Total PnL:         ${pnl:+,.2f}")
        print(f"Return:            {100*pnl/starting:+.2f}%")

        if not positions:
            print("\nNo positions generated.")
            return

        # Positions
        wins = [p for p in positions if p.realized_pnl > 0]
        losses = [p for p in positions if p.realized_pnl < 0]

        print(f"\n{'=' * 70}")
        print("POSITION STATISTICS")
        print(f"{'=' * 70}")
        print(f"Total positions:   {len(positions)}")
        print(f"Winning:           {len(wins)}  (PnL: ${sum(p.realized_pnl for p in wins):+,.2f})")
        print(f"Losing:            {len(losses)}  (PnL: ${sum(p.realized_pnl for p in losses):+,.2f})")
        if positions:
            print(f"Win rate:          {100*len(wins)/len(positions):.1f}%")
        print()
        if wins:
            print(f"Avg win:           ${np.mean([p.realized_pnl for p in wins]):+,.2f}")
            print(f"Largest win:       ${max(p.realized_pnl for p in wins):+,.2f}")
        if losses:
            print(f"Avg loss:          ${np.mean([p.realized_pnl for p in losses]):+,.2f}")
            print(f"Largest loss:      ${min(p.realized_pnl for p in losses):+,.2f}")
        if losses and sum(p.realized_pnl for p in losses) != 0:
            print(f"Profit factor:     {sum(p.realized_pnl for p in wins) / abs(sum(p.realized_pnl for p in losses)):.2f}")

        # Per-instrument
        inst_stats = []
        for p in positions:
            inst_stats.append({
                "instrument": p.instrument_id,
                "entries": p.entry_count,
                "pnl": p.realized_pnl,
                "final_qty": p.signed_qty,
            })
        df = pd.DataFrame(inst_stats).sort_values("pnl", ascending=False)
        print(f"\n{'=' * 70}")
        print("PER-INSTRUMENT BREAKDOWN")
        print(f"{'=' * 70}")
        with pd.option_context("display.max_rows", 50, "display.width", 120, "display.float_format", "{:.2f}".format):
            print(df.to_string(index=False))

        # Per-event
        print(f"\n{'=' * 70}")
        print("PER-EVENT BREAKDOWN")
        print(f"{'=' * 70}")
        event_pnl: dict[str, list[float]] = {}
        for p in positions:
            event = "-".join(p.instrument_id.split("-")[:2])
            event_pnl.setdefault(event, []).append(p.realized_pnl)
        for event, pnls in sorted(event_pnl.items()):
            print(f"  {event:12s}  {len(pnls):2d} instruments  PnL: ${sum(pnls):+,.2f}")

        # Capital efficiency
        total_bought = sum(o.avg_fill_price * o.filled_qty for o in buys if o.avg_fill_price)
        total_sold = sum(o.avg_fill_price * o.filled_qty for o in sells if o.avg_fill_price)
        turnover = total_bought + total_sold

        print(f"\n{'=' * 70}")
        print("CAPITAL EFFICIENCY")
        print(f"{'=' * 70}")
        print(f"Total bought:      ${total_bought:,.2f} notional")
        print(f"Total sold:        ${total_sold:,.2f} notional")
        print(f"Turnover:          ${turnover:,.2f}")
        if turnover > 0:
            print(f"PnL / Turnover:    {100*pnl/turnover:.2f}%")
        print(f"PnL / Starting:    {100*pnl/starting:+.2f}%")
```

- [ ] **Step 3: Build and verify**

Run: `maturin develop`

Then:
```python
python -c "
from engine import BacktestEngine, Instrument, FairValueData, OrderBookDelta, BookAction, OrderSide, F_LAST

inst = Instrument('X', 'EV', 4, 2, 0)
engine = BacktestEngine([inst], 10000.0)
print('Engine created OK')
"
```

Expected: `Engine created OK`

- [ ] **Step 4: Commit**

```bash
git add engine/strategy.py engine/__init__.py
git commit -m "feat: Python Strategy base class and BacktestEngine orchestrator"
```

---

### Task 7: prepare.py — Data Loading

**Files:**
- Create: `prepare.py`

This task implements all data loading with parquet caching. It depends on MongoDB access and the Kalshi API.

- [ ] **Step 1: Write prepare.py**

```python
# prepare.py
"""
Data loading and caching for Kalshi backtests.

Loads fair values (via kxrt_fv), orderbook data (MongoDB), instruments
(Kalshi REST API), and settlement outcomes (MongoDB). All data is cached
to parquet/JSON files under ~/.cache/kalshi-backtest/.

This file is FROZEN — the training agent must not modify it.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

# Load env from sandbox
load_dotenv(Path(__file__).resolve().parent / ".." / "sandbox" / ".env")

# Add sandbox to path for kxrt_fv
_sandbox = str(Path(__file__).resolve().parent / ".." / "sandbox")
if _sandbox not in sys.path:
    sys.path.insert(0, _sandbox)

from kxrt_fv import backtest as kxrt_backtest  # noqa: E402

from engine._engine import (  # noqa: E402
    BookAction,
    F_LAST,
    F_SNAPSHOT,
    FairValueData,
    Instrument,
    OrderBookDelta,
    OrderSide,
)


CACHE_DIR = Path.home() / ".cache" / "kalshi-backtest"
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"


@dataclass
class BacktestData:
    instruments: list[Instrument]
    fair_values: list[FairValueData]
    orderbook_deltas: list[OrderBookDelta]


def load(event_tickers: list[str], refresh: bool = False) -> BacktestData:
    """Load all backtest data, using parquet cache when available."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Event → market tickers from MongoDB
    print("Fetching market tickers from MongoDB...")
    event_markets = _load_event_market_tickers(event_tickers)
    total = sum(len(v) for v in event_markets.values())
    print(f"Found {len(event_markets)} events with {total} markets\n")

    all_tickers = [t for tickers in event_markets.values() for t in tickers]

    # Phase 2: Fair values via kxrt_fv
    print("Loading fair values...")
    fair_values = _load_fair_values(event_markets, refresh)
    fv_tickers = {fv.instrument_id for fv in fair_values}
    print(f"  {len(fair_values)} FV records for {len(fv_tickers)} tickers\n")

    # Phase 3: Instruments from Kalshi API
    print("Loading instruments...")
    instruments = _load_instruments(all_tickers, event_markets, refresh)
    inst_map = {inst.id: inst for inst in instruments}
    print(f"  {len(instruments)} instruments\n")

    # Phase 4: Orderbook deltas from MongoDB
    print("Loading orderbook data...")
    ob_deltas = _load_orderbooks(event_markets, inst_map, refresh)
    print(f"  {len(ob_deltas)} orderbook deltas\n")

    # Phase 5: Settlement outcomes + synthetic deltas
    print("Loading settlement outcomes...")
    outcomes = _load_outcomes(refresh)
    settlement_deltas = _build_settlement_deltas(inst_map, outcomes, ob_deltas)
    print(f"  {len(settlement_deltas)} settlement deltas\n")

    # Combine OB deltas + settlement deltas
    all_deltas = ob_deltas + settlement_deltas

    # Filter to instruments that have both FV and OB data
    active_tickers = fv_tickers & set(inst_map.keys())
    instruments = [inst_map[t] for t in sorted(active_tickers) if t in inst_map]
    fair_values = [fv for fv in fair_values if fv.instrument_id in active_tickers]
    all_deltas = [d for d in all_deltas if d.instrument_id in active_tickers]

    print(f"Ready: {len(instruments)} instruments, {len(fair_values)} FV, {len(all_deltas)} OB deltas")
    return BacktestData(instruments=instruments, fair_values=fair_values, orderbook_deltas=all_deltas)


# ── Internal loaders ──


def _load_event_market_tickers(event_tickers: list[str]) -> dict[str, list[str]]:
    """Query MongoDB kxrt.events for market tickers."""
    client = MongoClient(host=os.environ["MONGODB_URI"], tz_aware=True)
    db = client["kxrt"]
    events = list(db["events"].find(
        {"_id": {"$in": event_tickers}},
        {"_id": 1, "markets.ticker": 1},
    ))
    client.close()
    return {
        e["_id"]: [m["ticker"] for m in e.get("markets", [])]
        for e in events
    }


def _load_fair_values(
    event_markets: dict[str, list[str]], refresh: bool,
) -> list[FairValueData]:
    """Generate fair values via kxrt_fv.backtest(), cache per event."""
    fv_dir = CACHE_DIR / "fair_values"
    fv_dir.mkdir(exist_ok=True)
    all_fv: list[FairValueData] = []

    for i, (event_ticker, tickers) in enumerate(event_markets.items(), 1):
        if not tickers:
            continue
        cache_file = fv_dir / f"{event_ticker}.parquet"
        if cache_file.exists() and not refresh:
            print(f"  [{i}/{len(event_markets)}] {event_ticker}: cached")
            df = pd.read_parquet(cache_file)
            all_fv.extend(_df_to_fair_values(df))
            continue

        print(f"  [{i}/{len(event_markets)}] {event_ticker} ({len(tickers)} markets)...", end=" ", flush=True)
        try:
            snapshots = kxrt_backtest(tickers=tickers)
            records = []
            for ticker, snaps in snapshots.items():
                for snap in snaps:
                    ts_ns = int(snap.timestamp.timestamp() * 1e9)
                    records.append({
                        "timestamp_ns": ts_ns,
                        "instrument_id": ticker,
                        "fv": snap.fv,
                        "theta": snap.theta,
                        "gamma_pos": snap.gamma_pos,
                        "gamma_neg": snap.gamma_neg,
                        "new_review": snap.new_review,
                        "hours_left": snap.hours_left,
                        "cur_score": snap.cur_score,
                        "total_reviews": snap.total_reviews,
                    })
            df = pd.DataFrame(records)
            df.to_parquet(cache_file)
            all_fv.extend(_df_to_fair_values(df))
            print(f"{len(records)} records")
        except Exception as e:
            print(f"error: {e}")

    all_fv.sort(key=lambda x: x.timestamp_ns)
    return all_fv


def _df_to_fair_values(df: pd.DataFrame) -> list[FairValueData]:
    """Convert DataFrame rows to FairValueData objects."""
    result = []
    for _, row in df.iterrows():
        result.append(FairValueData(
            timestamp_ns=int(row["timestamp_ns"]),
            instrument_id=str(row["instrument_id"]),
            fv=float(row["fv"]),
            theta=float(row["theta"]),
            gamma_pos=float(row["gamma_pos"]),
            gamma_neg=float(row["gamma_neg"]),
            new_review=bool(row["new_review"]),
            hours_left=float(row["hours_left"]),
            cur_score=float(row["cur_score"]),
            total_reviews=int(row["total_reviews"]),
        ))
    return result


def _load_instruments(
    all_tickers: list[str],
    event_markets: dict[str, list[str]],
    refresh: bool,
) -> list[Instrument]:
    """Fetch instruments from Kalshi REST API, cache to parquet."""
    cache_file = CACHE_DIR / "instruments.parquet"
    if cache_file.exists() and not refresh:
        df = pd.read_parquet(cache_file)
        cached_tickers = set(df["id"].tolist())
        missing = [t for t in all_tickers if t not in cached_tickers]
        if not missing:
            return _df_to_instruments(df)
    else:
        df = pd.DataFrame()
        missing = all_tickers

    # Build ticker → event_ticker map
    ticker_to_event = {}
    for event, tickers in event_markets.items():
        for t in tickers:
            ticker_to_event[t] = event

    records = []
    if not df.empty:
        records = df.to_dict("records")

    with httpx.Client() as client:
        for i, ticker in enumerate(missing, 1):
            print(f"  [{i}/{len(missing)}] {ticker}...", end=" ", flush=True)
            try:
                resp = client.get(f"{KALSHI_API_BASE}/markets/{ticker}")
                resp.raise_for_status()
                market = resp.json()["market"]
                close_time = market.get("close_time") or market.get("latest_expiration_time")
                exp_ns = _parse_iso_to_ns(close_time) if close_time else 0
                records.append({
                    "id": ticker,
                    "event_ticker": market.get("event_ticker", ticker_to_event.get(ticker, "")),
                    "price_precision": 4,
                    "size_precision": 2,
                    "expiration_ns": exp_ns,
                })
                print("ok")
            except Exception as e:
                print(f"error: {e}")

    df = pd.DataFrame(records)
    df.to_parquet(cache_file)
    return _df_to_instruments(df)


def _df_to_instruments(df: pd.DataFrame) -> list[Instrument]:
    return [
        Instrument(
            id=str(row["id"]),
            event_ticker=str(row["event_ticker"]),
            price_precision=int(row["price_precision"]),
            size_precision=int(row["size_precision"]),
            expiration_ns=int(row["expiration_ns"]),
        )
        for _, row in df.iterrows()
    ]


def _parse_iso_to_ns(iso_str: str) -> int:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1e9)


def _load_orderbooks(
    event_markets: dict[str, list[str]],
    instruments: dict[str, Instrument],
    refresh: bool,
) -> list[OrderBookDelta]:
    """Load L2 orderbook data from MongoDB, cache per event."""
    ob_dir = CACHE_DIR / "orderbooks"
    ob_dir.mkdir(exist_ok=True)
    all_deltas: list[OrderBookDelta] = []

    client = MongoClient(host=os.environ["MONGODB_URI"], tz_aware=True)
    db = client["kxrt-training"]
    coll = db["orderbook-active"]

    for event_ticker, tickers in event_markets.items():
        cache_file = ob_dir / f"{event_ticker}.parquet"
        if cache_file.exists() and not refresh:
            print(f"  {event_ticker}: cached")
            df = pd.read_parquet(cache_file)
            all_deltas.extend(_df_to_ob_deltas(df))
            continue

        ob_tickers = [t for t in tickers if t in instruments]
        if not ob_tickers:
            continue

        print(f"  {event_ticker}: {len(ob_tickers)} markets...", end=" ", flush=True)
        docs = list(coll.find(
            {
                "metadata.event_ticker": event_ticker,
                "metadata.market_ticker": {"$in": ob_tickers},
            },
        ).sort("timestamp", 1))

        records = _transform_ob_docs(docs)
        df = pd.DataFrame(records)
        if not df.empty:
            df.to_parquet(cache_file)
        all_deltas.extend(_df_to_ob_deltas(df))
        print(f"{len(records)} deltas")

    client.close()
    return all_deltas


def _transform_ob_docs(docs: list[dict]) -> list[dict]:
    """Transform MongoDB orderbook documents into flat records."""
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for doc in docs:
        key = (
            doc["metadata"]["market_ticker"],
            doc["timestamp"],
            doc["is_snapshot"],
        )
        groups[key].append(doc)

    sorted_keys = sorted(groups.keys(), key=lambda k: (k[1], k[0]))
    records = []

    for key in sorted_keys:
        market_ticker, timestamp, is_snapshot = key
        group_docs = groups[key]
        ts_ns = int(timestamp.timestamp() * 1_000_000_000)

        if is_snapshot:
            # CLEAR + ADD for each level
            records.append({
                "instrument_id": market_ticker,
                "timestamp_ns": ts_ns,
                "action": "CLEAR",
                "side": "BUY",
                "price": 0.0,
                "size": 0.0,
                "flags": 0,
            })
            for i, doc in enumerate(group_docs):
                is_last = i == len(group_docs) - 1
                flags = F_SNAPSHOT | (F_LAST if is_last else 0)
                raw_price = float(doc["metadata"]["price"])
                if doc["metadata"]["side"] == "no":
                    raw_price = 1.0 - raw_price
                side = "BUY" if doc["metadata"]["side"] == "yes" else "SELL"
                records.append({
                    "instrument_id": market_ticker,
                    "timestamp_ns": ts_ns,
                    "action": "ADD",
                    "side": side,
                    "price": raw_price,
                    "size": float(doc["quantity"]),
                    "flags": flags,
                })
        else:
            for i, doc in enumerate(group_docs):
                is_last = i == len(group_docs) - 1
                quantity = doc["quantity"]
                action = "DELETE" if quantity == 0 else "UPDATE"
                flags = F_LAST if is_last else 0
                raw_price = float(doc["metadata"]["price"])
                if doc["metadata"]["side"] == "no":
                    raw_price = 1.0 - raw_price
                side = "BUY" if doc["metadata"]["side"] == "yes" else "SELL"
                records.append({
                    "instrument_id": market_ticker,
                    "timestamp_ns": ts_ns,
                    "action": action,
                    "side": side,
                    "price": raw_price,
                    "size": float(quantity),
                    "flags": flags,
                })

    return records


def _df_to_ob_deltas(df: pd.DataFrame) -> list[OrderBookDelta]:
    """Convert DataFrame rows to OrderBookDelta objects."""
    action_map = {"CLEAR": BookAction.Clear, "ADD": BookAction.Add,
                  "UPDATE": BookAction.Update, "DELETE": BookAction.Delete}
    side_map = {"BUY": OrderSide.Buy, "SELL": OrderSide.Sell}
    result = []
    for _, row in df.iterrows():
        result.append(OrderBookDelta(
            instrument_id=str(row["instrument_id"]),
            timestamp_ns=int(row["timestamp_ns"]),
            action=action_map[row["action"]],
            side=side_map[row["side"]],
            price=float(row["price"]),
            size=float(row["size"]),
            flags=int(row["flags"]),
        ))
    return result


def _load_outcomes(refresh: bool) -> dict[str, float]:
    """Load settlement outcomes from MongoDB kxrt.events."""
    cache_file = CACHE_DIR / "outcomes.json"
    if cache_file.exists() and not refresh:
        with open(cache_file) as f:
            return json.load(f)

    client = MongoClient(host=os.environ["MONGODB_URI"], tz_aware=True)
    db = client["kxrt"]
    outcomes: dict[str, float] = {}
    for event in db["events"].find({}, {"markets.ticker": 1, "markets.result": 1}):
        for market in event.get("markets", []):
            if "result" in market:
                outcomes[market["ticker"]] = 1.0 if market["result"] == "yes" else 0.0
    client.close()

    with open(cache_file, "w") as f:
        json.dump(outcomes, f)
    return outcomes


def _build_settlement_deltas(
    instruments: dict[str, Instrument],
    outcomes: dict[str, float],
    ob_deltas: list[OrderBookDelta],
) -> list[OrderBookDelta]:
    """Build synthetic settlement deltas that snap the book to 0.99 or 0.01."""
    # Find last OB timestamp per instrument
    last_ts: dict[str, int] = {}
    for d in ob_deltas:
        if d.instrument_id in last_ts:
            last_ts[d.instrument_id] = max(last_ts[d.instrument_id], d.timestamp_ns)
        else:
            last_ts[d.instrument_id] = d.timestamp_ns

    deltas: list[OrderBookDelta] = []
    for ticker, inst in instruments.items():
        outcome = outcomes.get(ticker)
        if outcome is None:
            continue

        expiry_ns = inst.expiration_ns
        ob_last = last_ts.get(ticker, 0)
        ts_ns = max(expiry_ns, ob_last) + 1_000_000_000  # 1s after

        settlement_price = 0.99 if outcome == 1.0 else 0.01

        # CLEAR
        deltas.append(OrderBookDelta(
            instrument_id=ticker, timestamp_ns=ts_ns,
            action=BookAction.Clear, side=OrderSide.Buy,
            price=0.0, size=0.0, flags=0,
        ))
        # BID at settlement price
        deltas.append(OrderBookDelta(
            instrument_id=ticker, timestamp_ns=ts_ns,
            action=BookAction.Add, side=OrderSide.Buy,
            price=settlement_price, size=1_000_000.0, flags=F_SNAPSHOT,
        ))
        # ASK at settlement price
        deltas.append(OrderBookDelta(
            instrument_id=ticker, timestamp_ns=ts_ns,
            action=BookAction.Add, side=OrderSide.Sell,
            price=settlement_price, size=1_000_000.0, flags=F_SNAPSHOT | F_LAST,
        ))

    deltas.sort(key=lambda d: d.timestamp_ns)
    return deltas
```

- [ ] **Step 2: Verify prepare.py loads (smoke test)**

Run: `python -c "import prepare; print('prepare imported OK')"`

Expected: `prepare imported OK` (may warn about missing MONGODB_URI if .env not set up)

- [ ] **Step 3: Commit**

```bash
git add prepare.py
git commit -m "feat: prepare.py with parquet-cached data loading pipeline"
```

---

### Task 8: train.py — KxrtBaseline Strategy

**Files:**
- Create: `train.py`

This is a direct port of nautilus_gct's KxrtBaseline strategy using our engine API.

- [ ] **Step 1: Write train.py**

```python
# train.py
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
```

- [ ] **Step 2: Verify train.py imports**

Run: `python -c "import train; print('train imported OK')"`

Expected: `train imported OK`

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: train.py with KxrtBaseline strategy (port from nautilus_gct)"
```

---

### Task 9: Run nautilus_gct Baseline (Capture Reference Results)

**Files:** None (read-only)

- [ ] **Step 1: Run the nautilus_gct baseline backtest**

```bash
cd /home/bgram/dev/nautilus_gct
python examples/kalshitest/kxrt_baseline_backtest.py 2>&1 | tee /tmp/nautilus_baseline_results.txt
```

- [ ] **Step 2: Extract key metrics**

Record from the output:
- Final balance
- Total PnL
- Number of filled orders
- Maker/taker fill counts
- Win rate
- Per-event PnL

Save these to compare against our engine.

---

### Task 10: Run kalshi-backtest and Compare

**Files:** None modified (may debug/fix issues found)

- [ ] **Step 1: Build and run the kalshi-backtest**

```bash
cd /home/bgram/dev/kalshi-backtest
maturin develop
python train.py 2>&1 | tee /tmp/kalshi_backtest_results.txt
```

- [ ] **Step 2: Compare results**

Compare the key metrics from Task 9 against the output:
- Final balance and PnL should be in the same direction and similar magnitude
- Fill counts should be similar
- Per-event PnL breakdown should show similar winners/losers

- [ ] **Step 3: Debug differences (if any)**

If results diverge significantly, check:
- Orderbook delta processing order (timestamp sorting)
- Fill logic: resting order fill at order price vs book price
- Fee calculation: rounding behavior
- Settlement delta timing
- Position PnL: avg entry price calculation

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: verified kalshi-backtest engine matches nautilus_gct baseline"
```
