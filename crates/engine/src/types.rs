use pyo3::prelude::*;

// ── Flags ────────────────────────────────────────────────────────────
pub const F_SNAPSHOT: u8 = 1;
pub const F_LAST: u8 = 2;

// ── Enums ────────────────────────────────────────────────────────────

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

// ── Instrument ───────────────────────────────────────────────────────

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
        Self {
            id,
            event_ticker,
            price_precision,
            size_precision,
            expiration_ns,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Instrument(id='{}', event_ticker='{}', price_prec={}, size_prec={}, exp_ns={})",
            self.id, self.event_ticker, self.price_precision, self.size_precision, self.expiration_ns
        )
    }
}

// ── FairValueData ────────────────────────────────────────────────────

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
            timestamp_ns,
            instrument_id,
            fv,
            theta,
            gamma_pos,
            gamma_neg,
            new_review,
            hours_left,
            cur_score,
            total_reviews,
        }
    }
}

// ── OrderBookDelta ───────────────────────────────────────────────────

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
        Self {
            instrument_id,
            timestamp_ns,
            action,
            side,
            price,
            size,
            flags,
        }
    }
}

// ── Order ────────────────────────────────────────────────────────────

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
            "Order(id='{}', inst='{}', side={:?}, px={}, qty={}, filled={}, status={:?})",
            self.id, self.instrument_id, self.side, self.price, self.quantity, self.filled_qty, self.status
        )
    }

    #[getter]
    fn is_closed(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Filled | OrderStatus::Canceled | OrderStatus::Rejected
        )
    }
}

// ── Fill ─────────────────────────────────────────────────────────────

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

// ── Position ─────────────────────────────────────────────────────────

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

#[pymethods]
impl Position {
    fn __repr__(&self) -> String {
        format!(
            "Position(inst='{}', qty={}, pnl={:.4}, avg_entry={:.4}, entries={})",
            self.instrument_id, self.signed_qty, self.realized_pnl, self.avg_entry_price, self.entry_count
        )
    }
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
