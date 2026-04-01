use std::collections::HashMap;

use crate::account::CashAccount;
use crate::orderbook::OrderBook;
use crate::types::*;

#[cfg_attr(not(test), pyo3::pyclass)]
#[derive(Debug, Clone)]
pub struct EngineCore {
    books: HashMap<String, OrderBook>,
    resting_orders: Vec<Order>,
    account: CashAccount,
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
        for inst in instruments {
            books.insert(inst.id.clone(), OrderBook::new());
            inst_map.insert(inst.id.clone(), inst);
        }
        Self {
            books,
            resting_orders: Vec::new(),
            account: CashAccount::new(starting_balance),
            instruments: inst_map,
            next_order_id: 1,
            fee_rate,
            all_orders: Vec::new(),
            all_fills: Vec::new(),
        }
    }

    pub fn apply_delta(&mut self, delta: &OrderBookDelta) {
        if let Some(book) = self.books.get_mut(&delta.instrument_id) {
            book.apply(delta);
        }
    }

    pub fn best_bid(&self, instrument_id: &str) -> Option<(f64, f64)> {
        self.books.get(instrument_id).and_then(|b| b.best_bid())
    }

    pub fn best_ask(&self, instrument_id: &str) -> Option<(f64, f64)> {
        self.books.get(instrument_id).and_then(|b| b.best_ask())
    }

    pub fn compute_taker_fee(&self, price: f64, qty: f64) -> f64 {
        (self.fee_rate * qty * price * (1.0 - price) * 100.0).ceil() / 100.0
    }

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
            id: order_id.clone(),
            instrument_id: instrument_id.to_string(),
            side: side,
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

        // Check if order crosses the book
        let crosses = match side {
            OrderSide::Buy => book
                .best_ask()
                .map_or(false, |(ask_px, _)| price >= ask_px),
            OrderSide::Sell => book
                .best_bid()
                .map_or(false, |(bid_px, _)| price <= bid_px),
        };

        if crosses {
            if post_only {
                order.status = OrderStatus::Rejected;
                self.all_orders.push(order.clone());
                return (order, None);
            }

            // Aggressive fill at book price (taker)
            let (fill_price, _) = match side {
                OrderSide::Buy => book.best_ask().unwrap(),
                OrderSide::Sell => book.best_bid().unwrap(),
            };

            let consumed = match side {
                OrderSide::Buy => book.consume_ask(quantity),
                OrderSide::Sell => book.consume_bid(quantity),
            };

            let fee = self.compute_taker_fee(fill_price, consumed);

            order.filled_qty = consumed;
            order.status = if consumed >= quantity {
                OrderStatus::Filled
            } else {
                // Partial fill - still mark as filled for simplicity
                // (depth-limited fill)
                OrderStatus::Filled
            };
            order.is_maker = Some(false);
            order.avg_fill_price = Some(fill_price);
            order.fill_timestamp_ns = Some(timestamp_ns);

            let fill = Fill {
                order_id: order_id.clone(),
                instrument_id: instrument_id.to_string(),
                side: side,
                price: fill_price,
                quantity: consumed,
                fee,
                is_maker: false,
                timestamp_ns,
            };

            self.account
                .process_fill(instrument_id, &side, fill_price, consumed, fee);
            self.all_fills.push(fill.clone());
            self.all_orders.push(order.clone());
            (order, Some(fill))
        } else {
            // Resting order
            order.status = OrderStatus::Resting;
            self.resting_orders.push(order.clone());
            self.all_orders.push(order.clone());
            (order, None)
        }
    }

    pub fn check_resting_orders(
        &mut self,
        instrument_id: &str,
        timestamp_ns: i64,
    ) -> Vec<Fill> {
        let mut fills = Vec::new();
        let mut filled_indices = Vec::new();

        for (i, order) in self.resting_orders.iter().enumerate() {
            if order.instrument_id != instrument_id {
                continue;
            }

            let book = match self.books.get(&order.instrument_id) {
                Some(b) => b,
                None => continue,
            };

            let crosses = match order.side {
                OrderSide::Buy => book
                    .best_ask()
                    .map_or(false, |(ask_px, _)| order.price >= ask_px),
                OrderSide::Sell => book
                    .best_bid()
                    .map_or(false, |(bid_px, _)| order.price <= bid_px),
            };

            if crosses {
                filled_indices.push(i);
            }
        }

        // Process fills in reverse to preserve indices
        for &i in filled_indices.iter().rev() {
            let mut order = self.resting_orders.remove(i);
            let remaining = order.quantity - order.filled_qty;

            let book = self.books.get_mut(&order.instrument_id).unwrap();
            let consumed = match order.side {
                OrderSide::Buy => book.consume_ask(remaining),
                OrderSide::Sell => book.consume_bid(remaining),
            };

            // Maker fill at ORDER's price, zero fee
            let fill = Fill {
                order_id: order.id.clone(),
                instrument_id: order.instrument_id.clone(),
                side: order.side,
                price: order.price,
                quantity: consumed,
                fee: 0.0,
                is_maker: true,
                timestamp_ns,
            };

            order.filled_qty += consumed;
            order.status = OrderStatus::Filled;
            order.is_maker = Some(true);
            order.avg_fill_price = Some(order.price);
            order.fill_timestamp_ns = Some(timestamp_ns);

            self.account.process_fill(
                &order.instrument_id,
                &order.side,
                order.price,
                consumed,
                0.0,
            );

            // Update the order in all_orders
            if let Some(ao) = self.all_orders.iter_mut().find(|o| o.id == order.id) {
                ao.filled_qty = order.filled_qty;
                ao.status = order.status.clone();
                ao.is_maker = order.is_maker;
                ao.avg_fill_price = order.avg_fill_price;
                ao.fill_timestamp_ns = order.fill_timestamp_ns;
            }

            self.all_fills.push(fill.clone());
            fills.push(fill);
        }

        fills
    }

    pub fn modify_order(&mut self, order_id: &str, new_quantity: f64) -> bool {
        if let Some(order) = self.resting_orders.iter_mut().find(|o| o.id == order_id) {
            order.quantity = new_quantity;
            // Also update in all_orders
            if let Some(ao) = self.all_orders.iter_mut().find(|o| o.id == order_id) {
                ao.quantity = new_quantity;
            }
            true
        } else {
            false
        }
    }

    pub fn cancel_order(&mut self, order_id: &str) -> bool {
        if let Some(idx) = self.resting_orders.iter().position(|o| o.id == order_id) {
            let mut order = self.resting_orders.remove(idx);
            order.status = OrderStatus::Canceled;
            // Update in all_orders
            if let Some(ao) = self.all_orders.iter_mut().find(|o| o.id == order_id) {
                ao.status = OrderStatus::Canceled;
            }
            true
        } else {
            false
        }
    }

    // Accessors
    pub fn balance(&self) -> f64 {
        self.account.balance
    }

    pub fn starting_balance(&self) -> f64 {
        self.account.starting_balance
    }

    pub fn get_position(&self, instrument_id: &str) -> Position {
        self.account.get_position(instrument_id)
    }

    pub fn get_all_orders(&self) -> Vec<Order> {
        self.all_orders.clone()
    }

    pub fn get_all_fills(&self) -> Vec<Fill> {
        self.all_fills.clone()
    }

    pub fn all_positions(&self) -> Vec<Position> {
        self.account.all_positions()
    }
}

// ── PyO3 bindings ────────────────────────────────────────────────────
#[cfg(not(test))]
#[pyo3::pymethods]
impl EngineCore {
    #[new]
    fn py_new(
        instruments: Vec<Instrument>,
        starting_balance: f64,
        fee_rate: f64,
    ) -> Self {
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

    #[pyo3(name = "check_resting_orders")]
    fn py_check_resting_orders(
        &mut self,
        instrument_id: &str,
        timestamp_ns: i64,
    ) -> Vec<Fill> {
        self.check_resting_orders(instrument_id, timestamp_ns)
    }

    #[pyo3(name = "modify_order")]
    fn py_modify_order(&mut self, order_id: &str, new_quantity: f64) -> bool {
        self.modify_order(order_id, new_quantity)
    }

    #[pyo3(name = "cancel_order")]
    fn py_cancel_order(&mut self, order_id: &str) -> bool {
        self.cancel_order(order_id)
    }

    #[pyo3(name = "compute_taker_fee")]
    fn py_compute_taker_fee(&self, price: f64, qty: f64) -> f64 {
        self.compute_taker_fee(price, qty)
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
        self.get_all_orders()
    }

    #[pyo3(name = "all_fills")]
    fn py_all_fills(&self) -> Vec<Fill> {
        self.get_all_fills()
    }

    #[pyo3(name = "all_positions")]
    fn py_all_positions(&self) -> Vec<Position> {
        self.all_positions()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_instrument(id: &str) -> Instrument {
        Instrument {
            id: id.to_string(),
            event_ticker: "EV".to_string(),
            price_precision: 4,
            size_precision: 2,
            expiration_ns: 0,
        }
    }

    fn setup_engine() -> EngineCore {
        let inst = make_instrument("X");
        let mut engine = EngineCore::new(vec![inst], 10_000.0, 0.07);
        // Add ask at 0.55 with 100 contracts
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(),
            timestamp_ns: 0,
            action: BookAction::Add,
            side: OrderSide::Sell,
            price: 0.55,
            size: 100.0,
            flags: 0,
        });
        // Add bid at 0.45 with 100 contracts
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(),
            timestamp_ns: 0,
            action: BookAction::Add,
            side: OrderSide::Buy,
            price: 0.45,
            size: 100.0,
            flags: 0,
        });
        engine
    }

    #[test]
    fn test_post_only_resting() {
        let mut engine = setup_engine();
        let (order, fill) =
            engine.submit_order("X", OrderSide::Buy, 0.50, 10.0, true, 1000);
        assert_eq!(order.status, OrderStatus::Resting);
        assert!(fill.is_none());
    }

    #[test]
    fn test_post_only_reject_on_cross() {
        let mut engine = setup_engine();
        // Buy at 0.55 would cross the ask at 0.55
        let (order, fill) =
            engine.submit_order("X", OrderSide::Buy, 0.55, 10.0, true, 1000);
        assert_eq!(order.status, OrderStatus::Rejected);
        assert!(fill.is_none());
    }

    #[test]
    fn test_aggressive_fill() {
        let mut engine = setup_engine();
        // Aggressive buy at 0.55, crosses ask at 0.55
        let (order, fill) =
            engine.submit_order("X", OrderSide::Buy, 0.55, 10.0, false, 1000);
        assert_eq!(order.status, OrderStatus::Filled);
        let fill = fill.unwrap();
        assert_eq!(fill.price, 0.55); // filled at book price
        assert_eq!(fill.quantity, 10.0);
        assert!(!fill.is_maker);
    }

    #[test]
    fn test_resting_order_fills_on_book_move() {
        let mut engine = setup_engine();
        // Place resting buy at 0.50 (doesn't cross ask at 0.55)
        let (order, _) =
            engine.submit_order("X", OrderSide::Buy, 0.50, 10.0, true, 1000);
        assert_eq!(order.status, OrderStatus::Resting);

        // Move ask down to 0.50 (now crosses our resting buy)
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(),
            timestamp_ns: 2000,
            action: BookAction::Add,
            side: OrderSide::Sell,
            price: 0.50,
            size: 50.0,
            flags: 0,
        });

        let fills = engine.check_resting_orders("X", 2000);
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].price, 0.50); // filled at ORDER's price
        assert!(fills[0].is_maker);
        assert_eq!(fills[0].fee, 0.0); // maker fee = 0
    }

    #[test]
    fn test_cancel_order() {
        let mut engine = setup_engine();
        let (order, _) =
            engine.submit_order("X", OrderSide::Buy, 0.50, 10.0, true, 1000);
        assert!(engine.cancel_order(&order.id));
        // Second cancel should fail
        assert!(!engine.cancel_order(&order.id));
    }

    #[test]
    fn test_modify_order() {
        let mut engine = setup_engine();
        let (order, _) =
            engine.submit_order("X", OrderSide::Buy, 0.50, 10.0, true, 1000);
        assert!(engine.modify_order(&order.id, 20.0));
        // Verify via all_orders
        let orders = engine.get_all_orders();
        let modified = orders.iter().find(|o| o.id == order.id).unwrap();
        assert_eq!(modified.quantity, 20.0);
    }

    #[test]
    fn test_taker_fee_calculation() {
        let engine = EngineCore::new(vec![make_instrument("X")], 10_000.0, 0.07);
        // 10 contracts @ 0.50: fee = ceil(0.07 * 10 * 0.50 * 0.50 * 100) / 100
        // = ceil(0.07 * 10 * 0.25 * 100) / 100 = ceil(17.5) / 100 = 18/100 = 0.18
        let fee = engine.compute_taker_fee(0.50, 10.0);
        assert!((fee - 0.18).abs() < 1e-9);
    }

    #[test]
    fn test_fill_limited_by_depth() {
        let mut engine = setup_engine();
        // Ask has 100.0, but let's set up a book with only 5
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(),
            timestamp_ns: 0,
            action: BookAction::Clear,
            side: OrderSide::Sell,
            price: 0.0,
            size: 0.0,
            flags: 0,
        });
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(),
            timestamp_ns: 0,
            action: BookAction::Add,
            side: OrderSide::Sell,
            price: 0.55,
            size: 5.0,
            flags: 0,
        });

        let (order, fill) =
            engine.submit_order("X", OrderSide::Buy, 0.55, 10.0, false, 1000);
        assert_eq!(order.status, OrderStatus::Filled);
        let fill = fill.unwrap();
        assert_eq!(fill.quantity, 5.0); // only got 5 of 10
        assert_eq!(order.filled_qty, 5.0);
    }
}
