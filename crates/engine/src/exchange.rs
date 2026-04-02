use std::collections::HashMap;

use crate::account::CashAccount;
use crate::orderbook::OrderBook;
use crate::types::*;

#[cfg_attr(not(test), pyo3::pyclass)]
#[derive(Debug, Clone)]
pub struct EngineCore {
    books: HashMap<String, OrderBook>,
    resting_orders: Vec<Order>,
    /// Capital locked per resting order (computed at submission time).
    order_locks: HashMap<String, f64>,
    account: CashAccount,
    instruments: HashMap<String, Instrument>,
    next_order_id: u64,
    fee_rate: f64,
    all_orders: Vec<Order>,
    all_fills: Vec<Fill>,
    /// Queue position tracking: order_id → (price_tick, qty_ahead)
    queue_ahead: HashMap<String, (i64, f64)>,
    consumed_asks: HashMap<String, HashMap<i64, f64>>,
    consumed_bids: HashMap<String, HashMap<i64, f64>>,
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
            order_locks: HashMap::new(),
            account: CashAccount::new(starting_balance),
            instruments: inst_map,
            next_order_id: 1,
            fee_rate,
            all_orders: Vec::new(),
            all_fills: Vec::new(),
            queue_ahead: HashMap::new(),
            consumed_asks: HashMap::new(),
            consumed_bids: HashMap::new(),
        }
    }

    pub fn apply_delta(&mut self, delta: &OrderBookDelta) {
        if let Some(book) = self.books.get_mut(&delta.instrument_id) {
            book.apply(delta);
        }

        // Consumption clearing: fresh delta data invalidates our consumption tracking
        match delta.action {
            BookAction::Add | BookAction::Update | BookAction::Delete => {
                let tick = crate::orderbook::price_to_tick(delta.price);
                let consumed_map = match delta.side {
                    OrderSide::Buy => self.consumed_bids.get_mut(&delta.instrument_id),
                    OrderSide::Sell => self.consumed_asks.get_mut(&delta.instrument_id),
                };
                if let Some(map) = consumed_map {
                    map.remove(&tick);
                }
            }
            BookAction::Clear => {
                self.consumed_asks.remove(&delta.instrument_id);
                self.consumed_bids.remove(&delta.instrument_id);
            }
        }

        // Queue position maintenance (mirrors nautilus cap_queue_ahead / clear_queue_on_delete)
        match delta.action {
            BookAction::Add | BookAction::Update => {
                if delta.size > 0.0 {
                    self.cap_queue_ahead(&delta.instrument_id, delta.price, delta.size, &delta.side);
                } else {
                    self.clear_queue_on_delete(&delta.instrument_id, delta.price, &delta.side);
                }
            }
            BookAction::Delete => {
                self.clear_queue_on_delete(&delta.instrument_id, delta.price, &delta.side);
            }
            BookAction::Clear => {
                self.clear_instrument_queues(&delta.instrument_id);
            }
        }
    }

    /// Cap queue_ahead to the new book depth at a price level.
    fn cap_queue_ahead(&mut self, instrument_id: &str, price: f64, new_size: f64, side: &OrderSide) {
        let price_tick = crate::orderbook::price_to_tick(price);
        for order in &self.resting_orders {
            if order.instrument_id != instrument_id || order.side != *side {
                continue;
            }
            if let Some((tracked_tick, ahead)) = self.queue_ahead.get_mut(&order.id) {
                if *tracked_tick == price_tick && *ahead > new_size {
                    *ahead = new_size;
                }
            }
        }
    }

    /// When a price level is deleted, all queue positions at that level clear to 0.
    fn clear_queue_on_delete(&mut self, instrument_id: &str, price: f64, side: &OrderSide) {
        let price_tick = crate::orderbook::price_to_tick(price);
        for order in &self.resting_orders {
            if order.instrument_id != instrument_id || order.side != *side {
                continue;
            }
            if let Some((tracked_tick, ahead)) = self.queue_ahead.get_mut(&order.id) {
                if *tracked_tick == price_tick {
                    *ahead = 0.0;
                }
            }
        }
    }

    /// When the book is cleared, clear all queue positions for that instrument.
    fn clear_instrument_queues(&mut self, instrument_id: &str) {
        for order in &self.resting_orders {
            if order.instrument_id != instrument_id {
                continue;
            }
            if let Some((_, ahead)) = self.queue_ahead.get_mut(&order.id) {
                *ahead = 0.0;
            }
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

    /// Compute the balance required to place an order.
    /// For binary options:
    ///   BUY:  cost = price * qty
    ///   SELL: cost = (1 - price) * qty for the position-opening portion
    ///         (position-reducing sells are free — you already own the contracts)
    fn order_cost(&self, instrument_id: &str, side: &OrderSide, price: f64, quantity: f64) -> f64 {
        match side {
            OrderSide::Buy => price * quantity,
            OrderSide::Sell => {
                let pos = self.account.get_position(instrument_id);
                // Only the portion that opens/increases a short costs capital
                let reducing_qty = if pos.signed_qty > 0.0 {
                    quantity.min(pos.signed_qty)
                } else {
                    0.0
                };
                let opening_qty = quantity - reducing_qty;
                (1.0 - price) * opening_qty
            }
        }
    }

    /// Free balance = cash balance minus capital locked by resting orders.
    fn free_balance(&self) -> f64 {
        let locked: f64 = self.order_locks.values().sum();
        self.account.balance - locked
    }

    pub fn submit_order(
        &mut self,
        instrument_id: &str,
        side: OrderSide,
        price: f64,
        quantity: f64,
        time_in_force: TimeInForce,
        reduce_only: bool,
        timestamp_ns: i64,
    ) -> (Order, Option<Fill>) {
        let order_id = format!("O-{}", self.next_order_id);
        self.next_order_id += 1;

        let order = Order {
            id: order_id.clone(),
            instrument_id: instrument_id.to_string(),
            side,
            price,
            quantity,
            filled_qty: 0.0,
            time_in_force,
            reduce_only,
            status: OrderStatus::Submitted,
            is_maker: None,
            avg_fill_price: None,
            submit_timestamp_ns: timestamp_ns,
            fill_timestamp_ns: None,
        };

        // reduce_only validation: reject if position doesn't allow reduction
        if reduce_only {
            let pos = self.account.get_position(instrument_id);
            let can_reduce = match side {
                OrderSide::Sell => pos.signed_qty > 1e-9,   // must be long to sell-reduce
                OrderSide::Buy => pos.signed_qty < -1e-9,   // must be short to buy-reduce
            };
            if !can_reduce {
                let mut order = order;
                order.status = OrderStatus::Rejected;
                self.all_orders.push(order.clone());
                return (order, None);
            }
        }

        // Balance validation: deny if insufficient free balance
        let cost = self.order_cost(instrument_id, &side, price, quantity);
        if cost > self.free_balance() + 1e-9 {
            let mut order = order;
            order.status = OrderStatus::Rejected;
            self.all_orders.push(order.clone());
            return (order, None);
        }

        let book = match self.books.get(instrument_id) {
            Some(b) => b,
            None => {
                let mut order = order;
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

        match time_in_force {
            TimeInForce::POST_ONLY => {
                if crosses {
                    let mut order = order;
                    order.status = OrderStatus::Rejected;
                    self.all_orders.push(order.clone());
                    (order, None)
                } else {
                    self.rest_order(order, cost)
                }
            }
            TimeInForce::FOK => {
                if !crosses {
                    let mut order = order;
                    order.status = OrderStatus::Canceled;
                    self.all_orders.push(order.clone());
                    return (order, None);
                }
                let same_side_depth = book.depth_at(&side, price);
                if same_side_depth > 1e-9 {
                    let mut order = order;
                    order.status = OrderStatus::Canceled;
                    self.all_orders.push(order.clone());
                    return (order, None);
                }
                self.fill_aggressive(order, instrument_id, side, price, quantity, true, timestamp_ns)
            }
            TimeInForce::IOC => {
                if !crosses {
                    let mut order = order;
                    order.status = OrderStatus::Canceled;
                    self.all_orders.push(order.clone());
                    return (order, None);
                }
                let same_side_depth = book.depth_at(&side, price);
                if same_side_depth > 1e-9 {
                    let mut order = order;
                    order.status = OrderStatus::Canceled;
                    self.all_orders.push(order.clone());
                    return (order, None);
                }
                self.fill_aggressive(order, instrument_id, side, price, quantity, false, timestamp_ns)
            }
            TimeInForce::GTC => {
                if crosses {
                    let same_side_depth = book.depth_at(&side, price);
                    if same_side_depth > 1e-9 {
                        return self.rest_order(order, cost);
                    }
                    self.fill_aggressive(order, instrument_id, side, price, quantity, false, timestamp_ns)
                } else {
                    self.rest_order(order, cost)
                }
            }
        }
    }

    fn rest_order(&mut self, mut order: Order, cost: f64) -> (Order, Option<Fill>) {
        order.status = OrderStatus::Resting;
        self.order_locks.insert(order.id.clone(), cost);
        let book = self.books.get(&order.instrument_id).unwrap();
        let depth_ahead = book.depth_at(&order.side, order.price);
        let tick = crate::orderbook::price_to_tick(order.price);
        self.queue_ahead.insert(order.id.clone(), (tick, depth_ahead));
        self.resting_orders.push(order.clone());
        self.all_orders.push(order.clone());
        (order, None)
    }

    fn fill_aggressive(
        &mut self,
        mut order: Order,
        instrument_id: &str,
        side: OrderSide,
        price: f64,
        quantity: f64,
        fok: bool,
        timestamp_ns: i64,
    ) -> (Order, Option<Fill>) {
        let empty = HashMap::new();
        let consumed = match side {
            OrderSide::Buy => self.consumed_asks.get(instrument_id).unwrap_or(&empty),
            OrderSide::Sell => self.consumed_bids.get(instrument_id).unwrap_or(&empty),
        };
        let book = self.books.get(instrument_id).unwrap();
        let fills = match side {
            OrderSide::Buy => book.simulate_fills_ask(price, quantity, consumed),
            OrderSide::Sell => book.simulate_fills_bid(price, quantity, consumed),
        };

        let total_available: f64 = fills.iter().map(|(_, q)| q).sum();

        // FOK requires full fill
        if fok && total_available < quantity - 1e-9 {
            order.status = OrderStatus::Canceled;
            self.all_orders.push(order.clone());
            return (order, None);
        }

        let fill_qty = quantity.min(total_available);
        if fill_qty <= 1e-9 {
            order.status = OrderStatus::Canceled;
            self.all_orders.push(order.clone());
            return (order, None);
        }

        let fill_price = Self::vwap(&fills);
        let fee = self.compute_taker_fee(fill_price, fill_qty);

        order.filled_qty = fill_qty;
        order.status = OrderStatus::Filled;
        order.is_maker = Some(false);
        order.avg_fill_price = Some(fill_price);
        order.fill_timestamp_ns = Some(timestamp_ns);

        let fill = Fill {
            order_id: order.id.clone(),
            instrument_id: instrument_id.to_string(),
            side,
            price: fill_price,
            quantity: fill_qty,
            fee,
            is_maker: false,
            timestamp_ns,
        };

        self.record_consumption(instrument_id, &side, &fills);
        self.account
            .process_fill(instrument_id, &side, fill_price, fill_qty, fee);
        self.all_fills.push(fill.clone());
        self.all_orders.push(order.clone());
        (order, Some(fill))
    }

    fn record_consumption(&mut self, instrument_id: &str, side: &OrderSide, fills: &[(f64, f64)]) {
        let consumed_map = match side {
            OrderSide::Buy => self.consumed_asks.entry(instrument_id.to_string()).or_insert_with(HashMap::new),
            OrderSide::Sell => self.consumed_bids.entry(instrument_id.to_string()).or_insert_with(HashMap::new),
        };
        for &(price, qty) in fills {
            let tick = crate::orderbook::price_to_tick(price);
            *consumed_map.entry(tick).or_insert(0.0) += qty;
        }
    }

    fn vwap(fills: &[(f64, f64)]) -> f64 {
        let total_qty: f64 = fills.iter().map(|(_, q)| q).sum();
        if total_qty <= 1e-9 { return 0.0; }
        let total_notional: f64 = fills.iter().map(|(p, q)| p * q).sum();
        total_notional / total_qty
    }

    pub fn check_resting_orders(
        &mut self,
        instrument_id: &str,
        timestamp_ns: i64,
    ) -> Vec<Fill> {
        self.check_resting_orders_inner(instrument_id, timestamp_ns)
    }

    /// Book-triggered fill check: fills resting orders whose price crosses the
    /// current book. Reads available depth WITHOUT consuming (liquidity_consumption=False,
    /// matching nautilus behavior where the book is read-only market data).
    fn check_resting_orders_inner(
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

            if !crosses {
                continue;
            }

            // Queue position gate: only fill if queue ahead is exhausted
            if let Some(&(_, ahead)) = self.queue_ahead.get(&order.id) {
                if ahead > 1e-9 {
                    continue; // still behind in queue
                }
            }

            // Read available depth without consuming (matches nautilus liquidity_consumption=False)
            let available = match order.side {
                OrderSide::Buy => book.available_ask_depth(order.price),
                OrderSide::Sell => book.available_bid_depth(order.price),
            };
            if available <= 0.0 {
                continue;
            }

            let remaining = order.quantity - order.filled_qty;
            let fill_qty = remaining.min(available);

            filled_indices.push((i, fill_qty));
        }

        // Process fills in reverse to preserve indices
        for &(i, fill_qty) in filled_indices.iter().rev() {
            let mut order = self.resting_orders.remove(i);

            let fill = Fill {
                order_id: order.id.clone(),
                instrument_id: order.instrument_id.clone(),
                side: order.side,
                price: order.price,
                quantity: fill_qty,
                fee: 0.0,
                is_maker: true,
                timestamp_ns,
            };

            order.filled_qty += fill_qty;
            order.avg_fill_price = Some(order.price);
            order.fill_timestamp_ns = Some(timestamp_ns);
            order.is_maker = Some(true);

            if order.filled_qty >= order.quantity - 1e-9 {
                order.status = OrderStatus::Filled;
                self.order_locks.remove(&order.id);
                self.queue_ahead.remove(&order.id);
            } else {
                // Partial fill — keep resting
                self.resting_orders.push(order.clone());
            }

            self.account.process_fill(
                &order.instrument_id,
                &order.side,
                order.price,
                fill_qty,
                0.0,
            );

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

    /// Process a historical trade tick: decrement queue positions, then fill
    /// matching resting orders using trade price for crossing detection (not the book).
    /// This mirrors nautilus's temporary bid/ask override during trade execution.
    pub fn process_trade_tick(&mut self, trade: &TradeTick) -> Vec<Fill> {
        let price_tick = crate::orderbook::price_to_tick(trade.price);

        // Buyer aggressor consumes ask-side depth → decrements SELL order queues.
        // Seller aggressor consumes bid-side depth → decrements BUY order queues.
        let target_side = match trade.aggressor_side {
            AggressorSide::Buyer => Some(OrderSide::Sell),
            AggressorSide::Seller => Some(OrderSide::Buy),
            AggressorSide::NoAggressor => None,
        };

        // ── Phase 1: Decrement queue positions and compute queue excess ──
        let mut queue_excess: HashMap<String, f64> = HashMap::new();

        let mut to_decrement: Vec<(String, f64, f64)> = Vec::new(); // (order_id, ahead, leaves)
        for order in &self.resting_orders {
            if order.instrument_id != trade.instrument_id {
                continue;
            }
            if let Some(ref ts) = target_side {
                if order.side != *ts {
                    continue;
                }
            }
            if let Some(&(order_tick, ahead)) = self.queue_ahead.get(&order.id) {
                if order_tick == price_tick && ahead > 1e-9 {
                    let leaves = order.quantity - order.filled_qty;
                    to_decrement.push((order.id.clone(), ahead, leaves));
                }
            }
        }
        to_decrement.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut remaining = trade.size;
        let mut prev_position: f64 = 0.0;

        for (order_id, ahead, leaves) in &to_decrement {
            if remaining <= 1e-9 {
                let new_ahead = (*ahead - trade.size).max(0.0);
                self.queue_ahead.insert(order_id.clone(), (price_tick, new_ahead));
                if new_ahead <= 1e-9 {
                    queue_excess.insert(order_id.clone(), 0.0);
                }
                continue;
            }

            let gap = (*ahead - prev_position).max(0.0);
            let queue_consumed = remaining.min(gap);
            remaining -= queue_consumed;

            if remaining <= 1e-9 && queue_consumed < gap {
                let new_ahead = (*ahead - trade.size).max(0.0);
                self.queue_ahead.insert(order_id.clone(), (price_tick, new_ahead));
                continue;
            }

            // Queue cleared for this order
            self.queue_ahead.insert(order_id.clone(), (price_tick, 0.0));
            let excess = remaining.min(*leaves);
            queue_excess.insert(order_id.clone(), excess);
            remaining -= excess;
            prev_position = *ahead + excess;
        }

        // ── Phase 2: Fill resting orders using trade price for crossing ──
        // This mirrors nautilus's temporary bid/ask override: a seller aggressor
        // at price P means ask = P, so buy orders at >= P can fill.
        let mut fills = Vec::new();
        let mut trade_consumption: f64 = 0.0;
        let mut filled_indices: Vec<(usize, f64)> = Vec::new();

        for (i, order) in self.resting_orders.iter().enumerate() {
            if order.instrument_id != trade.instrument_id {
                continue;
            }

            // Check side: seller aggressor fills buy orders, buyer fills sell orders
            let side_ok = match trade.aggressor_side {
                AggressorSide::Buyer => order.side == OrderSide::Sell,
                AggressorSide::Seller => order.side == OrderSide::Buy,
                AggressorSide::NoAggressor => true,
            };
            if !side_ok {
                continue;
            }

            // Check price: buy at >= trade price, sell at <= trade price
            let price_ok = match order.side {
                OrderSide::Buy => order.price >= trade.price - 1e-9,
                OrderSide::Sell => order.price <= trade.price + 1e-9,
            };
            if !price_ok {
                continue;
            }

            // Queue position gate
            if let Some(&(tracked_tick, ahead)) = self.queue_ahead.get(&order.id) {
                let order_tick = crate::orderbook::price_to_tick(order.price);
                if tracked_tick == order_tick && ahead > 1e-9 {
                    // Allow fill if trade "crossed through" the order's price
                    let crossed = match order.side {
                        OrderSide::Buy => trade.price < order.price - 1e-9,
                        OrderSide::Sell => trade.price > order.price + 1e-9,
                    };
                    if !crossed {
                        continue;
                    }
                }
            }

            let leaves = order.quantity - order.filled_qty;
            if leaves <= 1e-9 {
                continue;
            }

            let mut fill_qty = leaves;

            // Cap by queue excess (only for orders at the trade's price level)
            if let Some(&excess) = queue_excess.get(&order.id) {
                if excess <= 1e-9 {
                    continue;
                }
                fill_qty = fill_qty.min(excess);
            }

            // Cap by remaining trade budget
            let budget_remaining = trade.size - trade_consumption;
            if budget_remaining <= 1e-9 {
                continue;
            }
            fill_qty = fill_qty.min(budget_remaining);

            if fill_qty > 1e-9 {
                filled_indices.push((i, fill_qty));
                trade_consumption += fill_qty;
            }
        }

        // Process fills in reverse to preserve indices
        for &(i, fill_qty) in filled_indices.iter().rev() {
            let mut order = self.resting_orders.remove(i);

            let fill = Fill {
                order_id: order.id.clone(),
                instrument_id: order.instrument_id.clone(),
                side: order.side,
                price: order.price,
                quantity: fill_qty,
                fee: 0.0,
                is_maker: true,
                timestamp_ns: trade.timestamp_ns,
            };

            order.filled_qty += fill_qty;
            order.avg_fill_price = Some(order.price);
            order.fill_timestamp_ns = Some(trade.timestamp_ns);
            order.is_maker = Some(true);

            if order.filled_qty >= order.quantity - 1e-9 {
                order.status = OrderStatus::Filled;
                self.order_locks.remove(&order.id);
                self.queue_ahead.remove(&order.id);
            } else {
                // Partial fill — keep resting
                self.resting_orders.push(order.clone());
            }

            self.account.process_fill(
                &order.instrument_id,
                &order.side,
                order.price,
                fill_qty,
                0.0,
            );

            if let Some(ao) = self.all_orders.iter_mut().find(|o| o.id == order.id) {
                ao.filled_qty = order.filled_qty;
                ao.status = order.status.clone();
                ao.is_maker = order.is_maker;
                ao.avg_fill_price = order.avg_fill_price;
                ao.fill_timestamp_ns = order.fill_timestamp_ns;
            }

            // Record consumption: resting BUY filled = bid consumed, resting SELL = ask consumed
            let tick = crate::orderbook::price_to_tick(order.price);
            let consumed_map = match order.side {
                OrderSide::Buy => self.consumed_bids.entry(order.instrument_id.clone()).or_insert_with(HashMap::new),
                OrderSide::Sell => self.consumed_asks.entry(order.instrument_id.clone()).or_insert_with(HashMap::new),
            };
            *consumed_map.entry(tick).or_insert(0.0) += fill_qty;

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
            self.order_locks.remove(order_id);
            self.queue_ahead.remove(order_id);
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

    /// Settle a single instrument at a specific settlement price (1.00 or 0.00).
    /// Cancels resting orders and directly closes the position (no book crossing check).
    /// Fee at settlement price is 0 because price * (1 - price) = 0.
    pub fn settle_instrument(&mut self, instrument_id: &str, settlement_price: f64, timestamp_ns: i64) -> Vec<Fill> {
        // Cancel all resting orders for this instrument
        let order_ids: Vec<String> = self.resting_orders.iter()
            .filter(|o| o.instrument_id == instrument_id)
            .map(|o| o.id.clone())
            .collect();
        for id in order_ids {
            self.cancel_order(&id);
        }

        // Close the position directly at settlement price (no fee)
        let pos = self.account.get_position(instrument_id);
        if pos.signed_qty.abs() < 1e-9 {
            return Vec::new();
        }

        let side = if pos.signed_qty > 0.0 {
            OrderSide::Sell
        } else {
            OrderSide::Buy
        };
        let qty = pos.signed_qty.abs();

        let order_id = format!("O-{}", self.next_order_id);
        self.next_order_id += 1;

        // Fee = 0 at settlement prices (price * (1-price) = 0)
        let fee = self.compute_taker_fee(settlement_price, qty);

        let order = Order {
            id: order_id.clone(),
            instrument_id: instrument_id.to_string(),
            side,
            price: settlement_price,
            quantity: qty,
            filled_qty: qty,
            time_in_force: TimeInForce::FOK,
            reduce_only: false,
            status: OrderStatus::Filled,
            is_maker: Some(false),
            avg_fill_price: Some(settlement_price),
            submit_timestamp_ns: timestamp_ns,
            fill_timestamp_ns: Some(timestamp_ns),
        };

        let fill = Fill {
            order_id: order_id.clone(),
            instrument_id: instrument_id.to_string(),
            side,
            price: settlement_price,
            quantity: qty,
            fee,
            is_maker: false,
            timestamp_ns,
        };

        self.account.process_fill(instrument_id, &side, settlement_price, qty, fee);
        self.all_fills.push(fill.clone());
        self.all_orders.push(order);

        vec![fill]
    }

    /// Close all open positions at the current book's best price.
    /// Cancels all resting orders first, then submits market-like closes.
    pub fn settle_all(&mut self) -> Vec<Fill> {
        // Cancel all resting orders
        let order_ids: Vec<String> = self.resting_orders.iter().map(|o| o.id.clone()).collect();
        for id in order_ids {
            self.cancel_order(&id);
        }

        // Close each open position
        let mut fills = Vec::new();
        let positions = self.account.all_positions();
        for pos in positions {
            if pos.signed_qty.abs() < 1e-9 {
                continue;
            }

            let (side, close_price) = if pos.signed_qty > 0.0 {
                // Long → sell at best bid
                let price = self.best_bid(&pos.instrument_id)
                    .map(|(p, _)| p)
                    .unwrap_or(0.01);
                (OrderSide::Sell, price)
            } else {
                // Short → buy at best ask
                let price = self.best_ask(&pos.instrument_id)
                    .map(|(p, _)| p)
                    .unwrap_or(0.99);
                (OrderSide::Buy, price)
            };

            let qty = pos.signed_qty.abs();
            let (_, fill) = self.submit_order(
                &pos.instrument_id, side, close_price, qty, TimeInForce::IOC, false, 0,
            );
            if let Some(f) = fill {
                fills.push(f);
            }
        }
        fills
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
        time_in_force: TimeInForce,
        reduce_only: bool,
        timestamp_ns: i64,
    ) -> (Order, Option<Fill>) {
        self.submit_order(instrument_id, side, price, quantity, time_in_force, reduce_only, timestamp_ns)
    }

    #[pyo3(name = "check_resting_orders")]
    fn py_check_resting_orders(
        &mut self,
        instrument_id: &str,
        timestamp_ns: i64,
    ) -> Vec<Fill> {
        self.check_resting_orders(instrument_id, timestamp_ns)
    }

    #[pyo3(name = "process_trade_tick")]
    fn py_process_trade_tick(&mut self, trade: &TradeTick) -> Vec<Fill> {
        self.process_trade_tick(trade)
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

    #[pyo3(name = "free_balance")]
    fn py_free_balance(&self) -> f64 {
        self.free_balance()
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

    #[pyo3(name = "settle_instrument")]
    fn py_settle_instrument(&mut self, instrument_id: &str, settlement_price: f64, timestamp_ns: i64) -> Vec<Fill> {
        self.settle_instrument(instrument_id, settlement_price, timestamp_ns)
    }

    #[pyo3(name = "settle_all")]
    fn py_settle_all(&mut self) -> Vec<Fill> {
        self.settle_all()
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
            engine.submit_order("X", OrderSide::Buy, 0.50, 10.0, TimeInForce::POST_ONLY, false, 1000);
        assert_eq!(order.status, OrderStatus::Resting);
        assert!(fill.is_none());
    }

    #[test]
    fn test_post_only_reject_on_cross() {
        let mut engine = setup_engine();
        // Buy at 0.55 would cross the ask at 0.55
        let (order, fill) =
            engine.submit_order("X", OrderSide::Buy, 0.55, 10.0, TimeInForce::POST_ONLY, false, 1000);
        assert_eq!(order.status, OrderStatus::Rejected);
        assert!(fill.is_none());
    }

    #[test]
    fn test_aggressive_fill() {
        let mut engine = setup_engine();
        // Aggressive buy at 0.55, crosses ask at 0.55
        let (order, fill) =
            engine.submit_order("X", OrderSide::Buy, 0.55, 10.0, TimeInForce::FOK, false, 1000);
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
            engine.submit_order("X", OrderSide::Buy, 0.50, 10.0, TimeInForce::POST_ONLY, false, 1000);
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
        assert_eq!(fills[0].quantity, 10.0); // capped by depth (50 available, need 10)
        assert!(fills[0].is_maker);
        assert_eq!(fills[0].fee, 0.0); // maker fee = 0
    }

    #[test]
    fn test_cancel_order() {
        let mut engine = setup_engine();
        let (order, _) =
            engine.submit_order("X", OrderSide::Buy, 0.50, 10.0, TimeInForce::POST_ONLY, false, 1000);
        assert!(engine.cancel_order(&order.id));
        // Second cancel should fail
        assert!(!engine.cancel_order(&order.id));
    }

    #[test]
    fn test_modify_order() {
        let mut engine = setup_engine();
        let (order, _) =
            engine.submit_order("X", OrderSide::Buy, 0.50, 10.0, TimeInForce::POST_ONLY, false, 1000);
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
    fn test_balance_rejection() {
        // Engine with only $1 balance
        let inst = make_instrument("X");
        let mut engine = EngineCore::new(vec![inst], 1.0, 0.07);
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(),
            timestamp_ns: 0,
            action: BookAction::Add,
            side: OrderSide::Sell,
            price: 0.55,
            size: 100.0,
            flags: 0,
        });
        // Buy 10 @ 0.50 costs $5, but only $1 available
        let (order, fill) =
            engine.submit_order("X", OrderSide::Buy, 0.50, 10.0, TimeInForce::POST_ONLY, false, 1000);
        assert_eq!(order.status, OrderStatus::Rejected);
        assert!(fill.is_none());
    }

    #[test]
    fn test_locked_balance_prevents_second_order() {
        let mut engine = setup_engine();
        // Place resting buy at 0.50 for 100 contracts = $50 locked
        // Starting balance is $10,000
        let (o1, _) =
            engine.submit_order("X", OrderSide::Buy, 0.50, 19_990.0, TimeInForce::POST_ONLY, false, 1000);
        assert_eq!(o1.status, OrderStatus::Resting);
        // Free balance should be 10000 - 0.50*19990 = 10000 - 9995 = 5
        // Try to place another order for $10 → should be rejected
        let (o2, _) =
            engine.submit_order("X", OrderSide::Buy, 0.50, 20.0, TimeInForce::POST_ONLY, false, 2000);
        assert_eq!(o2.status, OrderStatus::Rejected);
    }

    #[test]
    fn test_cancel_releases_locked_balance() {
        let mut engine = setup_engine();
        // Lock nearly all balance
        let (o1, _) =
            engine.submit_order("X", OrderSide::Buy, 0.50, 19_990.0, TimeInForce::POST_ONLY, false, 1000);
        assert_eq!(o1.status, OrderStatus::Resting);
        // Cancel it
        engine.cancel_order(&o1.id);
        // Now we should be able to place the same order again
        let (o2, _) =
            engine.submit_order("X", OrderSide::Buy, 0.50, 19_990.0, TimeInForce::POST_ONLY, false, 2000);
        assert_eq!(o2.status, OrderStatus::Resting);
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

        // FOK: need 10 but only 5 available → canceled
        let (order, fill) =
            engine.submit_order("X", OrderSide::Buy, 0.55, 10.0, TimeInForce::FOK, false, 1000);
        assert_eq!(order.status, OrderStatus::Canceled);
        assert!(fill.is_none());
    }

    #[test]
    fn test_queue_position_blocks_fill() {
        let mut engine = setup_engine();
        // Book: bid 0.45 x100, ask 0.55 x100
        // Place resting buy at 0.50 — queue_ahead = 0 (no bids at 0.50)
        let (order, _) =
            engine.submit_order("X", OrderSide::Buy, 0.50, 10.0, TimeInForce::POST_ONLY, false, 1000);
        assert_eq!(order.status, OrderStatus::Resting);
        assert_eq!(engine.queue_ahead.get(&order.id).unwrap().1, 0.0);

        // Move ask down to 0.50 — crosses our buy
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(),
            timestamp_ns: 2000,
            action: BookAction::Add,
            side: OrderSide::Sell,
            price: 0.50,
            size: 50.0,
            flags: 0,
        });

        // Queue is 0, so check_resting_orders should fill it
        let fills = engine.check_resting_orders("X", 2000);
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].quantity, 10.0);
    }

    #[test]
    fn test_queue_position_with_depth_ahead() {
        let mut engine = setup_engine();
        // Add existing bid depth at 0.50 (20 contracts ahead of us)
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(),
            timestamp_ns: 500,
            action: BookAction::Add,
            side: OrderSide::Buy,
            price: 0.50,
            size: 20.0,
            flags: 0,
        });

        // Place resting buy at 0.50 — queue_ahead = 20
        let (order, _) =
            engine.submit_order("X", OrderSide::Buy, 0.50, 10.0, TimeInForce::POST_ONLY, false, 1000);
        assert_eq!(order.status, OrderStatus::Resting);
        assert_eq!(engine.queue_ahead.get(&order.id).unwrap().1, 20.0);

        // Move ask down to 0.50 — crosses our buy
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(),
            timestamp_ns: 2000,
            action: BookAction::Add,
            side: OrderSide::Sell,
            price: 0.50,
            size: 50.0,
            flags: 0,
        });

        // Queue is 20, so check_resting_orders should NOT fill
        let fills = engine.check_resting_orders("X", 2000);
        assert_eq!(fills.len(), 0);

        // Process a seller-aggressor trade at 0.50 for 15 contracts
        // This decrements BUY queue from 20 to 5
        let trade = TradeTick {
            instrument_id: "X".to_string(),
            price: 0.50,
            size: 15.0,
            aggressor_side: AggressorSide::Seller,
            timestamp_ns: 2500,
        };
        let fills = engine.process_trade_tick(&trade);
        assert_eq!(fills.len(), 0); // still 5 ahead
        assert!(engine.queue_ahead.get(&order.id).unwrap().1 > 0.0);

        // Process another trade for 10 — clears queue (5 ahead), excess = min(5, 10) = 5
        let trade2 = TradeTick {
            instrument_id: "X".to_string(),
            price: 0.50,
            size: 10.0,
            aggressor_side: AggressorSide::Seller,
            timestamp_ns: 3000,
        };
        let fills = engine.process_trade_tick(&trade2);
        assert_eq!(fills.len(), 1); // queue cleared, order fills
        assert_eq!(fills[0].quantity, 5.0); // capped by queue excess (5 remaining after clearing queue of 5)
    }

    #[test]
    fn test_fok_canceled_on_crossed_book() {
        let mut engine = setup_engine();
        // Create a crossed book: bid 0.55 x 10, ask 0.50 x 5
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(),
            timestamp_ns: 0,
            action: BookAction::Add,
            side: OrderSide::Buy,
            price: 0.55,
            size: 10.0,
            flags: 0,
        });
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(),
            timestamp_ns: 0,
            action: BookAction::Add,
            side: OrderSide::Sell,
            price: 0.50,
            size: 5.0,
            flags: 0,
        });

        // BUY at 0.50 crosses (ask=0.50), but bid depth at 0.50 is 0 → no queue ahead → fills
        // Wait — there are no bids at 0.50 in our book, bids are at 0.45 and 0.55
        // So depth_at(Buy, 0.50) = 0 → should fill
        let (order, fill) =
            engine.submit_order("X", OrderSide::Buy, 0.50, 3.0, TimeInForce::FOK, false, 1000);
        assert_eq!(order.status, OrderStatus::Filled);
        assert!(fill.is_some());

        // SELL at 0.55 crosses (bid=0.55), ask depth at 0.55 is 0.55x100 from setup
        // depth_at(Sell, 0.55) = asks at 0.55 = 100 from setup → queue ahead > 0 → canceled
        let (order2, fill2) =
            engine.submit_order("X", OrderSide::Sell, 0.55, 3.0, TimeInForce::FOK, false, 2000);
        assert_eq!(order2.status, OrderStatus::Canceled);
        assert!(fill2.is_none());
    }

    #[test]
    fn test_multilevel_aggressive_fill() {
        let inst = make_instrument("X");
        let mut engine = EngineCore::new(vec![inst], 10_000.0, 0.07);
        // Build multi-level ask book: 5@0.50 + 20@0.51 + 75@0.52
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(), timestamp_ns: 0,
            action: BookAction::Add, side: OrderSide::Sell, price: 0.50, size: 5.0, flags: 0,
        });
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(), timestamp_ns: 0,
            action: BookAction::Add, side: OrderSide::Sell, price: 0.51, size: 20.0, flags: 0,
        });
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(), timestamp_ns: 0,
            action: BookAction::Add, side: OrderSide::Sell, price: 0.52, size: 75.0, flags: 0,
        });

        // FOK buy 30 up to 0.52 → sweeps 5@0.50 + 20@0.51 + 5@0.52
        let (order, fill) = engine.submit_order("X", OrderSide::Buy, 0.52, 30.0, TimeInForce::FOK, false, 1000);
        assert_eq!(order.status, OrderStatus::Filled);
        let fill = fill.unwrap();
        assert_eq!(fill.quantity, 30.0);
        // VWAP = (5*0.50 + 20*0.51 + 5*0.52) / 30 = (2.5 + 10.2 + 2.6) / 30 = 15.3 / 30 = 0.51
        assert!((fill.price - 0.51).abs() < 1e-9);
    }

    #[test]
    fn test_ioc_partial_fill() {
        let inst = make_instrument("X");
        let mut engine = EngineCore::new(vec![inst], 10_000.0, 0.07);
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(), timestamp_ns: 0,
            action: BookAction::Add, side: OrderSide::Sell, price: 0.50, size: 100.0, flags: 0,
        });

        // IOC buy 200 against 100 available → fills 100
        let (order, fill) = engine.submit_order("X", OrderSide::Buy, 0.50, 200.0, TimeInForce::IOC, false, 1000);
        assert_eq!(order.status, OrderStatus::Filled);
        let fill = fill.unwrap();
        assert_eq!(fill.quantity, 100.0);
    }

    #[test]
    fn test_gtc_crosses_fills_immediately() {
        let inst = make_instrument("X");
        let mut engine = EngineCore::new(vec![inst], 10_000.0, 0.07);
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(), timestamp_ns: 0,
            action: BookAction::Add, side: OrderSide::Sell, price: 0.50, size: 100.0, flags: 0,
        });

        // GTC buy at ask → fills immediately
        let (order, fill) = engine.submit_order("X", OrderSide::Buy, 0.50, 10.0, TimeInForce::GTC, false, 1000);
        assert_eq!(order.status, OrderStatus::Filled);
        assert!(fill.is_some());
        assert_eq!(fill.unwrap().quantity, 10.0);
    }

    #[test]
    fn test_gtc_rests_when_no_cross() {
        let inst = make_instrument("X");
        let mut engine = EngineCore::new(vec![inst], 10_000.0, 0.07);
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(), timestamp_ns: 0,
            action: BookAction::Add, side: OrderSide::Sell, price: 0.55, size: 100.0, flags: 0,
        });

        // GTC buy below ask → rests
        let (order, fill) = engine.submit_order("X", OrderSide::Buy, 0.50, 10.0, TimeInForce::GTC, false, 1000);
        assert_eq!(order.status, OrderStatus::Resting);
        assert!(fill.is_none());
    }

    #[test]
    fn test_reduce_only_rejects_when_no_position() {
        let mut engine = setup_engine();
        // reduce_only sell with no position → rejected
        let (order, fill) = engine.submit_order("X", OrderSide::Sell, 0.45, 10.0, TimeInForce::FOK, true, 1000);
        assert_eq!(order.status, OrderStatus::Rejected);
        assert!(fill.is_none());
    }

    #[test]
    fn test_consumption_reduces_available_depth() {
        let inst = make_instrument("X");
        let mut engine = EngineCore::new(vec![inst], 10_000.0, 0.07);
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(), timestamp_ns: 0,
            action: BookAction::Add, side: OrderSide::Sell, price: 0.50, size: 100.0, flags: 0,
        });

        // First IOC buy 60 → fills 60
        let (o1, f1) = engine.submit_order("X", OrderSide::Buy, 0.50, 60.0, TimeInForce::IOC, false, 1000);
        assert_eq!(o1.status, OrderStatus::Filled);
        assert_eq!(f1.unwrap().quantity, 60.0);

        // Second IOC buy 60 → only 40 left (100 - 60 consumed)
        let (o2, f2) = engine.submit_order("X", OrderSide::Buy, 0.50, 60.0, TimeInForce::IOC, false, 2000);
        assert_eq!(o2.status, OrderStatus::Filled);
        assert_eq!(f2.unwrap().quantity, 40.0);
    }

    #[test]
    fn test_consumption_resets_on_delta() {
        let inst = make_instrument("X");
        let mut engine = EngineCore::new(vec![inst], 10_000.0, 0.07);
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(), timestamp_ns: 0,
            action: BookAction::Add, side: OrderSide::Sell, price: 0.50, size: 100.0, flags: 0,
        });

        // Consume all 100
        let (o1, f1) = engine.submit_order("X", OrderSide::Buy, 0.50, 100.0, TimeInForce::IOC, false, 1000);
        assert_eq!(o1.status, OrderStatus::Filled);
        assert_eq!(f1.unwrap().quantity, 100.0);

        // Fresh delta resets consumption at that level
        engine.apply_delta(&OrderBookDelta {
            instrument_id: "X".to_string(), timestamp_ns: 2000,
            action: BookAction::Add, side: OrderSide::Sell, price: 0.50, size: 50.0, flags: 0,
        });

        // Should now have 50 available again
        let (o2, f2) = engine.submit_order("X", OrderSide::Buy, 0.50, 50.0, TimeInForce::IOC, false, 3000);
        assert_eq!(o2.status, OrderStatus::Filled);
        assert_eq!(f2.unwrap().quantity, 50.0);
    }
}
