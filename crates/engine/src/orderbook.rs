use std::collections::BTreeMap;

use crate::types::{BookAction, OrderBookDelta, OrderSide};

/// L2 order book backed by BTreeMaps.
/// Prices are stored as integer ticks (price * 10_000) for exact comparison.
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub bids: BTreeMap<i64, f64>, // tick → size
    pub asks: BTreeMap<i64, f64>, // tick → size
}

pub fn price_to_tick(price: f64) -> i64 {
    (price * 10_000.0).round() as i64
}

pub fn tick_to_price(tick: i64) -> f64 {
    tick as f64 / 10_000.0
}

impl OrderBook {
    pub fn new() -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
        }
    }

    /// Apply a raw action to the book.
    pub fn apply_raw(&mut self, action: &BookAction, side: &OrderSide, price: f64, size: f64) {
        match action {
            BookAction::Clear => {
                self.bids.clear();
                self.asks.clear();
            }
            BookAction::Add | BookAction::Update => {
                let tick = price_to_tick(price);
                let book = match side {
                    OrderSide::Buy => &mut self.bids,
                    OrderSide::Sell => &mut self.asks,
                };
                if size == 0.0 {
                    book.remove(&tick);
                } else {
                    book.insert(tick, size);
                }
            }
            BookAction::Delete => {
                let tick = price_to_tick(price);
                let book = match side {
                    OrderSide::Buy => &mut self.bids,
                    OrderSide::Sell => &mut self.asks,
                };
                book.remove(&tick);
            }
        }
    }

    /// Apply an OrderBookDelta.
    pub fn apply(&mut self, delta: &OrderBookDelta) {
        self.apply_raw(&delta.action, &delta.side, delta.price, delta.size);
    }

    /// Best (highest) bid: returns (price, size).
    pub fn best_bid(&self) -> Option<(f64, f64)> {
        self.bids
            .iter()
            .next_back()
            .map(|(&tick, &size)| (tick_to_price(tick), size))
    }

    /// Get depth (size) at a specific price level on a given side.
    pub fn depth_at(&self, side: &OrderSide, price: f64) -> f64 {
        let tick = price_to_tick(price);
        let book = match side {
            OrderSide::Buy => &self.bids,
            OrderSide::Sell => &self.asks,
        };
        book.get(&tick).copied().unwrap_or(0.0)
    }

    /// Best (lowest) ask: returns (price, size).
    pub fn best_ask(&self) -> Option<(f64, f64)> {
        self.asks
            .iter()
            .next()
            .map(|(&tick, &size)| (tick_to_price(tick), size))
    }

    /// Total available ask depth at prices ≤ limit_price (read-only, no consumption).
    pub fn available_ask_depth(&self, limit_price: f64) -> f64 {
        let limit_tick = price_to_tick(limit_price);
        let mut total = 0.0;
        for (&tick, &size) in self.asks.iter() {
            if tick > limit_tick {
                break;
            }
            total += size;
        }
        total
    }

    /// Total available bid depth at prices ≥ limit_price (read-only, no consumption).
    pub fn available_bid_depth(&self, limit_price: f64) -> f64 {
        let limit_tick = price_to_tick(limit_price);
        let mut total = 0.0;
        for (&tick, &size) in self.bids.iter().rev() {
            if tick < limit_tick {
                break;
            }
            total += size;
        }
        total
    }

    pub fn simulate_fills_ask(
        &self,
        limit_price: f64,
        max_qty: f64,
        consumed: &std::collections::HashMap<i64, f64>,
    ) -> Vec<(f64, f64)> {
        let limit_tick = price_to_tick(limit_price);
        let mut remaining = max_qty;
        let mut fills = Vec::new();
        for (&tick, &size) in self.asks.iter() {
            if tick > limit_tick || remaining <= 1e-9 {
                break;
            }
            let already_consumed = consumed.get(&tick).copied().unwrap_or(0.0);
            let available = (size - already_consumed).max(0.0);
            if available <= 1e-9 {
                continue;
            }
            let fill_qty = remaining.min(available);
            fills.push((tick_to_price(tick), fill_qty));
            remaining -= fill_qty;
        }
        fills
    }

    pub fn simulate_fills_bid(
        &self,
        limit_price: f64,
        max_qty: f64,
        consumed: &std::collections::HashMap<i64, f64>,
    ) -> Vec<(f64, f64)> {
        let limit_tick = price_to_tick(limit_price);
        let mut remaining = max_qty;
        let mut fills = Vec::new();
        for (&tick, &size) in self.bids.iter().rev() {
            if tick < limit_tick || remaining <= 1e-9 {
                break;
            }
            let already_consumed = consumed.get(&tick).copied().unwrap_or(0.0);
            let available = (size - already_consumed).max(0.0);
            if available <= 1e-9 {
                continue;
            }
            let fill_qty = remaining.min(available);
            fills.push((tick_to_price(tick), fill_qty));
            remaining -= fill_qty;
        }
        fills
    }

    /// Consume up to `qty` from ask levels at or below `limit_price`.
    /// Sweeps from best (lowest) ask upward. Returns total quantity consumed.
    pub fn consume_ask(&mut self, qty: f64, limit_price: f64) -> f64 {
        let limit_tick = price_to_tick(limit_price);
        let mut remaining = qty;
        let mut consumed = 0.0;
        let mut to_remove = Vec::new();

        for (&tick, size) in self.asks.iter_mut() {
            if tick > limit_tick || remaining <= 0.0 {
                break;
            }
            if remaining >= *size {
                consumed += *size;
                remaining -= *size;
                to_remove.push(tick);
            } else {
                consumed += remaining;
                *size -= remaining;
                remaining = 0.0;
            }
        }

        for tick in to_remove {
            self.asks.remove(&tick);
        }
        consumed
    }

    /// Consume up to `qty` from bid levels at or above `limit_price`.
    /// Sweeps from best (highest) bid downward. Returns total quantity consumed.
    pub fn consume_bid(&mut self, qty: f64, limit_price: f64) -> f64 {
        let limit_tick = price_to_tick(limit_price);
        let mut remaining = qty;
        let mut consumed = 0.0;
        let mut to_remove = Vec::new();

        for (&tick, size) in self.bids.iter_mut().rev() {
            if tick < limit_tick || remaining <= 0.0 {
                break;
            }
            if remaining >= *size {
                consumed += *size;
                remaining -= *size;
                to_remove.push(tick);
            } else {
                consumed += remaining;
                *size -= remaining;
                remaining = 0.0;
            }
        }

        for tick in to_remove {
            self.bids.remove(&tick);
        }
        consumed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_book() {
        let book = OrderBook::new();
        assert!(book.best_bid().is_none());
        assert!(book.best_ask().is_none());
    }

    #[test]
    fn test_add_levels() {
        let mut book = OrderBook::new();
        book.apply_raw(&BookAction::Add, &OrderSide::Buy, 0.45, 10.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Buy, 0.50, 5.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.55, 8.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.60, 3.0);

        let (bp, bs) = book.best_bid().unwrap();
        assert_eq!(bp, 0.50);
        assert_eq!(bs, 5.0);

        let (ap, as_) = book.best_ask().unwrap();
        assert_eq!(ap, 0.55);
        assert_eq!(as_, 8.0);
    }

    #[test]
    fn test_clear() {
        let mut book = OrderBook::new();
        book.apply_raw(&BookAction::Add, &OrderSide::Buy, 0.45, 10.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.55, 8.0);
        book.apply_raw(&BookAction::Clear, &OrderSide::Buy, 0.0, 0.0);
        assert!(book.best_bid().is_none());
        assert!(book.best_ask().is_none());
    }

    #[test]
    fn test_update_and_delete() {
        let mut book = OrderBook::new();
        book.apply_raw(&BookAction::Add, &OrderSide::Buy, 0.50, 10.0);

        // Update size
        book.apply_raw(&BookAction::Update, &OrderSide::Buy, 0.50, 20.0);
        let (_, bs) = book.best_bid().unwrap();
        assert_eq!(bs, 20.0);

        // Delete level
        book.apply_raw(&BookAction::Delete, &OrderSide::Buy, 0.50, 0.0);
        assert!(book.best_bid().is_none());
    }

    #[test]
    fn test_simulate_fills_asks() {
        let mut book = OrderBook::new();
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.50, 5.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.51, 20.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.52, 75.0);

        let consumed = std::collections::HashMap::new();
        let fills = book.simulate_fills_ask(0.52, 30.0, &consumed);
        assert_eq!(fills, vec![(0.50, 5.0), (0.51, 20.0), (0.52, 5.0)]);
    }

    #[test]
    fn test_simulate_fills_bids() {
        let mut book = OrderBook::new();
        book.apply_raw(&BookAction::Add, &OrderSide::Buy, 0.50, 10.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Buy, 0.49, 20.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Buy, 0.48, 30.0);

        let consumed = std::collections::HashMap::new();
        let fills = book.simulate_fills_bid(0.49, 25.0, &consumed);
        assert_eq!(fills, vec![(0.50, 10.0), (0.49, 15.0)]);
    }

    #[test]
    fn test_simulate_fills_with_consumption() {
        let mut book = OrderBook::new();
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.50, 10.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.51, 20.0);

        let mut consumed = std::collections::HashMap::new();
        consumed.insert(price_to_tick(0.50), 7.0);

        let fills = book.simulate_fills_ask(0.51, 10.0, &consumed);
        assert_eq!(fills, vec![(0.50, 3.0), (0.51, 7.0)]);
    }

    #[test]
    fn test_consume_ask() {
        let mut book = OrderBook::new();
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.55, 10.0);

        // Partial consume
        let consumed = book.consume_ask(3.0, 0.55);
        assert_eq!(consumed, 3.0);
        let (_, size) = book.best_ask().unwrap();
        assert_eq!(size, 7.0);

        // Consume remaining
        let consumed = book.consume_ask(10.0, 0.55);
        assert_eq!(consumed, 7.0);
        assert!(book.best_ask().is_none());
    }

    #[test]
    fn test_consume_ask_multilevel() {
        let mut book = OrderBook::new();
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.48, 3.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.49, 5.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.50, 10.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.51, 20.0);

        // Sweep asks up to 0.50 for 15 contracts
        let consumed = book.consume_ask(15.0, 0.50);
        // 3 @ 0.48 + 5 @ 0.49 + 7 @ 0.50 = 15
        assert_eq!(consumed, 15.0);
        // 0.50 level should have 3 remaining, 0.51 untouched
        let (ap, as_) = book.best_ask().unwrap();
        assert_eq!(ap, 0.50);
        assert_eq!(as_, 3.0);
    }

    #[test]
    fn test_consume_bid() {
        let mut book = OrderBook::new();
        book.apply_raw(&BookAction::Add, &OrderSide::Buy, 0.45, 5.0);

        // Consume more than available
        let consumed = book.consume_bid(10.0, 0.45);
        assert_eq!(consumed, 5.0);
        assert!(book.best_bid().is_none());
    }

    #[test]
    fn test_consume_bid_multilevel() {
        let mut book = OrderBook::new();
        book.apply_raw(&BookAction::Add, &OrderSide::Buy, 0.50, 10.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Buy, 0.49, 5.0);
        book.apply_raw(&BookAction::Add, &OrderSide::Buy, 0.48, 3.0);

        // Sweep bids down to 0.49 for 12 contracts
        let consumed = book.consume_bid(12.0, 0.49);
        // 10 @ 0.50 + 2 @ 0.49 = 12
        assert_eq!(consumed, 12.0);
        let (bp, bs) = book.best_bid().unwrap();
        assert_eq!(bp, 0.49);
        assert_eq!(bs, 3.0);
    }
}
