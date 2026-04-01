use std::collections::BTreeMap;

use crate::types::{BookAction, OrderBookDelta, OrderSide};

/// L2 order book backed by BTreeMaps.
/// Prices are stored as integer ticks (price * 10_000) for exact comparison.
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub bids: BTreeMap<i64, f64>, // tick → size
    pub asks: BTreeMap<i64, f64>, // tick → size
}

fn price_to_tick(price: f64) -> i64 {
    (price * 10_000.0).round() as i64
}

fn tick_to_price(tick: i64) -> f64 {
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

    /// Best (lowest) ask: returns (price, size).
    pub fn best_ask(&self) -> Option<(f64, f64)> {
        self.asks
            .iter()
            .next()
            .map(|(&tick, &size)| (tick_to_price(tick), size))
    }

    /// Consume up to `qty` from the best ask level. Returns quantity actually consumed.
    pub fn consume_ask(&mut self, qty: f64) -> f64 {
        let tick = match self.asks.iter().next() {
            Some((&t, _)) => t,
            None => return 0.0,
        };
        let avail = self.asks.get_mut(&tick).unwrap();
        if qty >= *avail {
            let consumed = *avail;
            self.asks.remove(&tick);
            consumed
        } else {
            *avail -= qty;
            qty
        }
    }

    /// Consume up to `qty` from the best bid level. Returns quantity actually consumed.
    pub fn consume_bid(&mut self, qty: f64) -> f64 {
        let tick = match self.bids.iter().next_back() {
            Some((&t, _)) => t,
            None => return 0.0,
        };
        let avail = self.bids.get_mut(&tick).unwrap();
        if qty >= *avail {
            let consumed = *avail;
            self.bids.remove(&tick);
            consumed
        } else {
            *avail -= qty;
            qty
        }
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
    fn test_consume_ask() {
        let mut book = OrderBook::new();
        book.apply_raw(&BookAction::Add, &OrderSide::Sell, 0.55, 10.0);

        // Partial consume
        let consumed = book.consume_ask(3.0);
        assert_eq!(consumed, 3.0);
        let (_, size) = book.best_ask().unwrap();
        assert_eq!(size, 7.0);

        // Consume remaining
        let consumed = book.consume_ask(10.0);
        assert_eq!(consumed, 7.0);
        assert!(book.best_ask().is_none());
    }

    #[test]
    fn test_consume_bid() {
        let mut book = OrderBook::new();
        book.apply_raw(&BookAction::Add, &OrderSide::Buy, 0.45, 5.0);

        // Consume more than available
        let consumed = book.consume_bid(10.0);
        assert_eq!(consumed, 5.0);
        assert!(book.best_bid().is_none());
    }
}
