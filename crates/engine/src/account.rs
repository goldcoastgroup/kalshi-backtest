use std::collections::HashMap;

use crate::types::{OrderSide, Position};

#[derive(Debug, Clone)]
pub struct CashAccount {
    pub starting_balance: f64,
    pub balance: f64,
    pub positions: HashMap<String, Position>,
}

impl CashAccount {
    pub fn new(starting_balance: f64) -> Self {
        Self {
            starting_balance,
            balance: starting_balance,
            positions: HashMap::new(),
        }
    }

    /// Process a fill: update cash balance and position.
    pub fn process_fill(
        &mut self,
        instrument_id: &str,
        side: &OrderSide,
        price: f64,
        qty: f64,
        fee: f64,
    ) {
        // Update cash balance
        match side {
            OrderSide::Buy => {
                self.balance -= price * qty + fee;
            }
            OrderSide::Sell => {
                self.balance += price * qty - fee;
            }
        }

        // Update position
        let pos = self
            .positions
            .entry(instrument_id.to_string())
            .or_insert_with(|| Position::new(instrument_id.to_string()));

        let fill_signed = match side {
            OrderSide::Buy => qty,
            OrderSide::Sell => -qty,
        };

        let old_qty = pos.signed_qty;
        let new_qty = old_qty + fill_signed;

        // Determine if this fill is closing, opening, or flipping
        let is_reducing = (old_qty > 0.0 && fill_signed < 0.0)
            || (old_qty < 0.0 && fill_signed > 0.0);

        if is_reducing {
            let close_qty = qty.min(old_qty.abs());

            // Compute realized PnL on the closed portion
            let pnl = if old_qty > 0.0 {
                // Closing long: sell - avg_entry
                (price - pos.avg_entry_price) * close_qty
            } else {
                // Closing short: avg_entry - buy
                (pos.avg_entry_price - price) * close_qty
            };
            pos.realized_pnl += pnl;

            // If position flips sides, set avg_entry to fill price
            if (old_qty > 0.0 && new_qty < 0.0) || (old_qty < 0.0 && new_qty > 0.0) {
                pos.avg_entry_price = price;
                pos.entry_count += 1;
            }
            // If fully closed (new_qty == 0), avg_entry stays (doesn't matter)
        } else {
            // Increasing or opening position: weighted average entry
            let old_abs = old_qty.abs();
            let new_abs = new_qty.abs();
            if new_abs > 0.0 {
                pos.avg_entry_price =
                    (pos.avg_entry_price * old_abs + price * qty) / new_abs;
            }
            pos.entry_count += 1;
        }

        pos.signed_qty = new_qty;
    }

    pub fn get_position(&self, instrument_id: &str) -> Position {
        self.positions
            .get(instrument_id)
            .cloned()
            .unwrap_or_else(|| Position::new(instrument_id.to_string()))
    }

    pub fn all_positions(&self) -> Vec<Position> {
        self.positions.values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_balance() {
        let acct = CashAccount::new(10_000.0);
        assert_eq!(acct.balance, 10_000.0);
        assert_eq!(acct.starting_balance, 10_000.0);
    }

    #[test]
    fn test_buy_fill_deducts_balance() {
        let mut acct = CashAccount::new(10_000.0);
        // buy 10 @ 0.50, fee 0.175
        acct.process_fill("X", &OrderSide::Buy, 0.50, 10.0, 0.175);
        // balance = 10000 - (0.50 * 10 + 0.175) = 10000 - 5.175 = 9994.825
        assert!((acct.balance - 9994.825).abs() < 1e-9);
    }

    #[test]
    fn test_sell_fill_adds_balance() {
        let mut acct = CashAccount::new(10_000.0);
        acct.process_fill("X", &OrderSide::Buy, 0.50, 10.0, 0.0);
        acct.process_fill("X", &OrderSide::Sell, 0.60, 10.0, 0.0);
        // balance = 10000 - 5.0 + 6.0 = 10001.0
        assert!((acct.balance - 10001.0).abs() < 1e-9);
    }

    #[test]
    fn test_position_tracking_long() {
        let mut acct = CashAccount::new(10_000.0);
        acct.process_fill("X", &OrderSide::Buy, 0.40, 10.0, 0.0);
        let pos = acct.get_position("X");
        assert_eq!(pos.signed_qty, 10.0);
        assert!((pos.avg_entry_price - 0.40).abs() < 1e-9);
    }

    #[test]
    fn test_realized_pnl_on_close() {
        let mut acct = CashAccount::new(10_000.0);
        acct.process_fill("X", &OrderSide::Buy, 0.40, 10.0, 0.0);
        acct.process_fill("X", &OrderSide::Sell, 0.60, 10.0, 0.0);
        let pos = acct.get_position("X");
        // pnl = (0.60 - 0.40) * 10 = 2.0
        assert!((pos.realized_pnl - 2.0).abs() < 1e-9);
        assert_eq!(pos.signed_qty, 0.0);
    }

    #[test]
    fn test_partial_close_pnl() {
        let mut acct = CashAccount::new(10_000.0);
        acct.process_fill("X", &OrderSide::Buy, 0.40, 10.0, 0.0);
        acct.process_fill("X", &OrderSide::Sell, 0.60, 5.0, 0.0);
        let pos = acct.get_position("X");
        // pnl = (0.60 - 0.40) * 5 = 1.0
        assert!((pos.realized_pnl - 1.0).abs() < 1e-9);
        assert_eq!(pos.signed_qty, 5.0);
    }

    #[test]
    fn test_short_position() {
        let mut acct = CashAccount::new(10_000.0);
        acct.process_fill("X", &OrderSide::Sell, 0.60, 10.0, 0.0);
        let pos = acct.get_position("X");
        assert_eq!(pos.signed_qty, -10.0);
        assert!((pos.avg_entry_price - 0.60).abs() < 1e-9);
    }

    #[test]
    fn test_close_short_pnl() {
        let mut acct = CashAccount::new(10_000.0);
        acct.process_fill("X", &OrderSide::Sell, 0.60, 10.0, 0.0);
        acct.process_fill("X", &OrderSide::Buy, 0.40, 10.0, 0.0);
        let pos = acct.get_position("X");
        // pnl = (0.60 - 0.40) * 10 = 2.0
        assert!((pos.realized_pnl - 2.0).abs() < 1e-9);
        assert_eq!(pos.signed_qty, 0.0);
    }

    #[test]
    fn test_fee_deducted() {
        let mut acct = CashAccount::new(10_000.0);
        acct.process_fill("X", &OrderSide::Buy, 0.50, 10.0, 1.0);
        // balance = 10000 - (5.0 + 1.0) = 9994.0
        assert!((acct.balance - 9994.0).abs() < 1e-9);

        acct.process_fill("X", &OrderSide::Sell, 0.50, 10.0, 1.0);
        // balance = 9994.0 + (5.0 - 1.0) = 9998.0
        assert!((acct.balance - 9998.0).abs() < 1e-9);
    }
}
