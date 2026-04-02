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
    ///
    /// Binary option (fully collateralized) cash flows:
    ///   BUY  opening long:  balance -= price * qty       (pay for YES contract)
    ///   BUY  closing short: balance += (1-price) * qty   (NO margin returned)
    ///   SELL closing long:  balance += price * qty        (sell YES contract)
    ///   SELL opening short: balance -= (1-price) * qty   (post NO margin)
    ///   Fees always deducted.
    pub fn process_fill(
        &mut self,
        instrument_id: &str,
        side: &OrderSide,
        price: f64,
        qty: f64,
        fee: f64,
    ) {
        // Determine closing vs opening quantities BEFORE updating position
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

        let (close_qty, open_qty) = if is_reducing {
            let close = qty.min(old_qty.abs());
            (close, qty - close)
        } else {
            (0.0, qty)
        };

        // Update cash balance using binary-option collateral model
        match side {
            OrderSide::Buy => {
                // Closing short: margin (1-price) returned per contract
                // Opening long: pay price per contract
                self.balance += close_qty * (1.0 - price) - open_qty * price - fee;
            }
            OrderSide::Sell => {
                // Closing long: receive price per contract
                // Opening short: post (1-price) margin per contract
                self.balance += close_qty * price - open_qty * (1.0 - price) - fee;
            }
        }

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
        // Sell to open short: balance -= (1-0.60)*10 = -4.0 (post NO margin)
        acct.process_fill("X", &OrderSide::Sell, 0.60, 10.0, 0.0);
        let pos = acct.get_position("X");
        assert_eq!(pos.signed_qty, -10.0);
        assert!((pos.avg_entry_price - 0.60).abs() < 1e-9);
        assert!((acct.balance - 9996.0).abs() < 1e-9); // 10000 - 4.0
    }

    #[test]
    fn test_close_short_pnl() {
        let mut acct = CashAccount::new(10_000.0);
        // Sell to open short @ 0.60: balance -= (1-0.60)*10 = -4.0
        acct.process_fill("X", &OrderSide::Sell, 0.60, 10.0, 0.0);
        assert!((acct.balance - 9996.0).abs() < 1e-9);
        // Buy to close short @ 0.40: balance += (1-0.40)*10 = +6.0 (margin returned)
        acct.process_fill("X", &OrderSide::Buy, 0.40, 10.0, 0.0);
        let pos = acct.get_position("X");
        // pnl = (0.60 - 0.40) * 10 = 2.0
        assert!((pos.realized_pnl - 2.0).abs() < 1e-9);
        assert_eq!(pos.signed_qty, 0.0);
        // balance = 9996 + 6 = 10002 = 10000 + 2.0 pnl
        assert!((acct.balance - 10002.0).abs() < 1e-9);
    }

    #[test]
    fn test_fee_deducted() {
        let mut acct = CashAccount::new(10_000.0);
        // Buy to open long: balance -= price*qty + fee = 5 + 1 = 6
        acct.process_fill("X", &OrderSide::Buy, 0.50, 10.0, 1.0);
        assert!((acct.balance - 9994.0).abs() < 1e-9);

        // Sell to close long: balance += price*qty - fee = 5 - 1 = 4
        acct.process_fill("X", &OrderSide::Sell, 0.50, 10.0, 1.0);
        assert!((acct.balance - 9998.0).abs() < 1e-9);
    }

    #[test]
    fn test_short_open_close_balance() {
        // Full cycle: open short, close short, verify balance = starting + pnl - fees
        let mut acct = CashAccount::new(10_000.0);
        // Sell to open short @ 0.90: balance -= (1-0.90)*100 = -10
        acct.process_fill("X", &OrderSide::Sell, 0.90, 100.0, 1.0);
        assert!((acct.balance - 9989.0).abs() < 1e-9); // 10000 - 10 - 1
        // Buy to close short @ 0.80: balance += (1-0.80)*100 = +20
        acct.process_fill("X", &OrderSide::Buy, 0.80, 100.0, 1.0);
        // balance = 9989 + 20 - 1 = 10008
        // Expected: 10000 + (0.90-0.80)*100 pnl - 2 fees = 10000 + 10 - 2 = 10008
        assert!((acct.balance - 10008.0).abs() < 1e-9);
    }
}
