/// Order management and fill matching.
///
/// Key difference from Python version: orders are indexed by market_id
/// for O(1) lookup instead of scanning all pending orders per trade.
///
/// Fill model: limit orders fill at the **trade price** of the matching trade.
/// Because on_trade fires before check_fills in the engine hot loop, a strategy
/// that places an order in on_trade will fill at the same trade's price — exactly
/// "buy at the price you see."
///
/// Commission model — two variants selected by `flat_commission`:
///   false (Kalshi): fee = commission_rate × P × (1 − P) × qty
///     default rate 0.07; max 1.75¢/contract at P=0.50.
///   true  (Polymarket): fee = commission_rate × P × qty  (flat % of premium)
///     default rate 0.001 (0.10%); negligible at low prices.
/// `apply_market_impact` is kept for future support of aggressive / market-order
/// strategies.

use std::collections::HashMap;

use crate::models::{Fill, Order, OrderAction, OrderStatus, Side, Trade};

pub struct Broker {
    /// Orders indexed by market_id for fast lookup.
    pending: HashMap<String, Vec<Order>>,
    commission_rate: f64,
    /// true  → flat fee: commission_rate × P × qty           (Polymarket)
    /// false → parabolic: commission_rate × P × (1−P) × qty  (Kalshi)
    flat_commission: bool,
    liquidity_cap: bool,
    next_id: u64,
    /// Exponential moving average of trade size per market.
    /// Retained for potential future taker-order market-impact model.
    ema_trade_size: HashMap<String, f64>,
    ema_decay: f64,
}

/// Check if an order should fill against a trade. Returns fill price if matched.
///
/// Pure price-condition matching: a resting limit order fills whenever the trade
/// price satisfies the order's limit. This is the correct model for backtesting
/// calibration strategies with discrete trade data — the strategy fires on_trade,
/// immediately places a limit order, and that order fills on the same trade event
/// (because on_trade fires before check_fills). The fill price therefore equals
/// the trade price that triggered the strategy, giving clean "buy at the price
/// you see" semantics with no look-ahead or selection bias.
///
///   YES bid fills when trade.yes_price <= order.price  (price is in our range)
///   YES ask fills when trade.yes_price >= order.price
///   NO  bid fills when trade.no_price  <= order.price
///   NO  ask fills when trade.no_price  >= order.price
fn match_order(order: &Order, trade: &Trade) -> Option<f64> {
    match (order.action, order.side) {
        (OrderAction::Buy, Side::Yes) => {
            if trade.yes_price <= order.price {
                Some(trade.yes_price)
            } else {
                None
            }
        }
        (OrderAction::Sell, Side::Yes) => {
            if trade.yes_price >= order.price {
                Some(trade.yes_price)
            } else {
                None
            }
        }
        (OrderAction::Buy, Side::No) => {
            if trade.no_price <= order.price {
                Some(trade.no_price)
            } else {
                None
            }
        }
        (OrderAction::Sell, Side::No) => {
            if trade.no_price >= order.price {
                Some(trade.no_price)
            } else {
                None
            }
        }
    }
}

/// Apply market impact to a fill price (for aggressive / market-order strategies).
///
/// Combines two effects:
///   1. Price-proportional spread: spread widens at extreme prices (p near 0 or 1).
///      At p=0.50 the multiplier is 1×; at p=0.15 it is ~2×; at p=0.05 it is ~5×.
///   2. Square-root size impact: large orders relative to the market's average
///      trade size pay more (standard Almgren-Chriss / Kyle-lambda approach).
///
/// NOT called for resting limit (maker) orders — those fill at the trade price.
/// Reserved for future market-order execution paths.
#[allow(dead_code)]
pub fn apply_market_impact(
    base_slippage: f64,
    price: f64,
    action: OrderAction,
    order_qty: f64,
    avg_trade_size: f64,
) -> f64 {
    if base_slippage == 0.0 {
        return price;
    }
    let variance = (price * (1.0 - price)).max(0.01);
    let spread_factor = (0.25 / variance).max(1.0);
    let size_ratio = order_qty / avg_trade_size.max(0.01);
    let size_factor = size_ratio.sqrt().max(1.0);
    let impact = base_slippage * spread_factor * size_factor;
    match action {
        OrderAction::Buy => (price + impact).min(0.99),
        OrderAction::Sell => (price - impact).max(0.01),
    }
}

impl Broker {
    pub fn new(commission_rate: f64, flat_commission: bool, liquidity_cap: bool, ema_decay: f64) -> Self {
        Self {
            pending: HashMap::new(),
            commission_rate,
            flat_commission,
            liquidity_cap,
            next_id: 1,
            ema_trade_size: HashMap::new(),
            ema_decay,
        }
    }

    /// Update the EMA of trade size for a market. Call on every trade before check_fills.
    /// Retained for potential future taker-order market-impact calculations.
    pub fn update_trade_size(&mut self, market_id: &str, trade_qty: f64) {
        let entry = self.ema_trade_size.entry(market_id.to_string()).or_insert(trade_qty);
        *entry = *entry * (1.0 - self.ema_decay) + trade_qty * self.ema_decay;
    }

    pub fn place_order(
        &mut self,
        market_id: &str,
        action: OrderAction,
        side: Side,
        price: f64,
        quantity: f64,
        timestamp: f64,
    ) -> Order {
        let order = Order {
            order_id: self.next_id.to_string(),
            market_id: market_id.to_string(),
            action,
            side,
            price,
            quantity,
            status: OrderStatus::Pending,
            created_at: timestamp,
            filled_at: None,
            fill_price: None,
            filled_quantity: 0.0,
        };
        self.next_id += 1;
        self.pending
            .entry(market_id.to_string())
            .or_default()
            .push(order.clone());
        order
    }

    pub fn cancel_order(&mut self, order_id: &str) -> bool {
        for orders in self.pending.values_mut() {
            if let Some(pos) = orders.iter().position(|o| o.order_id == order_id) {
                orders.remove(pos);
                return true;
            }
        }
        false
    }

    pub fn cancel_all(&mut self, market_id: Option<&str>) -> usize {
        match market_id {
            Some(mid) => {
                if let Some(orders) = self.pending.remove(mid) {
                    orders.len()
                } else {
                    0
                }
            }
            None => {
                let count: usize = self.pending.values().map(|v| v.len()).sum();
                self.pending.clear();
                count
            }
        }
    }

    /// Check all pending orders for this trade's market. O(orders_for_market).
    ///
    /// Resting limit orders fill at the actual trade price — no market impact is
    /// added. Commission formula selected by `flat_commission`:
    ///   false → commission_rate × P × (1−P) × qty  (Kalshi)
    ///   true  → commission_rate × P × qty           (Polymarket flat %)
    pub fn check_fills(&mut self, trade: &Trade, available_cash: f64) -> Vec<Fill> {
        let commission_rate = self.commission_rate;
        let flat_commission = self.flat_commission;
        let liquidity_cap = self.liquidity_cap;

        let orders = match self.pending.get_mut(&trade.market_id) {
            Some(orders) if !orders.is_empty() => orders,
            _ => return vec![],
        };

        let mut fills: Vec<Fill> = Vec::new();
        let mut cash = available_cash;
        let mut remaining_liq = if liquidity_cap {
            trade.quantity
        } else {
            f64::INFINITY
        };
        let mut to_remove: Vec<usize> = Vec::new();

        for (idx, order) in orders.iter_mut().enumerate() {
            // fill_price = actual trade price (maker fill — no additional impact).
            let fill_price = match match_order(order, trade) {
                Some(p) => p,
                None => continue,
            };

            let mut fill_qty = if liquidity_cap {
                order.quantity.min(remaining_liq)
            } else {
                order.quantity
            };
            if fill_qty <= 0.0 {
                continue;
            }

            let mut cost = fill_price * fill_qty;
            let fee_factor = if flat_commission { 1.0 } else { 1.0 - fill_price };
            let mut commission = commission_rate * fill_price * fee_factor * fill_qty;

            if order.action == OrderAction::Buy && cost + commission > cash {
                if liquidity_cap {
                    let max_qty = cash / (fill_price * (1.0 + commission_rate * fee_factor));
                    fill_qty = fill_qty.min(max_qty);
                    if fill_qty < 1.0 {
                        continue;
                    }
                    fill_qty = fill_qty.floor();
                    cost = fill_price * fill_qty;
                    commission = commission_rate * fill_price * fee_factor * fill_qty;
                } else {
                    continue;
                }
            }

            if order.action == OrderAction::Buy {
                cash -= cost + commission;
            }
            remaining_liq -= fill_qty;

            fills.push(Fill {
                order_id: order.order_id.clone(),
                market_id: order.market_id.clone(),
                action: order.action,
                side: order.side,
                price: fill_price,
                quantity: fill_qty,
                timestamp: trade.timestamp,
                commission,
            });

            order.status = OrderStatus::Filled;
            order.filled_at = Some(trade.timestamp);
            order.fill_price = Some(fill_price);
            order.filled_quantity = fill_qty;
            to_remove.push(idx);
        }

        // Remove filled orders (reverse order to preserve indices)
        for &idx in to_remove.iter().rev() {
            orders.remove(idx);
        }

        fills
    }

    /// Return references to all pending orders (flattened).
    pub fn all_pending(&self) -> Vec<&Order> {
        self.pending.values().flat_map(|v| v.iter()).collect()
    }
}
