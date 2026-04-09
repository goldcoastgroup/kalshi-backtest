"""
Kalshi backtest strategy — batch-edge detection.

HybridFV: passive FV market maker + aggressive directional taker.
"""

from __future__ import annotations

import math

import prepare
from engine import FairValueData, Fill, OrderSide, OrderStatus, TimeInForce
from engine.strategy import Strategy


# ── Config ──────────────────────────────────────────────────────────

BASE_BANKROLL = 10_000.0

# Passive maker
HALF_SPREAD = 0.10
MAX_POSITION_BASE = 300
REQUOTE_PRICE_DELTA = 0.01
SKEW_PER_CONTRACT = 0.0015

# Aggressive directional
AGG_EDGE_THRESHOLD = 0.20
AGG_MAX_POSITION_BASE = 25
AGG_RAMP_BASE = 8000
UNWIND_THRESH_BASE = 22


# ── Strategy ────────────────────────────────────────────────────────

class HybridFV(Strategy):
    """Passive FV market maker + aggressive directional taker."""

    def __init__(self, instrument_id: str):
        super().__init__(instrument_id)
        self._bid_order_id: str | None = None
        self._ask_order_id: str | None = None
        self._bid_price: float = 0.0
        self._ask_price: float = 0.0
        self._last_fv: float | None = None
        self._last_gamma: float = 0.0
        self._last_theta: float = 0.0
        self._hours_left: float = 0.0
        # Batch tracking
        self._last_review_ts: int = 0
        self._batch_size: int = 0
        self._prev_total_reviews: int = 0

    def _batch_freshness(self, timestamp_ns: int) -> float:
        """0-1: how recently a batch completed."""
        if self._batch_size == 0 or self._last_review_ts == 0:
            return 0.0
        minutes_since = (timestamp_ns - self._last_review_ts) / (60 * 1_000_000_000)
        if minutes_since < 10:
            return 0.0  # still in batch
        freshness = max(0.0, 1.0 - (minutes_since - 10) / 60)
        return freshness * min(1.0, self._batch_size / 3.0)

    def on_data(self, data: FairValueData) -> None:
        self._last_fv = data.fv
        self._last_gamma = max(abs(data.gamma_pos), abs(data.gamma_neg))
        self._last_theta = data.theta
        self._hours_left = data.hours_left

        # Track review batches
        if data.total_reviews > self._prev_total_reviews:
            new_count = data.total_reviews - self._prev_total_reviews
            gap = (data.timestamp_ns - self._last_review_ts) / (60 * 1_000_000_000) if self._last_review_ts > 0 else 999
            self._batch_size = new_count if gap > 15 else self._batch_size + new_count
            self._last_review_ts = data.timestamp_ns
            self._prev_total_reviews = data.total_reviews

        if data.new_review or self._quotes_stale(data.timestamp_ns):
            self._requote(data.timestamp_ns)

        self._check_aggress(data.timestamp_ns)

    def on_book_update(self, instrument_id: str, timestamp_ns: int = 0) -> None:
        self._check_aggress(timestamp_ns)

    def on_fill(self, fill: Fill) -> None:
        self._check_aggress(fill.timestamp_ns)
        self._requote(fill.timestamp_ns)

    def _quotes_stale(self, timestamp_ns: int) -> bool:
        """True if resting quotes have drifted from ideal by >= REQUOTE_PRICE_DELTA."""
        fv = self._last_fv
        if fv is None:
            return False
        if self._bid_order_id is None and self._ask_order_id is None:
            return True
        qty = self.get_position().signed_qty
        skew = qty * SKEW_PER_CONTRACT
        batch_fresh = self._batch_freshness(timestamp_ns)
        spread = HALF_SPREAD + batch_fresh * 0.12 + self._last_gamma * 0.4
        ideal_bid = math.floor((fv - spread - skew) * 100) / 100
        ideal_ask = math.ceil((fv + spread - skew) * 100) / 100
        bid_drift = abs(ideal_bid - self._bid_price) if self._bid_order_id else 0
        ask_drift = abs(ideal_ask - self._ask_price) if self._ask_order_id else 0
        return max(bid_drift, ask_drift) >= REQUOTE_PRICE_DELTA

    def _scale(self, agg: bool = False) -> float:
        """Scale factor with power dampening for large balances."""
        raw = self.get_balance() / BASE_BANKROLL
        exp = 0.13 if agg else 0.15
        return raw if raw <= 1.0 else raw ** exp

    def _requote(self, timestamp_ns: int) -> None:
        fv = self._last_fv
        if fv is None:
            return

        pos = self.get_position()
        qty = pos.signed_qty
        s = self._scale()
        batch_fresh = self._batch_freshness(timestamp_ns)

        max_pos = max(1, int(MAX_POSITION_BASE * s))

        # Cancel maker quotes when gamma very low (FV is near-certain)
        if self._last_gamma < 0.01 and (fv > 0.85 or fv < 0.15):
            if self._bid_order_id is not None:
                self.cancel_order(self._bid_order_id)
                self._bid_order_id = None
            if self._ask_order_id is not None:
                self.cancel_order(self._ask_order_id)
                self._ask_order_id = None
            return

        skew = qty * SKEW_PER_CONTRACT
        spread = HALF_SPREAD + batch_fresh * 0.12 + self._last_gamma * 0.4

        bid_price = math.floor((fv - spread - skew) * 100) / 100
        ask_price = math.ceil((fv + spread - skew) * 100) / 100
        bid_price = max(0.01, min(0.99, bid_price))
        ask_price = max(0.01, min(0.99, ask_price))

        if bid_price >= ask_price:
            if self._bid_order_id is not None:
                self.cancel_order(self._bid_order_id)
                self._bid_order_id = None
            if self._ask_order_id is not None:
                self.cancel_order(self._ask_order_id)
                self._ask_order_id = None
            return

        best_ask = self.best_ask()
        best_bid = self.best_bid()

        # Bid side
        bid_qty = min(max_pos, max_pos - int(qty)) if qty < max_pos else 0
        can_bid = bid_qty >= 1 and (best_ask is None or bid_price < best_ask[0])
        if can_bid:
            if self._bid_order_id is not None and abs(bid_price - self._bid_price) <= 0.01:
                self.modify_order(self._bid_order_id, bid_qty)
            else:
                if self._bid_order_id is not None:
                    self.cancel_order(self._bid_order_id)
                    self._bid_order_id = None
                order, fill = self.submit_order(
                    OrderSide.Buy, bid_price, bid_qty,
                    time_in_force=TimeInForce.POST_ONLY,
                    timestamp_ns=timestamp_ns,
                )
                if order.status not in (OrderStatus.Rejected, OrderStatus.Canceled):
                    self._bid_order_id = order.id
                    self._bid_price = bid_price
        elif self._bid_order_id is not None:
            self.cancel_order(self._bid_order_id)
            self._bid_order_id = None

        # Ask side
        ask_qty = min(max_pos, max_pos + int(qty)) if qty > -max_pos else 0
        can_ask = ask_qty >= 1 and (best_bid is None or ask_price > best_bid[0])
        if can_ask:
            if self._ask_order_id is not None and abs(ask_price - self._ask_price) <= 0.01:
                self.modify_order(self._ask_order_id, ask_qty)
            else:
                if self._ask_order_id is not None:
                    self.cancel_order(self._ask_order_id)
                    self._ask_order_id = None
                order, fill = self.submit_order(
                    OrderSide.Sell, ask_price, ask_qty,
                    time_in_force=TimeInForce.POST_ONLY,
                    timestamp_ns=timestamp_ns,
                )
                if order.status not in (OrderStatus.Rejected, OrderStatus.Canceled):
                    self._ask_order_id = order.id
                    self._ask_price = ask_price
        elif self._ask_order_id is not None:
            self.cancel_order(self._ask_order_id)
            self._ask_order_id = None

    def _check_aggress(self, timestamp_ns: int) -> None:
        fv = self._last_fv
        if fv is None:
            return

        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return

        bid_price, bid_size = bid
        ask_price, ask_size = ask

        pos = self.get_position()
        qty = pos.signed_qty
        s = self._scale(agg=True)

        # Unwind positions when FV crosses the book price
        unwind_thresh = int(UNWIND_THRESH_BASE * s)
        if qty > unwind_thresh and fv < bid_price:
            size = min(int(bid_size), int(qty))
            if size >= 1:
                self.submit_order(
                    OrderSide.Sell, bid_price, size,
                    time_in_force=TimeInForce.IOC,
                    timestamp_ns=timestamp_ns,
                )
        elif qty < -unwind_thresh and fv > ask_price:
            size = min(int(ask_size), int(-qty))
            if size >= 1:
                self.submit_order(
                    OrderSide.Buy, ask_price, size,
                    time_in_force=TimeInForce.IOC,
                    timestamp_ns=timestamp_ns,
                )

        # Spread filter
        extremeness = 1.0 - 4.0 * fv * (1.0 - fv)
        max_spread = 0.05 + 0.25 * extremeness
        if ask_price - bid_price > max_spread:
            return

        # Aggressive directional: gamma-gated (low gamma = confident FV)
        fee_adj = 0.07 * fv * (1.0 - fv)
        gamma_dampen = max(0.05, 1.0 - self._last_gamma * 10.0)
        # Gamma confidence: sharp transition in [0.006, 0.009]
        gamma_conf = max(0.0, min(1.0, (0.009 - self._last_gamma) / 0.003))

        # Threshold: gamma-confident → lower threshold
        agg_thresh = AGG_EDGE_THRESHOLD + (fee_adj + self._last_gamma * 0.6) * (1.0 - gamma_conf)

        # Position limit: gamma confidence drives the ramp
        time_scale = min(1.0, self._hours_left / 2.0) if self._hours_left > 0 else 1.0
        agg_max = max(1, int((AGG_MAX_POSITION_BASE + AGG_RAMP_BASE * gamma_conf) * s * gamma_dampen * time_scale))

        # Confidence and sizing
        extremeness_boost = 1.0 - 4.0 * fv * (1.0 - fv)
        confidence_mult = 1.0 + gamma_conf * 11.0 * (1.0 + extremeness_boost * 0.2)
        base_min = (3 + 47 * gamma_conf) * s
        base_range = (10 + 110 * gamma_conf) * s

        free_bal = self.get_free_balance()

        # Theta confirmation: reduce threshold when theta aligns with direction
        theta_buy_conf = max(0.0, min(1.0, self._last_theta * 20.0))   # positive theta → FV trending up
        theta_sell_conf = max(0.0, min(1.0, -self._last_theta * 20.0))  # negative theta → FV trending down

        if fv > ask_price + agg_thresh * (1.0 - 0.1 * theta_buy_conf) and qty < agg_max:
            edge = fv - ask_price
            scaled = int((base_min + base_range * min(1.0, (edge - agg_thresh) / 0.04)) * confidence_mult)
            scaled = max(1, int(scaled * (1.0 - qty / agg_max) ** 0.02))
            sweep_price = min(0.99, ask_price + 0.02)
            bal_cap = int(free_bal / sweep_price) if sweep_price > 0 else 0
            size = min(scaled, agg_max - int(qty), bal_cap)
            if size >= 1:
                self.submit_order(
                    OrderSide.Buy, sweep_price, size,
                    time_in_force=TimeInForce.IOC,
                    timestamp_ns=timestamp_ns,
                )

        if fv < bid_price - agg_thresh * (1.0 - 0.1 * theta_sell_conf) and qty > -agg_max:
            edge = bid_price - fv
            scaled = int((base_min + base_range * min(1.0, (edge - agg_thresh) / 0.04)) * confidence_mult)
            scaled = max(1, int(scaled * (1.0 - (-qty) / agg_max) ** 0.02))
            sweep_price = max(0.01, bid_price - 0.02)
            bal_cap = int(free_bal / (1.0 - sweep_price)) if sweep_price < 1.0 else 0
            size = min(scaled, agg_max + int(qty), bal_cap)
            if size >= 1:
                self.submit_order(
                    OrderSide.Sell, sweep_price, size,
                    time_in_force=TimeInForce.IOC,
                    timestamp_ns=timestamp_ns,
                )


# ── Run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prepare.run_backtest(strategy_factory=HybridFV)
