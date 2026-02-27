"""Coin Flip strategy — buy YES near 50 cents (calibration control).

Analysis insight: At YES price = 50 cents, the actual win rate is exactly
50.00% across 1.8M+ trades. This is a perfectly calibrated price point —
zero edge.

Transaction cost model (resting limit orders, Kalshi):
  - Fill at trade price (maker fill — no bid-ask spread cost)
  - Commission: 1% of notional (Kalshi flat fee)
  - Slippage: 0 (passive side of the book)

At 50¢ with 0 pp edge and 1% commission:
  EV = 0.50 × $1.00 - 0.50 × 1.01 = -$0.005 per contract → ~-1% per trade.

This strategy has ZERO calibration edge and exists purely to measure the
commission drag. Any result significantly worse than ~-1% per trade points to
an engine accounting error; a result significantly better than -1% points to
look-ahead bias.

Hypothesis: SLIGHT NEGATIVE return (~-1 to -3%) — commission drag only.

Engine validation role: Neutral cost-only baseline.
"""

from src.backtesting.models import MarketInfo, Side, TradeEvent
from src.backtesting.strategy import Strategy


class CoinFlipStrategy(Strategy):
    def __init__(
        self,
        low: float = 0.48,
        high: float = 0.52,
        notional: float = 100.0,
        initial_cash: float = 10_000.0,
    ):
        super().__init__(
            name="coin_flip",
            description=f"Buy YES at {low:.0%}–{high:.0%} (no-edge baseline, commission drag only).",
            initial_cash=initial_cash,
        )
        self.low = low
        self.high = high
        self.notional = notional
        self._ordered: set[str] = set()

    def on_trade(self, trade: TradeEvent) -> None:
        if trade.market_id in self._ordered:
            return
        if self.low <= trade.yes_price <= self.high:
            qty = max(1, int(self.notional / trade.yes_price))
            self.buy_yes(
                market_id=trade.market_id,
                price=trade.yes_price,
                quantity=qty,
            )
            self._ordered.add(trade.market_id)

    def on_market_resolve(self, market: MarketInfo, result: Side) -> None:
        self._ordered.discard(market.market_id)
