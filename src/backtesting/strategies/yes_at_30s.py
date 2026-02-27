"""Buy YES at 30–37 cents — exploits the calibration sweet spot.

Analysis insight: At YES prices 30–37 cents, the actual resolution probability
is ~2–2.5 percentage points ABOVE the implied probability across 1M+ trades.
E.g., at 32 cents, YES resolves 34.38% of the time vs implied 32%.

Transaction cost model (resting limit orders, Kalshi):
  - Fill at trade price (maker fill — no bid-ask spread cost)
  - Commission: 1% of notional (Kalshi flat fee)
  - Slippage: 0 (passive side of the book)

At 32¢ with 2.4 pp edge and 1% commission:
  EV = 0.344 × $1.00 - 0.32 × 1.01 = +$0.021 per contract → ~+6.5% per trade.

Position sizing: fixed notional per market so that dollar risk is uniform
across price levels (vs fixed contract count which would create larger dollar
risk at higher prices).

Hypothesis: POSITIVE return (~+5 to +15%) net of commission.

Engine validation role: This should be the best-performing strategy.
"""

from src.backtesting.models import MarketInfo, Side, TradeEvent
from src.backtesting.strategy import Strategy


class YesAt30sStrategy(Strategy):
    def __init__(
        self,
        low: float = 0.30,
        high: float = 0.37,
        notional: float = 100.0,
        initial_cash: float = 10_000.0,
    ):
        super().__init__(
            name="yes_at_30s",
            description=f"Buy YES at {low:.0%}–{high:.0%} (~+2pp edge, fixed ${notional:.0f} notional).",
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
