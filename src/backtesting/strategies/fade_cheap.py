"""Fade Cheap YES — buy NO when YES is priced 5–15 cents.

Analysis insight: At YES prices 5–15 cents, the actual YES win rate is
~1 percentage point BELOW the implied probability (longshot bias). E.g.,
at 10 cents YES wins 8.91% vs implied 10%. So NO has a genuine edge.

Transaction cost model (resting limit orders, Kalshi):
  - Fill at trade price (maker fill — no bid-ask spread cost)
  - Commission: 1% of notional (Kalshi flat fee)
  - Slippage: 0 (passive side of the book)

With zero slippage, the 1 pp NO edge comfortably clears 1% commission:
  At YES=10¢, NO=90¢:
    EV = 0.9109 × $1.00 - 0.90 × 1.01 = +$0.0019 per contract → slightly positive
  At YES=15¢, NO=85¢:
    EV = 0.870 × $1.00 - 0.85 × 1.01 = +$0.0115 per contract → clearly positive

Note: earlier analysis predicted a NEGATIVE return because it included
taker-style slippage (~3× amplified at extreme prices). With correct maker
fill modeling (resting limit orders fill at trade price), the genuine 1 pp
calibration edge survives costs.

Hypothesis: SMALL POSITIVE return (~+1 to +5%) net of commission.

Engine validation role: Should sit between yes_at_30s and coin_flip.
"""

from src.backtesting.models import MarketInfo, Side, TradeEvent
from src.backtesting.strategy import Strategy


class FadeCheapStrategy(Strategy):
    def __init__(
        self,
        low: float = 0.05,
        high: float = 0.15,
        notional: float = 100.0,
        initial_cash: float = 10_000.0,
    ):
        super().__init__(
            name="fade_cheap",
            description=f"Buy NO when YES ≤ {high:.0%} (longshot bias ~+1pp edge for NO).",
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
            qty = max(1, int(self.notional / trade.no_price))
            self.buy_no(
                market_id=trade.market_id,
                price=trade.no_price,
                quantity=qty,
            )
            self._ordered.add(trade.market_id)

    def on_market_resolve(self, market: MarketInfo, result: Side) -> None:
        self._ordered.discard(market.market_id)
