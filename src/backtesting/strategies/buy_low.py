"""Buy Low — buy YES when price < 20 cents (negative-EV control).

Analysis insight: At YES prices below 20 cents, the actual YES win rate is
~1–2 percentage points BELOW the implied probability (longshot bias inverts
the YES side). E.g., at 10 cents YES wins 8.91% vs implied 10%.

Transaction cost model (resting limit orders, Kalshi):
  - Fill at trade price (maker fill — no bid-ask spread cost)
  - Commission: 0.07 × P × (1 − P) per contract  (Kalshi taker-fee formula)
      At YES=10¢: fee = 0.07 × 0.10 × 0.90 = 0.63¢/contract (6.3% of notional)
      At YES=20¢: fee = 0.07 × 0.20 × 0.80 = 1.12¢/contract (5.6% of notional)
  - Slippage: 0 (resting limit orders fill at quoted price, no market impact)

Commission alone is sufficient to overcome the calibration shortfall:
  At YES=10¢: EV = 0.0891 × $1.00 - 0.10 - 0.0063 = -$0.017 per contract
  At YES=19¢: EV = 0.175  × $1.00 - 0.19 - 0.0106 = -$0.026 per contract

This is the wrong-direction strategy — it buys the side with NEGATIVE
calibration edge. Expected to lose money at every price level.

Hypothesis: NEGATIVE return (~-5 to -15%) net of commission.

Engine validation role: Should rank last. If it beats yes_at_30s, either
the engine is wrong or the market sample is unrepresentative of the
general-population calibration data (e.g., sports-market underdogs at these
prices may have different win rates than the all-market average).
"""

from src.backtesting.models import MarketInfo, Side, TradeEvent
from src.backtesting.strategy import Strategy


class BuyLowStrategy(Strategy):
    def __init__(
        self,
        threshold: float = 0.20,
        notional: float = 100.0,
        initial_cash: float = 10_000.0,
    ):
        super().__init__(
            name="buy_low",
            description=f"Buy YES when price < {threshold:.0%} (negative-EV control, wrong direction).",
            initial_cash=initial_cash,
        )
        self.threshold = threshold
        self.notional = notional
        self._ordered: set[str] = set()

    def on_trade(self, trade: TradeEvent) -> None:
        if trade.market_id in self._ordered:
            return
        if trade.yes_price < self.threshold:
            qty = max(1, int(self.notional / trade.yes_price))
            self.buy_yes(
                market_id=trade.market_id,
                price=trade.yes_price,
                quantity=qty,
            )
            self._ordered.add(trade.market_id)

    def on_market_resolve(self, market: MarketInfo, result: Side) -> None:
        self._ordered.discard(market.market_id)
