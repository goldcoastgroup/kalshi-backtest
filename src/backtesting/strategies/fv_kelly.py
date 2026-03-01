"""FV-Kelly — size bets with Kelly criterion using model fair values.

For each market, looks up the model's fair value (fv_T) from pre-computed
per-minute FV timeseries parquets. When the observed trade price deviates
from fair value, places a Kelly-sized limit order:

  YES underpriced (p_yes < fv):  buy YES,  f* = (fv - p_yes) / (1 - p_yes)
  NO  underpriced (p_no < 1-fv): buy NO,   f* = ((1-fv) - p_no) / (1-p_no)

The two conditions are mutually exclusive.

Gamma stress-testing (when gamma_stress > 0):
  Before entering, the fair value is stressed by one review's worth of adverse
  gamma to ask "does this trade still have edge if the next review goes against
  me?"

  YES bet: fv_stressed = fv_T + gamma_neg * gamma_stress  (gamma_neg ≤ 0)
  NO  bet: no_fv_stressed = (1-fv_T) - gamma_pos * gamma_stress  (gamma_pos ≥ 0)

  The stressed fv must still clear min_edge against the market price.
  Set gamma_stress=0.0 to disable and revert to plain fv_T behaviour.

Per-event notional cap (max_event_fraction):
  Multiple strike markets for the same film (e.g. KXRTMICKEY17-82 and
  KXRTMICKEY17-85) are correlated — if the model is wrong about the score
  all strikes lose together.  Total notional committed across all active
  positions for one event is capped at max_event_fraction × initial_cash.

  The event ticker is derived as market_id.rsplit('-', 1)[0], so
  KXRTMICKEY17-82 and KXRTMICKEY17-85 share event KXRTMICKEY17.
  Set max_event_fraction=1.0 to disable.

Leverage controls:
  1. One pending order per market (skip if already pending).
  2. One open position per market until resolution (skip if already have a
     position — prevents sequential re-betting the same market).
  3. Position sized against FREE cash = portfolio.cash − committed notional
     of all outstanding unfilled orders.
  4. Per-bet Kelly fraction and hard cap applied to free cash.
  5. Per-event notional cap applied on top of Kelly sizing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.backtesting.strategy import Strategy

if TYPE_CHECKING:
    from src.backtesting.models import Fill, MarketInfo, Side, TradeEvent

_DEFAULT_FV_DATA_DIR = (
    Path(__file__).parent.parent.parent.parent / "fv-timeseries" / "data"
)


class FVKellyStrategy(Strategy):
    def __init__(
        self,
        fv_data_dir: Path | str | None = None,
        kelly_fraction: float = 0.5,
        max_position_fraction: float = 0.25,
        min_edge: float = 0.17,
        gamma_stress: float = 1.2,
        max_event_fraction: float = 0.20,
        initial_cash: float = 10_000.0,
    ):
        super().__init__(
            name="fv_kelly",
            description=(
                f"Kelly bets on FV mispricing, one position per market "
                f"(kelly={kelly_fraction}, cap={max_position_fraction:.0%}, "
                f"min_edge={min_edge:.0%}, gamma_stress={gamma_stress}, "
                f"event_cap={max_event_fraction:.0%})."
            ),
            initial_cash=initial_cash,
        )
        self._fv_data_dir = Path(fv_data_dir) if fv_data_dir is not None else _DEFAULT_FV_DATA_DIR
        self.kelly_fraction = kelly_fraction
        self.max_position_fraction = max_position_fraction
        self.min_edge = min_edge
        self.gamma_stress = gamma_stress
        self.max_event_fraction = max_event_fraction
        self._initial_cash = initial_cash

        # ticker → {minute → (fv_T, gamma_pos, gamma_neg)}
        self._fv_lookup: dict[str, dict[pd.Timestamp, tuple[float, float, float]]] = {}

        # market_id → committed notional for its one pending order
        self._market_committed: dict[str, float] = {}
        # markets with an open position (pending or filled, until resolution)
        self._active_markets: set[str] = set()
        # event_ticker → total FILLED notional at risk across all active strikes.
        # Incremented at fill time (not submission) to avoid over-counting partial fills
        # on low-price bets where submitted qty >> filled qty.
        self._event_notional: dict[str, float] = {}
        # market_id → filled notional (set at fill time, popped at resolve for cleanup)
        self._market_notional: dict[str, float] = {}
        # market_id → fv_T at the moment the bet was placed (for calibration reporting)
        self.fv_at_fill: dict[str, float] = {}

    @staticmethod
    def _event_ticker(market_id: str) -> str:
        """Strip the strike suffix: 'KXRTMICKEY17-82' → 'KXRTMICKEY17'."""
        return market_id.rsplit("-", 1)[0]

    @property
    def _committed(self) -> float:
        return sum(self._market_committed.values())

    def initialize(self) -> None:
        parquets = list(self._fv_data_dir.glob("*.parquet"))
        if not parquets:
            print(f"[fv_kelly] WARNING: no parquets found in {self._fv_data_dir}")
            return

        for path in parquets:
            df = pd.read_parquet(path, columns=["timestamp", "ticker", "fv_T", "gamma_pos", "gamma_neg"])
            df = df[df["ticker"].str.len() > 0]
            df["minute"] = df["timestamp"].dt.floor("min")
            for ticker, grp in df.groupby("ticker"):
                self._fv_lookup[ticker] = dict(
                    zip(grp["minute"], zip(grp["fv_T"], grp["gamma_pos"], grp["gamma_neg"]))
                )

        print(f"[fv_kelly] Loaded FV data for {len(self._fv_lookup)} markets.")

    def on_trade(self, trade: TradeEvent) -> None:
        market_id = trade.market_id
        if market_id not in self._fv_lookup:
            return

        # Skip if we already have a pending order or open position for this market
        if market_id in self._active_markets:
            return

        ts = pd.Timestamp(trade.timestamp)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC")
        minute = ts.floor("min")

        entry = self._fv_lookup[market_id].get(minute)
        if entry is None:
            return

        fv, g_pos, g_neg = entry

        p_yes = trade.yes_price
        p_no = trade.no_price

        free_cash = max(0.0, self.portfolio.cash - self._committed)
        if free_cash <= 0:
            return

        # Apply gamma stress: shift fv by one adverse review before checking edge.
        # gamma_neg ≤ 0 (fresh→rotten review hurts a YES bet).
        fv_yes_stressed = fv + g_neg * self.gamma_stress
        # gamma_pos ≥ 0 (fresh review hurts a NO bet by pushing fv up).
        no_fv_stressed = (1.0 - fv) - g_pos * self.gamma_stress

        # Per-event notional cap: how much budget remains for this event?
        # Reference is initial_cash so the cap stays stable as capital is deployed.
        event = self._event_ticker(market_id)
        event_cap = self.max_event_fraction * self._initial_cash
        event_remaining = max(0.0, event_cap - self._event_notional.get(event, 0.0))

        if p_yes < fv and (fv_yes_stressed - p_yes) >= self.min_edge:
            if event_remaining <= 0:
                return  # event budget exhausted
            # Kelly sizing on raw fv (unstressed); gate on stressed fv
            edge = fv - p_yes
            f = min(self.kelly_fraction * edge / (1.0 - p_yes), self.max_position_fraction)
            qty = max(1, int(f * free_cash / p_yes))
            self.buy_yes(market_id, price=p_yes, quantity=qty)
            self._market_committed[market_id] = qty * p_yes
            self._active_markets.add(market_id)
            self.fv_at_fill[market_id] = fv

        elif p_no < (1.0 - fv) and (no_fv_stressed - p_no) >= self.min_edge:
            if event_remaining <= 0:
                return  # event budget exhausted
            # Kelly sizing on raw no_fv; gate on stressed no_fv
            no_fv = 1.0 - fv
            edge = no_fv - p_no
            f = min(self.kelly_fraction * edge / (1.0 - p_no), self.max_position_fraction)
            qty = max(1, int(f * free_cash / p_no))
            self.buy_no(market_id, price=p_no, quantity=qty)
            self._market_committed[market_id] = qty * p_no
            self._active_markets.add(market_id)
            self.fv_at_fill[market_id] = fv

    def on_fill(self, fill: Fill) -> None:
        # Order filled — release committed cash; record actual filled notional for event cap.
        # We track filled qty × price (not submitted) to avoid over-counting partial fills
        # on low-price bets where submitted qty >> filled qty.
        self._market_committed.pop(fill.market_id, None)
        event = self._event_ticker(fill.market_id)
        filled_notional = fill.price * fill.quantity
        self._event_notional[event] = self._event_notional.get(event, 0.0) + filled_notional
        self._market_notional[fill.market_id] = (
            self._market_notional.get(fill.market_id, 0.0) + filled_notional
        )

    def on_market_resolve(self, market: MarketInfo, result: Side) -> None:
        # Position resolved — market is free to bet again if re-opened (never happens for KXRT)
        if market.market_id in self._active_markets:
            self._active_markets.discard(market.market_id)
            event = self._event_ticker(market.market_id)
            notional = self._market_notional.pop(market.market_id, 0.0)
            remaining = self._event_notional.get(event, 0.0) - notional
            if remaining <= 0.0:
                self._event_notional.pop(event, None)
            else:
                self._event_notional[event] = remaining
        self._market_committed.pop(market.market_id, None)

    def on_market_close(self, market: MarketInfo) -> None:
        # Cancel unfilled order and release its committed slot.
        # _event_notional only contains filled notional so no adjustment needed here.
        self.cancel_all(market.market_id)
        self._market_committed.pop(market.market_id, None)
        # Keep in _active_markets if filled (position still open until resolve)
