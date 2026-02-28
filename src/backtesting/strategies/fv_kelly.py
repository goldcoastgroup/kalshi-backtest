"""FV-Kelly — size bets with Kelly criterion using model fair values.

For each market, looks up the model's fair value (fv_T) from pre-computed
per-minute FV timeseries parquets. When the observed trade price deviates
from fair value, places a Kelly-sized limit order:

  YES underpriced (p_yes < fv):  buy YES,  f* = (fv - p_yes) / (1 - p_yes)
  NO  underpriced (p_no < 1-fv): buy NO,   f* = ((1-fv) - p_no) / (1-p_no)

The two conditions are mutually exclusive.

Leverage controls:
  1. One pending order per market (skip if already pending).
  2. One open position per market until resolution (skip if already have a
     position — prevents sequential re-betting the same market).
  3. Position sized against FREE cash = portfolio.cash − committed notional
     of all outstanding unfilled orders.
  4. Per-bet Kelly fraction and hard cap applied to free cash.
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
        initial_cash: float = 10_000.0,
    ):
        super().__init__(
            name="fv_kelly",
            description=(
                f"Kelly bets on FV mispricing, one position per market "
                f"(kelly={kelly_fraction}, cap={max_position_fraction:.0%}, "
                f"min_edge={min_edge:.0%})."
            ),
            initial_cash=initial_cash,
        )
        self._fv_data_dir = Path(fv_data_dir) if fv_data_dir is not None else _DEFAULT_FV_DATA_DIR
        self.kelly_fraction = kelly_fraction
        self.max_position_fraction = max_position_fraction
        self.min_edge = min_edge

        self._fv_lookup: dict[str, dict[pd.Timestamp, float]] = {}

        # market_id → committed notional for its one pending order
        self._market_committed: dict[str, float] = {}
        # markets with an open position (pending or filled, until resolution)
        self._active_markets: set[str] = set()

    @property
    def _committed(self) -> float:
        return sum(self._market_committed.values())

    def initialize(self) -> None:
        parquets = list(self._fv_data_dir.glob("*.parquet"))
        if not parquets:
            print(f"[fv_kelly] WARNING: no parquets found in {self._fv_data_dir}")
            return

        for path in parquets:
            df = pd.read_parquet(path, columns=["timestamp", "ticker", "fv_T"])
            df = df[df["ticker"].str.len() > 0]
            df["minute"] = df["timestamp"].dt.floor("min")
            for ticker, grp in df.groupby("ticker"):
                self._fv_lookup[ticker] = dict(zip(grp["minute"], grp["fv_T"]))

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

        fv = self._fv_lookup[market_id].get(minute)
        if fv is None:
            return

        p_yes = trade.yes_price
        p_no = trade.no_price

        free_cash = max(0.0, self.portfolio.cash - self._committed)
        if free_cash <= 0:
            return

        if p_yes < fv and (fv - p_yes) >= self.min_edge:
            edge = fv - p_yes
            f = min(self.kelly_fraction * edge / (1.0 - p_yes), self.max_position_fraction)
            qty = max(1, int(f * free_cash / p_yes))
            self.buy_yes(market_id, price=p_yes, quantity=qty)
            self._market_committed[market_id] = qty * p_yes
            self._active_markets.add(market_id)

        elif p_no < (1.0 - fv) and ((1.0 - fv) - p_no) >= self.min_edge:
            no_fv = 1.0 - fv
            edge = no_fv - p_no
            f = min(self.kelly_fraction * edge / (1.0 - p_no), self.max_position_fraction)
            qty = max(1, int(f * free_cash / p_no))
            self.buy_no(market_id, price=p_no, quantity=qty)
            self._market_committed[market_id] = qty * p_no
            self._active_markets.add(market_id)

    def on_fill(self, fill: Fill) -> None:
        # Order filled — release committed cash; position stays active until resolution
        self._market_committed.pop(fill.market_id, None)

    def on_market_resolve(self, market: MarketInfo, result: Side) -> None:
        # Position resolved — market is free to bet again if re-opened (never happens for KXRT)
        self._active_markets.discard(market.market_id)
        self._market_committed.pop(market.market_id, None)

    def on_market_close(self, market: MarketInfo) -> None:
        # Cancel unfilled order and release its committed slot
        self.cancel_all(market.market_id)
        self._market_committed.pop(market.market_id, None)
        # Keep in _active_markets if filled (position still open until resolve)
