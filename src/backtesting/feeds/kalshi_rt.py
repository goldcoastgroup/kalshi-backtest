"""Kalshi RT feed — KalshiFeed restricted to markets with fv-timeseries data."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from src.backtesting.feeds.kalshi import KalshiFeed
from src.backtesting.models import MarketInfo, MarketStatus, Platform, Side, TradeEvent


class KalshiRTFeed(KalshiFeed):
    """KalshiFeed restricted to markets backed by fv-timeseries parquets.

    markets() returns only markets whose event_ticker has a corresponding
    parquet file under fv-timeseries/data/.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base_dir = Path(__file__).parent.parent.parent.parent
        self._fv_data_dir = base_dir / "fv-timeseries" / "data"

    def markets(self) -> dict[str, MarketInfo]:
        """Load Kalshi markets restricted to those with fv-timeseries data."""
        if self._markets is not None:
            return self._markets

        rt_tickers = {p.stem for p in self._fv_data_dir.glob("*.parquet")}
        if not rt_tickers:
            self._markets = {}
            return self._markets

        tickers_sql = ", ".join(f"'{t}'" for t in sorted(rt_tickers))

        con = self._get_con()
        rows = con.execute(
            f"""
            SELECT ticker, event_ticker, title, status, result,
                   open_time, close_time
            FROM '{self.markets_dir}/*.parquet'
            WHERE event_ticker IN ({tickers_sql})
            """
        ).fetchall()

        self._markets = {}
        for ticker, event_ticker, title, status, result, open_time, close_time in rows:
            if status == "finalized" and result == "yes":
                ms = MarketStatus.RESOLVED_YES
                side: Side | None = Side.YES
            elif status == "finalized" and result == "no":
                ms = MarketStatus.RESOLVED_NO
                side = Side.NO
            elif status == "closed":
                ms = MarketStatus.CLOSED
                side = None
            else:
                ms = MarketStatus.OPEN
                side = None

            self._markets[ticker] = MarketInfo(
                market_id=ticker,
                platform=Platform.KALSHI,
                title=title or "",
                open_time=open_time,
                close_time=close_time,
                result=side,
                status=ms,
                event_id=event_ticker,
            )

        return self._markets

    def _rt_market_ids(self) -> list[str]:
        return list(self.markets().keys())

    def trades(
        self,
        market_ids: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        batch_size: int = 50_000,
    ) -> Iterator[TradeEvent]:
        rt_ids = self._rt_market_ids()
        effective_ids = list(set(market_ids) & set(rt_ids)) if market_ids else rt_ids
        yield from super().trades(market_ids=effective_ids, start_time=start_time, end_time=end_time, batch_size=batch_size)

    def trade_count(
        self,
        market_ids: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> int:
        rt_ids = self._rt_market_ids()
        effective_ids = list(set(market_ids) & set(rt_ids)) if market_ids else rt_ids
        return super().trade_count(market_ids=effective_ids, start_time=start_time, end_time=end_time)

    def market_volumes(
        self,
        market_ids: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, int]:
        rt_ids = self._rt_market_ids()
        effective_ids = list(set(market_ids) & set(rt_ids)) if market_ids else rt_ids
        return super().market_volumes(market_ids=effective_ids, start_time=start_time, end_time=end_time)
