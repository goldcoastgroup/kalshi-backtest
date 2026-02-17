"""Live Polymarket WebSocket feed — streams real-time trades for front testing.

Connects to the Polymarket CLOB WebSocket (public MARKET channel — no auth
required) and yields TradeEvent objects as trades occur.

No environment variables required for public market data.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from src.backtesting.models import MarketInfo, MarketStatus, Platform, Side, TradeEvent

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore[assignment]

__all__ = ["PolymarketLiveFeed", "fetch_random_polymarket_condition"]

_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
_CLOB_API = "https://clob.polymarket.com"
_GAMMA_API = "https://gamma-api.polymarket.com"
_UA = "prediction-market-backtester/0.1"


def _get(url: str) -> dict | list:
    """GET JSON from a URL with proper headers."""
    import urllib.request

    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", _UA)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def fetch_random_polymarket_condition() -> str | None:
    """Fetch a random active market condition ID from the Polymarket Gamma API."""
    import random

    try:
        # Sort by 24h volume descending so we pick an active market
        data = _get(f"{_GAMMA_API}/markets?active=true&closed=false&limit=50&order=volume24hr&ascending=false")
        markets = data if isinstance(data, list) else data.get("markets", [])
        with_tokens = [m for m in markets if m.get("clobTokenIds")]
        if not with_tokens:
            with_tokens = markets
        if not with_tokens:
            return None
        # Pick from the top 10 by volume for a better chance of seeing trades
        top = with_tokens[:10] if len(with_tokens) >= 10 else with_tokens
        chosen = random.choice(top)
        title = chosen.get("question", "")
        if title:
            print(f"  -> {title}")
        return chosen.get("conditionId")
    except Exception:
        return None


class PolymarketLiveFeed:
    """Async live feed that streams Polymarket trades via WebSocket.

    Usage::

        feed = PolymarketLiveFeed(condition_ids=["0x1234...", ...])
        async for trade in feed.trades():
            ...

    ``condition_ids`` are the Polymarket condition IDs for the markets you
    want to track.  You can find them on the Polymarket UI or via the
    Gamma API (GET /markets).
    """

    def __init__(self, condition_ids: list[str]):
        if websockets is None:
            raise ImportError("Install websockets: pip install websockets")

        self.condition_ids = condition_ids
        self._markets: dict[str, MarketInfo] = {}
        # Maps token_id -> (condition_id, outcome_index)
        self._token_map: dict[str, tuple[str, int]] = {}

    def _fetch_market_clob(self, cid: str) -> None:
        """Fetch a single market's metadata from the CLOB API."""
        try:
            raw = _get(f"{_CLOB_API}/markets/{cid}")
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        data: dict = raw

        condition_id = data.get("condition_id", cid)
        tokens = data.get("tokens", [])
        for tok in tokens:
            token_id = str(tok["token_id"])
            outcome = tok.get("outcome", "")
            idx = 0 if outcome == "Yes" else 1
            self._token_map[token_id] = (condition_id, idx)

        closed = data.get("closed", False)
        active = data.get("active", True)
        # CLOB API doesn't have resolution info directly — assume open if active
        status = MarketStatus.OPEN
        if closed or not active:
            status = MarketStatus.CLOSED

        end_date = None
        if data.get("end_date_iso"):
            try:
                end_date = datetime.fromisoformat(data["end_date_iso"].replace("Z", "+00:00"))
            except Exception:
                pass

        self._markets[condition_id] = MarketInfo(
            market_id=condition_id,
            platform=Platform.POLYMARKET,
            title=data.get("question", condition_id),
            open_time=None,
            close_time=end_date,
            result=None,
            status=status,
        )

    def markets(self) -> dict[str, MarketInfo]:
        """Return cached market metadata (call connect() first)."""
        return dict(self._markets)

    async def connect(self) -> None:
        """Fetch market metadata before streaming."""
        for cid in self.condition_ids:
            self._fetch_market_clob(cid)

    async def trades(self) -> AsyncIterator[TradeEvent]:
        """Connect to WebSocket and yield TradeEvent objects for live trades."""
        if not self._markets:
            await self.connect()

        asset_ids = list(self._token_map.keys())
        if not asset_ids:
            raise ValueError(
                "No token IDs found for the given condition_ids. "
                "Check that the condition IDs are valid and have CLOB tokens."
            )

        async for trade in self._stream_with_reconnect(asset_ids):
            yield trade

    async def _stream_with_reconnect(self, asset_ids: list[str]) -> AsyncIterator[TradeEvent]:
        """Stream trades with automatic reconnection on disconnect."""
        while True:
            try:
                async with websockets.connect(
                    _WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    subscribe_msg = {
                        "type": "subscribe",
                        "assets_ids": asset_ids,
                    }
                    await ws.send(json.dumps(subscribe_msg))

                    async for raw in ws:
                        msgs = json.loads(raw)
                        if not isinstance(msgs, list):
                            msgs = [msgs]

                        for msg in msgs:
                            event_type = msg.get("event_type", "")

                            if event_type == "last_trade_price":
                                trade = self._parse_trade(msg)
                                if trade is not None:
                                    yield trade
                            elif event_type == "price_change":
                                # price_change events also indicate activity —
                                # extract last_trade_price if present
                                ltp = msg.get("last_trade_price")
                                if ltp is not None:
                                    trade = self._parse_trade(msg)
                                    if trade is not None:
                                        yield trade
                            # book snapshots are silently ignored

            except websockets.exceptions.ConnectionClosed:
                await asyncio.sleep(2)
                continue
            except Exception:
                await asyncio.sleep(5)
                continue

    def _parse_trade(self, msg: dict) -> TradeEvent | None:
        """Parse a WebSocket message into a TradeEvent."""
        asset_id = msg.get("asset_id", "")
        token_info = self._token_map.get(asset_id)
        if token_info is None:
            return None

        condition_id, outcome_index = token_info
        if condition_id not in self._markets:
            return None

        # last_trade_price can be a standalone field or nested
        price_val = msg.get("price") or msg.get("last_trade_price")
        if price_val is None:
            return None
        price = float(price_val)
        if price <= 0:
            return None

        size = float(msg.get("size", msg.get("quantity", 1)))

        # outcome_index 0 = YES token, 1 = NO token
        if outcome_index == 0:
            yes_price = price
            no_price = 1.0 - price
            taker_side = Side.YES
        else:
            no_price = price
            yes_price = 1.0 - price
            taker_side = Side.NO

        ts_raw = msg.get("timestamp", "")
        try:
            if ts_raw:
                ts_val = str(ts_raw)
                if ts_val.isdigit():
                    # Polymarket timestamps are in milliseconds
                    timestamp = datetime.fromtimestamp(int(ts_val) / 1000, tz=timezone.utc)
                else:
                    timestamp = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
            else:
                timestamp = datetime.now(timezone.utc)
        except Exception:
            timestamp = datetime.now(timezone.utc)

        return TradeEvent(
            timestamp=timestamp,
            market_id=condition_id,
            platform=Platform.POLYMARKET,
            yes_price=yes_price,
            no_price=no_price,
            quantity=size,
            taker_side=taker_side,
            raw_id=msg.get("id"),
        )
