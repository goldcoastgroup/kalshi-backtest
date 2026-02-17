"""Live Kalshi WebSocket feed — streams real-time trades for front testing.

Connects to the Kalshi WebSocket API and yields TradeEvent objects as trades
occur on the platform.  Requires API credentials (key + RSA private key file).

Environment variables:
    KALSHI_API_KEY            API key ID (UUID format)
    KALSHI_PRIVATE_KEY_PATH   Path to the RSA .key file
"""

from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

from src.backtesting.models import MarketInfo, MarketStatus, Platform, Side, TradeEvent

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore[assignment]

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
except ImportError:
    hashes = None  # type: ignore[assignment]

__all__ = ["KalshiLiveFeed", "fetch_random_kalshi_ticker"]

_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
_REST_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def fetch_random_kalshi_ticker() -> str | None:
    """Fetch a random active market ticker from the Kalshi public API."""
    import random
    import urllib.request

    url = f"{_REST_BASE}/markets?status=open&limit=100"
    try:
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "prediction-market-backtester/0.1")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        markets = data.get("markets", [])
        if not markets:
            return None
        chosen = random.choice(markets)
        return chosen.get("ticker")
    except Exception:
        return None


def _sign(private_key_pem: bytes, timestamp_ms: int, method: str, path: str) -> str:
    """Create RSA-PSS signature for Kalshi authentication."""
    message = f"{timestamp_ms}{method}{path}".encode()
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    signature = private_key.sign(  # type: ignore[union-attr]
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode()


class KalshiLiveFeed:
    """Async live feed that streams Kalshi trades via WebSocket.

    Usage::

        feed = KalshiLiveFeed(market_tickers=["KXBTC-25FEB14-T96000", ...])
        async for trade in feed.trades():
            ...
    """

    def __init__(
        self,
        market_tickers: list[str],
        api_key: str | None = None,
        private_key_path: str | None = None,
    ):
        if websockets is None:
            raise ImportError("Install websockets: pip install websockets")
        if hashes is None:
            raise ImportError("Install cryptography: pip install cryptography")

        self.market_tickers = market_tickers
        self.api_key = api_key or os.environ.get("KALSHI_API_KEY", "")
        pk_path = private_key_path or os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
        if not self.api_key or not pk_path:
            raise ValueError(
                "Kalshi API credentials required. Set KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH environment variables."
            )
        self._private_key_pem = Path(pk_path).expanduser().read_bytes()
        self._markets: dict[str, MarketInfo] = {}

    async def _fetch_markets_rest(self) -> dict[str, MarketInfo]:
        """Fetch market metadata via REST for the subscribed tickers."""
        import urllib.request

        markets: dict[str, MarketInfo] = {}
        for ticker in self.market_tickers:
            ts = int(time.time() * 1000)
            path = f"/trade-api/v2/markets/{ticker}"
            sig = _sign(self._private_key_pem, ts, "GET", path)
            url = f"{_REST_BASE}/markets/{ticker}"
            req = urllib.request.Request(url)
            req.add_header("KALSHI-ACCESS-KEY", self.api_key)
            req.add_header("KALSHI-ACCESS-TIMESTAMP", str(ts))
            req.add_header("KALSHI-ACCESS-SIGNATURE", sig)
            req.add_header("Accept", "application/json")
            req.add_header("User-Agent", "prediction-market-backtester/0.1")
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read())
            except Exception:
                continue

            m = data.get("market", data)
            result_str = m.get("result", "")
            result: Side | None = None
            if result_str == "yes":
                result = Side.YES
            elif result_str == "no":
                result = Side.NO

            status = MarketStatus.OPEN
            st = m.get("status", "")
            if st == "closed":
                status = MarketStatus.CLOSED
            elif result == Side.YES:
                status = MarketStatus.RESOLVED_YES
            elif result == Side.NO:
                status = MarketStatus.RESOLVED_NO

            open_time = None
            close_time = None
            if m.get("open_time"):
                try:
                    open_time = datetime.fromisoformat(m["open_time"].replace("Z", "+00:00"))
                except Exception:
                    pass
            if m.get("close_time"):
                try:
                    close_time = datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))
                except Exception:
                    pass

            markets[ticker] = MarketInfo(
                market_id=ticker,
                platform=Platform.KALSHI,
                title=m.get("title", ticker),
                open_time=open_time,
                close_time=close_time,
                result=result,
                status=status,
                event_id=m.get("event_ticker"),
            )
        return markets

    def markets(self) -> dict[str, MarketInfo]:
        """Return cached market metadata (call connect() first)."""
        return dict(self._markets)

    async def connect(self) -> None:
        """Fetch market metadata before streaming."""
        self._markets = await self._fetch_markets_rest()

    async def trades(self) -> AsyncIterator[TradeEvent]:
        """Connect to the WebSocket and yield TradeEvent objects."""
        if not self._markets:
            await self.connect()

        ts = int(time.time() * 1000)
        path = "/trade-api/ws/v2"
        sig = _sign(self._private_key_pem, ts, "GET", path)

        headers = {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
            "KALSHI-ACCESS-SIGNATURE": sig,
        }

        async with websockets.connect(_WS_URL, additional_headers=headers) as ws:
            # Subscribe to trade channel for each ticker
            for ticker in self.market_tickers:
                await ws.send(
                    json.dumps(
                        {
                            "id": 1,
                            "cmd": "subscribe",
                            "params": {
                                "channels": ["trade"],
                                "market_ticker": ticker,
                            },
                        }
                    )
                )

            async for raw in ws:
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type != "trade":
                    continue

                data = msg.get("msg", msg)
                ticker = data.get("market_ticker", "")
                if ticker not in self._markets:
                    continue

                yes_price = data.get("yes_price", 0)
                # Kalshi prices are in cents (1-99) — normalize to [0, 1]
                if yes_price > 1:
                    yes_price = yes_price / 100.0
                no_price = 1.0 - yes_price

                ts_str = data.get("created_time") or data.get("ts", "")
                try:
                    timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except Exception:
                    timestamp = datetime.now(timezone.utc)

                taker_side_str = data.get("taker_side", "yes").lower()
                taker_side = Side.YES if taker_side_str == "yes" else Side.NO

                yield TradeEvent(
                    timestamp=timestamp,
                    market_id=ticker,
                    platform=Platform.KALSHI,
                    yes_price=yes_price,
                    no_price=no_price,
                    quantity=float(data.get("count", data.get("quantity", 1))),
                    taker_side=taker_side,
                    raw_id=data.get("trade_id"),
                )
