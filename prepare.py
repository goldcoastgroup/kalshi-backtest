"""
Data loading, engine setup, and evaluation for Kalshi backtests.

Loads fair values (via kxrt_fv), orderbook data (MongoDB), instruments
(Kalshi REST API), and settlement outcomes (MongoDB). All data is cached
to parquet/JSON files under ~/.cache/kalshi-backtest/.

Also provides run_backtest() which wires up the engine, runs the
strategy, and prints results. train.py calls this as its entry point.

This file is FROZEN — the training agent must not modify it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

if TYPE_CHECKING:
    from engine.strategy import Strategy

# Load env from sandbox
load_dotenv(Path(__file__).resolve().parent / ".." / "sandbox" / ".env")

# Add sandbox to path for kxrt_fv
_sandbox = str(Path(__file__).resolve().parent / ".." / "sandbox")
if _sandbox not in sys.path:
    sys.path.insert(0, _sandbox)

from kxrt_fv import backtest as kxrt_backtest  # noqa: E402

from engine._engine import (  # noqa: E402
    AggressorSide,
    BookAction,
    F_LAST,
    F_SNAPSHOT,
    FairValueData,
    Instrument,
    OrderBookDelta,
    OrderSide,
    TradeTick,
)

CACHE_DIR = Path.home() / ".cache" / "kalshi-backtest"
LOCAL_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
OOS_CACHE_DIR = Path(__file__).resolve().parent / ".cache-oos"
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
MODEL_NAME = "xgb_ff99dd2"

# ── Frozen constants (do NOT modify in train.py) ──

EVENT_TICKERS = [
    "KXRT-BRI",
    "KXRT-HOP",
    "KXRT-PRO",
    "KXRT-REA",
    "KXRT-REM",
]
STARTING_BALANCE = 10_000


@dataclass
class BacktestData:
    instruments: list[Instrument]
    fair_values: list[FairValueData]
    orderbook_deltas: list[OrderBookDelta]
    trades: list[TradeTick]


def load(event_tickers: list[str], refresh: bool = False) -> BacktestData:
    """Load all backtest data, using parquet cache when available."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Event → market tickers from MongoDB
    print("Fetching market tickers from MongoDB...")
    event_markets = _load_event_market_tickers(event_tickers)
    total = sum(len(v) for v in event_markets.values())
    print(f"Found {len(event_markets)} events with {total} markets\n")

    all_tickers = [t for tickers in event_markets.values() for t in tickers]

    # Phase 2: Fair values via kxrt_fv
    print("Loading fair values...")
    fair_values = _load_fair_values(event_markets, refresh)
    fv_tickers = {fv.instrument_id for fv in fair_values}
    print(f"  {len(fair_values)} FV records for {len(fv_tickers)} tickers\n")

    # Phase 3: Instruments from Kalshi API
    print("Loading instruments...")
    instruments = _load_instruments(all_tickers, event_markets, refresh)
    inst_map = {inst.id: inst for inst in instruments}
    print(f"  {len(instruments)} instruments\n")

    # Phase 4: Orderbook deltas from MongoDB
    print("Loading orderbook data...")
    ob_deltas = _load_orderbooks(event_markets, inst_map, refresh)
    print(f"  {len(ob_deltas)} orderbook deltas\n")

    # Phase 5: Trade data from Kalshi API
    print("Loading trade data...")
    trades = _load_trades(all_tickers, refresh)
    print(f"  {len(trades)} trades\n")

    # Phase 6: Settlement outcomes + synthetic deltas
    print("Loading settlement outcomes...")
    outcomes = _load_outcomes(refresh)
    settlement_deltas = _build_settlement_deltas(inst_map, outcomes, ob_deltas)
    print(f"  {len(settlement_deltas)} settlement deltas\n")

    # Combine OB deltas + settlement deltas
    all_deltas = ob_deltas + settlement_deltas

    # Filter to instruments that have both FV and OB data
    active_tickers = fv_tickers & set(inst_map.keys())
    instruments = [inst_map[t] for t in sorted(active_tickers) if t in inst_map]
    fair_values = [fv for fv in fair_values if fv.instrument_id in active_tickers]
    all_deltas = [d for d in all_deltas if d.instrument_id in active_tickers]

    trades = [t for t in trades if t.instrument_id in active_tickers]

    print(f"Ready: {len(instruments)} instruments, {len(fair_values)} FV, {len(all_deltas)} OB deltas, {len(trades)} trades\n")
    return BacktestData(instruments=instruments, fair_values=fair_values, orderbook_deltas=all_deltas, trades=trades)


# ── Internal loaders ──


def _load_event_market_tickers(event_tickers: list[str]) -> dict[str, list[str]]:
    """Query MongoDB kxrt.events for market tickers."""
    client = MongoClient(host=os.environ["MONGODB_URI"], tz_aware=True)
    db = client["kxrt"]
    events = list(db["events"].find(
        {"_id": {"$in": event_tickers}},
        {"_id": 1, "markets.ticker": 1},
    ))
    client.close()
    return {
        e["_id"]: [m["ticker"] for m in e.get("markets", [])]
        for e in events
    }


def _load_fair_values(
    event_markets: dict[str, list[str]], refresh: bool,
) -> list[FairValueData]:
    """Generate fair values via kxrt_fv.backtest(), cache per event."""
    fv_dir = CACHE_DIR / "fair_values"
    fv_dir.mkdir(exist_ok=True)
    all_fv: list[FairValueData] = []

    for i, (event_ticker, tickers) in enumerate(event_markets.items(), 1):
        if not tickers:
            continue
        cache_file = fv_dir / f"{event_ticker}.parquet"
        if cache_file.exists() and not refresh:
            print(f"  [{i}/{len(event_markets)}] {event_ticker}: cached")
            df = pd.read_parquet(cache_file)
            all_fv.extend(_df_to_fair_values(df))
            continue

        print(f"  [{i}/{len(event_markets)}] {event_ticker} ({len(tickers)} markets)...", end=" ", flush=True)
        try:
            snapshots = kxrt_backtest(tickers=tickers, model_name=MODEL_NAME)
            records = []
            for ticker, snaps in snapshots.items():
                for snap in snaps:
                    ts_ns = int(snap.timestamp.timestamp() * 1e9)
                    records.append({
                        "timestamp_ns": ts_ns,
                        "instrument_id": ticker,
                        "fv": snap.fv,
                        "theta": snap.theta,
                        "gamma_pos": snap.gamma_pos,
                        "gamma_neg": snap.gamma_neg,
                        "new_review": snap.new_review,
                        "hours_left": snap.hours_left,
                        "cur_score": snap.cur_score,
                        "total_reviews": snap.total_reviews,
                    })
            df = pd.DataFrame(records)
            df.to_parquet(cache_file)
            all_fv.extend(_df_to_fair_values(df))
            print(f"{len(records)} records")
        except Exception as e:
            print(f"error: {e}")

    all_fv.sort(key=lambda x: x.timestamp_ns)
    return all_fv


def _df_to_fair_values(df: pd.DataFrame) -> list[FairValueData]:
    """Convert DataFrame rows to FairValueData objects."""
    if df.empty:
        return []
    ts = df["timestamp_ns"].values
    ids = df["instrument_id"].values
    fvs = df["fv"].values
    thetas = df["theta"].values
    gpos = df["gamma_pos"].values
    gneg = df["gamma_neg"].values
    nrs = df["new_review"].values
    hrs = df["hours_left"].values
    scores = df["cur_score"].values
    reviews = df["total_reviews"].values
    n = len(df)
    result = [None] * n
    for i in range(n):
        result[i] = FairValueData(
            timestamp_ns=int(ts[i]),
            instrument_id=str(ids[i]),
            fv=float(fvs[i]),
            theta=float(thetas[i]),
            gamma_pos=float(gpos[i]),
            gamma_neg=float(gneg[i]),
            new_review=bool(nrs[i]),
            hours_left=float(hrs[i]),
            cur_score=float(scores[i]),
            total_reviews=int(reviews[i]),
        )
    return result


def _load_instruments(
    all_tickers: list[str],
    event_markets: dict[str, list[str]],
    refresh: bool,
) -> list[Instrument]:
    """Fetch instruments from Kalshi REST API, cache to parquet."""
    cache_file = CACHE_DIR / "instruments.parquet"
    if cache_file.exists() and not refresh:
        df = pd.read_parquet(cache_file)
        cached_tickers = set(df["id"].tolist())
        missing = [t for t in all_tickers if t not in cached_tickers]
        if not missing:
            return _df_to_instruments(df)
    else:
        df = pd.DataFrame()
        missing = all_tickers

    # Build ticker → event_ticker map
    ticker_to_event = {}
    for event, tickers in event_markets.items():
        for t in tickers:
            ticker_to_event[t] = event

    records = []
    if not df.empty:
        records = df.to_dict("records")

    with httpx.Client() as client:
        for i, ticker in enumerate(missing, 1):
            print(f"  [{i}/{len(missing)}] {ticker}...", end=" ", flush=True)
            try:
                resp = client.get(f"{KALSHI_API_BASE}/markets/{ticker}")
                resp.raise_for_status()
                market = resp.json()["market"]
                close_time = market.get("close_time") or market.get("latest_expiration_time")
                exp_ns = _parse_iso_to_ns(close_time) if close_time else 0
                records.append({
                    "id": ticker,
                    "event_ticker": market.get("event_ticker", ticker_to_event.get(ticker, "")),
                    "price_precision": 4,
                    "size_precision": 2,
                    "expiration_ns": exp_ns,
                })
                print("ok")
            except Exception as e:
                print(f"error: {e}")

    df = pd.DataFrame(records)
    df.to_parquet(cache_file)
    return _df_to_instruments(df)


def _df_to_instruments(df: pd.DataFrame) -> list[Instrument]:
    if df.empty:
        return []
    ids = df["id"].values
    events = df["event_ticker"].values
    pp = df["price_precision"].values
    sp = df["size_precision"].values
    exp = df["expiration_ns"].values
    n = len(df)
    result = [None] * n
    for i in range(n):
        result[i] = Instrument(
            id=str(ids[i]),
            event_ticker=str(events[i]),
            price_precision=int(pp[i]),
            size_precision=int(sp[i]),
            expiration_ns=int(exp[i]),
        )
    return result


def _parse_iso_to_ns(iso_str: str) -> int:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1e9)


def _load_orderbooks(
    event_markets: dict[str, list[str]],
    instruments: dict[str, Instrument],
    refresh: bool,
) -> list[OrderBookDelta]:
    """Load L2 orderbook data from MongoDB, cache per event."""
    ob_dir = CACHE_DIR / "orderbooks"
    ob_dir.mkdir(exist_ok=True)
    all_deltas: list[OrderBookDelta] = []

    client = MongoClient(host=os.environ["MONGODB_URI"], tz_aware=True)
    db = client["kxrt-training"]
    coll = db["orderbook-active"]

    for event_ticker, tickers in event_markets.items():
        cache_file = ob_dir / f"{event_ticker}.parquet"
        if cache_file.exists() and not refresh:
            print(f"  {event_ticker}: cached")
            df = pd.read_parquet(cache_file)
            all_deltas.extend(_df_to_ob_deltas(df))
            continue

        ob_tickers = [t for t in tickers if t in instruments]
        if not ob_tickers:
            continue

        print(f"  {event_ticker}: {len(ob_tickers)} markets...", end=" ", flush=True)
        docs = list(coll.find(
            {
                "metadata.event_ticker": event_ticker,
                "metadata.market_ticker": {"$in": ob_tickers},
            },
        ).sort("timestamp", 1))

        records = _transform_ob_docs(docs)
        df = pd.DataFrame(records)
        if not df.empty:
            df.to_parquet(cache_file)
        all_deltas.extend(_df_to_ob_deltas(df))
        print(f"{len(records)} deltas")

    client.close()
    return all_deltas


def _transform_ob_docs(docs: list[dict]) -> list[dict]:
    """Transform MongoDB orderbook documents into flat records."""
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for doc in docs:
        key = (
            doc["metadata"]["market_ticker"],
            doc["timestamp"],
            doc["is_snapshot"],
        )
        groups[key].append(doc)

    sorted_keys = sorted(groups.keys(), key=lambda k: (k[1], k[0]))
    records = []

    for key in sorted_keys:
        market_ticker, timestamp, is_snapshot = key
        group_docs = groups[key]
        ts_ns = int(timestamp.timestamp() * 1_000_000_000)

        if is_snapshot:
            # CLEAR + ADD for each level
            records.append({
                "instrument_id": market_ticker,
                "timestamp_ns": ts_ns,
                "action": "CLEAR",
                "side": "BUY",
                "price": 0.0,
                "size": 0.0,
                "flags": 0,
            })
            for i, doc in enumerate(group_docs):
                is_last = i == len(group_docs) - 1
                flags = F_SNAPSHOT | (F_LAST if is_last else 0)
                raw_price = float(doc["metadata"]["price"])
                if doc["metadata"]["side"] == "no":
                    raw_price = 1.0 - raw_price
                side = "BUY" if doc["metadata"]["side"] == "yes" else "SELL"
                records.append({
                    "instrument_id": market_ticker,
                    "timestamp_ns": ts_ns,
                    "action": "ADD",
                    "side": side,
                    "price": raw_price,
                    "size": float(doc["quantity"]),
                    "flags": flags,
                })
        else:
            for i, doc in enumerate(group_docs):
                is_last = i == len(group_docs) - 1
                quantity = doc["quantity"]
                action = "DELETE" if quantity == 0 else "UPDATE"
                flags = F_LAST if is_last else 0
                raw_price = float(doc["metadata"]["price"])
                if doc["metadata"]["side"] == "no":
                    raw_price = 1.0 - raw_price
                side = "BUY" if doc["metadata"]["side"] == "yes" else "SELL"
                records.append({
                    "instrument_id": market_ticker,
                    "timestamp_ns": ts_ns,
                    "action": action,
                    "side": side,
                    "price": raw_price,
                    "size": float(quantity),
                    "flags": flags,
                })

    return records


def _df_to_ob_deltas(df: pd.DataFrame) -> list[OrderBookDelta]:
    """Convert DataFrame rows to OrderBookDelta objects."""
    if df.empty:
        return []
    action_map = {"CLEAR": BookAction.Clear, "ADD": BookAction.Add,
                  "UPDATE": BookAction.Update, "DELETE": BookAction.Delete}
    side_map = {"BUY": OrderSide.Buy, "SELL": OrderSide.Sell}
    # Extract columns as arrays for fast iteration (avoid iterrows overhead)
    ids = df["instrument_id"].values
    ts = df["timestamp_ns"].values
    actions = df["action"].values
    sides = df["side"].values
    prices = df["price"].values
    sizes = df["size"].values
    flags = df["flags"].values
    n = len(df)
    result = [None] * n
    for i in range(n):
        result[i] = OrderBookDelta(
            instrument_id=str(ids[i]),
            timestamp_ns=int(ts[i]),
            action=action_map[actions[i]],
            side=side_map[sides[i]],
            price=float(prices[i]),
            size=float(sizes[i]),
            flags=int(flags[i]),
        )
    return result


def _load_trades(all_tickers: list[str], refresh: bool) -> list[TradeTick]:
    """Fetch historical trades from Kalshi REST API, cache to parquet."""
    trade_dir = CACHE_DIR / "trades"
    trade_dir.mkdir(exist_ok=True)
    all_trades: list[TradeTick] = []

    # Group tickers by event for caching
    ticker_events: dict[str, str] = {}
    for ticker in all_tickers:
        event = "-".join(ticker.split("-")[:2])
        ticker_events[ticker] = event

    events = sorted(set(ticker_events.values()))
    event_tickers: dict[str, list[str]] = {e: [] for e in events}
    for ticker, event in ticker_events.items():
        event_tickers[event].append(ticker)

    for event in events:
        cache_file = trade_dir / f"{event}.parquet"
        if cache_file.exists() and not refresh:
            print(f"  {event}: cached")
            df = pd.read_parquet(cache_file)
            all_trades.extend(_df_to_trades(df))
            continue

        tickers = event_tickers[event]
        print(f"  {event}: fetching {len(tickers)} markets...", end=" ", flush=True)
        records = []
        with httpx.Client(timeout=30.0) as client:
            for ticker in tickers:
                ticker_trades = _fetch_trades_for_ticker(client, ticker)
                records.extend(ticker_trades)

        df = pd.DataFrame(records)
        if not df.empty:
            df.to_parquet(cache_file)
        all_trades.extend(_df_to_trades(df))
        print(f"{len(records)} trades")

    all_trades.sort(key=lambda t: t.timestamp_ns)
    return all_trades


def _fetch_trades_for_ticker(client: httpx.Client, ticker: str) -> list[dict]:
    """Fetch all trades for a ticker with cursor pagination."""
    import time as _time

    records = []
    cursor = None
    while True:
        params = {"ticker": ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor

        resp = client.get(f"{KALSHI_API_BASE}/markets/trades", params=params)
        resp.raise_for_status()
        data = resp.json()

        for trade in data.get("trades", []):
            # Handle both legacy and current API formats
            if "created_time" in trade:
                ts_ns = _parse_iso_to_ns(trade["created_time"])
            elif "ts" in trade:
                ts_ns = int(trade["ts"]) * 1_000_000_000
            else:
                continue

            # Price
            if "yes_price_dollars" in trade:
                price = float(trade["yes_price_dollars"])
            elif "yes_price" in trade:
                price = int(trade["yes_price"]) / 100.0
            else:
                continue

            # Size
            if "count_fp" in trade:
                size = float(trade["count_fp"])
            elif "count" in trade:
                size = float(trade["count"])
            else:
                continue

            # Aggressor side
            taker = trade.get("taker_side", "")
            if taker == "yes":
                side = "BUYER"
            elif taker == "no":
                side = "SELLER"
            else:
                side = "NO_AGGRESSOR"

            records.append({
                "instrument_id": ticker,
                "price": price,
                "size": size,
                "aggressor_side": side,
                "timestamp_ns": ts_ns,
            })

        cursor = data.get("cursor")
        if not cursor or not data.get("trades"):
            break
        _time.sleep(0.1)  # rate limit

    return records


def _df_to_trades(df: pd.DataFrame) -> list[TradeTick]:
    """Convert DataFrame rows to TradeTick objects."""
    if df.empty:
        return []
    side_map = {
        "BUYER": AggressorSide.Buyer,
        "SELLER": AggressorSide.Seller,
        "NO_AGGRESSOR": AggressorSide.NoAggressor,
    }
    ids = df["instrument_id"].values
    prices = df["price"].values
    sizes = df["size"].values
    sides = df["aggressor_side"].values
    ts = df["timestamp_ns"].values
    n = len(df)
    result = [None] * n
    for i in range(n):
        result[i] = TradeTick(
            instrument_id=str(ids[i]),
            price=float(prices[i]),
            size=float(sizes[i]),
            aggressor_side=side_map[sides[i]],
            timestamp_ns=int(ts[i]),
        )
    return result


def _load_outcomes(refresh: bool) -> dict[str, float]:
    """Load settlement outcomes from MongoDB kxrt.events."""
    cache_file = CACHE_DIR / "outcomes.json"
    if cache_file.exists() and not refresh:
        with open(cache_file) as f:
            return json.load(f)

    client = MongoClient(host=os.environ["MONGODB_URI"], tz_aware=True)
    db = client["kxrt"]
    outcomes: dict[str, float] = {}
    for event in db["events"].find({}, {"markets.ticker": 1, "markets.result": 1}):
        for market in event.get("markets", []):
            if "result" in market:
                outcomes[market["ticker"]] = 1.0 if market["result"] == "yes" else 0.0
    client.close()

    with open(cache_file, "w") as f:
        json.dump(outcomes, f)
    return outcomes


def _build_settlement_deltas(
    instruments: dict[str, Instrument],
    outcomes: dict[str, float],
    ob_deltas: list[OrderBookDelta],
) -> list[OrderBookDelta]:
    """Build synthetic settlement deltas that snap the book to 0.99 or 0.01."""
    # Find last OB timestamp per instrument
    last_ts: dict[str, int] = {}
    for d in ob_deltas:
        if d.instrument_id in last_ts:
            last_ts[d.instrument_id] = max(last_ts[d.instrument_id], d.timestamp_ns)
        else:
            last_ts[d.instrument_id] = d.timestamp_ns

    deltas: list[OrderBookDelta] = []
    for ticker, inst in instruments.items():
        outcome = outcomes.get(ticker)
        if outcome is None:
            continue

        expiry_ns = inst.expiration_ns
        ob_last = last_ts.get(ticker, 0)
        ts_ns = max(expiry_ns, ob_last) + 1_000_000_000  # 1s after

        settlement_price = 0.99 if outcome == 1.0 else 0.01

        # CLEAR
        deltas.append(OrderBookDelta(
            instrument_id=ticker, timestamp_ns=ts_ns,
            action=BookAction.Clear, side=OrderSide.Buy,
            price=0.0, size=0.0, flags=0,
        ))
        # BID at settlement price
        deltas.append(OrderBookDelta(
            instrument_id=ticker, timestamp_ns=ts_ns,
            action=BookAction.Add, side=OrderSide.Buy,
            price=settlement_price, size=1_000_000.0, flags=F_SNAPSHOT,
        ))
        # ASK at settlement price
        deltas.append(OrderBookDelta(
            instrument_id=ticker, timestamp_ns=ts_ns,
            action=BookAction.Add, side=OrderSide.Sell,
            price=settlement_price, size=1_000_000.0, flags=F_SNAPSHOT | F_LAST,
        ))

    deltas.sort(key=lambda d: d.timestamp_ns)
    return deltas


# ── Engine setup & evaluation ──


def _load_cached_data() -> BacktestData:
    """Load backtest data, using local parquet cache for fast subsequent runs.

    The cache lives in .cache/ (gitignored, persists across branch changes).
    First run fetches from MongoDB/API and writes parquet files.
    Subsequent runs deserialize from parquet in ~2-3s instead of ~30s.
    """
    LOCAL_CACHE_DIR.mkdir(exist_ok=True)
    marker = LOCAL_CACHE_DIR / "_ready"

    if marker.exists():
        print("Loading data from local cache...")
        t0 = time.time()
        data = _read_local_cache()
        print(f"Data loaded in {time.time() - t0:.1f}s\n")
        return data

    print("Loading data (first run, building local cache)...")
    t0 = time.time()
    data = load(EVENT_TICKERS)
    load_time = time.time() - t0
    print(f"Data loaded in {load_time:.1f}s")

    print("Saving local cache...")
    _write_local_cache(data)
    marker.touch()
    print(f"Cache saved to {LOCAL_CACHE_DIR}\n")

    return data


def _load_oos_cached_data(event_tickers: list[str]) -> BacktestData:
    """Load out-of-sample backtest data, using .cache-oos/ for caching.

    Same pattern as _load_cached_data but parameterized by event tickers
    and using a separate cache directory to avoid contaminating in-sample data.
    """
    OOS_CACHE_DIR.mkdir(exist_ok=True)
    marker = OOS_CACHE_DIR / "_ready"

    if marker.exists():
        print("Loading OOS data from local cache...")
        t0 = time.time()
        data = _read_local_cache(OOS_CACHE_DIR)
        print(f"OOS data loaded in {time.time() - t0:.1f}s\n")
        return data

    print(f"Loading OOS data for {event_tickers} (first run, building cache)...")
    t0 = time.time()
    data = load(event_tickers)
    load_time = time.time() - t0
    print(f"OOS data loaded in {load_time:.1f}s")

    print("Saving OOS local cache...")
    _write_local_cache(data, OOS_CACHE_DIR)
    marker.touch()
    print(f"OOS cache saved to {OOS_CACHE_DIR}\n")

    return data


def _write_local_cache(data: BacktestData, cache_dir: Path | None = None) -> None:
    """Serialize BacktestData to parquet files."""
    if cache_dir is None:
        cache_dir = LOCAL_CACHE_DIR
    # Instruments
    inst_records = [{
        "id": i.id, "event_ticker": i.event_ticker,
        "price_precision": i.price_precision, "size_precision": i.size_precision,
        "expiration_ns": i.expiration_ns,
    } for i in data.instruments]
    pd.DataFrame(inst_records).to_parquet(cache_dir / "instruments.parquet")

    # Fair values
    fv_records = [{
        "timestamp_ns": f.timestamp_ns, "instrument_id": f.instrument_id,
        "fv": f.fv, "theta": f.theta, "gamma_pos": f.gamma_pos,
        "gamma_neg": f.gamma_neg, "new_review": f.new_review,
        "hours_left": f.hours_left, "cur_score": f.cur_score,
        "total_reviews": f.total_reviews,
    } for f in data.fair_values]
    pd.DataFrame(fv_records).to_parquet(cache_dir / "fair_values.parquet")

    # Orderbook deltas — use str() for PyO3 enums (not hashable for dict keys)
    ob_records = [{
        "instrument_id": d.instrument_id, "timestamp_ns": d.timestamp_ns,
        "action": str(d.action).split(".")[-1].upper(), "side": "BUY" if int(d.side) == int(OrderSide.Buy) else "SELL",
        "price": d.price, "size": d.size, "flags": d.flags,
    } for d in data.orderbook_deltas]
    pd.DataFrame(ob_records).to_parquet(cache_dir / "orderbook_deltas.parquet")

    # Trades
    ob_records = [{
        "instrument_id": t.instrument_id, "price": t.price, "size": t.size,
        "aggressor_side": "NO_AGGRESSOR" if int(t.aggressor_side) == int(AggressorSide.NoAggressor) else ("BUYER" if int(t.aggressor_side) == int(AggressorSide.Buyer) else "SELLER"),
        "timestamp_ns": t.timestamp_ns,
    } for t in data.trades]
    pd.DataFrame(ob_records).to_parquet(cache_dir / "trades.parquet")


def _read_local_cache(cache_dir: Path | None = None) -> BacktestData:
    """Deserialize BacktestData from parquet files."""
    if cache_dir is None:
        cache_dir = LOCAL_CACHE_DIR
    instruments = _df_to_instruments(pd.read_parquet(cache_dir / "instruments.parquet"))
    fair_values = _df_to_fair_values(pd.read_parquet(cache_dir / "fair_values.parquet"))
    orderbook_deltas = _df_to_ob_deltas(pd.read_parquet(cache_dir / "orderbook_deltas.parquet"))
    trades = _df_to_trades(pd.read_parquet(cache_dir / "trades.parquet"))
    return BacktestData(
        instruments=instruments, fair_values=fair_values,
        orderbook_deltas=orderbook_deltas, trades=trades,
    )


def run_backtest(
    strategy_factory: callable,
    fee_rate: float = 0.07,
    starting_balance: float | None = None,
) -> dict:
    """Load data, build engine, run strategy, print results.

    Args:
        strategy_factory: callable(instrument_id: str) -> Strategy instance.
        fee_rate: taker fee rate (maker fee = 0 for KXRT markets).
        starting_balance: override starting balance (default: STARTING_BALANCE).

    Returns baseline metrics dict (invisible to train.py which ignores return value).
    """
    from engine import BacktestEngine

    total_t0 = time.time()

    data = _load_cached_data()

    balance = starting_balance if starting_balance is not None else STARTING_BALANCE

    # ── Build engine ──
    engine = BacktestEngine(data.instruments, balance, fee_rate)
    for inst in data.instruments:
        engine.add_strategy(strategy_factory(inst.id))

    # ── Run ──
    print(f"Running backtest with {len(data.instruments)} instruments...")
    t0 = time.time()
    engine.run(data.fair_values, data.orderbook_deltas, data.trades)
    run_time = time.time() - t0
    print(f"Backtest completed in {run_time:.1f}s")

    # ── Results ──
    turnover = _print_results(engine)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print("STANDARDIZED OUTPUT")
    print(f"{'=' * 70}")
    starting = engine._core.starting_balance()
    final = engine._core.balance()
    pnl = final - starting
    all_fills = engine._core.all_fills()
    total_fees = sum(f.fee for f in all_fills)
    positions = engine._core.all_positions()
    wins = [p for p in positions if p.realized_pnl > 0]
    losses = [p for p in positions if p.realized_pnl < 0]
    # ── Max drawdown (realized equity curve) ──
    from collections import defaultdict
    from engine._engine import OrderSide as _OrdSide

    sorted_fills = sorted(all_fills, key=lambda x: x.timestamp_ns)
    inst_pos: dict[str, float] = {}
    inst_entry: dict[str, float] = {}
    cum_rpnl = 0.0
    cum_fees = 0.0
    peak_equity = starting
    max_dd = 0.0

    for f in sorted_fills:
        iid = f.instrument_id
        is_buy = f.side == _OrdSide.Buy
        fqty = f.quantity
        signed = fqty if is_buy else -fqty

        pos = inst_pos.get(iid, 0.0)
        avg = inst_entry.get(iid, 0.0)

        if pos == 0.0:
            inst_pos[iid] = signed
            inst_entry[iid] = f.price
        elif (pos > 0 and signed > 0) or (pos < 0 and signed < 0):
            total = abs(pos) + fqty
            inst_entry[iid] = (avg * abs(pos) + f.price * fqty) / total
            inst_pos[iid] = pos + signed
        else:
            close_qty = min(fqty, abs(pos))
            if pos > 0:
                cum_rpnl += (f.price - avg) * close_qty
            else:
                cum_rpnl += (avg - f.price) * close_qty
            remaining = fqty - close_qty
            inst_pos[iid] = pos + signed
            if remaining > 0 and abs(inst_pos[iid]) > 1e-9:
                inst_entry[iid] = f.price

        cum_fees += f.fee
        equity = starting + cum_rpnl - cum_fees
        if equity > peak_equity:
            peak_equity = equity
        dd = peak_equity - equity
        if dd > max_dd:
            max_dd = dd

    max_dd_pct = 100.0 * max_dd / starting if starting > 0 else 0.0

    print(f"---")
    print(f"pnl:              {pnl:+.2f}")
    print(f"return_pct:       {100*pnl/starting:+.2f}")
    print(f"max_dd_pct:       {max_dd_pct:.2f}")
    print(f"total_fees:       {total_fees:.2f}")
    print(f"total_fills:      {len(all_fills)}")
    print(f"win_rate:         {100*len(wins)/len(positions):.1f}" if positions else "win_rate:         0.0")
    print(f"n_instruments:    {len(data.instruments)}")
    print(f"run_seconds:      {run_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")

    # Return baseline metrics (train.py ignores this, but it's available)
    positions = engine._core.all_positions()
    wins = [p for p in positions if p.realized_pnl > 0]
    actual_wr = len(wins) / len(positions) if positions else 0.0

    # Expected win rate: entry_price (buy) or 1-entry_price (sell)
    # Build first-fill-side lookup once (O(fills) not O(positions*fills))
    from engine._engine import OrderSide as _OS
    first_fill_side: dict[str, int] = {}
    for f in all_fills:
        if f.instrument_id not in first_fill_side:
            first_fill_side[f.instrument_id] = int(f.side)

    expected_wrs = []
    for p in positions:
        side = first_fill_side.get(p.instrument_id)
        if side is not None and side == int(_OS.Buy):
            expected_wrs.append(p.avg_entry_price)
        elif side is not None:
            expected_wrs.append(1.0 - p.avg_entry_price)
        else:
            expected_wrs.append(0.5)
    mean_expected_wr = sum(expected_wrs) / len(expected_wrs) if expected_wrs else 0.0

    return {
        "pnl": pnl,
        "return_pct": round(100 * pnl / starting, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "total_fees": total_fees,
        "total_fills": len(all_fills),
        "win_rate": round(100 * actual_wr, 1),
        "expected_win_rate": round(100 * mean_expected_wr, 1),
        "win_rate_over_expected": round(100 * (actual_wr - mean_expected_wr), 1),
        "turnover": round(turnover, 2),
        "n_instruments": len(data.instruments),
    }


def run_backtest_analysis(
    strategy_factory: callable,
    event_tickers: list[str] | None = None,
    output_dir: str = "analyze",
    fee_rate: float = 0.07,
) -> dict:
    """Run backtest and generate full tearsheet reports (HTML, markdown, JSON).

    Args:
        strategy_factory: callable(instrument_id: str) -> Strategy instance.
        event_tickers: list of event tickers to backtest. None = in-sample (EVENT_TICKERS).
        output_dir: directory for output files.
        fee_rate: taker fee rate.

    Returns baseline metrics dict.
    """
    import subprocess

    from engine import BacktestEngine

    import report

    total_t0 = time.time()

    # Determine in-sample vs OOS
    is_oos = event_tickers is not None and set(event_tickers) != set(EVENT_TICKERS)

    if is_oos:
        data = _load_oos_cached_data(event_tickers)
    else:
        data = _load_cached_data()

    # Build engine
    engine = BacktestEngine(data.instruments, STARTING_BALANCE, fee_rate)
    for inst in data.instruments:
        engine.add_strategy(strategy_factory(inst.id))

    # Run
    print(f"Running analysis backtest with {len(data.instruments)} instruments...")
    t0 = time.time()
    engine.run(data.fair_values, data.orderbook_deltas, data.trades)
    run_time = time.time() - t0
    print(f"Backtest completed in {run_time:.1f}s")

    # Get commit hash for filenames
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
        ).strip()
    except Exception:
        commit_hash = "unknown"

    prefix = f"oos-{commit_hash}" if is_oos else f"in-sample-{commit_hash}"

    # Generate reports
    summary = report.create_report(
        engine=engine,
        fair_values=data.fair_values,
        output_dir=output_dir,
        prefix=prefix,
        starting_balance=STARTING_BALANCE,
    )

    total_time = time.time() - total_t0
    print(f"Analysis completed in {total_time:.1f}s")

    return summary


def _print_results(engine) -> float:
    """Print comprehensive backtest statistics."""
    from engine._engine import OrderSide, OrderStatus

    starting = engine._core.starting_balance()
    final = engine._core.balance()
    pnl = final - starting

    all_orders = engine._core.all_orders()
    all_fills = engine._core.all_fills()
    positions = engine._core.all_positions()

    filled = [o for o in all_orders if o.status == OrderStatus.Filled]
    canceled = [o for o in all_orders if o.status == OrderStatus.Canceled]
    rejected = [o for o in all_orders if o.status == OrderStatus.Rejected]

    buys = [o for o in filled if o.side == OrderSide.Buy]
    sells = [o for o in filled if o.side == OrderSide.Sell]
    maker_fills = [o for o in filled if o.is_maker is True]
    taker_fills = [o for o in filled if o.is_maker is False]

    fill_qtys = [o.filled_qty for o in filled]
    fill_prices = [o.avg_fill_price for o in filled if o.avg_fill_price is not None]

    print(f"\n{'=' * 70}")
    print("ORDER STATISTICS")
    print(f"{'=' * 70}")
    print(f"Total orders:      {len(all_orders):,}")
    if all_orders:
        print(f"Filled:            {len(filled):,}  ({100*len(filled)/len(all_orders):.1f}%)")
    print(f"Canceled:          {len(canceled):,}")
    print(f"Rejected:          {len(rejected):,}")
    print()
    print(f"Buy fills:         {len(buys):,}  ({sum(o.filled_qty for o in buys):,.0f} contracts)")
    print(f"Sell fills:        {len(sells):,}  ({sum(o.filled_qty for o in sells):,.0f} contracts)")
    if filled:
        print(f"Maker fills:       {len(maker_fills):,}  ({100*len(maker_fills)/len(filled):.1f}%)")
        print(f"Taker fills:       {len(taker_fills):,}  ({100*len(taker_fills)/len(filled):.1f}%)")
    print()
    if fill_qtys:
        print(f"Total fill qty:    {sum(fill_qtys):,.0f} contracts")
        print(f"Avg fill qty:      {np.mean(fill_qtys):,.1f}")
        print(f"Median fill qty:   {np.median(fill_qtys):,.1f}")
        print(f"Max fill qty:      {max(fill_qtys):,.0f}")
    if fill_prices:
        print(f"Avg fill price:    {np.mean(fill_prices):.4f}")
        print(f"Median fill price: {np.median(fill_prices):.4f}")

    # Account
    print(f"\n{'=' * 70}")
    print("ACCOUNT & PNL")
    print(f"{'=' * 70}")
    print(f"Starting balance:  ${starting:,.2f}")
    print(f"Final balance:     ${final:,.2f}")
    print(f"Total PnL:         ${pnl:+,.2f}")
    print(f"Return:            {100*pnl/starting:+.2f}%")

    total_fees = sum(f.fee for f in all_fills)
    print(f"Total fees:        ${total_fees:,.2f}")

    if not positions:
        print("\nNo positions generated.")
        return

    # Positions
    wins = [p for p in positions if p.realized_pnl > 0]
    losses = [p for p in positions if p.realized_pnl < 0]

    print(f"\n{'=' * 70}")
    print("POSITION STATISTICS")
    print(f"{'=' * 70}")
    print(f"Total positions:   {len(positions)}")
    print(f"Winning:           {len(wins)}  (PnL: ${sum(p.realized_pnl for p in wins):+,.2f})")
    print(f"Losing:            {len(losses)}  (PnL: ${sum(p.realized_pnl for p in losses):+,.2f})")
    if positions:
        print(f"Win rate:          {100*len(wins)/len(positions):.1f}%")
    print()
    if wins:
        print(f"Avg win:           ${np.mean([p.realized_pnl for p in wins]):+,.2f}")
        print(f"Largest win:       ${max(p.realized_pnl for p in wins):+,.2f}")
    if losses:
        print(f"Avg loss:          ${np.mean([p.realized_pnl for p in losses]):+,.2f}")
        print(f"Largest loss:      ${min(p.realized_pnl for p in losses):+,.2f}")
    if losses and sum(p.realized_pnl for p in losses) != 0:
        print(f"Profit factor:     {sum(p.realized_pnl for p in wins) / abs(sum(p.realized_pnl for p in losses)):.2f}")

    # Per-instrument
    inst_stats = []
    for p in positions:
        inst_stats.append({
            "instrument": p.instrument_id,
            "entries": p.entry_count,
            "pnl": p.realized_pnl,
            "final_qty": p.signed_qty,
        })
    df = pd.DataFrame(inst_stats).sort_values("pnl", ascending=False)
    print(f"\n{'=' * 70}")
    print("PER-INSTRUMENT BREAKDOWN")
    print(f"{'=' * 70}")
    with pd.option_context("display.max_rows", 50, "display.width", 120, "display.float_format", "{:.2f}".format):
        print(df.to_string(index=False))

    # Per-event
    print(f"\n{'=' * 70}")
    print("PER-EVENT BREAKDOWN")
    print(f"{'=' * 70}")
    event_pnl: dict[str, list[float]] = {}
    for p in positions:
        event = "-".join(p.instrument_id.split("-")[:2])
        event_pnl.setdefault(event, []).append(p.realized_pnl)
    for event, pnls in sorted(event_pnl.items()):
        print(f"  {event:12s}  {len(pnls):2d} instruments  PnL: ${sum(pnls):+,.2f}")

    # Capital efficiency
    total_bought = sum(o.avg_fill_price * o.filled_qty for o in buys if o.avg_fill_price)
    total_sold = sum(o.avg_fill_price * o.filled_qty for o in sells if o.avg_fill_price)
    turnover = total_bought + total_sold

    print(f"\n{'=' * 70}")
    print("CAPITAL EFFICIENCY")
    print(f"{'=' * 70}")
    print(f"Total bought:      ${total_bought:,.2f} notional")
    print(f"Total sold:        ${total_sold:,.2f} notional")
    print(f"Turnover:          ${turnover:,.2f}")
    if turnover > 0:
        print(f"PnL / Turnover:    {100*pnl/turnover:.2f}%")
    print(f"PnL / Starting:    {100*pnl/starting:+.2f}%")
    return turnover
