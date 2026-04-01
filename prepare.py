"""
Data loading and caching for Kalshi backtests.

Loads fair values (via kxrt_fv), orderbook data (MongoDB), instruments
(Kalshi REST API), and settlement outcomes (MongoDB). All data is cached
to parquet/JSON files under ~/.cache/kalshi-backtest/.

This file is FROZEN — the training agent must not modify it.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

# Load env from sandbox
load_dotenv(Path(__file__).resolve().parent / ".." / "sandbox" / ".env")

# Add sandbox to path for kxrt_fv
_sandbox = str(Path(__file__).resolve().parent / ".." / "sandbox")
if _sandbox not in sys.path:
    sys.path.insert(0, _sandbox)

from kxrt_fv import backtest as kxrt_backtest  # noqa: E402

from engine._engine import (  # noqa: E402
    BookAction,
    F_LAST,
    F_SNAPSHOT,
    FairValueData,
    Instrument,
    OrderBookDelta,
    OrderSide,
)

CACHE_DIR = Path.home() / ".cache" / "kalshi-backtest"
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
MODEL_NAME = "xgb_ff99dd2"


@dataclass
class BacktestData:
    instruments: list[Instrument]
    fair_values: list[FairValueData]
    orderbook_deltas: list[OrderBookDelta]


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

    # Phase 5: Settlement outcomes + synthetic deltas
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

    print(f"Ready: {len(instruments)} instruments, {len(fair_values)} FV, {len(all_deltas)} OB deltas")
    return BacktestData(instruments=instruments, fair_values=fair_values, orderbook_deltas=all_deltas)


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
    result = []
    for _, row in df.iterrows():
        result.append(FairValueData(
            timestamp_ns=int(row["timestamp_ns"]),
            instrument_id=str(row["instrument_id"]),
            fv=float(row["fv"]),
            theta=float(row["theta"]),
            gamma_pos=float(row["gamma_pos"]),
            gamma_neg=float(row["gamma_neg"]),
            new_review=bool(row["new_review"]),
            hours_left=float(row["hours_left"]),
            cur_score=float(row["cur_score"]),
            total_reviews=int(row["total_reviews"]),
        ))
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
    return [
        Instrument(
            id=str(row["id"]),
            event_ticker=str(row["event_ticker"]),
            price_precision=int(row["price_precision"]),
            size_precision=int(row["size_precision"]),
            expiration_ns=int(row["expiration_ns"]),
        )
        for _, row in df.iterrows()
    ]


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
    action_map = {"CLEAR": BookAction.Clear, "ADD": BookAction.Add,
                  "UPDATE": BookAction.Update, "DELETE": BookAction.Delete}
    side_map = {"BUY": OrderSide.Buy, "SELL": OrderSide.Sell}
    result = []
    for _, row in df.iterrows():
        result.append(OrderBookDelta(
            instrument_id=str(row["instrument_id"]),
            timestamp_ns=int(row["timestamp_ns"]),
            action=action_map[row["action"]],
            side=side_map[row["side"]],
            price=float(row["price"]),
            size=float(row["size"]),
            flags=int(row["flags"]),
        ))
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
