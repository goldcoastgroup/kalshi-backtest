"""
Pure compute functions for backtest analysis metrics.

No plotting, no file I/O — engine objects in, dicts out.
"""

from __future__ import annotations

import statistics
from bisect import bisect_left
from collections import defaultdict


def build_fv_lookup(
    fair_values: list,
) -> dict[str, list[tuple[int, float, float, float]]]:
    """Build lookup: instrument_id -> sorted [(timestamp_ns, fv, hours_left, gamma)].

    Args:
        fair_values: list of FairValueData objects from backtest data loading.
    """
    lookup: dict[str, list[tuple[int, float, float, float]]] = defaultdict(list)
    for fv in fair_values:
        gamma = max(abs(fv.gamma_pos), abs(fv.gamma_neg))
        lookup[fv.instrument_id].append((fv.timestamp_ns, fv.fv, fv.hours_left, gamma))
    for entries in lookup.values():
        entries.sort(key=lambda e: e[0])
    return dict(lookup)


def _find_nearest_idx(
    instrument_id: str,
    timestamp_ns: int,
    fv_lookup: dict[str, list[tuple[int, float, float, float]]],
    max_delta_ns: int = 120_000_000_000,  # 2 minutes
) -> int | None:
    """Find index of nearest FV entry within tolerance. Returns None if no match."""
    entries = fv_lookup.get(instrument_id)
    if not entries:
        return None

    timestamps = [e[0] for e in entries]
    idx = bisect_left(timestamps, timestamp_ns)

    best_idx = None
    best_delta = max_delta_ns + 1

    for candidate in (idx - 1, idx):
        if 0 <= candidate < len(entries):
            delta = abs(entries[candidate][0] - timestamp_ns)
            if delta < best_delta:
                best_delta = delta
                best_idx = candidate

    if best_delta <= max_delta_ns:
        return best_idx
    return None


def find_closest_fv(
    instrument_id: str,
    timestamp_ns: int,
    fv_lookup: dict[str, list[tuple[int, float, float, float]]],
) -> float | None:
    """Return FV at closest timestamp within 2-minute tolerance."""
    idx = _find_nearest_idx(instrument_id, timestamp_ns, fv_lookup)
    if idx is None:
        return None
    return fv_lookup[instrument_id][idx][1]


def find_closest_hours_left(
    instrument_id: str,
    timestamp_ns: int,
    fv_lookup: dict[str, list[tuple[int, float, float, float]]],
) -> float | None:
    """Return hours_left at closest timestamp within 2-minute tolerance."""
    idx = _find_nearest_idx(instrument_id, timestamp_ns, fv_lookup)
    if idx is None:
        return None
    return fv_lookup[instrument_id][idx][2]


def compute_fill_details(
    engine,
    fv_lookup: dict[str, list[tuple[int, float, float, float]]],
) -> list[dict]:
    """Compute per-fill detail for all non-settlement fills.

    Returns one dict per fill with keys: instrument_id, side, fill_price,
    fill_qty, fv_at_fill, edge, hours_left, is_maker, time_to_fill_sec.
    """
    from engine._engine import OrderSide

    order_map: dict[str, object] = {}
    for order in engine._core.all_orders():
        order_map[order.id] = order

    results = []
    for fill in engine._core.all_fills():
        if fill.price <= 0.01 or fill.price >= 0.99:
            continue

        iid = fill.instrument_id
        fv = find_closest_fv(iid, fill.timestamp_ns, fv_lookup)
        if fv is None:
            continue

        is_buy = fill.side == OrderSide.Buy
        side_str = "BUY" if is_buy else "SELL"
        edge = (fv - fill.price) if is_buy else (fill.price - fv)

        hours_left = find_closest_hours_left(iid, fill.timestamp_ns, fv_lookup)

        time_to_fill_sec = None
        order = order_map.get(fill.order_id)
        if order is not None and order.submit_timestamp_ns > 0:
            duration_ns = fill.timestamp_ns - order.submit_timestamp_ns
            time_to_fill_sec = round(duration_ns / 1_000_000_000, 2)

        results.append({
            "instrument_id": iid,
            "side": side_str,
            "fill_price": round(fill.price, 4),
            "fill_qty": fill.quantity,
            "fv_at_fill": round(fv, 4),
            "edge": round(edge, 4),
            "hours_left": round(hours_left, 2) if hours_left is not None else None,
            "is_maker": fill.is_maker,
            "time_to_fill_sec": time_to_fill_sec,
        })

    return results


def compute_position_details(
    engine,
    fv_lookup: dict[str, list[tuple[int, float, float, float]]],
) -> list[dict]:
    """Compute per-position lifecycle detail.

    Returns one dict per position with keys: instrument_id, entry_side,
    avg_entry_price, realized_pnl, fv_at_entry, theoretical_edge,
    exit_type, hold_time_hours, expected_win_rate.
    """
    from engine._engine import OrderSide

    positions = engine._core.all_positions()
    all_fills = engine._core.all_fills()

    fills_by_inst: dict[str, list] = defaultdict(list)
    for fill in all_fills:
        fills_by_inst[fill.instrument_id].append(fill)
    for fills in fills_by_inst.values():
        fills.sort(key=lambda f: f.timestamp_ns)

    results = []
    for pos in positions:
        iid = pos.instrument_id
        inst_fills = fills_by_inst.get(iid, [])
        if not inst_fills:
            continue

        first_fill = inst_fills[0]
        is_buy = first_fill.side == OrderSide.Buy
        entry_side = "BUY" if is_buy else "SELL"

        first_ts = inst_fills[0].timestamp_ns
        last_ts = inst_fills[-1].timestamp_ns
        hold_time_hours = (last_ts - first_ts) / 3_600_000_000_000

        fv_at_entry = find_closest_fv(iid, first_ts, fv_lookup)

        theoretical_edge = None
        if fv_at_entry is not None:
            if is_buy:
                theoretical_edge = fv_at_entry - pos.avg_entry_price
            else:
                theoretical_edge = pos.avg_entry_price - fv_at_entry

        last_fill = inst_fills[-1]
        exit_type = "settlement" if (last_fill.price <= 0.01 or last_fill.price >= 0.99) else "trade"

        if is_buy:
            expected_win_rate = pos.avg_entry_price
        else:
            expected_win_rate = 1.0 - pos.avg_entry_price

        results.append({
            "instrument_id": iid,
            "entry_side": entry_side,
            "avg_entry_price": round(pos.avg_entry_price, 4),
            "realized_pnl": round(pos.realized_pnl, 4),
            "signed_qty": pos.signed_qty,
            "entry_count": pos.entry_count,
            "fv_at_entry": round(fv_at_entry, 4) if fv_at_entry is not None else None,
            "theoretical_edge": round(theoretical_edge, 4) if theoretical_edge is not None else None,
            "exit_type": exit_type,
            "hold_time_hours": round(hold_time_hours, 2),
            "expected_win_rate": round(expected_win_rate, 4),
        })

    return results


def compute_instrument_summary(
    fills: list[dict],
    positions: list[dict],
    engine,
) -> list[dict]:
    """Compute per-instrument summary metrics."""
    orders_per_inst: dict[str, int] = defaultdict(int)
    for order in engine._core.all_orders():
        orders_per_inst[order.instrument_id] += 1

    fill_agg: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "total_edge": 0.0, "total_qty": 0.0, "total_capital": 0.0,
    })
    for f in fills:
        iid = f["instrument_id"]
        fa = fill_agg[iid]
        fa["count"] += 1
        fa["total_edge"] += f["edge"] * f["fill_qty"]
        fa["total_qty"] += f["fill_qty"]
        if f["side"] == "BUY":
            fa["total_capital"] += f["fill_price"] * f["fill_qty"]
        else:
            fa["total_capital"] += (1 - f["fill_price"]) * f["fill_qty"]

    pos_agg: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "total_pnl": 0.0, "wins": 0,
    })
    for p in positions:
        iid = p["instrument_id"]
        pa = pos_agg[iid]
        pa["count"] += 1
        pa["total_pnl"] += p["realized_pnl"]
        if p["realized_pnl"] > 0:
            pa["wins"] += 1

    all_iids = set(fill_agg.keys()) | set(pos_agg.keys())

    results = []
    for iid in sorted(all_iids):
        fa = fill_agg.get(iid, {"count": 0, "total_edge": 0.0, "total_qty": 0.0, "total_capital": 0.0})
        pa = pos_agg.get(iid, {"count": 0, "total_pnl": 0.0, "wins": 0})

        avg_edge = (fa["total_edge"] / fa["total_qty"]) if fa["total_qty"] > 0 else 0.0
        fill_rate = fa["count"] / orders_per_inst[iid] if orders_per_inst.get(iid, 0) > 0 else 0.0
        capital = fa["total_capital"]
        pnl_per_dollar = pa["total_pnl"] / capital if capital > 0 else None

        results.append({
            "instrument_id": iid,
            "num_fills": fa["count"],
            "num_positions": pa["count"],
            "total_pnl": round(pa["total_pnl"], 4),
            "avg_edge_at_fill": round(avg_edge, 4),
            "fill_rate": round(fill_rate, 4),
            "capital_deployed": round(capital, 4),
            "pnl_per_dollar": round(pnl_per_dollar, 4) if pnl_per_dollar is not None else None,
        })

    return results


def compute_win_rate_vs_expected(
    positions: list[dict],
    bucket_width: float = 0.05,
) -> list[dict]:
    """Compute win rate vs expected by effective probability bucket."""
    if not positions:
        return []

    buckets: dict[float, list[dict]] = defaultdict(list)
    for p in positions:
        ew = p["expected_win_rate"]
        bucket_mid = round((int(ew / bucket_width) + 0.5) * bucket_width, 4)
        bucket_mid = max(bucket_width / 2, min(1.0 - bucket_width / 2, bucket_mid))
        buckets[bucket_mid].append(p)

    results = []
    for mid in sorted(buckets.keys()):
        bucket_pos = buckets[mid]
        n = len(bucket_pos)
        wins = sum(1 for p in bucket_pos if p["realized_pnl"] > 0)
        actual_wr = wins / n
        expected_wr = statistics.mean(p["expected_win_rate"] for p in bucket_pos)

        results.append({
            "bucket_mid": mid,
            "count": n,
            "actual_win_rate": round(actual_wr, 4),
            "expected_win_rate": round(expected_wr, 4),
            "delta": round(actual_wr - expected_wr, 4),
        })

    return results


def compute_summary_stats(
    fills: list[dict],
    positions: list[dict],
    instruments: list[dict],
    engine,
) -> dict:
    """Compute aggregate summary statistics across all metrics."""
    starting = engine._core.starting_balance()
    final = engine._core.balance()
    pnl = final - starting
    all_fills_raw = engine._core.all_fills()
    total_fees = sum(f.fee for f in all_fills_raw)

    from engine._engine import OrderSide, OrderStatus
    all_orders = engine._core.all_orders()
    filled_orders = [o for o in all_orders if o.status == OrderStatus.Filled]
    buys = [o for o in filled_orders if o.side == OrderSide.Buy]
    sells = [o for o in filled_orders if o.side == OrderSide.Sell]
    total_bought = sum(o.avg_fill_price * o.filled_qty for o in buys if o.avg_fill_price)
    total_sold = sum(o.avg_fill_price * o.filled_qty for o in sells if o.avg_fill_price)
    turnover = total_bought + total_sold

    stats: dict = {
        "pnl": round(pnl, 2),
        "return_pct": round(100 * pnl / starting, 2),
        "starting_balance": starting,
        "final_balance": round(final, 2),
        "total_fees": round(total_fees, 2),
        "turnover": round(turnover, 2),
    }

    if fills:
        edges = [f["edge"] for f in fills]
        total_qty = sum(f["fill_qty"] for f in fills)
        qty_weighted_edge = sum(f["edge"] * f["fill_qty"] for f in fills) / total_qty if total_qty > 0 else 0.0
        stats["total_fills"] = len(fills)
        stats["mean_edge"] = round(statistics.mean(edges), 4)
        stats["median_edge"] = round(statistics.median(edges), 4)
        stats["qty_weighted_edge"] = round(qty_weighted_edge, 4)
        stats["pct_positive_edge"] = round(100 * sum(1 for e in edges if e > 0) / len(edges), 1)

        buy_edges = [f["edge"] for f in fills if f["side"] == "BUY"]
        sell_edges = [f["edge"] for f in fills if f["side"] == "SELL"]
        stats["mean_edge_buy"] = round(statistics.mean(buy_edges), 4) if buy_edges else None
        stats["mean_edge_sell"] = round(statistics.mean(sell_edges), 4) if sell_edges else None
    else:
        stats["total_fills"] = 0

    if positions:
        stats["total_positions"] = len(positions)
        wins = [p for p in positions if p["realized_pnl"] > 0]
        losses = [p for p in positions if p["realized_pnl"] < 0]

        actual_wr = len(wins) / len(positions)
        expected_wrs = [p["expected_win_rate"] for p in positions]
        mean_expected_wr = statistics.mean(expected_wrs)

        stats["win_rate"] = round(100 * actual_wr, 1)
        stats["expected_win_rate"] = round(100 * mean_expected_wr, 1)
        stats["win_rate_over_expected"] = round(100 * (actual_wr - mean_expected_wr), 1)

        settled = [p for p in positions if p["exit_type"] == "settlement"]
        traded = [p for p in positions if p["exit_type"] == "trade"]
        stats["settled_count"] = len(settled)
        stats["traded_count"] = len(traded)
        stats["settled_pnl"] = round(sum(p["realized_pnl"] for p in settled), 2)
        stats["traded_pnl"] = round(sum(p["realized_pnl"] for p in traded), 2)

        if wins:
            stats["avg_win"] = round(statistics.mean(p["realized_pnl"] for p in wins), 2)
        if losses:
            stats["avg_loss"] = round(statistics.mean(p["realized_pnl"] for p in losses), 2)
        if losses and sum(p["realized_pnl"] for p in losses) != 0:
            stats["profit_factor"] = round(
                sum(p["realized_pnl"] for p in wins) / abs(sum(p["realized_pnl"] for p in losses)), 2
            )
    else:
        stats["total_positions"] = 0

    if instruments:
        stats["active_instruments"] = len(instruments)
        stats["profitable_instruments"] = sum(1 for i in instruments if i["total_pnl"] > 0)
        total_capital = sum(i["capital_deployed"] for i in instruments)
        stats["total_capital_deployed"] = round(total_capital, 2)
        stats["pnl_per_dollar"] = round(pnl / total_capital, 4) if total_capital > 0 else None

    return stats
