"""
Backtest report generator.

Orchestrates metrics computation and produces three output artifacts:
- HTML tearsheet (charts + tables)
- Markdown summary (for LLM analysis)
- JSON sidecar (full detail dump)
"""

from __future__ import annotations

import json
import os

from metrics import (
    build_fv_lookup,
    compute_fill_details,
    compute_instrument_summary,
    compute_position_details,
    compute_summary_stats,
    compute_win_rate_vs_expected,
)


def create_report(
    engine,
    fair_values: list,
    output_dir: str,
    prefix: str,
    starting_balance: float,
) -> dict:
    """Generate a complete backtest report.

    Produces three files in output_dir:
    - {prefix}.html — visual charts and tables
    - {prefix}.md — markdown summary for LLM analysis
    - {prefix}.json — full detail dump

    Returns the summary stats dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build shared data structures
    fv_lookup = build_fv_lookup(fair_values)

    # Compute all metrics
    fills = compute_fill_details(engine, fv_lookup)
    positions = compute_position_details(engine, fv_lookup)
    instruments = compute_instrument_summary(fills, positions, engine)
    wr_buckets = compute_win_rate_vs_expected(positions)
    summary = compute_summary_stats(fills, positions, instruments, engine)

    # Write outputs
    md_path = os.path.join(output_dir, f"{prefix}.md")
    _write_markdown(summary, instruments, wr_buckets, positions, md_path, prefix)

    json_path = os.path.join(output_dir, f"{prefix}.json")
    _write_json(summary, fills, positions, instruments, wr_buckets, json_path)

    html_path = os.path.join(output_dir, f"{prefix}.html")
    _write_html(
        engine, fills, positions, instruments, wr_buckets, summary,
        starting_balance, html_path, prefix,
    )

    print(f"Report written to {output_dir}/")
    print(f"  - {html_path}")
    print(f"  - {md_path}")
    print(f"  - {json_path}")

    return summary


def _write_markdown(
    summary: dict,
    instruments: list[dict],
    wr_buckets: list[dict],
    positions: list[dict],
    output_path: str,
    prefix: str,
) -> None:
    """Write markdown summary for agent consumption."""
    s = summary
    lines = [
        f"# Backtest Analysis — {prefix}",
        "",
        "## Top-Line Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| PnL | ${s.get('pnl', 0):+,.2f} |",
        f"| Return % | {s.get('return_pct', 0):+.2f}% |",
        f"| Win Rate | {s.get('win_rate', 0):.1f}% |",
        f"| Expected Win Rate | {s.get('expected_win_rate', 0):.1f}% |",
        f"| **Win Rate Over Expected** | **{s.get('win_rate_over_expected', 0):+.1f}%** |",
        f"| Mean Edge at Fill | {s.get('mean_edge', 0):.4f} |",
        f"| Median Edge at Fill | {s.get('median_edge', 0):.4f} |",
        f"| Qty-Weighted Edge | {s.get('qty_weighted_edge', 0):.4f} |",
        f"| % Positive Edge Fills | {s.get('pct_positive_edge', 0):.1f}% |",
        f"| Total Fees | ${s.get('total_fees', 0):,.2f} |",
        f"| Turnover | ${s.get('turnover', 0):,.2f} |",
        f"| Total Fills | {s.get('total_fills', 0):,} |",
        f"| Total Positions | {s.get('total_positions', 0)} |",
        f"| Profit Factor | {s.get('profit_factor', 'N/A')} |",
        "",
    ]

    # Per-event breakdown
    event_agg: dict[str, dict] = {}
    for p in positions:
        event = "-".join(p["instrument_id"].split("-")[:2])
        if event not in event_agg:
            event_agg[event] = {"instruments": set(), "pnl": 0.0, "wins": 0, "total": 0}
        event_agg[event]["instruments"].add(p["instrument_id"])
        event_agg[event]["pnl"] += p["realized_pnl"]
        event_agg[event]["total"] += 1
        if p["realized_pnl"] > 0:
            event_agg[event]["wins"] += 1

    if event_agg:
        lines.extend([
            "## Per-Event Breakdown",
            "",
            "| Event | Instruments | PnL | Win Rate |",
            "|-------|-------------|-----|----------|",
        ])
        for event in sorted(event_agg.keys()):
            ea = event_agg[event]
            wr = 100 * ea["wins"] / ea["total"] if ea["total"] > 0 else 0
            lines.append(
                f"| {event} | {len(ea['instruments'])} | ${ea['pnl']:+,.2f} | {wr:.0f}% |"
            )
        lines.append("")

    # Top/bottom instruments
    if instruments:
        sorted_inst = sorted(instruments, key=lambda x: x["total_pnl"], reverse=True)
        top_n = min(5, len(sorted_inst))
        lines.extend([
            "## Top Instruments",
            "",
            "| Instrument | PnL | Avg Edge | Fills | PnL/$ |",
            "|------------|-----|----------|-------|-------|",
        ])
        for inst in sorted_inst[:top_n]:
            pnl_d = f"{inst['pnl_per_dollar']:.4f}" if inst["pnl_per_dollar"] is not None else "N/A"
            lines.append(
                f"| {inst['instrument_id']} | ${inst['total_pnl']:+,.2f} | {inst['avg_edge_at_fill']:.4f} | {inst['num_fills']} | {pnl_d} |"
            )
        lines.append("")

        lines.extend([
            "## Bottom Instruments",
            "",
            "| Instrument | PnL | Avg Edge | Fills | PnL/$ |",
            "|------------|-----|----------|-------|-------|",
        ])
        for inst in sorted_inst[-top_n:]:
            pnl_d = f"{inst['pnl_per_dollar']:.4f}" if inst["pnl_per_dollar"] is not None else "N/A"
            lines.append(
                f"| {inst['instrument_id']} | ${inst['total_pnl']:+,.2f} | {inst['avg_edge_at_fill']:.4f} | {inst['num_fills']} | {pnl_d} |"
            )
        lines.append("")

    # Win rate vs expected buckets
    if wr_buckets:
        lines.extend([
            "## Win Rate vs Expected",
            "",
            "| Probability Bucket | Count | Actual WR | Expected WR | Delta |",
            "|--------------------|-------|-----------|-------------|-------|",
        ])
        for b in wr_buckets:
            lines.append(
                f"| {b['bucket_mid']:.2f} | {b['count']} | {100*b['actual_win_rate']:.0f}% | {100*b['expected_win_rate']:.0f}% | {100*b['delta']:+.1f}% |"
            )
        lines.append("")

    # Capital efficiency
    lines.extend([
        "## Capital Efficiency",
        "",
        f"- Starting balance: ${s.get('starting_balance', 0):,.2f}",
        f"- Final balance: ${s.get('final_balance', 0):,.2f}",
        f"- Total capital deployed: ${s.get('total_capital_deployed', 0):,.2f}",
        f"- PnL per dollar deployed: {s.get('pnl_per_dollar', 'N/A')}",
        f"- PnL / Turnover: {100 * s.get('pnl', 0) / s['turnover']:.2f}%" if s.get("turnover", 0) > 0 else "- PnL / Turnover: N/A",
        "",
    ])

    # Settlement vs trade
    lines.extend([
        "## Settlement vs Trade Exits",
        "",
        f"- Settled at expiry: {s.get('settled_count', 0)} positions, PnL ${s.get('settled_pnl', 0):+,.2f}",
        f"- Closed by trade: {s.get('traded_count', 0)} positions, PnL ${s.get('traded_pnl', 0):+,.2f}",
        "",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def _write_json(
    summary: dict,
    fills: list[dict],
    positions: list[dict],
    instruments: list[dict],
    wr_buckets: list[dict],
    output_path: str,
) -> None:
    """Write full detail JSON sidecar."""
    data = {
        "summary": summary,
        "fills": fills,
        "positions": positions,
        "instruments": instruments,
        "win_rate_buckets": wr_buckets,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _write_html(
    engine,
    fills: list[dict],
    positions: list[dict],
    instruments: list[dict],
    wr_buckets: list[dict],
    summary: dict,
    starting_balance: float,
    output_path: str,
    prefix: str,
) -> None:
    """Write combined HTML tearsheet with plotly charts and tables."""
    import plotly.graph_objects as go
    from plotly.io import to_html

    charts = []

    # 1. Cumulative PnL over time (from fill sequence)
    fig1 = go.Figure()
    if fills:
        cum_pnl = []
        running = 0.0
        for f in fills:
            edge_pnl = f["edge"] * f["fill_qty"]
            running += edge_pnl
            cum_pnl.append(running)
        fig1.add_trace(go.Scatter(
            y=cum_pnl, mode="lines", name="Cumulative Edge PnL",
            line={"color": "#2196F3", "width": 2},
        ))
    fig1.update_layout(
        title="Cumulative Edge PnL Over Fills",
        xaxis_title="Fill #", yaxis_title="Cumulative Edge ($)",
        template="plotly_white", height=400,
    )
    charts.append(fig1)

    # 2. Win rate vs expected
    fig2 = go.Figure()
    if wr_buckets:
        mids = [b["bucket_mid"] for b in wr_buckets]
        fig2.add_trace(go.Bar(
            x=mids, y=[100 * b["actual_win_rate"] for b in wr_buckets],
            name="Actual Win Rate", marker_color="#2196F3",
        ))
        fig2.add_trace(go.Scatter(
            x=mids, y=[100 * b["expected_win_rate"] for b in wr_buckets],
            mode="lines+markers", name="Expected Win Rate",
            line={"color": "#FF9800", "width": 2, "dash": "dash"},
        ))
        fig2.add_trace(go.Bar(
            x=mids, y=[b["count"] for b in wr_buckets],
            name="Count", yaxis="y2",
            marker_color="rgba(200,200,200,0.4)",
        ))
    fig2.update_layout(
        title="Win Rate vs Expected by Effective Probability",
        xaxis_title="Effective Probability", yaxis_title="Win Rate (%)",
        template="plotly_white", height=400,
        yaxis2={"title": "Count", "overlaying": "y", "side": "right", "showgrid": False},
        barmode="overlay",
    )
    charts.append(fig2)

    # 3. PnL by effective probability
    fig3 = go.Figure()
    if wr_buckets:
        bucket_pnl: dict[float, float] = {}
        for p in positions:
            ew = p["expected_win_rate"]
            bucket_mid = round((int(ew / 0.05) + 0.5) * 0.05, 4)
            bucket_mid = max(0.025, min(0.975, bucket_mid))
            bucket_pnl[bucket_mid] = bucket_pnl.get(bucket_mid, 0) + p["realized_pnl"]
        mids = sorted(bucket_pnl.keys())
        pnls = [bucket_pnl[m] for m in mids]
        colors = ["#4CAF50" if p >= 0 else "#F44336" for p in pnls]
        fig3.add_trace(go.Bar(
            x=mids, y=pnls, marker_color=colors,
            text=[f"${p:+.0f}" for p in pnls], textposition="outside",
        ))
    fig3.update_layout(
        title="PnL by Effective Probability",
        xaxis_title="Effective Probability", yaxis_title="PnL ($)",
        template="plotly_white", height=400,
    )
    charts.append(fig3)

    # 4. Edge at fill distribution
    fig4 = go.Figure()
    if fills:
        edges = [f["edge"] for f in fills]
        fig4.add_trace(go.Histogram(
            x=edges, nbinsx=50, name="Edge at Fill",
            marker_color="#2196F3",
        ))
        fig4.add_vline(x=0, line_dash="dash", line_color="red")
    fig4.update_layout(
        title="Edge at Fill Distribution",
        xaxis_title="Edge (FV - Fill Price)", yaxis_title="Count",
        template="plotly_white", height=400,
    )
    charts.append(fig4)

    # 5. Per-instrument PnL
    fig5 = go.Figure()
    if instruments:
        sorted_inst = sorted(instruments, key=lambda x: x["total_pnl"], reverse=True)
        labels = [i["instrument_id"] for i in sorted_inst]
        pnls = [i["total_pnl"] for i in sorted_inst]
        colors = ["#4CAF50" if p >= 0 else "#F44336" for p in pnls]
        fig5.add_trace(go.Bar(
            y=labels, x=pnls, orientation="h", marker_color=colors,
            text=[f"${p:+.2f}" for p in pnls], textposition="outside",
        ))
    fig5.update_layout(
        title="PnL by Instrument",
        xaxis_title="PnL ($)", template="plotly_white",
        height=max(400, len(instruments) * 25 + 100) if instruments else 400,
    )
    charts.append(fig5)

    # 6. PnL by hours_left
    fig6 = go.Figure()
    if fills:
        hours_pnl: dict[int, float] = {}
        for f in fills:
            if f["hours_left"] is not None:
                bucket = int(f["hours_left"] / 12) * 12
                hours_pnl[bucket] = hours_pnl.get(bucket, 0) + f["edge"] * f["fill_qty"]
        if hours_pnl:
            buckets_sorted = sorted(hours_pnl.keys())
            pnls = [hours_pnl[b] for b in buckets_sorted]
            colors = ["#4CAF50" if p >= 0 else "#F44336" for p in pnls]
            fig6.add_trace(go.Bar(
                x=[f"{b}-{b+12}h" for b in buckets_sorted], y=pnls,
                marker_color=colors,
            ))
    fig6.update_layout(
        title="Edge PnL by Hours to Expiry",
        xaxis_title="Hours Left", yaxis_title="Edge PnL ($)",
        template="plotly_white", height=400,
    )
    charts.append(fig6)

    # 7. Maker vs taker breakdown
    fig7 = go.Figure()
    if fills:
        maker_fills = [f for f in fills if f["is_maker"]]
        taker_fills = [f for f in fills if not f["is_maker"]]
        maker_pnl = sum(f["edge"] * f["fill_qty"] for f in maker_fills)
        taker_pnl = sum(f["edge"] * f["fill_qty"] for f in taker_fills)
        fig7.add_trace(go.Bar(
            x=["Maker", "Taker"],
            y=[len(maker_fills), len(taker_fills)],
            name="Fill Count", marker_color=["#2196F3", "#FF9800"],
            text=[f"n={len(maker_fills)}", f"n={len(taker_fills)}"],
            textposition="outside",
        ))
        fig7.add_trace(go.Bar(
            x=["Maker", "Taker"],
            y=[maker_pnl, taker_pnl],
            name="Edge PnL", yaxis="y2",
            marker_color=["#90CAF9", "#FFCC80"],
            text=[f"${maker_pnl:+.2f}", f"${taker_pnl:+.2f}"],
            textposition="outside",
        ))
    fig7.update_layout(
        title="Maker vs Taker Fills",
        template="plotly_white", height=400,
        yaxis={"title": "Count"},
        yaxis2={"title": "Edge PnL ($)", "overlaying": "y", "side": "right"},
        barmode="group",
    )
    charts.append(fig7)

    # 8. Settlement vs trade exit PnL
    fig8 = go.Figure()
    s = summary
    if s.get("settled_count", 0) > 0 or s.get("traded_count", 0) > 0:
        fig8.add_trace(go.Bar(
            x=["Settled at Expiry", "Closed by Trade"],
            y=[s.get("settled_pnl", 0), s.get("traded_pnl", 0)],
            text=[
                f"${s.get('settled_pnl', 0):+.2f} ({s.get('settled_count', 0)} pos)",
                f"${s.get('traded_pnl', 0):+.2f} ({s.get('traded_count', 0)} pos)",
            ],
            textposition="outside",
            marker_color=["#FF9800", "#2196F3"],
        ))
    fig8.update_layout(
        title="PnL: Settlement vs Trade Exit",
        yaxis_title="PnL ($)", template="plotly_white", height=400,
    )
    charts.append(fig8)

    # 9. Fills over time scatter (color by instrument, hover details)
    fig9 = go.Figure()
    if fills:
        from datetime import datetime, timezone

        # Group fills by instrument for color coding
        fills_by_inst: dict[str, list[dict]] = {}
        for f in fills:
            fills_by_inst.setdefault(f["instrument_id"], []).append(f)

        for iid in sorted(fills_by_inst.keys()):
            inst_fills = fills_by_inst[iid]
            timestamps = [
                datetime.fromtimestamp(f["timestamp_ns"] / 1_000_000_000, tz=timezone.utc)
                for f in inst_fills
            ]
            prices = [f["fill_price"] for f in inst_fills]
            hover = [
                f"{iid}<br>Price: {f['fill_price']:.2f}<br>Side: {f['side']}<br>"
                f"Qty: {f['fill_qty']:.0f}<br>Edge: {f['edge']:.4f}<br>FV: {f['fv_at_fill']:.4f}"
                for f in inst_fills
            ]
            fig9.add_trace(go.Scatter(
                x=timestamps, y=prices, mode="markers",
                name=iid,
                marker={"size": 5, "opacity": 0.7},
                hovertext=hover, hoverinfo="text",
            ))
    fig9.update_layout(
        title="Fills Over Time by Price",
        xaxis_title="Time (UTC)", yaxis_title="Fill Price",
        template="plotly_white", height=500,
        legend={"title": "Instrument", "font": {"size": 10}},
    )
    charts.append(fig9)

    # Build HTML
    html_parts = [
        "<html><head>",
        f"<title>Backtest Tearsheet — {prefix}</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
        "table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 14px; }",
        "th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }",
        "th { background-color: #f5f5f5; font-weight: 600; text-align: center; }",
        "td:first-child, th:first-child { text-align: left; }",
        "tr:nth-child(even) { background-color: #fafafa; }",
        ".positive { color: #4CAF50; } .negative { color: #F44336; }",
        "h1, h2 { text-align: center; }",
        "</style>",
        "</head><body>",
        f"<h1>Backtest Tearsheet — {prefix}</h1>",
    ]

    for i, fig in enumerate(charts):
        html_parts.append(to_html(fig, full_html=False, include_plotlyjs=(i == 0)))

    # Summary stats table
    html_parts.append("<h2>Summary Statistics</h2>")
    html_parts.append("<table>")
    for key, val in summary.items():
        if isinstance(val, float):
            if "pct" in key or "rate" in key or key == "win_rate_over_expected":
                display = f"{val:+.1f}%" if "over" in key else f"{val:.1f}%"
            elif "pnl" in key or "balance" in key or "fees" in key or "turnover" in key or key in ("avg_win", "avg_loss") or "capital" in key:
                display = f"${val:+,.2f}" if "pnl" in key else f"${val:,.2f}"
            else:
                display = f"{val:.4f}"
        else:
            display = str(val)
        label = key.replace("_", " ").title()
        html_parts.append(f"<tr><td>{label}</td><td>{display}</td></tr>")
    html_parts.append("</table>")

    # Instrument scorecard table
    if instruments:
        html_parts.append("<h2>Instrument Scorecard</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Instrument</th><th>Fills</th><th>PnL</th><th>Avg Price</th><th>Avg Edge</th><th>Fill Rate</th><th>Capital</th><th>PnL/$</th></tr>")
        sorted_inst = sorted(instruments, key=lambda x: x["total_pnl"], reverse=True)
        for inst in sorted_inst:
            pnl_class = "positive" if inst["total_pnl"] >= 0 else "negative"
            pnl_d = f"{inst['pnl_per_dollar']:.4f}" if inst["pnl_per_dollar"] is not None else "N/A"
            avg_p = f"{inst['avg_price']:.4f}" if inst["avg_price"] is not None else "N/A"
            html_parts.append(
                f"<tr><td>{inst['instrument_id']}</td>"
                f"<td>{inst['num_fills']}</td>"
                f"<td class='{pnl_class}'>${inst['total_pnl']:+,.2f}</td>"
                f"<td>{avg_p}</td>"
                f"<td>{inst['avg_edge_at_fill']:.4f}</td>"
                f"<td>{100*inst['fill_rate']:.1f}%</td>"
                f"<td>${inst['capital_deployed']:,.2f}</td>"
                f"<td>{pnl_d}</td></tr>"
            )
        html_parts.append("</table>")

    html_parts.append("</body></html>")

    with open(output_path, "w") as f:
        f.write("\n".join(html_parts))
