"""KXRT effective spread proxy (median consecutive trade price gap) by hours to close."""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.kalshi.util.kxrt_trades import kxrt_base_cte
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


class KxrtEffectiveSpreadRtAnalysis(Analysis):
    """Median consecutive trade price gap as a realized spread proxy across KXRT markets."""

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
    ):
        super().__init__(
            name="kxrt_effective_spread_rt",
            description="KXRT effective spread proxy by hours to close",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")

    def run(self) -> AnalysisOutput:
        con = duckdb.connect()
        with self.progress("Querying KXRT trades"):
            df = con.execute(
                f"""
                WITH {kxrt_base_cte(self.trades_dir, self.markets_dir)},
                trade_sequence AS (
                    SELECT
                        (hours_to_close / 6) * 6 AS hours_bucket,
                        yes_price,
                        LAG(yes_price) OVER (PARTITION BY ticker ORDER BY created_time) AS prev_price
                    FROM kxrt_trades
                )
                SELECT
                    hours_bucket,
                    MEDIAN(ABS(yes_price - prev_price)) AS median_spread_cents
                FROM trade_sequence
                WHERE prev_price IS NOT NULL
                GROUP BY hours_bucket
                ORDER BY hours_bucket DESC
                """
            ).df()
        return AnalysisOutput(figure=self._figure(df), data=df, chart=self._chart(df))

    def _figure(self, df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 6))
        labels = [f"{int(b)}-{int(b) + 6}h" for b in df["hours_bucket"]]
        ax.plot(labels, df["median_spread_cents"], marker="o", color="#4C72B0")
        ax.set_xlabel("Hours to Close")
        ax.set_ylabel("Median |ΔPrice| (cents)")
        ax.set_title("KXRT Effective Spread Proxy by Hours to Close")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    def _chart(self, df: pd.DataFrame) -> ChartConfig:
        return ChartConfig(
            type=ChartType.LINE,
            data=[
                {
                    "hours_to_close": f"{int(r['hours_bucket'])}-{int(r['hours_bucket']) + 6}h",
                    "median_spread_cents": float(r["median_spread_cents"]),
                }
                for _, r in df.iterrows()
            ],
            xKey="hours_to_close",
            yKeys=["median_spread_cents"],
            title="KXRT Effective Spread Proxy",
            xLabel="Hours to Close",
            yLabel="Median |ΔPrice| (cents)",
            yUnit=UnitType.CENTS,
        )
