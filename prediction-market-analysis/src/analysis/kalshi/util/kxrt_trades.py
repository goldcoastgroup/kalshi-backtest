# prediction-market-analysis/src/analysis/kalshi/util/kxrt_trades.py
"""Shared DuckDB CTE for KXRT market trades in the last 5 days before close."""

from __future__ import annotations


def kxrt_base_cte(trades_dir: str, markets_dir: str) -> str:
    """Return a WITH-clause fragment for KXRT trades within 0–119 hours of market close.

    The CTE is named ``kxrt_trades`` and exposes all trade columns plus:
      - hours_to_close (INTEGER): whole hours remaining until market close

    Usage::

        sql = f'''
            WITH {kxrt_base_cte(self.trades_dir, self.markets_dir)}
            SELECT (hours_to_close / 6) * 6 AS bucket, SUM(count) AS vol
            FROM kxrt_trades GROUP BY bucket
        '''
    """
    return f"""
        kxrt_trades AS (
            SELECT
                t.*,
                date_diff('hour', t.created_time, m.close_time) AS hours_to_close
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN '{markets_dir}/*.parquet' m ON t.ticker = m.ticker
            WHERE t.ticker LIKE 'KXRT%'
              AND date_diff('hour', t.created_time, m.close_time) BETWEEN 0 AND 119
        )
    """
