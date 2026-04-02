"""Kalshi backtest engine — Rust core with Python orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine._engine import (  # noqa: F401
    AggressorSide,
    BookAction,
    EngineCore,
    F_LAST,
    F_SNAPSHOT,
    FairValueData,
    Fill,
    Instrument,
    Order,
    OrderBookDelta,
    OrderSide,
    OrderStatus,
    Position,
    TradeTick,
)
from engine.strategy import Strategy


class BacktestEngine:
    """Orchestrates event replay, calling into the Rust EngineCore."""

    def __init__(
        self,
        instruments: list[Instrument],
        starting_balance: float,
        fee_rate: float = 0.07,
    ):
        self._core = EngineCore(instruments, starting_balance, fee_rate)
        self._instruments = {inst.id: inst for inst in instruments}
        self._strategies: dict[str, Strategy] = {}

    def add_strategy(self, strategy: Strategy) -> None:
        strategy._bind(self._core)
        self._strategies[strategy.instrument_id] = strategy

    def run(
        self,
        fair_values: list[FairValueData],
        orderbook_deltas: list[OrderBookDelta],
        trades: list[TradeTick] | None = None,
    ) -> None:
        """Run the backtest: merge events, replay chronologically."""
        # Build sorted event list: (timestamp_ns, type_priority, event)
        # type_priority: 0=OB delta (book first), 1=trade tick, 2=FV data
        events: list[tuple[int, int, object]] = []
        for delta in orderbook_deltas:
            events.append((delta.timestamp_ns, 0, delta))
        if trades:
            for trade in trades:
                events.append((trade.timestamp_ns, 1, trade))
        for fv in fair_values:
            events.append((fv.timestamp_ns, 2, fv))

        events.sort(key=lambda e: (e[0], e[1]))

        # Start strategies
        for strategy in self._strategies.values():
            strategy.on_start()

        # Event loop
        for _, type_prio, event in events:
            if type_prio == 0:
                # Orderbook delta
                delta = event
                self._core.apply_delta(delta)

                if delta.flags & F_SNAPSHOT:
                    settled = False
                    if delta.flags & F_LAST:
                        bb = self._core.best_bid(delta.instrument_id)
                        ba = self._core.best_ask(delta.instrument_id)
                        is_settlement = (
                            bb is not None and ba is not None
                            and abs(bb[0] - ba[0]) < 0.001
                            and (bb[0] >= 0.98 or ba[0] <= 0.02)
                        )
                        if is_settlement:
                            settled = True
                            # Settle at actual price: 1.00 (YES) or 0.00 (NO)
                            settlement_price = 1.0 if bb[0] >= 0.50 else 0.0
                            settlement_fills = self._core.settle_instrument(
                                delta.instrument_id, settlement_price,
                                delta.timestamp_ns,
                            )
                            for fill in settlement_fills:
                                strat = self._strategies.get(fill.instrument_id)
                                if strat:
                                    strat.on_fill(fill)
                        else:
                            pass  # Resting fills only from trade ticks
                    # Fire strategy on every snapshot delta (matches nautilus
                    # buffer_deltas=False), skip only settlement F_LAST
                    if not settled:
                        strat = self._strategies.get(delta.instrument_id)
                        if strat:
                            strat.on_book_update(delta.instrument_id, delta.timestamp_ns)
                else:
                    # Nautilus default: buffer_deltas=False means strategy
                    # fires on every individual delta, not just F_LAST batches
                    strat = self._strategies.get(delta.instrument_id)
                    if strat:
                        strat.on_book_update(delta.instrument_id, delta.timestamp_ns)

            elif type_prio == 1:
                # Trade tick — decrement queues and check fills
                trade = event
                fills = self._core.process_trade_tick(trade)
                for fill in fills:
                    strat = self._strategies.get(fill.instrument_id)
                    if strat:
                        strat.on_fill(fill)

            else:
                # Fair value data
                fv = event
                strat = self._strategies.get(fv.instrument_id)
                if strat:
                    strat.on_data(fv)

        # Stop strategies
        for strategy in self._strategies.values():
            strategy.on_stop()

        # Engine-level settlement: close any remaining open positions
        # at the final book price (matches nautilus InstrumentClose behavior)
        self._core.settle_all()

    def print_results(self) -> None:
        """Print comprehensive backtest statistics."""
        starting = self._core.starting_balance()
        final = self._core.balance()
        pnl = final - starting

        all_orders = self._core.all_orders()
        all_fills = self._core.all_fills()
        positions = self._core.all_positions()

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
