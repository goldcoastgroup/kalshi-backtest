"""Batch initialization script: generate FV timeseries parquets for all KXRT events."""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from pymongo import MongoClient

# Allow running as a script from the fv-timeseries/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fv_timeseries.generate import _DEFAULT_ENV, generate_timeseries
from fv_timeseries.storage import _parquet_path


def _prompt_float(label: str, default: float) -> float:
    raw = input(f"{label} [{default}]: ").strip()
    return float(raw) if raw else default


def _prompt_bool(label: str, default: bool) -> bool:
    default_str = "Y/n" if default else "y/N"
    raw = input(f"{label} [{default_str}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def main() -> None:
    print("=== initrt: batch FV timeseries generation ===\n")

    T = _prompt_float("Temperature scaling factor (T)", 1.0)
    theta_delta_hours = _prompt_float("Theta finite-difference step in hours (theta_delta_hours)", 5.0)
    overwrite = _prompt_bool("Overwrite existing parquet files?", True)

    print(f"\nParameters: T={T}, theta_delta_hours={theta_delta_hours}, overwrite={overwrite}\n")

    load_dotenv(_DEFAULT_ENV)
    client = MongoClient(host=os.environ["MONGODB_URI"], tz_aware=True)
    db = client["kxrt"]

    events = [e for e in db["events"].find({"status": "finalized"}) if e.get("ems_id")]

    total = len(events)
    print(f"Found {total} events to process.\n")

    for i, event in enumerate(events, start=1):
        event_ticker = event["_id"]
        ems_id = event["ems_id"]
        title = event.get("title", "")
        strikes = sorted(int(m["strike"]) for m in event.get("markets", []))

        if not overwrite and _parquet_path("data/", event_ticker, T).exists():
            print(f"[{i}/{total}] {event_ticker} — {title} ... skipped (exists)")
            continue

        print(f"[{i}/{total}] {event_ticker} — {title} ...")

        try:
            generate_timeseries(
                ems_id=ems_id,
                event_ticker=event_ticker,
                start_delta_hours=168.0,
                strikes=strikes,
                T=T,
                theta_delta_hours=theta_delta_hours,
                output_dir="data/",
                save=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")
            continue

        print(f"  done.")

    print(f"\nFinished. Processed {total} events.")


if __name__ == "__main__":
    main()
