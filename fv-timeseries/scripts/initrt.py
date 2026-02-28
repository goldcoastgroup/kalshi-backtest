"""Batch initialization script: generate FV timeseries parquets for all KXRT events."""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from pymongo import MongoClient

# Allow running as a script from the fv-timeseries/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fv_timeseries.generate import _DEFAULT_ENV, generate_timeseries


def _prompt_float(label: str, default: float) -> float:
    raw = input(f"{label} [{default}]: ").strip()
    return float(raw) if raw else default


def main() -> None:
    print("=== initrt: batch FV timeseries generation ===\n")

    T = _prompt_float("Temperature scaling factor (T)", 1.0)
    theta_delta_hours = _prompt_float("Theta finite-difference step in hours (theta_delta_hours)", 5.0)

    print(f"\nParameters: T={T}, theta_delta_hours={theta_delta_hours}\n")

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
