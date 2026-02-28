"""Core generation logic for per-movie fair value & Greeks time-series."""
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from scipy.special import expit

UTC = ZoneInfo("UTC")
_DEFAULT_LAGS = ["1h", "3h", "8h", "1d", "3d"]
_DEFAULT_SANDBOX = "/mnt/c/Users/bgram/projects/sandbox"
_DEFAULT_MODEL = (
    "/mnt/c/Users/bgram/projects/sandbox/dist-calibration/logistic_mixture_m5.pkl"
)
_DEFAULT_ENV = "/mnt/c/Users/bgram/projects/sandbox/.env"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _setup_sandbox_path() -> str:
    sandbox = os.environ.get("SANDBOX_PATH", _DEFAULT_SANDBOX)
    if sandbox not in sys.path:
        sys.path.insert(0, sandbox)
    return sandbox


def _load_model():
    model_path = os.environ.get("MODEL_PATH", _DEFAULT_MODEL)
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _get_collections():
    load_dotenv(_DEFAULT_ENV)
    client = MongoClient(host=os.environ["MONGODB_URI"], tz_aware=True)
    db = client["kxrt-training"]
    return db["movies-all"], db["reviews-all-tz"]


def _load_movie_reviews(ems_id: str, movie_coll, review_coll):
    """Fetch the movie document and all reviews, sorted ascending by creation_date."""
    movie = movie_coll.find_one({"_id": ems_id})
    if movie is None:
        raise ValueError(f"Movie {ems_id!r} not found in movies-all")
    reviews = list(review_coll.find({"ems_id": ems_id}, sort=[("creation_date", 1)]))
    return movie, reviews


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _build_review_features(
    reviews: list, premiere: datetime, market_close_utc: datetime, lags: list
) -> pd.DataFrame:
    """Build per-review feature DataFrame for a single movie.

    Replicates the get_base_df + per_movie_features pipeline for one movie.
    Returns a DataFrame indexed by creation_date (UTC-aware).
    """
    _setup_sandbox_path()
    from helpers.data import per_movie_features  # noqa: PLC0415

    rows = []
    dates = []
    first_review_date = _to_utc(reviews[0]["creation_date"])

    for review in reviews:
        cd = _to_utc(review["creation_date"])
        if cd >= market_close_utc:
            continue
        hours_left = (market_close_utc - cd).total_seconds() / 3600
        rows.append(
            {
                "score": float(review["score"]),
                "hours_left": hours_left,
                "days_since_first_review": float((cd - first_review_date).days),
                "days_since_premiere": float((cd - premiere).days),
            }
        )
        dates.append(cd)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
    df["total_reviews"] = range(1, len(df) + 1)
    df["total_score"] = df["score"].cumsum() / df["total_reviews"]
    # final_score required by per_movie_features; use last known score
    df["final_score"] = float(df["total_score"].iloc[-1])

    return per_movie_features(df, lags)


def _predict_batch(xgblss, X_df: pd.DataFrame, train_cols: list, n_cpu: int):
    """Run XGBoostLSS inference. Returns (locs, scales, probs) each (n, 5)."""
    import xgboost as xgb  # noqa: PLC0415

    d = xgb.DMatrix(X_df[train_cols], nthread=n_cpu)
    y_dist = xgblss.predict(d, pred_type="parameters")
    locs = y_dist.filter(regex=r"^loc").to_numpy()
    scales = y_dist.filter(regex=r"^scale").to_numpy()
    probs = y_dist.filter(regex=r"^mix_prob").to_numpy()
    return locs, scales, probs


def _load_strike_helper():
    sandbox = _setup_sandbox_path()
    spec = importlib.util.spec_from_file_location(
        "strike_helper",
        os.path.join(sandbox, "strikes/helpers/strike_helper.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_timeseries(
    ems_id: str,
    event_ticker: str,
    start_delta_hours: float,
    strikes: list[int],
    T: float = 1.0,
    theta_delta_hours: float = 5.0,
    output_dir: str = "data/",
    save: bool = True,
) -> pd.DataFrame:
    """Generate per-minute fair value & Greeks time-series for a single movie.

    Parameters
    ----------
    ems_id : str
        Movie identifier (MongoDB ``_id`` in movies-all).
    event_ticker : str
        Kalshi event ticker; used as the parquet filename stem.
    start_delta_hours : float
        How many hours before market close to begin the simulation.
    strikes : list[int]
        Strike prices (0–99) to include in the output.
    T : float
        Temperature scaling factor applied to distribution scales (default 1.0).
    theta_delta_hours : float
        Hour step used for finite-difference theta approximation (default 5.0).
    output_dir : str
        Directory for parquet output (relative to caller or absolute).
    save : bool
        Whether to save the result to ``{output_dir}/{event_ticker}_T{T}.parquet``.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame, one row per (timestamp, strike).
    """
    _setup_sandbox_path()
    from helpers.data import (  # noqa: PLC0415
        get_next_monday_10am,
        get_training_cols,
        perturb_for_gamma,
        perturb_for_theta,
    )

    sh = _load_strike_helper()
    compute_fair_values = sh.compute_fair_values
    compute_theta = sh.compute_theta
    compute_gamma_pos = sh.compute_gamma_pos
    compute_gamma_neg = sh.compute_gamma_neg

    n_cpu = multiprocessing.cpu_count()
    lags = _DEFAULT_LAGS
    train_cols = get_training_cols(lags)

    xgblss = _load_model()
    movie_coll, review_coll = _get_collections()
    movie, reviews = _load_movie_reviews(ems_id, movie_coll, review_coll)

    premiere = _to_utc(movie["premiere"])
    market_close_utc = _to_utc(get_next_monday_10am(premiere))
    sim_start = market_close_utc - timedelta(hours=start_delta_hours)

    # All reviews strictly before market close, sorted ascending
    reviews_before_close = [
        r for r in reviews if _to_utc(r["creation_date"]) < market_close_utc
    ]
    if not reviews_before_close:
        raise ValueError(f"No reviews before market close for {ems_id!r}")

    review_feat_df = _build_review_features(
        reviews_before_close, premiere, market_close_utc, lags
    )
    if review_feat_df.empty:
        raise ValueError(f"Could not build features for {ems_id!r}")

    first_review_date = _to_utc(reviews_before_close[0]["creation_date"])

    # Split reviews into pre-simulation (state at sim_start) and in-window (new epochs)
    pre_sim_reviews = [
        r for r in reviews_before_close if _to_utc(r["creation_date"]) <= sim_start
    ]
    sim_reviews = [
        r for r in reviews_before_close if _to_utc(r["creation_date"]) > sim_start
    ]

    if not pre_sim_reviews:
        # All reviews arrived after the requested sim_start; begin the window
        # at the first review instead.
        sim_start = _to_utc(reviews_before_close[0]["creation_date"])
        pre_sim_reviews = reviews_before_close[:1]
        sim_reviews = reviews_before_close[1:]

    # Epoch boundaries expressed as integer minute offsets from sim_start.
    # Using raw review timestamps as epoch starts causes gaps: if a review
    # arrives at sim_start + 7m30s, epoch 0 covers minutes [0..6] and epoch 1
    # starts at 7m30s, so the minute at 7m00s is never generated.
    # Instead, each epoch starts at the first full minute AFTER the review.
    sim_cds = [_to_utc(r["creation_date"]) for r in sim_reviews]
    total_minutes = int((market_close_utc - sim_start).total_seconds() // 60)
    epoch_start_mins = [0]
    for cd in sim_cds:
        epoch_start_mins.append(int((cd - sim_start).total_seconds() // 60) + 1)
    epoch_end_mins = epoch_start_mins[1:] + [total_minutes]

    # Base review date for each epoch
    epoch_base_cds = [_to_utc(pre_sim_reviews[-1]["creation_date"])] + sim_cds

    strikes = list(strikes)
    n_strikes = len(strikes)
    strike_arr = np.array(strikes, dtype=np.int16)
    all_dfs: list[pd.DataFrame] = []

    for epoch_idx, (m_start, m_end, base_cd) in enumerate(
        zip(epoch_start_mins, epoch_end_mins, epoch_base_cds)
    ):
        n_minutes = m_end - m_start
        if n_minutes <= 0:
            continue

        timestamps = [sim_start + timedelta(minutes=m_start + m) for m in range(n_minutes)]
        n_min = len(timestamps)

        # Locate base feature row in review_feat_df
        feat_idx = review_feat_df.index.searchsorted(base_cd, side="right") - 1
        feat_idx = max(0, feat_idx)
        base_row = review_feat_df.iloc[feat_idx]

        cur_score_logit_val = float(base_row["cur_score_logit"])
        cur_score_val = float(expit(cur_score_logit_val))
        total_reviews_val = int(base_row["total_reviews"])

        # Only generate rows once the model's training constraints are met:
        # at least 50 reviews and a non-perfect score.
        if total_reviews_val < 50 or cur_score_val >= 1.0:
            continue

        # Build feature matrix: copy base row, override time features per minute
        X_dict = {col: np.full(n_min, float(base_row[col])) for col in train_cols}

        hours_left_arr = np.array(
            [(market_close_utc - t).total_seconds() / 3600 for t in timestamps]
        )
        days_premiere_arr = np.array(
            [float((t - premiere).days) for t in timestamps]
        )
        days_first_arr = np.array(
            [float((t - first_review_date).days) for t in timestamps]
        )

        X_dict["hours_left"] = hours_left_arr
        X_dict["days_since_premiere"] = days_premiere_arr
        X_dict["days_since_first_review"] = days_first_arr

        X_current = pd.DataFrame(X_dict)[train_cols]
        csl = np.full(n_min, cur_score_logit_val)

        # Four batched predictions
        locs_c, scales_c, probs_c = _predict_batch(xgblss, X_current, train_cols, n_cpu)

        X_theta = perturb_for_theta(X_current, delta_hours=theta_delta_hours)
        locs_th, scales_th, probs_th = _predict_batch(xgblss, X_theta, train_cols, n_cpu)

        X_gp, csl_gp = perturb_for_gamma(X_current, score=1)
        locs_gp, scales_gp, probs_gp = _predict_batch(xgblss, X_gp, train_cols, n_cpu)

        X_gn, csl_gn = perturb_for_gamma(X_current, score=0)
        locs_gn, scales_gn, probs_gn = _predict_batch(xgblss, X_gn, train_cols, n_cpu)

        # Compute Greeks
        scale_T = scales_c * T
        fv = compute_fair_values(locs_c, scale_T, probs_c, csl)

        fv_th = compute_fair_values(locs_th, scales_th * T, probs_th, csl)
        theta = compute_theta(fv, fv_th)

        fv_gp = compute_fair_values(locs_gp, scales_gp * T, probs_gp, csl_gp)
        fv_gn = compute_fair_values(locs_gn, scales_gn * T, probs_gn, csl_gn)
        gamma_p = compute_gamma_pos(fv, fv_gp)
        gamma_n = compute_gamma_neg(fv, fv_gn)

        # Slice to requested strikes: (n_min, n_strikes)
        fv_s = fv[:, strike_arr].astype(np.float32)
        th_s = theta[:, strike_arr].astype(np.float32)
        gp_s = gamma_p[:, strike_arr].astype(np.float32)
        gn_s = gamma_n[:, strike_arr].astype(np.float32)

        # new_review flag: True only at the first minute of epoch > 0
        new_review_col = np.zeros(n_min, dtype=bool)
        if epoch_idx > 0:
            new_review_col[0] = True

        # Vectorised long-format construction
        ts_tiled = np.repeat(timestamps, n_strikes)
        s_tiled = np.tile(strike_arr, n_min)
        nr_tiled = np.repeat(new_review_col, n_strikes)
        hl_tiled = np.repeat(hours_left_arr, n_strikes).astype(np.float32)

        epoch_df = pd.DataFrame(
            {
                "timestamp": ts_tiled,
                "ems_id": ems_id,
                "strike": s_tiled,
                "fv_T": fv_s.ravel(),
                "theta": th_s.ravel(),
                "gamma_pos": gp_s.ravel(),
                "gamma_neg": gn_s.ravel(),
                "new_review": nr_tiled,
                "hours_left": hl_tiled,
                "cur_score": np.float32(cur_score_val),
                "total_reviews": np.int32(total_reviews_val),
            }
        )
        all_dfs.append(epoch_df)

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["strike"] = df["strike"].astype(np.int16)
    df["total_reviews"] = df["total_reviews"].astype(np.int32)

    if save:
        try:
            from .storage import save_timeseries  # noqa: PLC0415
        except ImportError:
            from storage import save_timeseries  # noqa: PLC0415

        save_timeseries(df, output_dir, event_ticker, T)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate FV timeseries for a single movie"
    )
    parser.add_argument("--ems_id", required=True, help="Movie EMS ID")
    parser.add_argument("--event_ticker", required=True, help="Kalshi event ticker (used as parquet filename stem)")
    parser.add_argument(
        "--start_delta",
        type=float,
        default=168.0,
        help="Hours before market close to start simulation (default 168 = 1 week)",
    )
    parser.add_argument(
        "--strikes",
        type=int,
        nargs="+",
        default=list(range(1, 100)),
        help="Strike prices to include (default 1-99)",
    )
    parser.add_argument(
        "--T",
        type=float,
        default=1.0,
        help="Temperature scaling factor (default 1.0)",
    )
    parser.add_argument(
        "--theta_delta_hours",
        type=float,
        default=5.0,
        help="Hour step for finite-difference theta approximation (default 5.0)",
    )
    parser.add_argument("--output_dir", default="data/", help="Output directory")
    parser.add_argument(
        "--no_save", action="store_true", help="Skip saving to parquet"
    )
    args = parser.parse_args()

    result = generate_timeseries(
        ems_id=args.ems_id,
        event_ticker=args.event_ticker,
        start_delta_hours=args.start_delta,
        strikes=args.strikes,
        T=args.T,
        theta_delta_hours=args.theta_delta_hours,
        output_dir=args.output_dir,
        save=not args.no_save,
    )
    print(f"Generated {len(result)} rows for {args.ems_id!r}")
    if not result.empty:
        print(result.head(10).to_string())
