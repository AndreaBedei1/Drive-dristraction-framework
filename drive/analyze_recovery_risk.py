#!/usr/bin/env python3
from __future__ import annotations

"""
Estimate baseline error probability and post-distraction recovery risk.
Now with confidence intervals, bootstrap, and robust recovery criteria.
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta  # for Clopper‑Pearson CI

NANOS_PER_SECOND = 1_000_000_000


# ----------------------------------------------------------------------
#  Confidence intervals and bootstrap
# ----------------------------------------------------------------------
def binomial_ci(count: int, nobs: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Clopper‑Pearson exact binomial confidence interval."""
    if nobs == 0:
        return (np.nan, np.nan)
    if count == 0:
        return (0.0, beta.ppf(1 - alpha / 2, 1, nobs))
    if count == nobs:
        return (beta.ppf(alpha / 2, count, 1), 1.0)
    lower = beta.ppf(alpha / 2, count, nobs - count + 1)
    upper = beta.ppf(1 - alpha / 2, count + 1, nobs - count)
    return lower, upper


def expand_binary_outcomes(n_windows: int, error_windows: int) -> np.ndarray:
    """Expand aggregated counts into an array of 0/1 outcomes."""
    arr = np.zeros(n_windows, dtype=int)
    if error_windows > 0:
        arr[:error_windows] = 1
    return arr


def bootstrap_ratio_ci(
    baseline_outcomes: np.ndarray,
    offset_outcomes: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
) -> Tuple[float, float, float, float]:
    """
    Bootstrap confidence interval for the ratio p_offset / p_baseline.
    Returns (lower, upper) of the ratio, plus the 2.5th and 97.5th percentiles.
    """
    if baseline_outcomes.size == 0 or offset_outcomes.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    n_baseline = len(baseline_outcomes)
    n_offset = len(offset_outcomes)

    ratios = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        bs_base = np.random.choice(baseline_outcomes, size=n_baseline, replace=True)
        bs_off = np.random.choice(offset_outcomes, size=n_offset, replace=True)
        p_base = bs_base.mean()
        p_off = bs_off.mean()
        if p_base > 0:
            ratios[i] = p_off / p_base
        else:
            ratios[i] = np.nan

    ratios = ratios[~np.isnan(ratios)]
    if len(ratios) == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    lower = np.percentile(ratios, 100 * alpha / 2)
    upper = np.percentile(ratios, 100 * (1 - alpha / 2))
    return lower, upper, np.percentile(ratios, 2.5), np.percentile(ratios, 97.5)


# ----------------------------------------------------------------------
#  Argument parsing (extended)
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze baseline risk and post-distraction recovery risk with confidence intervals."
    )
    parser.add_argument(
        "--baseline-errors",
        type=Path,
        default=Path("data") / "Dataset Errors_baseline.csv",
    )
    parser.add_argument(
        "--baseline-driving-time",
        type=Path,
        default=Path("data") / "Dataset Driving Time_baseline.csv",
        help="Path to baseline driving-time CSV.",
    )
    parser.add_argument(
        "--distraction-errors",
        type=Path,
        default=Path("data") / "Dataset Errors_distraction.csv",
    )
    parser.add_argument(
        "--distractions",
        type=Path,
        default=Path("data") / "Dataset Distractions_distraction.csv",
    )
    parser.add_argument(
        "--offsets-seconds",
        type=str,
        default="1,3,5,7,10,15",
        help="Comma-separated offsets after distraction end, in seconds.",
    )
    parser.add_argument(
        "--delta-t-seconds",
        type=float,
        default=1.0,
        help="Window length in seconds.",
    )
    parser.add_argument(
        "--users",
        type=str,
        default="",
        help="Optional comma-separated user_id filter.",
    )
    parser.add_argument(
        "--state-col",
        type=str,
        default="",
        help="Optional error column used as state filter.",
    )
    parser.add_argument(
        "--state-values",
        type=str,
        default="",
        help="Optional comma-separated accepted values for --state-col.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis_outputs") / "recovery_risk",
        help="Output directory.",
    )
    # ----- Robustness parameters -----
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Confidence level for intervals (default 0.05 → 95% CI).",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=1000,
        help="Number of bootstrap replicates for normalized risk CI.",
    )
    parser.add_argument(
        "--min-windows",
        type=int,
        default=10,
        help="Minimum number of windows required to include an offset.",
    )
    parser.add_argument(
        "--require-consecutive",
        action="store_true",
        help="Require two consecutive offsets with risk ≤ baseline to declare recovery.",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip bootstrap CI calculation (only point estimates).",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
#  Data I/O and cleaning (unchanged, except added user ID consistency warning)
# ----------------------------------------------------------------------
def clean_text(value: object) -> str:
    if pd.isna(value):
        return "None"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return "None"
    return text


def parse_list(raw: str) -> List[str]:
    if not raw.strip():
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_offsets(raw: str) -> List[float]:
    offsets = []
    for token in parse_list(raw):
        value = float(token)
        if value < 0.0:
            raise ValueError(f"Offset must be >= 0, got {value}")
        offsets.append(value)
    if not offsets:
        raise ValueError("At least one offset is required.")
    return sorted(set(offsets))


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def read_errors(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    text_cols = ["user_id", "run_id", "error_type", "model_pred", "emotion_label", "details"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].map(clean_text)
        else:
            df[col] = "None"
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    return df


def read_distractions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    text_cols = ["user_id", "run_id", "details", "model_pred_start", "emotion_label_start"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].map(clean_text)
        else:
            df[col] = "None"
    df["timestamp_start"] = pd.to_datetime(df.get("timestamp_start"), errors="coerce")
    df["timestamp_end"] = pd.to_datetime(df.get("timestamp_end"), errors="coerce")
    return df


def read_baseline_driving_time(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    required_cols = ["user_id", "run_id", "run_duration_seconds"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Baseline driving-time CSV missing required columns: {', '.join(missing)}"
        )

    df["user_id"] = df["user_id"].map(clean_text)
    df["run_id"] = df["run_id"].map(clean_text)
    df["run_duration_seconds"] = pd.to_numeric(df["run_duration_seconds"], errors="coerce")
    df = df[df["run_duration_seconds"].notna()].copy()
    return df


def apply_user_filter(df: pd.DataFrame, users: Sequence[str]) -> pd.DataFrame:
    if not users:
        return df
    return df[df["user_id"].isin(users)].copy()


def apply_state_filter(df: pd.DataFrame, state_col: str, state_values: Sequence[str]) -> pd.DataFrame:
    if not state_col:
        return df
    if state_col not in df.columns:
        raise ValueError(f"State column '{state_col}' not found in errors dataset.")
    if not state_values:
        return df
    normalized = df[state_col].map(clean_text)
    accepted = set(clean_text(v) for v in state_values)
    return df[normalized.isin(accepted)].copy()


def warn_missing_users(baseline_users: set, distraction_users: set) -> None:
    only_baseline = baseline_users - distraction_users
    only_distraction = distraction_users - baseline_users
    if only_baseline:
        warnings.warn(f"Users in baseline but not in distraction: {only_baseline}")
    if only_distraction:
        warnings.warn(f"Users in distraction but not in baseline: {only_distraction}")


def compute_baseline_run_probabilities(
    errors: pd.DataFrame,
    baseline_driving_time: pd.DataFrame,
    delta_t_seconds: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    duration_by_run = (
        baseline_driving_time.groupby(["user_id", "run_id"], dropna=False)["run_duration_seconds"]
        .sum()
        .to_dict()
    )

    timestamps_by_run: Dict[Tuple[str, str], pd.Series] = {}
    for (user_id, run_id), grp in errors.groupby(["user_id", "run_id"], dropna=False):
        ts = grp["timestamp"].dropna().sort_values()
        if ts.empty:
            continue
        timestamps_by_run[(str(user_id), str(run_id))] = ts

    for (user_id, run_id), run_duration_seconds in duration_by_run.items():
        duration_s = float(run_duration_seconds)
        if duration_s <= 0.0:
            continue
        n_windows = max(1, int(np.ceil(duration_s / delta_t_seconds)))

        ts = timestamps_by_run.get((str(user_id), str(run_id)))
        error_windows = 0
        if ts is not None and not ts.empty:
            start_ns = int(ts.iloc[0].value)
            rel_s = (ts.astype("int64") - start_ns) / NANOS_PER_SECOND
            idx = np.floor(rel_s / delta_t_seconds).astype(int).to_numpy()
            idx = np.clip(idx, 0, n_windows - 1)
            error_windows = int(np.unique(idx).size)

        p_error = float(error_windows / n_windows)

        rows.append(
            {
                "user_id": user_id,
                "run_id": run_id,
                "duration_seconds": float(duration_s),
                "n_windows": n_windows,
                "error_windows": error_windows,
                "p_error": p_error,
            }
        )
    return pd.DataFrame(rows)


def aggregate_baseline_by_user(run_probs: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    if run_probs.empty:
        return pd.DataFrame(
            columns=["user_id", "n_runs", "n_windows", "error_windows",
                     "baseline_p_error", "baseline_p_error_lower", "baseline_p_error_upper"]
        )
    for user_id, grp in run_probs.groupby("user_id", dropna=False):
        n_windows = int(grp["n_windows"].sum())
        error_windows = int(grp["error_windows"].sum())
        baseline_p = error_windows / n_windows if n_windows > 0 else np.nan
        lower, upper = binomial_ci(error_windows, n_windows, alpha)
        rows.append(
            {
                "user_id": user_id,
                "n_runs": int(len(grp)),
                "n_windows": n_windows,
                "error_windows": error_windows,
                "baseline_p_error": baseline_p,
                "baseline_p_error_lower": lower,
                "baseline_p_error_upper": upper,
            }
        )
    return pd.DataFrame(rows)


def aggregate_baseline_overall(run_probs: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if run_probs.empty:
        return pd.DataFrame(
            [{"n_runs": 0, "n_windows": 0, "error_windows": 0,
              "baseline_p_error": np.nan, "baseline_p_error_lower": np.nan,
              "baseline_p_error_upper": np.nan}]
        )
    n_windows = int(run_probs["n_windows"].sum())
    error_windows = int(run_probs["error_windows"].sum())
    baseline_p = error_windows / n_windows if n_windows > 0 else np.nan
    lower, upper = binomial_ci(error_windows, n_windows, alpha)
    return pd.DataFrame(
        [
            {
                "n_runs": int(len(run_probs)),
                "n_windows": n_windows,
                "error_windows": error_windows,
                "baseline_p_error": baseline_p,
                "baseline_p_error_lower": lower,
                "baseline_p_error_upper": upper,
            }
        ]
    )


# ----------------------------------------------------------------------
#  Offset probabilities (with CI)
# ----------------------------------------------------------------------
def compute_offset_probabilities(
    errors: pd.DataFrame,
    distractions: pd.DataFrame,
    offsets_seconds: Sequence[float],
    delta_t_seconds: float,
    alpha: float,
) -> pd.DataFrame:
    delta_ns = int(round(delta_t_seconds * NANOS_PER_SECOND))
    offsets = list(offsets_seconds)
    hits = {offset: 0 for offset in offsets}
    totals = {offset: 0 for offset in offsets}

    error_ns_by_run = {}
    for key, grp in errors.groupby(["user_id", "run_id"], dropna=False):
        error_ns_by_run[(str(key[0]), str(key[1]))] = _timestamps_to_ns(grp["timestamp"].dropna())

    for key, grp in distractions.groupby(["user_id", "run_id"], dropna=False):
        run_key = (str(key[0]), str(key[1]))
        end_ns = _timestamps_to_ns(grp["timestamp_end"].dropna())
        if end_ns.size == 0:
            continue
        err_ns = error_ns_by_run.get(run_key, np.empty(0, dtype=np.int64))

        for end_value in end_ns:
            for offset in offsets:
                start_ns = int(end_value + round(offset * NANOS_PER_SECOND))
                stop_ns = start_ns + delta_ns
                totals[offset] += 1
                if err_ns.size == 0:
                    continue
                left = int(np.searchsorted(err_ns, start_ns, side="left"))
                right = int(np.searchsorted(err_ns, stop_ns, side="left"))
                if right > left:
                    hits[offset] += 1

    rows = []
    for offset in offsets:
        total = totals[offset]
        hit = hits[offset]
        p_error = hit / total if total > 0 else np.nan
        lower, upper = binomial_ci(hit, total, alpha)
        rows.append(
            {
                "offset_seconds": float(offset),
                "n_windows": total,
                "error_windows": hit,
                "p_error": p_error,
                "p_error_lower": lower,
                "p_error_upper": upper,
            }
        )
    return pd.DataFrame(rows)


def compute_offset_probabilities_by_user(
    errors: pd.DataFrame,
    distractions: pd.DataFrame,
    offsets_seconds: Sequence[float],
    delta_t_seconds: float,
    alpha: float,
) -> pd.DataFrame:
    rows = []
    if distractions.empty:
        return pd.DataFrame(
            columns=["user_id", "offset_seconds", "n_windows", "error_windows",
                     "p_error", "p_error_lower", "p_error_upper"]
        )
    for user_id, d_grp in distractions.groupby("user_id", dropna=False):
        e_grp = errors[errors["user_id"] == user_id].copy()
        probs = compute_offset_probabilities(e_grp, d_grp, offsets_seconds, delta_t_seconds, alpha)
        probs.insert(0, "user_id", user_id)
        rows.append(probs)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _timestamps_to_ns(series: pd.Series) -> np.ndarray:
    if series.empty:
        return np.empty(0, dtype=np.int64)
    return np.sort(series.astype("int64").to_numpy(dtype=np.int64, copy=False))


# ----------------------------------------------------------------------
#  Build per‑window outcomes for bootstrap
# ----------------------------------------------------------------------
def build_offset_hits_per_distraction(
    errors: pd.DataFrame,
    distractions: pd.DataFrame,
    offsets_seconds: Sequence[float],
    delta_t_seconds: float,
) -> pd.DataFrame:
    """Return one row per (distraction, offset) with binary has_error."""
    delta_ns = int(round(delta_t_seconds * NANOS_PER_SECOND))
    offsets = list(offsets_seconds)

    error_ns_by_run = {}
    for key, grp in errors.groupby(["user_id", "run_id"], dropna=False):
        error_ns_by_run[(str(key[0]), str(key[1]))] = _timestamps_to_ns(grp["timestamp"].dropna())

    rows = []
    for distraction_idx, row in distractions.iterrows():
        end_ts = row.get("timestamp_end")
        if pd.isna(end_ts):
            continue
        user_id = str(row.get("user_id", "None"))
        run_id = str(row.get("run_id", "None"))
        details = clean_text(row.get("details", "None"))
        end_ns = int(pd.Timestamp(end_ts).value)
        err_ns = error_ns_by_run.get((user_id, run_id), np.empty(0, dtype=np.int64))

        for offset in offsets:
            start_ns = int(end_ns + round(offset * NANOS_PER_SECOND))
            stop_ns = start_ns + delta_ns
            has_error = False
            if err_ns.size > 0:
                left = int(np.searchsorted(err_ns, start_ns, side="left"))
                right = int(np.searchsorted(err_ns, stop_ns, side="left"))
                has_error = right > left
            rows.append(
                {
                    "user_id": user_id,
                    "run_id": run_id,
                    "distraction_index": int(distraction_idx),
                    "distraction_details": details,
                    "timestamp_end": pd.Timestamp(end_ns),
                    "offset_seconds": float(offset),
                    "window_start": pd.Timestamp(start_ns),
                    "window_end": pd.Timestamp(stop_ns),
                    "has_error": int(has_error),
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
#  Normalized risk with bootstrap CI
# ----------------------------------------------------------------------
def add_normalized_risk_overall(
    offset_probs_overall: pd.DataFrame,
    baseline_overall: pd.DataFrame,
    baseline_run: pd.DataFrame,          # for expanding outcomes
    offset_hits: pd.DataFrame,           # per‑window outcomes
    alpha: float,
    n_bootstrap: int,
    no_bootstrap: bool,
    min_windows: int,
) -> pd.DataFrame:
    out = offset_probs_overall.copy()
    baseline_p = baseline_overall.iloc[0]["baseline_p_error"]
    out["baseline_p_error"] = baseline_p
    out["baseline_p_error_lower"] = baseline_overall.iloc[0]["baseline_p_error_lower"]
    out["baseline_p_error_upper"] = baseline_overall.iloc[0]["baseline_p_error_upper"]

    # Point estimate
    valid = out["p_error"].notna() & (baseline_p > 0) if pd.notna(baseline_p) else False
    out["normalized_risk"] = np.where(valid, out["p_error"] / baseline_p, np.nan)

    # Bootstrap CIs for normalized risk (if requested)
    if no_bootstrap or baseline_run.empty or offset_hits.empty:
        out["normalized_risk_lower"] = np.nan
        out["normalized_risk_upper"] = np.nan
        return out

    # Expand baseline outcomes once
    baseline_outcomes_list = []
    for _, row in baseline_run.iterrows():
        if row["n_windows"] > 0:
            baseline_outcomes_list.append(
                expand_binary_outcomes(int(row["n_windows"]), int(row["error_windows"]))
            )
    if baseline_outcomes_list:
        baseline_outcomes = np.concatenate(baseline_outcomes_list)
    else:
        baseline_outcomes = np.array([])

    # For each offset, bootstrap ratio
    lower_list, upper_list = [], []
    for offset in out["offset_seconds"]:
        off_mask = offset_hits["offset_seconds"] == offset
        off_outcomes = offset_hits.loc[off_mask, "has_error"].to_numpy(dtype=int)
        if len(off_outcomes) < min_windows:
            lower_list.append(np.nan)
            upper_list.append(np.nan)
            continue
        lo, hi, _, _ = bootstrap_ratio_ci(
            baseline_outcomes, off_outcomes, alpha, n_bootstrap
        )
        lower_list.append(lo)
        upper_list.append(hi)

    out["normalized_risk_lower"] = lower_list
    out["normalized_risk_upper"] = upper_list
    return out


def add_normalized_risk_by_user(
    offset_probs_by_user: pd.DataFrame,
    baseline_by_user: pd.DataFrame,
    alpha: float,
    min_windows: int,
) -> pd.DataFrame:
    out = offset_probs_by_user.copy()
    baseline_map = baseline_by_user.set_index("user_id")[
        ["baseline_p_error", "baseline_p_error_lower", "baseline_p_error_upper"]
    ].to_dict(orient="index")
    out["baseline_p_error"] = out["user_id"].map(lambda uid: baseline_map.get(uid, {}).get("baseline_p_error", np.nan))
    out["baseline_p_error_lower"] = out["user_id"].map(lambda uid: baseline_map.get(uid, {}).get("baseline_p_error_lower", np.nan))
    out["baseline_p_error_upper"] = out["user_id"].map(lambda uid: baseline_map.get(uid, {}).get("baseline_p_error_upper", np.nan))

    # Point estimate
    valid = out["p_error"].notna() & (out["baseline_p_error"] > 0)
    out["normalized_risk"] = np.where(valid, out["p_error"] / out["baseline_p_error"], np.nan)

    # Per‑user bootstrap is skipped by default (small samples). Add warning if needed.
    out["normalized_risk_lower"] = np.nan
    out["normalized_risk_upper"] = np.nan

    # Warn if any user‑offset combination has too few windows
    small = out[out["n_windows"] < min_windows]
    if not small.empty:
        warnings.warn(
            f"{len(small)} user‑offset combinations have n_windows < {min_windows} "
            "(CIs and recovery estimates may be unreliable)."
        )
    return out


# ----------------------------------------------------------------------
#  Recovery time (point and conservative)
# ----------------------------------------------------------------------
def compute_recovery_time_overall(
    normalized_overall: pd.DataFrame,
    require_consecutive: bool,
) -> pd.DataFrame:
    if normalized_overall.empty:
        return pd.DataFrame(
            [{"baseline_p_error": np.nan,
              "recovery_offset_seconds": np.nan,
              "recovery_offset_conservative_seconds": np.nan}]
        )
    g = normalized_overall.sort_values("offset_seconds").copy()
    baseline_p = g["baseline_p_error"].iloc[0] if not g.empty else np.nan

    # ---- Point estimate: first offset where normalized_risk <= 1.0 ----
    point_candidates = g[g["normalized_risk"].notna() & (g["normalized_risk"] <= 1.0)]
    if require_consecutive:
        # need two consecutive offsets with risk <= 1.0
        conv = g["normalized_risk"].notna() & (g["normalized_risk"] <= 1.0)
        conv = conv.to_numpy()
        found = False
        for i in range(len(conv) - 1):
            if conv[i] and conv[i + 1]:
                recovery_offset = g.iloc[i]["offset_seconds"]
                found = True
                break
        if not found:
            recovery_offset = np.nan
    else:
        recovery_offset = point_candidates["offset_seconds"].iloc[0] if not point_candidates.empty else np.nan

    # ---- Conservative estimate: first offset where normalized_risk_upper <= 1.0 ----
    if "normalized_risk_upper" in g.columns:
        cons_candidates = g[g["normalized_risk_upper"].notna() & (g["normalized_risk_upper"] <= 1.0)]
        if require_consecutive:
            conv_cons = g["normalized_risk_upper"].notna() & (g["normalized_risk_upper"] <= 1.0)
            conv_cons = conv_cons.to_numpy()
            found_cons = False
            for i in range(len(conv_cons) - 1):
                if conv_cons[i] and conv_cons[i + 1]:
                    recovery_offset_cons = g.iloc[i]["offset_seconds"]
                    found_cons = True
                    break
            if not found_cons:
                recovery_offset_cons = np.nan
        else:
            recovery_offset_cons = cons_candidates["offset_seconds"].iloc[0] if not cons_candidates.empty else np.nan
    else:
        recovery_offset_cons = np.nan

    return pd.DataFrame(
        [{
            "baseline_p_error": baseline_p,
            "recovery_offset_seconds": recovery_offset,
            "recovery_offset_conservative_seconds": recovery_offset_cons,
        }]
    )


def compute_recovery_time_by_user(
    normalized_by_user: pd.DataFrame,
    require_consecutive: bool,
) -> pd.DataFrame:
    rows = []
    if normalized_by_user.empty:
        return pd.DataFrame(
            columns=["user_id", "baseline_p_error", "recovery_offset_seconds",
                     "recovery_offset_conservative_seconds"]
        )
    for user_id, grp in normalized_by_user.groupby("user_id", dropna=False):
        g = grp.sort_values("offset_seconds")
        baseline_p = g["baseline_p_error"].iloc[0] if not g.empty else np.nan

        # Point estimate
        point_candidates = g[g["normalized_risk"].notna() & (g["normalized_risk"] <= 1.0)]
        if require_consecutive:
            conv = g["normalized_risk"].notna() & (g["normalized_risk"] <= 1.0)
            conv = conv.to_numpy()
            found = False
            for i in range(len(conv) - 1):
                if conv[i] and conv[i + 1]:
                    recovery_offset = g.iloc[i]["offset_seconds"]
                    found = True
                    break
            if not found:
                recovery_offset = np.nan
        else:
            recovery_offset = point_candidates["offset_seconds"].iloc[0] if not point_candidates.empty else np.nan

        # Conservative (no per‑user CI yet → set to NaN)
        recovery_offset_cons = np.nan

        rows.append(
            {
                "user_id": user_id,
                "baseline_p_error": baseline_p,
                "recovery_offset_seconds": recovery_offset,
                "recovery_offset_conservative_seconds": recovery_offset_cons,
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
#  Plotting (with error bars / bands)
# ----------------------------------------------------------------------
def plot_overall_probability_curve(
    overall_probs: pd.DataFrame,
    baseline_p: float,
    baseline_p_lower: float,
    baseline_p_upper: float,
    out_path: Path,
) -> None:
    if overall_probs.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = overall_probs["offset_seconds"]
    y = overall_probs["p_error"]
    yerr_lower = y - overall_probs["p_error_lower"]
    yerr_upper = overall_probs["p_error_upper"] - y
    ax.errorbar(x, y, yerr=[yerr_lower, yerr_upper], fmt="o-", capsize=3, label="P(error | t)")
    if pd.notna(baseline_p):
        ax.axhline(baseline_p, color="#d62828", linestyle="--", label="P(error | baseline)")
        if pd.notna(baseline_p_lower) and pd.notna(baseline_p_upper):
            ax.axhspan(baseline_p_lower, baseline_p_upper, color="#d62828", alpha=0.2, label="Baseline 95% CI")
    ax.set_xlabel("Offset after distraction end (s)")
    ax.set_ylabel("Probability")
    ax.set_title("Error Probability vs Offset (with 95% CI)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_overall_normalized_curve(
    overall_norm: pd.DataFrame,
    out_path: Path,
) -> None:
    if overall_norm.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = overall_norm["offset_seconds"]
    y = overall_norm["normalized_risk"]
    ax.plot(x, y, "o-", label="normalized_risk(t)")

    if "normalized_risk_lower" in overall_norm.columns and "normalized_risk_upper" in overall_norm.columns:
        ax.fill_between(
            x,
            overall_norm["normalized_risk_lower"],
            overall_norm["normalized_risk_upper"],
            alpha=0.2,
            label="95% bootstrap CI",
        )

    ax.axhline(1.0, color="#2a9d8f", linestyle="--", label="Baseline parity (1.0)")
    ax.set_xlabel("Offset after distraction end (s)")
    ax.set_ylabel("Normalized risk")
    ax.set_title("Normalized Recovery Risk vs Offset")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_user_normalized_curve(
    user_norm: pd.DataFrame,
    out_path: Path,
) -> None:
    if user_norm.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for user_id, grp in user_norm.groupby("user_id", dropna=False):
        g = grp.sort_values("offset_seconds")
        ax.plot(g["offset_seconds"], g["normalized_risk"], marker="o", label=str(user_id))
    ax.axhline(1.0, color="#2a9d8f", linestyle="--", label="Baseline parity (1.0)")
    ax.set_xlabel("Offset after distraction end (s)")
    ax.set_ylabel("Normalized risk")
    ax.set_title("Normalized Recovery Risk by Driver (point estimates)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ----------------------------------------------------------------------
#  Report (extended)
# ----------------------------------------------------------------------
def write_report(
    out_path: Path,
    delta_t_seconds: float,
    offsets: Sequence[float],
    baseline_overall: pd.DataFrame,
    overall_norm: pd.DataFrame,
    recovery_overall: pd.DataFrame,
    state_col: str,
    state_values: Sequence[str],
    min_windows: int,
    alpha: float,
) -> None:
    baseline_p = baseline_overall.iloc[0]["baseline_p_error"] if not baseline_overall.empty else np.nan
    baseline_low = baseline_overall.iloc[0]["baseline_p_error_lower"] if not baseline_overall.empty else np.nan
    baseline_high = baseline_overall.iloc[0]["baseline_p_error_upper"] if not baseline_overall.empty else np.nan
    recovery_offset = recovery_overall.iloc[0]["recovery_offset_seconds"] if not recovery_overall.empty else np.nan
    recovery_offset_cons = recovery_overall.iloc[0]["recovery_offset_conservative_seconds"] if not recovery_overall.empty else np.nan

    lines = []
    lines.append("# Recovery Risk Report (Robust)")
    lines.append("")
    lines.append("## Definitions")
    lines.append(f"- Baseline window length: {delta_t_seconds:g}s")
    lines.append(f"- Offsets analyzed: {', '.join(f'{x:g}' for x in offsets)} seconds.")
    lines.append(f"- Minimum windows per offset: {min_windows} (fewer excluded)")
    lines.append(f"- Confidence level: {(1-alpha)*100:.0f}%")
    lines.append("- Normalized risk = P(error | t) / P(error | baseline)")
    lines.append("")

    if state_col:
        state_desc = ", ".join(state_values) if state_values else "(all values)"
        lines.append("## State filter")
        lines.append(f"- State column: {state_col}")
        lines.append(f"- Accepted values: {state_desc}")
        lines.append("")

    lines.append("## Overall baseline")
    if pd.notna(baseline_p):
        lines.append(f"- P(error | baseline) = {baseline_p:.6f}  [{baseline_low:.6f}, {baseline_high:.6f}]")
    else:
        lines.append("- P(error | baseline) = NaN")
    if pd.notna(recovery_offset):
        lines.append(f"- Estimated recovery offset (point, risk ≤ 1): {recovery_offset:g}s")
    else:
        lines.append("- Estimated recovery offset (point): not reached")
    if pd.notna(recovery_offset_cons):
        lines.append(f"- Estimated recovery offset (conservative, upper CI ≤ 1): {recovery_offset_cons:g}s")
    else:
        lines.append("- Estimated recovery offset (conservative): not reached")
    lines.append("")

    lines.append("## Overall curve (with CIs)")
    if overall_norm.empty:
        lines.append("- No offset result available.")
    else:
        for _, row in overall_norm.sort_values("offset_seconds").iterrows():
            offset = row["offset_seconds"]
            p = row["p_error"]
            p_low = row["p_error_lower"]
            p_high = row["p_error_upper"]
            norm = row["normalized_risk"]
            norm_low = row.get("normalized_risk_lower", np.nan)
            norm_high = row.get("normalized_risk_upper", np.nan)
            win = int(row["n_windows"])
            line = (f"- t={offset:g}s (n={win}): "
                    f"P={p:.4f} [{p_low:.4f}, {p_high:.4f}], "
                    f"norm={norm:.4f}")
            if pd.notna(norm_low):
                line += f" [{norm_low:.4f}, {norm_high:.4f}]"
            lines.append(line)
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    offsets = parse_offsets(args.offsets_seconds)
    users = parse_list(args.users)
    state_values = parse_list(args.state_values)
    alpha = args.alpha
    min_windows = args.min_windows
    require_consecutive = args.require_consecutive
    n_bootstrap = args.bootstrap_reps
    no_bootstrap = args.no_bootstrap

    ensure_exists(args.baseline_errors, "baseline errors")
    ensure_exists(args.baseline_driving_time, "baseline driving time")
    ensure_exists(args.distraction_errors, "distraction errors")
    ensure_exists(args.distractions, "distraction windows")

    if args.delta_t_seconds <= 0.0:
        raise ValueError("--delta-t-seconds must be > 0")

    out_dir = args.out_dir
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ---- Read and filter ----
    baseline_errors = read_errors(args.baseline_errors)
    baseline_driving_time = read_baseline_driving_time(args.baseline_driving_time)
    distraction_errors = read_errors(args.distraction_errors)
    distractions = read_distractions(args.distractions)

    baseline_errors = apply_user_filter(baseline_errors, users)
    baseline_driving_time = apply_user_filter(baseline_driving_time, users)
    distraction_errors = apply_user_filter(distraction_errors, users)
    distractions = apply_user_filter(distractions, users)

    baseline_errors = apply_state_filter(baseline_errors, args.state_col, state_values)
    distraction_errors = apply_state_filter(distraction_errors, args.state_col, state_values)

    baseline_run = compute_baseline_run_probabilities(
        baseline_errors,
        baseline_driving_time,
        args.delta_t_seconds,
    )
    baseline_user = aggregate_baseline_by_user(baseline_run)
    baseline_overall = aggregate_baseline_overall(baseline_run)

    # ---- Baseline ----
    baseline_run = compute_baseline_run_probabilities(baseline_errors, args.delta_t_seconds)
    baseline_user = aggregate_baseline_by_user(baseline_run, alpha)
    baseline_overall = aggregate_baseline_overall(baseline_run, alpha)

    # ---- Offset probabilities and per‑window outcomes ----
    offset_overall = compute_offset_probabilities(
        distraction_errors, distractions, offsets, args.delta_t_seconds, alpha
    )
    offset_by_user = compute_offset_probabilities_by_user(
        distraction_errors, distractions, offsets, args.delta_t_seconds, alpha
    )
    offset_hits = build_offset_hits_per_distraction(
        distraction_errors, distractions, offsets, args.delta_t_seconds
    )

    # ---- Filter offsets with too few windows ----
    offset_overall = offset_overall[offset_overall["n_windows"] >= min_windows].copy()
    offset_by_user = offset_by_user[offset_by_user["n_windows"] >= min_windows].copy()
    offset_hits = offset_hits[offset_hits["offset_seconds"].isin(offset_overall["offset_seconds"])].copy()

    if offset_overall.empty:
        warnings.warn("No offset passed the minimum‑window threshold. Exiting.")
        return

    # ---- Normalized risk (with bootstrap for overall) ----
    normalized_overall = add_normalized_risk_overall(
        offset_overall,
        baseline_overall,
        baseline_run,
        offset_hits,
        alpha,
        n_bootstrap,
        no_bootstrap,
        min_windows,
    )
    normalized_by_user = add_normalized_risk_by_user(
        offset_by_user, baseline_user, alpha, min_windows
    )

    # ---- Recovery times ----
    recovery_overall = compute_recovery_time_overall(normalized_overall, require_consecutive)
    recovery_by_user = compute_recovery_time_by_user(normalized_by_user, require_consecutive)

    # ---- Save CSVs ----
    baseline_run.to_csv(out_dir / "baseline_run_probabilities.csv", index=False)
    baseline_user.to_csv(out_dir / "baseline_user_probabilities.csv", index=False)
    baseline_overall.to_csv(out_dir / "baseline_overall_probability.csv", index=False)
    offset_by_user.to_csv(out_dir / "offset_probabilities_by_user.csv", index=False)
    offset_overall.to_csv(out_dir / "offset_probabilities_overall.csv", index=False)
    offset_hits.to_csv(out_dir / "offset_hits_per_distraction.csv", index=False)
    normalized_by_user.to_csv(out_dir / "normalized_risk_by_user.csv", index=False)
    normalized_overall.to_csv(out_dir / "normalized_risk_overall.csv", index=False)
    recovery_by_user.to_csv(out_dir / "recovery_time_by_user.csv", index=False)
    recovery_overall.to_csv(out_dir / "recovery_time_overall.csv", index=False)

    # ---- Plots ----
    baseline_p = baseline_overall.iloc[0]["baseline_p_error"] if not baseline_overall.empty else np.nan
    baseline_low = baseline_overall.iloc[0]["baseline_p_error_lower"] if not baseline_overall.empty else np.nan
    baseline_high = baseline_overall.iloc[0]["baseline_p_error_upper"] if not baseline_overall.empty else np.nan
    plot_overall_probability_curve(
        overall_probs=offset_overall,
        baseline_p=baseline_p,
        baseline_p_lower=baseline_low,
        baseline_p_upper=baseline_high,
        out_path=plots_dir / "overall_error_probability_curve.png",
    )
    plot_overall_normalized_curve(
        overall_norm=normalized_overall,
        out_path=plots_dir / "overall_normalized_risk_curve.png",
    )
    plot_user_normalized_curve(
        user_norm=normalized_by_user,
        out_path=plots_dir / "user_normalized_risk_curve.png",
    )

    # ---- Report ----
    write_report(
        out_path=out_dir / "recovery_report.md",
        delta_t_seconds=args.delta_t_seconds,
        offsets=offsets,
        baseline_overall=baseline_overall,
        overall_norm=normalized_overall,
        recovery_overall=recovery_overall,
        state_col=args.state_col,
        state_values=state_values,
        min_windows=min_windows,
        alpha=alpha,
    )

    # ---- JSON summary ----
    summary = {
        "delta_t_seconds": float(args.delta_t_seconds),
        "offsets_seconds": [float(x) for x in offsets],
        "min_windows": min_windows,
        "alpha": alpha,
        "require_consecutive": require_consecutive,
        "baseline_p_error_overall": baseline_p,
        "baseline_p_error_lower": baseline_low,
        "baseline_p_error_upper": baseline_high,
        "recovery_offset_seconds_overall": (
            float(recovery_overall.iloc[0]["recovery_offset_seconds"])
            if not recovery_overall.empty and pd.notna(recovery_overall.iloc[0]["recovery_offset_seconds"])
            else None
        ),
        "recovery_offset_conservative_seconds_overall": (
            float(recovery_overall.iloc[0]["recovery_offset_conservative_seconds"])
            if not recovery_overall.empty and pd.notna(recovery_overall.iloc[0]["recovery_offset_conservative_seconds"])
            else None
        ),
        "n_users_baseline": int(baseline_user["user_id"].nunique()) if not baseline_user.empty else 0,
        "n_users_distraction": int(distractions["user_id"].nunique()) if not distractions.empty else 0,
        "n_distraction_windows": int(distractions["timestamp_end"].notna().sum()),
        "n_bootstrap_reps": n_bootstrap if not no_bootstrap else 0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] Baseline runs analyzed: {len(baseline_run)}")
    print(f"[done] Distraction windows analyzed: {summary['n_distraction_windows']}")
    print(f"[done] Offsets after filtering (≥{min_windows} windows): {list(offset_overall['offset_seconds'])}")
    print(f"[done] Baseline P(error): {baseline_p:.6f}  [{baseline_low:.6f}, {baseline_high:.6f}]")
    print(f"[done] Output directory: {out_dir}")


if __name__ == "__main__":
    main()