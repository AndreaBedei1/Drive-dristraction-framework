#!/usr/bin/env python3
from __future__ import annotations

"""
Estimate baseline error probability and post-distraction recovery risk.

Core definitions:
- P_baseline = probability of >=1 error in a baseline window of length delta_t.
- P_t        = probability of >=1 error in [distraction_end + t, distraction_end + t + delta_t).
- normalized_risk(t) = P_t / P_baseline.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NANOS_PER_SECOND = 1_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze baseline risk and post-distraction recovery risk."
    )
    parser.add_argument(
        "--baseline-errors",
        type=Path,
        default=Path("data") / "Dataset Errors_baseline.csv",
        help="Path to baseline Dataset Errors CSV.",
    )
    parser.add_argument(
        "--distraction-errors",
        type=Path,
        default=Path("data") / "Dataset Errors_distraction.csv",
        help="Path to distraction Dataset Errors CSV.",
    )
    parser.add_argument(
        "--distractions",
        type=Path,
        default=Path("data") / "Dataset Distractions_distraction.csv",
        help="Path to distraction Dataset Distractions CSV.",
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
        help="Window length in seconds used for baseline and offset probabilities.",
    )
    parser.add_argument(
        "--users",
        type=str,
        default="",
        help="Optional comma-separated user_id filter (e.g. participant_01,participant_03).",
    )
    parser.add_argument(
        "--state-col",
        type=str,
        default="",
        help="Optional error column used as state filter (e.g. model_pred, emotion_label, error_type).",
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
    return parser.parse_args()


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
    offsets: List[float] = []
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


def _timestamps_to_ns(series: pd.Series) -> np.ndarray:
    if series.empty:
        return np.empty(0, dtype=np.int64)
    return np.sort(series.astype("int64").to_numpy(dtype=np.int64, copy=False))


def compute_baseline_run_probabilities(errors: pd.DataFrame, delta_t_seconds: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (user_id, run_id), grp in errors.groupby(["user_id", "run_id"], dropna=False):
        ts = grp["timestamp"].dropna().sort_values()
        if ts.empty:
            continue

        start_ns = int(ts.iloc[0].value)
        end_ns = int(ts.iloc[-1].value)
        duration_s = max((end_ns - start_ns) / NANOS_PER_SECOND, delta_t_seconds)
        n_windows = max(1, int(np.ceil(duration_s / delta_t_seconds)))

        rel_s = (ts.astype("int64") - start_ns) / NANOS_PER_SECOND
        idx = np.floor(rel_s / delta_t_seconds).astype(int).to_numpy()
        idx = np.clip(idx, 0, n_windows - 1)

        has_error = np.zeros(n_windows, dtype=bool)
        has_error[np.unique(idx)] = True
        error_windows = int(has_error.sum())
        p_error = float(error_windows / n_windows)

        rows.append(
            {
                "user_id": user_id,
                "run_id": run_id,
                "duration_seconds": float(duration_s),
                "n_windows": int(n_windows),
                "error_windows": error_windows,
                "p_error": p_error,
            }
        )

    cols = ["user_id", "run_id", "duration_seconds", "n_windows", "error_windows", "p_error"]
    return pd.DataFrame(rows, columns=cols)


def aggregate_baseline_by_user(run_probs: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if run_probs.empty:
        return pd.DataFrame(
            columns=["user_id", "n_runs", "n_windows", "error_windows", "baseline_p_error"]
        )

    for user_id, grp in run_probs.groupby("user_id", dropna=False):
        n_windows = int(grp["n_windows"].sum())
        error_windows = int(grp["error_windows"].sum())
        baseline_p = float(error_windows / n_windows) if n_windows > 0 else np.nan
        rows.append(
            {
                "user_id": user_id,
                "n_runs": int(len(grp)),
                "n_windows": n_windows,
                "error_windows": error_windows,
                "baseline_p_error": baseline_p,
            }
        )
    return pd.DataFrame(rows)


def aggregate_baseline_overall(run_probs: pd.DataFrame) -> pd.DataFrame:
    if run_probs.empty:
        return pd.DataFrame(
            [{"n_runs": 0, "n_windows": 0, "error_windows": 0, "baseline_p_error": np.nan}]
        )
    n_windows = int(run_probs["n_windows"].sum())
    error_windows = int(run_probs["error_windows"].sum())
    baseline_p = float(error_windows / n_windows) if n_windows > 0 else np.nan
    return pd.DataFrame(
        [
            {
                "n_runs": int(len(run_probs)),
                "n_windows": n_windows,
                "error_windows": error_windows,
                "baseline_p_error": baseline_p,
            }
        ]
    )


def compute_offset_probabilities(
    errors: pd.DataFrame,
    distractions: pd.DataFrame,
    offsets_seconds: Sequence[float],
    delta_t_seconds: float,
) -> pd.DataFrame:
    delta_ns = int(round(delta_t_seconds * NANOS_PER_SECOND))
    offsets = list(offsets_seconds)
    hits: Dict[float, int] = {offset: 0 for offset in offsets}
    totals: Dict[float, int] = {offset: 0 for offset in offsets}

    error_ns_by_run: Dict[Tuple[str, str], np.ndarray] = {}
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

    rows: List[Dict[str, object]] = []
    for offset in offsets:
        total = int(totals[offset])
        hit = int(hits[offset])
        p_error = float(hit / total) if total > 0 else np.nan
        rows.append(
            {
                "offset_seconds": float(offset),
                "n_windows": total,
                "error_windows": hit,
                "p_error": p_error,
            }
        )
    return pd.DataFrame(rows)


def compute_offset_probabilities_by_user(
    errors: pd.DataFrame,
    distractions: pd.DataFrame,
    offsets_seconds: Sequence[float],
    delta_t_seconds: float,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    if distractions.empty:
        return pd.DataFrame(columns=["user_id", "offset_seconds", "n_windows", "error_windows", "p_error"])

    for user_id, d_grp in distractions.groupby("user_id", dropna=False):
        e_grp = errors[errors["user_id"] == user_id].copy()
        probs = compute_offset_probabilities(e_grp, d_grp, offsets_seconds, delta_t_seconds)
        probs.insert(0, "user_id", user_id)
        rows.append(probs)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_offset_hits_per_distraction(
    errors: pd.DataFrame,
    distractions: pd.DataFrame,
    offsets_seconds: Sequence[float],
    delta_t_seconds: float,
) -> pd.DataFrame:
    delta_ns = int(round(delta_t_seconds * NANOS_PER_SECOND))
    offsets = list(offsets_seconds)

    error_ns_by_run: Dict[Tuple[str, str], np.ndarray] = {}
    for key, grp in errors.groupby(["user_id", "run_id"], dropna=False):
        error_ns_by_run[(str(key[0]), str(key[1]))] = _timestamps_to_ns(grp["timestamp"].dropna())

    rows: List[Dict[str, object]] = []
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


def add_normalized_risk_by_user(
    offset_probs_by_user: pd.DataFrame,
    baseline_by_user: pd.DataFrame,
) -> pd.DataFrame:
    out = offset_probs_by_user.copy()
    baseline_map = baseline_by_user.set_index("user_id")["baseline_p_error"].to_dict()
    out["baseline_p_error"] = out["user_id"].map(baseline_map)

    valid = out["baseline_p_error"] > 0
    out["normalized_risk"] = np.where(
        valid,
        out["p_error"] / out["baseline_p_error"],
        np.nan,
    )
    out["risk_gap_vs_baseline"] = out["normalized_risk"] - 1.0
    out["at_or_below_baseline"] = out["normalized_risk"] <= 1.0
    return out


def add_normalized_risk_overall(
    offset_probs_overall: pd.DataFrame,
    baseline_overall: pd.DataFrame,
) -> pd.DataFrame:
    out = offset_probs_overall.copy()
    baseline_p = float(baseline_overall.iloc[0]["baseline_p_error"]) if not baseline_overall.empty else np.nan
    out["baseline_p_error"] = baseline_p
    if pd.notna(baseline_p) and baseline_p > 0:
        out["normalized_risk"] = out["p_error"] / baseline_p
    else:
        out["normalized_risk"] = np.nan
    out["risk_gap_vs_baseline"] = out["normalized_risk"] - 1.0
    out["at_or_below_baseline"] = out["normalized_risk"] <= 1.0
    return out


def compute_recovery_time_by_user(normalized_by_user: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if normalized_by_user.empty:
        return pd.DataFrame(columns=["user_id", "baseline_p_error", "recovery_offset_seconds"])

    for user_id, grp in normalized_by_user.groupby("user_id", dropna=False):
        g = grp.sort_values("offset_seconds")
        baseline_p = float(g["baseline_p_error"].iloc[0]) if not g.empty else np.nan
        recovered = g[(g["normalized_risk"].notna()) & (g["normalized_risk"] <= 1.0)]
        recovery_offset = float(recovered["offset_seconds"].iloc[0]) if not recovered.empty else np.nan
        rows.append(
            {
                "user_id": user_id,
                "baseline_p_error": baseline_p,
                "recovery_offset_seconds": recovery_offset,
            }
        )
    return pd.DataFrame(rows)


def compute_recovery_time_overall(normalized_overall: pd.DataFrame) -> pd.DataFrame:
    if normalized_overall.empty:
        return pd.DataFrame([{"baseline_p_error": np.nan, "recovery_offset_seconds": np.nan}])
    g = normalized_overall.sort_values("offset_seconds")
    baseline_p = float(g["baseline_p_error"].iloc[0]) if not g.empty else np.nan
    recovered = g[(g["normalized_risk"].notna()) & (g["normalized_risk"] <= 1.0)]
    recovery_offset = float(recovered["offset_seconds"].iloc[0]) if not recovered.empty else np.nan
    return pd.DataFrame(
        [{"baseline_p_error": baseline_p, "recovery_offset_seconds": recovery_offset}]
    )


def plot_overall_probability_curve(
    overall_probs: pd.DataFrame,
    baseline_p: float,
    out_path: Path,
) -> None:
    if overall_probs.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(overall_probs["offset_seconds"], overall_probs["p_error"], marker="o", label="P(error | t)")
    if pd.notna(baseline_p):
        ax.axhline(baseline_p, color="#d62828", linestyle="--", label="P(error | baseline)")
    ax.set_xlabel("Offset after distraction end (s)")
    ax.set_ylabel("Probability")
    ax.set_title("Error Probability vs Offset")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_overall_normalized_curve(overall_norm: pd.DataFrame, out_path: Path) -> None:
    if overall_norm.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        overall_norm["offset_seconds"],
        overall_norm["normalized_risk"],
        marker="o",
        label="normalized_risk(t)",
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


def plot_user_normalized_curve(user_norm: pd.DataFrame, out_path: Path) -> None:
    if user_norm.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for user_id, grp in user_norm.groupby("user_id", dropna=False):
        g = grp.sort_values("offset_seconds")
        ax.plot(g["offset_seconds"], g["normalized_risk"], marker="o", label=str(user_id))
    ax.axhline(1.0, color="#2a9d8f", linestyle="--", label="Baseline parity (1.0)")
    ax.set_xlabel("Offset after distraction end (s)")
    ax.set_ylabel("Normalized risk")
    ax.set_title("Normalized Recovery Risk by Driver")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_report(
    out_path: Path,
    delta_t_seconds: float,
    offsets: Sequence[float],
    baseline_overall: pd.DataFrame,
    overall_norm: pd.DataFrame,
    recovery_overall: pd.DataFrame,
    state_col: str,
    state_values: Sequence[str],
) -> None:
    baseline_p = (
        float(baseline_overall.iloc[0]["baseline_p_error"])
        if not baseline_overall.empty
        else np.nan
    )
    recovery_offset = (
        float(recovery_overall.iloc[0]["recovery_offset_seconds"])
        if not recovery_overall.empty
        else np.nan
    )

    lines: List[str] = []
    lines.append("# Recovery Risk Report")
    lines.append("")
    lines.append("## Definitions")
    lines.append(
        f"- Baseline window length: {delta_t_seconds:g}s (used for baseline and post-distraction windows)."
    )
    lines.append(f"- Offsets analyzed: {', '.join(f'{x:g}' for x in offsets)} seconds.")
    lines.append("- Formula: normalized_risk(t) = P(error | t_after_distraction) / P(error | baseline).")
    lines.append("- Interpretation: normalized_risk(t) > 1 means residual impairment above baseline.")
    lines.append("- Interpretation: normalized_risk(t) = 1 means parity with baseline.")
    lines.append("- Interpretation: normalized_risk(t) < 1 means below-baseline risk (possible variance/noise).")
    lines.append("")

    if state_col:
        state_desc = ", ".join(state_values) if state_values else "(all values)"
        lines.append("## State filter")
        lines.append(f"- State column: {state_col}")
        lines.append(f"- Accepted values: {state_desc}")
        lines.append("")

    lines.append("## Overall baseline")
    lines.append(f"- P(error | baseline) = {baseline_p:.6f}" if pd.notna(baseline_p) else "- P(error | baseline) = NaN")
    lines.append(
        f"- Estimated recovery offset (first normalized_risk <= 1): {recovery_offset:g}s"
        if pd.notna(recovery_offset)
        else "- Estimated recovery offset: not reached in tested offsets"
    )
    lines.append("")

    lines.append("## Overall curve")
    if overall_norm.empty:
        lines.append("- No offset result available.")
    else:
        for _, row in overall_norm.sort_values("offset_seconds").iterrows():
            lines.append(
                f"- t={float(row['offset_seconds']):g}s: "
                f"P(error|t)={float(row['p_error']):.6f}, "
                f"normalized_risk={float(row['normalized_risk']):.6f}"
            )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    offsets = parse_offsets(args.offsets_seconds)
    users = parse_list(args.users)
    state_values = parse_list(args.state_values)

    ensure_exists(args.baseline_errors, "baseline errors")
    ensure_exists(args.distraction_errors, "distraction errors")
    ensure_exists(args.distractions, "distraction windows")

    if args.delta_t_seconds <= 0.0:
        raise ValueError("--delta-t-seconds must be > 0")

    out_dir: Path = args.out_dir
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline_errors = read_errors(args.baseline_errors)
    distraction_errors = read_errors(args.distraction_errors)
    distractions = read_distractions(args.distractions)

    baseline_errors = apply_user_filter(baseline_errors, users)
    distraction_errors = apply_user_filter(distraction_errors, users)
    distractions = apply_user_filter(distractions, users)

    baseline_errors = apply_state_filter(baseline_errors, args.state_col, state_values)
    distraction_errors = apply_state_filter(distraction_errors, args.state_col, state_values)

    baseline_run = compute_baseline_run_probabilities(baseline_errors, args.delta_t_seconds)
    baseline_user = aggregate_baseline_by_user(baseline_run)
    baseline_overall = aggregate_baseline_overall(baseline_run)

    offset_overall = compute_offset_probabilities(
        distraction_errors, distractions, offsets, args.delta_t_seconds
    )
    offset_by_user = compute_offset_probabilities_by_user(
        distraction_errors, distractions, offsets, args.delta_t_seconds
    )
    offset_hits = build_offset_hits_per_distraction(
        distraction_errors, distractions, offsets, args.delta_t_seconds
    )

    normalized_overall = add_normalized_risk_overall(offset_overall, baseline_overall)
    normalized_by_user = add_normalized_risk_by_user(offset_by_user, baseline_user)

    recovery_by_user = compute_recovery_time_by_user(normalized_by_user)
    recovery_overall = compute_recovery_time_overall(normalized_overall)

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

    baseline_p = (
        float(baseline_overall.iloc[0]["baseline_p_error"])
        if not baseline_overall.empty
        else np.nan
    )
    plot_overall_probability_curve(
        overall_probs=offset_overall,
        baseline_p=baseline_p,
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

    write_report(
        out_path=out_dir / "recovery_report.md",
        delta_t_seconds=args.delta_t_seconds,
        offsets=offsets,
        baseline_overall=baseline_overall,
        overall_norm=normalized_overall,
        recovery_overall=recovery_overall,
        state_col=args.state_col,
        state_values=state_values,
    )

    summary = {
        "delta_t_seconds": float(args.delta_t_seconds),
        "offsets_seconds": [float(x) for x in offsets],
        "baseline_p_error_overall": baseline_p,
        "recovery_offset_seconds_overall": (
            float(recovery_overall.iloc[0]["recovery_offset_seconds"])
            if not recovery_overall.empty
            and pd.notna(recovery_overall.iloc[0]["recovery_offset_seconds"])
            else None
        ),
        "n_users_baseline": int(baseline_user["user_id"].nunique()) if not baseline_user.empty else 0,
        "n_users_distraction": int(distractions["user_id"].nunique()) if not distractions.empty else 0,
        "n_distraction_windows": int(distractions["timestamp_end"].notna().sum()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] Baseline runs analyzed: {len(baseline_run)}")
    print(f"[done] Distraction windows analyzed: {summary['n_distraction_windows']}")
    print(f"[done] Baseline P(error): {baseline_p:.6f}" if pd.notna(baseline_p) else "[done] Baseline P(error): NaN")
    print(f"[done] Output directory: {out_dir}")


if __name__ == "__main__":
    main()
