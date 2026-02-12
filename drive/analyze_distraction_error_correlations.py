#!/usr/bin/env python3
from __future__ import annotations

"""
Comprehensive correlation analysis between distraction windows and driving errors.

Outputs:
- Merged event-level dataset (errors enriched with distraction context)
- Window-level dataset (distraction windows enriched with error counts)
- Statistical summaries and crosstabs
- Correlation matrices and figures
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

POST_DISTRACTION_GRACE_SECONDS = 2.0
POST_DISTRACTION_GRACE = pd.to_timedelta(POST_DISTRACTION_GRACE_SECONDS, unit="s")
PEDESTRIAN_COLLISION_MARKERS = (
    "walker.pedestrian",
    "controller.ai.walker",
    "pedestrian",
    "walker.",
)
VEHICLE_COLLISION_ERROR_TYPES = {
    "Vehicle collision",
    "Vehicle-pedestrian collision",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze correlations between Dataset Distractions and Dataset Errors."
    )
    parser.add_argument(
        "--distractions",
        type=Path,
        default=Path("data") / "Dataset Distractions.csv",
        help="Path to Dataset Distractions CSV.",
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=Path("data") / "Dataset Errors.csv",
        help="Path to Dataset Errors CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Output directory for tables and plots.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return "None"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return "None"
    return text


def normalize_error_type(error_type: object, details: object) -> str:
    """Normalize collision labels and recover pedestrian collisions from details."""
    normalized = clean_text(error_type)
    normalized_lower = normalized.lower()
    details_text = clean_text(details).lower()

    if "pedestrian" in normalized_lower or "walker" in normalized_lower:
        return "Vehicle-pedestrian collision"
    if any(marker in details_text for marker in PEDESTRIAN_COLLISION_MARKERS):
        return "Vehicle-pedestrian collision"
    if normalized == "Collision" and details_text.startswith("vehicle."):
        return "Vehicle collision"
    if normalized in {"Vehicle collision", "Collision"}:
        return normalized
    return normalized


def read_and_clean_distractions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    text_cols = [
        "user_id",
        "run_id",
        "weather",
        "map_name",
        "model_pred_start",
        "model_pred_end",
        "emotion_label_start",
        "emotion_label_end",
        "details",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].map(clean_text)

    numeric_cols = [
        "start_x",
        "start_y",
        "start_z",
        "end_x",
        "end_y",
        "end_z",
        "arousal_start",
        "arousal_end",
        "hr_bpm_start",
        "hr_bpm_end",
        "model_prob_start",
        "model_prob_end",
        "emotion_prob_start",
        "emotion_prob_end",
        "frame_start",
        "frame_end",
        "sim_time_start",
        "sim_time_end",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp_start"] = pd.to_datetime(df["timestamp_start"], errors="coerce")
    df["timestamp_end"] = pd.to_datetime(df["timestamp_end"], errors="coerce")
    df["distraction_duration_s"] = (
        df["timestamp_end"] - df["timestamp_start"]
    ).dt.total_seconds()

    df["hr_bpm_avg"] = df[["hr_bpm_start", "hr_bpm_end"]].mean(axis=1)
    df["arousal_avg"] = df[["arousal_start", "arousal_end"]].mean(axis=1)

    df["error_type"] = "distraction_window"
    df = df.reset_index(drop=True)
    df["distraction_index"] = df.index
    return df


def read_and_clean_errors(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    text_cols = [
        "user_id",
        "run_id",
        "weather",
        "map_name",
        "error_type",
        "model_pred",
        "model_pred_start",
        "model_pred_end",
        "emotion_label",
        "details",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].map(clean_text)

    # Normalize known label variants and recover pedestrian collisions from details.
    if "error_type" not in df.columns:
        df["error_type"] = "None"
    if "details" not in df.columns:
        df["details"] = "None"
    df["error_type"] = [
        normalize_error_type(error_type, details)
        for error_type, details in zip(df["error_type"], df["details"])
    ]

    numeric_cols = [
        "model_prob",
        "model_prob_start",
        "model_prob_end",
        "emotion_prob",
        "speed_kmh",
        "frame",
        "sim_time_seconds",
        "x",
        "y",
        "z",
        "road_id",
        "lane_id",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize error-level model columns:
    # new schema -> model_pred/model_prob
    # old schema -> model_pred_start/model_pred_end + model_prob_start/model_prob_end
    if "model_pred" not in df.columns:
        pred_candidates = [c for c in ["model_pred_start", "model_pred_end"] if c in df.columns]
        if pred_candidates:
            df["model_pred"] = df[pred_candidates].bfill(axis=1).iloc[:, 0].map(clean_text)
        else:
            df["model_pred"] = "None"
    else:
        df["model_pred"] = df["model_pred"].map(clean_text)

    if "model_prob" not in df.columns:
        prob_candidates = [c for c in ["model_prob_start", "model_prob_end"] if c in df.columns]
        if prob_candidates:
            df["model_prob"] = pd.to_numeric(
                df[prob_candidates].bfill(axis=1).iloc[:, 0], errors="coerce"
            )
        else:
            df["model_prob"] = np.nan
    else:
        df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce")

    # Backward-compatible aliases used by the rest of this analysis script.
    if "model_pred_start" not in df.columns:
        df["model_pred_start"] = df["model_pred"]
    if "model_pred_end" not in df.columns:
        df["model_pred_end"] = df["model_pred"]
    if "model_prob_start" not in df.columns:
        df["model_prob_start"] = df["model_prob"]
    if "model_prob_end" not in df.columns:
        df["model_prob_end"] = df["model_prob"]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.reset_index(drop=True)
    df["error_index"] = df.index
    return df


def contingency_and_cramers_v(table: pd.DataFrame) -> Dict[str, float]:
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"chi2": np.nan, "p_value": np.nan, "dof": np.nan, "cramers_v": np.nan}

    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    n = table.values.sum()
    if n == 0:
        return {"chi2": np.nan, "p_value": np.nan, "dof": np.nan, "cramers_v": np.nan}
    min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = np.sqrt((chi2 / n) / min_dim) if min_dim > 0 else np.nan
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": float(dof),
        "cramers_v": float(cramers_v),
    }


def mannwhitney_test(group_a: pd.Series, group_b: pd.Series) -> Dict[str, float]:
    a = pd.to_numeric(group_a, errors="coerce").dropna()
    b = pd.to_numeric(group_b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return {"statistic": np.nan, "p_value": np.nan}
    # Two-sided non-parametric test for distributions with possible non-normality.
    statistic, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"statistic": float(statistic), "p_value": float(p_value)}


def enrich_errors_with_distractions(
    errors: pd.DataFrame, distractions: pd.DataFrame
) -> pd.DataFrame:
    merged = errors.copy()
    merged["during_distraction"] = False
    merged["overlapping_distraction_count"] = 0
    merged["matched_distraction_index"] = np.nan
    merged["nearest_distraction_index"] = np.nan
    merged["seconds_from_distraction_start"] = np.nan
    merged["seconds_to_distraction_end"] = np.nan
    merged["seconds_to_nearest_distraction"] = np.nan

    matched_cols = [
        "timestamp_start",
        "timestamp_end",
        "distraction_duration_s",
        "hr_bpm_start",
        "hr_bpm_end",
        "hr_bpm_avg",
        "arousal_start",
        "arousal_end",
        "arousal_avg",
        "model_pred_start",
        "model_pred_end",
        "model_prob_start",
        "model_prob_end",
        "emotion_label_start",
        "emotion_label_end",
        "emotion_prob_start",
        "emotion_prob_end",
        "details",
    ]
    for col in matched_cols:
        merged[f"distraction_{col}"] = pd.NA

    group_cols = ["user_id", "run_id"]
    grouped_distractions = {
        key: grp.sort_values("timestamp_start").reset_index(drop=True)
        for key, grp in distractions.groupby(group_cols, dropna=False)
    }

    for key, error_group in merged.groupby(group_cols, dropna=False):
        if key not in grouped_distractions:
            continue

        d_grp = grouped_distractions[key]
        starts = d_grp["timestamp_start"]
        ends = d_grp["timestamp_end"]
        effective_ends = ends + POST_DISTRACTION_GRACE

        for error_idx in error_group.index:
            ts = merged.at[error_idx, "timestamp"]
            if pd.isna(ts):
                continue

            overlaps = (starts <= ts) & (ts <= effective_ends)
            overlap_count = int(overlaps.sum())
            merged.at[error_idx, "overlapping_distraction_count"] = overlap_count

            if overlap_count > 0:
                matched = d_grp.loc[overlaps].iloc[0]
                merged.at[error_idx, "during_distraction"] = True
                merged.at[error_idx, "matched_distraction_index"] = matched[
                    "distraction_index"
                ]
                merged.at[error_idx, "seconds_from_distraction_start"] = (
                    ts - matched["timestamp_start"]
                ).total_seconds()
                merged.at[error_idx, "seconds_to_distraction_end"] = (
                    matched["timestamp_end"] + POST_DISTRACTION_GRACE - ts
                ).total_seconds()
                merged.at[error_idx, "seconds_to_nearest_distraction"] = 0.0
                for col in matched_cols:
                    merged.at[error_idx, f"distraction_{col}"] = matched[col]
            else:
                # Distance to nearest distraction interval boundary.
                delta_to_start = (ts - starts).abs().dt.total_seconds()
                delta_to_end = (effective_ends - ts).abs().dt.total_seconds()
                nearest_dist = np.minimum(delta_to_start, delta_to_end)
                nearest_pos = nearest_dist.idxmin()
                nearest = d_grp.loc[nearest_pos]
                merged.at[error_idx, "nearest_distraction_index"] = nearest[
                    "distraction_index"
                ]
                merged.at[error_idx, "seconds_to_nearest_distraction"] = float(
                    nearest_dist.loc[nearest_pos]
                )

    merged["during_distraction_int"] = merged["during_distraction"].astype(int)
    return merged


def compute_errors_per_window(
    errors: pd.DataFrame, distractions: pd.DataFrame
) -> pd.DataFrame:
    windows = distractions.copy()
    windows["errors_in_window"] = 0

    grouped_errors = {
        key: grp.sort_values("timestamp").reset_index(drop=True)
        for key, grp in errors.groupby(["user_id", "run_id"], dropna=False)
    }

    for key, d_group in windows.groupby(["user_id", "run_id"], dropna=False):
        if key not in grouped_errors:
            continue
        e_grp = grouped_errors[key]
        e_ts = e_grp["timestamp"]
        for idx in d_group.index:
            start = windows.at[idx, "timestamp_start"]
            end = windows.at[idx, "timestamp_end"]
            if pd.isna(start) or pd.isna(end):
                continue
            effective_end = end + POST_DISTRACTION_GRACE
            count = ((e_ts >= start) & (e_ts <= effective_end)).sum()
            windows.at[idx, "errors_in_window"] = int(count)

    windows["analysis_window_duration_s"] = (
        pd.to_numeric(windows["distraction_duration_s"], errors="coerce")
        + POST_DISTRACTION_GRACE_SECONDS
    )
    windows["errors_per_minute_window"] = np.where(
        windows["analysis_window_duration_s"] > 0,
        windows["errors_in_window"] / (windows["analysis_window_duration_s"] / 60.0),
        np.nan,
    )
    return windows


def plot_heatmap(
    table: pd.DataFrame,
    title: str,
    out_path: Path,
    normalize_rows: bool = False,
    cmap: str = "viridis",
    annotate: bool = True,
) -> None:
    if table.empty:
        return

    values = table.values.astype(float)
    if normalize_rows:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)

    h, w = table.shape
    fig_w = min(24, max(8, 1.2 * w + 4))
    fig_h = min(20, max(6, 0.8 * h + 4))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(w))
    ax.set_xticklabels([str(c) for c in table.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(h))
    ax.set_yticklabels([str(r) for r in table.index])
    cbar_label = "Ratio" if normalize_rows else "Count"
    fig.colorbar(im, ax=ax, label=cbar_label)

    if annotate:
        for i in range(h):
            for j in range(w):
                if normalize_rows:
                    txt = f"{values[i, j]:.2f}"
                else:
                    txt = f"{int(table.values[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_bar_counts(merged: pd.DataFrame, out_path: Path) -> None:
    counts = merged["during_distraction"].value_counts().reindex([True, False], fill_value=0)
    labels = [f"During distraction (+{POST_DISTRACTION_GRACE_SECONDS:g}s)", "Outside distraction"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, counts.values, color=["#E76F51", "#264653"])
    ax.set_title(f"Errors During (+{POST_DISTRACTION_GRACE_SECONDS:g}s) vs Outside Distraction")
    ax.set_ylabel("Number of errors")
    for i, value in enumerate(counts.values):
        ax.text(i, value + 0.5, str(int(value)), ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_stacked_error_type(merged: pd.DataFrame, out_path: Path) -> None:
    table = pd.crosstab(merged["error_type"], merged["during_distraction"])
    if table.empty:
        return
    table = table.rename(columns={False: "Outside", True: "During"})
    table = table.sort_values(by=["During", "Outside"], ascending=False)

    fig, ax = plt.subplots(figsize=(12, 7))
    table.plot(kind="bar", stacked=True, ax=ax, color=["#457B9D", "#E63946"])
    ax.set_title("Error Type Split by Distraction State")
    ax.set_xlabel("Error type")
    ax.set_ylabel("Count")
    ax.legend(title="State")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_speed_boxplot(merged: pd.DataFrame, out_path: Path) -> None:
    during = merged.loc[merged["during_distraction"], "speed_kmh"].dropna()
    outside = merged.loc[~merged["during_distraction"], "speed_kmh"].dropna()
    if len(during) == 0 and len(outside) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([outside, during], labels=["Outside", "During"], patch_artist=True)
    ax.set_title("Speed Distribution by Distraction State")
    ax.set_ylabel("Speed (km/h)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_hr_vs_errors_windows(windows: pd.DataFrame, out_path: Path) -> None:
    plot_df = windows[["hr_bpm_avg", "errors_in_window"]].dropna()
    if len(plot_df) < 2:
        return
    x = plot_df["hr_bpm_avg"].values
    y = plot_df["errors_in_window"].values

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.75, color="#1D3557")
    ax.set_title("Errors in Distraction Window vs Average HR")
    ax.set_xlabel("Average HR in window (bpm)")
    ax.set_ylabel("Errors within same window")

    if len(np.unique(x)) >= 2:
        coef = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = coef[0] * x_line + coef[1]
        ax.plot(x_line, y_line, color="#E63946", linewidth=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_window_metric_by_error_presence(
    windows: pd.DataFrame, metric: str, title: str, y_label: str, out_path: Path
) -> None:
    subset = windows[[metric, "errors_in_window"]].copy()
    subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
    with_errors = subset.loc[subset["errors_in_window"] > 0, metric].dropna()
    without_errors = subset.loc[subset["errors_in_window"] == 0, metric].dropna()
    if len(with_errors) == 0 and len(without_errors) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([without_errors, with_errors], labels=["No errors", ">=1 error"], patch_artist=True)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_hr_by_error_type(merged: pd.DataFrame, out_path: Path) -> None:
    subset = merged[merged["during_distraction"]].copy()
    subset = subset.dropna(subset=["distraction_hr_bpm_avg"])
    if subset.empty:
        return

    counts = subset["error_type"].value_counts()
    keep = counts[counts >= 2].index.tolist()
    subset = subset[subset["error_type"].isin(keep)]
    if subset.empty:
        return

    ordered_types = subset["error_type"].value_counts().index.tolist()
    data = [
        subset.loc[subset["error_type"] == err, "distraction_hr_bpm_avg"].values
        for err in ordered_types
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data, labels=ordered_types, patch_artist=True)
    ax.set_title("Average HR in Matched Distraction Window by Error Type")
    ax.set_ylabel("HR (bpm)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    if corr.empty:
        return
    fig_w = min(18, max(8, len(corr.columns) * 0.8))
    fig_h = fig_w
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Spearman Correlation Matrix (Numeric Features)")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax, label="Spearman rho")

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(
                j,
                i,
                f"{corr.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def spearman_with_pvalues(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = df.columns.tolist()
    corr = pd.DataFrame(np.nan, index=cols, columns=cols)
    pval = pd.DataFrame(np.nan, index=cols, columns=cols)

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i > j:
                continue
            if c1 == c2:
                series = df[c1].dropna()
                if len(series) >= 3 and series.nunique(dropna=True) > 1:
                    corr.at[c1, c2] = 1.0
                    pval.at[c1, c2] = 0.0
                continue
            sub = df[[c1, c2]].dropna()
            if len(sub) < 3:
                continue
            if sub[c1].nunique(dropna=True) < 2 or sub[c2].nunique(dropna=True) < 2:
                continue
            result = stats.spearmanr(
                sub[c1].to_numpy(), sub[c2].to_numpy(), nan_policy="omit"
            )
            rho = getattr(result, "statistic", np.nan)
            p = getattr(result, "pvalue", np.nan)

            # scipy can return 2x2 arrays in some versions/edge-cases.
            if isinstance(rho, np.ndarray):
                if rho.shape == (2, 2):
                    rho = rho[0, 1]
                elif rho.size == 1:
                    rho = rho.item()
                else:
                    rho = np.nan
            if isinstance(p, np.ndarray):
                if p.shape == (2, 2):
                    p = p[0, 1]
                elif p.size == 1:
                    p = p.item()
                else:
                    p = np.nan

            rho = float(rho) if pd.notna(rho) else np.nan
            p = float(p) if pd.notna(p) else np.nan
            corr.at[c1, c2] = rho
            corr.at[c2, c1] = rho
            pval.at[c1, c2] = p
            pval.at[c2, c1] = p
    return corr, pval


def pairwise_stats_to_table(rows: Iterable[Dict[str, object]]) -> pd.DataFrame:
    cols = ["test_name", "chi2", "p_value", "dof", "cramers_v", "notes"]
    return pd.DataFrame(list(rows), columns=cols)


def generate_report(
    merged: pd.DataFrame,
    windows: pd.DataFrame,
    statistical_tests: pd.DataFrame,
    corr: pd.DataFrame,
    corr_p: pd.DataFrame,
    out_path: Path,
) -> None:
    total_errors = len(merged)
    during = int(merged["during_distraction"].sum())
    outside = total_errors - during
    during_pct = (during / total_errors * 100.0) if total_errors else 0.0

    lines: List[str] = []
    lines.append("# Correlation Analysis Report")
    lines.append("")
    lines.append("## Core counts")
    lines.append(f"- Total errors: {total_errors}")
    lines.append(
        f"- Errors during distraction (+{POST_DISTRACTION_GRACE_SECONDS:g}s): {during} ({during_pct:.2f}%)"
    )
    lines.append(f"- Errors outside distraction: {outside} ({100.0 - during_pct:.2f}%)")
    lines.append(f"- Total distraction windows: {len(windows)}")
    by_user = merged.groupby("user_id")["during_distraction"].agg(["sum", "count"])
    for user_id, row in by_user.iterrows():
        user_during = int(row["sum"])
        user_total = int(row["count"])
        user_pct = (user_during / user_total * 100.0) if user_total else 0.0
        lines.append(
            f"- {user_id}: {user_during}/{user_total} errors during distraction (+{POST_DISTRACTION_GRACE_SECONDS:g}s) ({user_pct:.2f}%)"
        )
    lines.append("")

    lines.append("## Error type split")
    split = pd.crosstab(merged["error_type"], merged["during_distraction"])
    split = split.rename(columns={False: "outside", True: "during"})
    split["total"] = split.sum(axis=1)
    split = split.sort_values("total", ascending=False)
    for err, row in split.iterrows():
        lines.append(
            f"- {err}: total={int(row['total'])}, during={int(row.get('during', 0))}, outside={int(row.get('outside', 0))}"
        )
    lines.append("")

    lines.append("## Designed correlation checks")
    vc_mask = merged["error_type"].isin(VEHICLE_COLLISION_ERROR_TYPES)
    vc_target = merged["emotion_label"].isin(["fear", "angry"])
    vc_total = int(vc_mask.sum())
    vc_match = int((vc_mask & vc_target).sum())
    vc_pct = (vc_match / vc_total * 100.0) if vc_total else 0.0
    lines.append(
        f"- Vehicle-related collisions with fear/angry: {vc_match}/{vc_total} ({vc_pct:.2f}%)"
    )

    sl_mask = merged["error_type"].eq("Solid line crossing")
    sl_target = merged["emotion_label"].isin(["neutral", "happy"])
    sl_total = int(sl_mask.sum())
    sl_match = int((sl_mask & sl_target).sum())
    sl_pct = (sl_match / sl_total * 100.0) if sl_total else 0.0
    lines.append(
        f"- Solid line crossing with neutral/happy: {sl_match}/{sl_total} ({sl_pct:.2f}%)"
    )

    start_model_allowed = windows["model_pred_start"].isin(["None", "sing"])
    allowed_model_count = int(start_model_allowed.sum())
    model_total = len(windows)
    allowed_model_pct = (allowed_model_count / model_total * 100.0) if model_total else 0.0
    lines.append(
        f"- Distraction model_pred_start in {{None, sing}}: {allowed_model_count}/{model_total} ({allowed_model_pct:.2f}%)"
    )

    start_emotion_allowed = windows["emotion_label_start"].isin(["neutral", "happy", "sad"])
    allowed_emotion_count = int(start_emotion_allowed.sum())
    allowed_emotion_pct = (allowed_emotion_count / model_total * 100.0) if model_total else 0.0
    sad_count = int((windows["emotion_label_start"] == "sad").sum())
    sad_pct = (sad_count / model_total * 100.0) if model_total else 0.0
    lines.append(
        f"- Distraction emotion_label_start in {{neutral, happy, sad}}: {allowed_emotion_count}/{model_total} ({allowed_emotion_pct:.2f}%)"
    )
    lines.append(f"- Distraction emotion_label_start == sad: {sad_count}/{model_total} ({sad_pct:.2f}%)")
    lines.append("")

    lines.append("## Statistical tests")
    during_n = int(merged["during_distraction"].sum())
    lines.append(
        f"- Note: tests on \"during distraction (+{POST_DISTRACTION_GRACE_SECONDS:g}s)\" subsets use n={during_n} events; interpret sparse crosstabs with caution."
    )
    if len(statistical_tests) == 0:
        lines.append("- No valid test computed.")
    else:
        for _, row in statistical_tests.iterrows():
            lines.append(
                f"- {row['test_name']}: p={row['p_value']:.4g}, chi2={row['chi2']:.4g}, Cramer's V={row['cramers_v']:.4g}. {row['notes']}"
            )
    lines.append("")

    lines.append("## Strongest numeric correlations (|rho| >= 0.30, p < 0.05)")
    pairs: List[Tuple[str, str, float, float]] = []
    for i, c1 in enumerate(corr.columns):
        for j, c2 in enumerate(corr.columns):
            if j <= i:
                continue
            rho = corr.iat[i, j]
            p = corr_p.iat[i, j]
            if pd.isna(rho) or pd.isna(p):
                continue
            if abs(rho) >= 0.30 and p < 0.05:
                pairs.append((c1, c2, float(rho), float(p)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    if not pairs:
        lines.append("- No correlation passed the threshold.")
    else:
        for c1, c2, rho, p in pairs[:20]:
            lines.append(f"- {c1} vs {c2}: rho={rho:.3f}, p={p:.4g}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    distractions = read_and_clean_distractions(args.distractions)
    errors = read_and_clean_errors(args.errors)

    merged = enrich_errors_with_distractions(errors, distractions)
    windows = compute_errors_per_window(errors, distractions)

    # Main contingency tables and stats.
    test_rows: List[Dict[str, object]] = []

    t_error_vs_during = pd.crosstab(merged["error_type"], merged["during_distraction"])
    stats_error_vs_during = contingency_and_cramers_v(t_error_vs_during)
    test_rows.append(
        {
            "test_name": "error_type_vs_during_distraction",
            "chi2": stats_error_vs_during["chi2"],
            "p_value": stats_error_vs_during["p_value"],
            "dof": stats_error_vs_during["dof"],
            "cramers_v": stats_error_vs_during["cramers_v"],
            "notes": "Association between error category and distraction state.",
        }
    )

    t_emotion_vs_error = pd.crosstab(merged["emotion_label"], merged["error_type"])
    stats_emotion_vs_error = contingency_and_cramers_v(t_emotion_vs_error)
    test_rows.append(
        {
            "test_name": "emotion_label_at_error_vs_error_type",
            "chi2": stats_emotion_vs_error["chi2"],
            "p_value": stats_emotion_vs_error["p_value"],
            "dof": stats_emotion_vs_error["dof"],
            "cramers_v": stats_emotion_vs_error["cramers_v"],
            "notes": "Association between emotion at error instant and error type.",
        }
    )

    t_model_vs_error = pd.crosstab(merged["model_pred_start"], merged["error_type"])
    stats_model_vs_error = contingency_and_cramers_v(t_model_vs_error)
    test_rows.append(
        {
            "test_name": "model_pred_at_error_vs_error_type",
            "chi2": stats_model_vs_error["chi2"],
            "p_value": stats_model_vs_error["p_value"],
            "dof": stats_model_vs_error["dof"],
            "cramers_v": stats_model_vs_error["cramers_v"],
            "notes": "Association between predicted distraction label at error event and error type.",
        }
    )

    t_vehicle_collision_vs_target_emotion = pd.crosstab(
        merged["error_type"].isin(VEHICLE_COLLISION_ERROR_TYPES).map(
            {True: "Vehicle-related collision", False: "Other errors"}
        ),
        merged["emotion_label"].isin(["fear", "angry"]).map(
            {True: "fear_or_angry", False: "other_emotions"}
        ),
    )
    stats_vehicle_collision_vs_target_emotion = contingency_and_cramers_v(
        t_vehicle_collision_vs_target_emotion
    )
    test_rows.append(
        {
            "test_name": "vehicle_collision_vs_fear_or_angry",
            "chi2": stats_vehicle_collision_vs_target_emotion["chi2"],
            "p_value": stats_vehicle_collision_vs_target_emotion["p_value"],
            "dof": stats_vehicle_collision_vs_target_emotion["dof"],
            "cramers_v": stats_vehicle_collision_vs_target_emotion["cramers_v"],
            "notes": "Targeted check: vehicle-related collisions should skew toward fear/angry.",
        }
    )

    t_solid_line_vs_target_emotion = pd.crosstab(
        merged["error_type"].eq("Solid line crossing").map(
            {True: "Solid line crossing", False: "Other errors"}
        ),
        merged["emotion_label"].isin(["neutral", "happy"]).map(
            {True: "neutral_or_happy", False: "other_emotions"}
        ),
    )
    stats_solid_line_vs_target_emotion = contingency_and_cramers_v(
        t_solid_line_vs_target_emotion
    )
    test_rows.append(
        {
            "test_name": "solid_line_crossing_vs_neutral_or_happy",
            "chi2": stats_solid_line_vs_target_emotion["chi2"],
            "p_value": stats_solid_line_vs_target_emotion["p_value"],
            "dof": stats_solid_line_vs_target_emotion["dof"],
            "cramers_v": stats_solid_line_vs_target_emotion["cramers_v"],
            "notes": "Targeted check: solid line crossings should skew toward neutral/happy.",
        }
    )

    during_only = merged[merged["during_distraction"]].copy()
    t_window_pred_start_vs_error = pd.crosstab(
        during_only["distraction_model_pred_start"], during_only["error_type"]
    )
    stats_window_pred_start_vs_error = contingency_and_cramers_v(
        t_window_pred_start_vs_error
    )
    test_rows.append(
        {
            "test_name": "window_model_pred_start_vs_error_type_during_only",
            "chi2": stats_window_pred_start_vs_error["chi2"],
            "p_value": stats_window_pred_start_vs_error["p_value"],
            "dof": stats_window_pred_start_vs_error["dof"],
            "cramers_v": stats_window_pred_start_vs_error["cramers_v"],
            "notes": "Association between distraction window start prediction and error type for matched events.",
        }
    )

    t_window_pred_end_vs_error = pd.crosstab(
        during_only["distraction_model_pred_end"], during_only["error_type"]
    )
    stats_window_pred_end_vs_error = contingency_and_cramers_v(t_window_pred_end_vs_error)
    test_rows.append(
        {
            "test_name": "window_model_pred_end_vs_error_type_during_only",
            "chi2": stats_window_pred_end_vs_error["chi2"],
            "p_value": stats_window_pred_end_vs_error["p_value"],
            "dof": stats_window_pred_end_vs_error["dof"],
            "cramers_v": stats_window_pred_end_vs_error["cramers_v"],
            "notes": "Association between distraction window end prediction and error type for matched events.",
        }
    )

    t_window_emotion_start_vs_error = pd.crosstab(
        during_only["distraction_emotion_label_start"], during_only["error_type"]
    )
    stats_window_emotion_start_vs_error = contingency_and_cramers_v(
        t_window_emotion_start_vs_error
    )
    test_rows.append(
        {
            "test_name": "window_emotion_start_vs_error_type_during_only",
            "chi2": stats_window_emotion_start_vs_error["chi2"],
            "p_value": stats_window_emotion_start_vs_error["p_value"],
            "dof": stats_window_emotion_start_vs_error["dof"],
            "cramers_v": stats_window_emotion_start_vs_error["cramers_v"],
            "notes": "Association between distraction window start emotion and error type for matched events.",
        }
    )

    t_window_emotion_end_vs_error = pd.crosstab(
        during_only["distraction_emotion_label_end"], during_only["error_type"]
    )
    stats_window_emotion_end_vs_error = contingency_and_cramers_v(
        t_window_emotion_end_vs_error
    )
    test_rows.append(
        {
            "test_name": "window_emotion_end_vs_error_type_during_only",
            "chi2": stats_window_emotion_end_vs_error["chi2"],
            "p_value": stats_window_emotion_end_vs_error["p_value"],
            "dof": stats_window_emotion_end_vs_error["dof"],
            "cramers_v": stats_window_emotion_end_vs_error["cramers_v"],
            "notes": "Association between distraction window end emotion and error type for matched events.",
        }
    )

    t_window_model_start_vs_emotion_start = pd.crosstab(
        windows["model_pred_start"], windows["emotion_label_start"]
    )
    stats_window_model_start_vs_emotion_start = contingency_and_cramers_v(
        t_window_model_start_vs_emotion_start
    )
    test_rows.append(
        {
            "test_name": "window_model_start_vs_emotion_start",
            "chi2": stats_window_model_start_vs_emotion_start["chi2"],
            "p_value": stats_window_model_start_vs_emotion_start["p_value"],
            "dof": stats_window_model_start_vs_emotion_start["dof"],
            "cramers_v": stats_window_model_start_vs_emotion_start["cramers_v"],
            "notes": "Association between distraction model prediction at window start and start emotion.",
        }
    )

    # Correlations on window-level and event-level numerics.
    if not windows.empty:
        hr_s = windows[["hr_bpm_avg", "errors_in_window"]].dropna()
        if len(hr_s) >= 3 and hr_s["hr_bpm_avg"].nunique() > 1 and hr_s["errors_in_window"].nunique() > 1:
            corr_hr_err_spearman = stats.spearmanr(
                hr_s["hr_bpm_avg"], hr_s["errors_in_window"], nan_policy="omit"
            )
            test_rows.append(
                {
                    "test_name": "window_hr_avg_vs_errors_in_window_spearman",
                    "chi2": float(corr_hr_err_spearman.statistic)
                    if hasattr(corr_hr_err_spearman, "statistic")
                    else np.nan,
                    "p_value": float(corr_hr_err_spearman.pvalue)
                    if hasattr(corr_hr_err_spearman, "pvalue")
                    else np.nan,
                    "dof": np.nan,
                    "cramers_v": np.nan,
                    "notes": "Spearman rho reported in chi2 column for compactness.",
                }
            )

        arousal_s = windows[["arousal_avg", "errors_in_window"]].dropna()
        if (
            len(arousal_s) >= 3
            and arousal_s["arousal_avg"].nunique() > 1
            and arousal_s["errors_in_window"].nunique() > 1
        ):
            corr_arousal_err_spearman = stats.spearmanr(
                arousal_s["arousal_avg"], arousal_s["errors_in_window"], nan_policy="omit"
            )
            test_rows.append(
                {
                    "test_name": "window_arousal_avg_vs_errors_in_window_spearman",
                    "chi2": float(corr_arousal_err_spearman.statistic)
                    if hasattr(corr_arousal_err_spearman, "statistic")
                    else np.nan,
                    "p_value": float(corr_arousal_err_spearman.pvalue)
                    if hasattr(corr_arousal_err_spearman, "pvalue")
                    else np.nan,
                    "dof": np.nan,
                    "cramers_v": np.nan,
                    "notes": "Spearman rho reported in chi2 column for compactness.",
                }
            )

        w_has_err = windows["errors_in_window"] > 0
        mw_hr = mannwhitney_test(
            windows.loc[w_has_err, "hr_bpm_avg"], windows.loc[~w_has_err, "hr_bpm_avg"]
        )
        test_rows.append(
            {
                "test_name": "window_hr_avg_distribution_errors_vs_no_errors_mannwhitney",
                "chi2": mw_hr["statistic"],
                "p_value": mw_hr["p_value"],
                "dof": np.nan,
                "cramers_v": np.nan,
                "notes": "U statistic reported in chi2 column.",
            }
        )

        mw_arousal = mannwhitney_test(
            windows.loc[w_has_err, "arousal_avg"], windows.loc[~w_has_err, "arousal_avg"]
        )
        test_rows.append(
            {
                "test_name": "window_arousal_avg_distribution_errors_vs_no_errors_mannwhitney",
                "chi2": mw_arousal["statistic"],
                "p_value": mw_arousal["p_value"],
                "dof": np.nan,
                "cramers_v": np.nan,
                "notes": "U statistic reported in chi2 column.",
            }
        )

    numeric_cols = [
        "during_distraction_int",
        "speed_kmh",
        "model_prob_start",
        "model_prob_end",
        "emotion_prob",
        "seconds_from_distraction_start",
        "seconds_to_distraction_end",
        "seconds_to_nearest_distraction",
        "distraction_duration_s",
        "distraction_hr_bpm_start",
        "distraction_hr_bpm_end",
        "distraction_hr_bpm_avg",
        "distraction_arousal_start",
        "distraction_arousal_end",
        "distraction_arousal_avg",
        "distraction_model_prob_start",
        "distraction_model_prob_end",
        "distraction_emotion_prob_start",
        "distraction_emotion_prob_end",
    ]
    numeric_cols = [c for c in numeric_cols if c in merged.columns]
    numeric_df = merged[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr, corr_p = spearman_with_pvalues(numeric_df)

    statistical_tests = pairwise_stats_to_table(test_rows)

    # Save datasets and tables.
    merged.to_csv(out_dir / "merged_error_distraction_events.csv", index=False)
    windows.to_csv(out_dir / "distraction_windows_with_error_counts.csv", index=False)
    statistical_tests.to_csv(out_dir / "statistical_tests.csv", index=False)
    t_error_vs_during.to_csv(out_dir / "crosstab_error_type_vs_during.csv")
    t_emotion_vs_error.to_csv(out_dir / "crosstab_emotion_vs_error_type.csv")
    t_model_vs_error.to_csv(out_dir / "crosstab_model_pred_event_vs_error_type.csv")
    t_window_pred_start_vs_error.to_csv(
        out_dir / "crosstab_window_model_pred_start_vs_error_type_during_only.csv"
    )
    t_window_pred_end_vs_error.to_csv(
        out_dir / "crosstab_window_model_pred_end_vs_error_type_during_only.csv"
    )
    t_window_emotion_start_vs_error.to_csv(
        out_dir / "crosstab_window_emotion_start_vs_error_type_during_only.csv"
    )
    t_window_emotion_end_vs_error.to_csv(
        out_dir / "crosstab_window_emotion_end_vs_error_type_during_only.csv"
    )
    t_vehicle_collision_vs_target_emotion.to_csv(
        out_dir / "crosstab_vehicle_collision_vs_fear_angry.csv"
    )
    t_solid_line_vs_target_emotion.to_csv(
        out_dir / "crosstab_solid_line_crossing_vs_neutral_happy.csv"
    )
    t_window_model_start_vs_emotion_start.to_csv(
        out_dir / "crosstab_window_model_start_vs_emotion_start.csv"
    )
    corr.to_csv(out_dir / "spearman_correlation_matrix.csv")
    corr_p.to_csv(out_dir / "spearman_pvalues_matrix.csv")

    summary = {
        "total_errors": int(len(merged)),
        "errors_during_distraction": int(merged["during_distraction"].sum()),
        "errors_outside_distraction": int((~merged["during_distraction"]).sum()),
        "total_distraction_windows": int(len(windows)),
        "avg_errors_per_window": float(windows["errors_in_window"].mean()),
        "median_errors_per_window": float(windows["errors_in_window"].median()),
    }
    (out_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Plots.
    plot_bar_counts(merged, plots_dir / "errors_during_vs_outside.png")
    plot_stacked_error_type(merged, plots_dir / "error_type_split_by_distraction_state.png")
    plot_heatmap(
        t_error_vs_during,
        "Error Type vs Distraction State (Counts)",
        plots_dir / "heatmap_error_type_vs_distraction_state.png",
        normalize_rows=False,
        cmap="magma",
        annotate=True,
    )
    plot_heatmap(
        t_error_vs_during,
        "Error Type vs Distraction State (Row-normalized)",
        plots_dir / "heatmap_error_type_vs_distraction_state_row_norm.png",
        normalize_rows=True,
        cmap="cividis",
        annotate=True,
    )
    plot_heatmap(
        t_emotion_vs_error,
        "Emotion at Error vs Error Type",
        plots_dir / "heatmap_emotion_vs_error_type.png",
        normalize_rows=False,
        cmap="viridis",
        annotate=True,
    )
    plot_heatmap(
        t_model_vs_error,
        "Model Prediction at Error vs Error Type",
        plots_dir / "heatmap_model_pred_at_error_vs_error_type.png",
        normalize_rows=False,
        cmap="plasma",
        annotate=True,
    )
    plot_heatmap(
        t_vehicle_collision_vs_target_emotion,
        "Vehicle Collision vs Fear/Angry Emotion",
        plots_dir / "heatmap_vehicle_collision_vs_fear_angry.png",
        normalize_rows=False,
        cmap="magma",
        annotate=True,
    )
    plot_heatmap(
        t_solid_line_vs_target_emotion,
        "Solid Line Crossing vs Neutral/Happy Emotion",
        plots_dir / "heatmap_solid_line_crossing_vs_neutral_happy.png",
        normalize_rows=False,
        cmap="cividis",
        annotate=True,
    )
    plot_heatmap(
        t_window_model_start_vs_emotion_start,
        "Window Start Model Prediction vs Start Emotion",
        plots_dir / "heatmap_window_model_start_vs_emotion_start.png",
        normalize_rows=False,
        cmap="viridis",
        annotate=True,
    )
    plot_heatmap(
        t_window_pred_start_vs_error,
        "Distraction Window Start Prediction vs Error Type (During Only)",
        plots_dir / "heatmap_window_model_start_vs_error_type_during_only.png",
        normalize_rows=False,
        cmap="inferno",
        annotate=True,
    )
    plot_heatmap(
        t_window_pred_end_vs_error,
        "Distraction Window End Prediction vs Error Type (During Only)",
        plots_dir / "heatmap_window_model_end_vs_error_type_during_only.png",
        normalize_rows=False,
        cmap="inferno",
        annotate=True,
    )
    plot_heatmap(
        t_window_emotion_start_vs_error,
        "Distraction Window Start Emotion vs Error Type (During Only)",
        plots_dir / "heatmap_window_emotion_start_vs_error_type_during_only.png",
        normalize_rows=False,
        cmap="viridis",
        annotate=True,
    )
    plot_heatmap(
        t_window_emotion_end_vs_error,
        "Distraction Window End Emotion vs Error Type (During Only)",
        plots_dir / "heatmap_window_emotion_end_vs_error_type_during_only.png",
        normalize_rows=False,
        cmap="viridis",
        annotate=True,
    )
    plot_speed_boxplot(merged, plots_dir / "boxplot_speed_by_distraction_state.png")
    plot_hr_by_error_type(merged, plots_dir / "boxplot_hr_by_error_type_during_only.png")
    plot_hr_vs_errors_windows(windows, plots_dir / "scatter_hr_vs_errors_per_window.png")
    plot_window_metric_by_error_presence(
        windows,
        metric="hr_bpm_avg",
        title="Window HR by Error Presence",
        y_label="HR (bpm)",
        out_path=plots_dir / "boxplot_window_hr_by_error_presence.png",
    )
    plot_window_metric_by_error_presence(
        windows,
        metric="arousal_avg",
        title="Window Arousal by Error Presence",
        y_label="Arousal",
        out_path=plots_dir / "boxplot_window_arousal_by_error_presence.png",
    )
    plot_correlation_heatmap(corr, plots_dir / "heatmap_spearman_numeric_correlations.png")

    generate_report(
        merged=merged,
        windows=windows,
        statistical_tests=statistical_tests,
        corr=corr,
        corr_p=corr_p,
        out_path=out_dir / "analysis_report.md",
    )

    total_errors = len(merged)
    during = int(merged["during_distraction"].sum())
    outside = total_errors - during
    print(f"[done] Total errors: {total_errors}")
    print(f"[done] During distraction (+{POST_DISTRACTION_GRACE_SECONDS:g}s): {during}")
    print(f"[done] Outside distraction: {outside}")
    print(f"[done] Outputs written to: {out_dir}")
if __name__ == "__main__":
    main()
