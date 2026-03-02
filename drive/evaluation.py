#!/usr/bin/env python3
"""
evaluate_impairment.py
======================
Construct-validity evaluation for the driver impairment FTD score.

The core problem: there is no continuous ground-truth impairment label.
Instead, we validate against the known CAUSAL STRUCTURE of post-distraction
recovery:

  1. Recovery curve fitting     — does FTD decay exponentially per distraction?
  2. Onset discrimination       — does FTD spike at distraction onset?
  3. Calibration curve          — does FTD decile predict aggregate error rate?
  4. Concordance index (C-index)— do higher-risk pairs rank above lower-risk ones?

These together form a principled evaluation of a CONTINUOUS RISK ESTIMATOR
without needing per-sample ground truth impairment labels.

Usage
-----
Call evaluate_construct_validity() from run_pipeline() after ftd_te is
computed. Or run standalone:

    python evaluate_impairment.py \\
        --artifact results/impairment_hmm.joblib \\
        --data-path data \\
        --output-dir results/construct_validity/

All plots are saved to output_dir. A summary dict is returned / saved to JSON.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, mannwhitneyu, spearmanr
from hmmlearn.hmm import GaussianHMM
from ftd_hmm import *

LOG = logging.getLogger("construct_validity")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Recovery curve fitting
# ─────────────────────────────────────────────────────────────────────────────

def _exp_decay(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    """FTD(t) = A * exp(-t / tau) + C"""
    return A * np.exp(-t / np.clip(tau, 0.1, 300.0)) + C


def fit_recovery_curves(
    df_te: pd.DataFrame,
    ftd_te: np.ndarray,
    min_points: int = 6,
) -> Tuple[pd.DataFrame, Dict]:
    """
    For each (user_id, run_id, distraction_event) in the test set, fit an
    exponential decay curve to the post-distraction FTD scores.

    Returns
    -------
    fits_df : DataFrame with columns [user_id, run_id, event_idx,
              A, tau, C, r2, n_points, dist_duration]
    summary : dict with aggregate stats
    """
    df = df_te.copy()
    df["ftd"] = ftd_te

    # Identify distraction events: consecutive within_distraction=True blocks
    # We label each event by change-point detection on (user_id, run_id, within_distraction)
    df = df.sort_values(["user_id", "run_id", "sample_ts"]).reset_index(drop=True)

    # Assign event index: each transition from within→post is a new distraction event
    df["_wd"] = df["within_distraction"].fillna(0).astype(int)
    df["_new_session"] = (
        (df["user_id"] != df["user_id"].shift()) |
        (df["run_id"]  != df["run_id"].shift())
    ).astype(int)
    # A new event starts when within_distraction transitions 1→0 (distraction ends)
    df["_event_end"] = (
        (df["_wd"].shift(1, fill_value=0) == 1) & (df["_wd"] == 0) |
        (df["_new_session"] == 1)
    ).astype(int)
    df["event_idx"] = df.groupby(["user_id", "run_id"])["_event_end"].cumsum()

    rows = []
    for (uid, rid, eidx), grp in df.groupby(["user_id", "run_id", "event_idx"]):
        post = grp[grp["_wd"] == 0].copy()
        if len(post) < min_points:
            continue

        t  = post["time_since_distraction_end"].values.astype(float)
        y  = post["ftd"].values.astype(float)

        # Sort by time
        order = np.argsort(t)
        t, y  = t[order], y[order]

        if t.max() - t.min() < 2.0:
            continue  # Too short a recovery window to fit

        # Initial guess: A = range, tau = 5s, C = min
        A0   = max(y.max() - y.min(), 0.01)
        tau0 = 5.0
        C0   = float(y.min())

        try:
            popt, _ = curve_fit(
                _exp_decay, t, y,
                p0=[A0, tau0, C0],
                bounds=([0, 0.1, 0.0], [2.0, 120.0, 1.0]),
                maxfev=2000,
            )
            A_fit, tau_fit, C_fit = popt
            y_pred = _exp_decay(t, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2     = 1.0 - ss_res / max(ss_tot, 1e-10)
        except (RuntimeError, ValueError):
            continue

        # Duration of the distraction itself
        within = grp[grp["_wd"] == 1]
        dist_dur = float(within["time_in_distraction"].max()) if (
            "time_in_distraction" in within.columns and len(within) > 0
        ) else np.nan

        rows.append({
            "user_id":       uid,
            "run_id":        rid,
            "event_idx":     eidx,
            "A":             round(float(A_fit),   4),
            "tau":           round(float(tau_fit), 4),
            "C":             round(float(C_fit),   4),
            "r2":            round(float(r2),      4),
            "n_points":      len(post),
            "dist_duration": round(float(dist_dur), 2) if np.isfinite(dist_dur) else np.nan,
        })

    fits_df = pd.DataFrame(rows)
    if fits_df.empty:
        LOG.warning("Recovery curve fitting: no valid fits found.")
        return fits_df, {}

    good = fits_df[fits_df["r2"] > 0.0]
    summary = {
        "n_events_total":    len(fits_df),
        "n_events_good_fit": len(good),
        "tau_median":        round(float(good["tau"].median()), 2) if len(good) else np.nan,
        "tau_iqr":           round(float(good["tau"].quantile(0.75) - good["tau"].quantile(0.25)), 2) if len(good) else np.nan,
        "A_median":          round(float(good["A"].median()),   2) if len(good) else np.nan,
        "C_median":          round(float(good["C"].median()),   2) if len(good) else np.nan,
        "r2_median":         round(float(good["r2"].median()),  3) if len(good) else np.nan,
        "tau_A_spearman_r":  None,
        "tau_A_spearman_p":  None,
    }

    # Does initial amplitude A correlate with distraction duration?
    if "dist_duration" in good.columns:
        valid = good.dropna(subset=["dist_duration", "A"])
        if len(valid) > 5:
            r, p = spearmanr(valid["dist_duration"], valid["A"])
            summary["A_distdur_spearman_r"] = round(float(r), 3)
            summary["A_distdur_spearman_p"] = round(float(p), 4)

    LOG.info("Recovery curves: %d events, %d good fits (R²>0)  "
             "τ median=%.1fs  A median=%.3f",
             summary["n_events_total"], summary["n_events_good_fit"],
             summary["tau_median"] or 0, summary["A_median"] or 0)

    return fits_df, summary


def plot_recovery_curves(
    fits_df: pd.DataFrame,
    df_te: pd.DataFrame,
    ftd_te: np.ndarray,
    out: Path,
    n_examples: int = 6,
    H: float = 14.0,
) -> None:
    """
    3-panel figure:
      Left  — distribution of fitted τ values
      Middle — scatter A vs distraction duration
      Right  — n_examples example recovery curves with fitted model overlaid
    """
    if fits_df.empty:
        return

    good = fits_df[fits_df["r2"] > 0.0].copy()
    df   = df_te.copy()
    df["ftd"] = ftd_te
    df["_wd"] = df["within_distraction"].fillna(0).astype(int)

    fig = plt.figure(figsize=(17, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel 1: τ distribution
    ax1 = fig.add_subplot(gs[0])
    tau_vals = good["tau"].clip(0, 60)
    ax1.hist(tau_vals, bins=25, color="#3498db", edgecolor="white", alpha=0.8)
    ax1.axvline(float(tau_vals.median()), color="#c0392b", lw=2,
                label=f"Median τ = {tau_vals.median():.1f}s")
    ax1.axvline(H, color="#e67e22", lw=1.5, ls="--", label=f"H = {H:.0f}s")
    ax1.set_xlabel("Fitted recovery time constant τ (s)")
    ax1.set_ylabel("Number of distraction events")
    ax1.set_title("Recovery Time Constant Distribution")
    ax1.legend(fontsize=9)

    # Panel 2: A vs distraction duration
    ax2 = fig.add_subplot(gs[1])
    has_dur = good.dropna(subset=["dist_duration"])
    if len(has_dur) > 3:
        ax2.scatter(has_dur["dist_duration"], has_dur["A"],
                    alpha=0.5, s=18, color="#9b59b6")
        # fit line
        m, b = np.polyfit(has_dur["dist_duration"], has_dur["A"], 1)
        xs   = np.linspace(has_dur["dist_duration"].min(),
                           has_dur["dist_duration"].max(), 100)
        ax2.plot(xs, m * xs + b, color="#c0392b", lw=1.5)
        r, p = spearmanr(has_dur["dist_duration"], has_dur["A"])
        ax2.set_title(f"Amplitude A vs Distraction Duration\n(Spearman r={r:.2f}, p={p:.3f})")
    else:
        ax2.set_title("Amplitude A vs Distraction Duration\n(insufficient data)")
    ax2.set_xlabel("Distraction duration (s)")
    ax2.set_ylabel("Fitted amplitude A")

    # Panel 3: example curves
    ax3  = fig.add_subplot(gs[2])
    cmap = plt.cm.tab10
    sample_fits = good.nlargest(min(n_examples, len(good)), "r2")

    # Reconstruct event grouping
    df = df.sort_values(["user_id", "run_id", "sample_ts"]).reset_index(drop=True)
    df["_new_session"] = (
        (df["user_id"] != df["user_id"].shift()) |
        (df["run_id"]  != df["run_id"].shift())
    ).astype(int)
    df["_event_end"] = (
        (df["_wd"].shift(1, fill_value=0) == 1) & (df["_wd"] == 0) |
        (df["_new_session"] == 1)
    ).astype(int)
    df["event_idx"] = df.groupby(["user_id", "run_id"])["_event_end"].cumsum()

    for k_idx, (_, row) in enumerate(sample_fits.iterrows()):
        subset = df[
            (df["user_id"] == row["user_id"]) &
            (df["run_id"]  == row["run_id"])  &
            (df["event_idx"] == row["event_idx"]) &
            (df["_wd"] == 0)
        ].copy()
        if len(subset) < 2:
            continue
        t = subset["time_since_distraction_end"].values.astype(float)
        y = subset["ftd"].values.astype(float)
        order = np.argsort(t)
        t, y  = t[order], y[order]

        color = cmap(k_idx / max(n_examples - 1, 1))
        ax3.scatter(t, y, s=12, color=color, alpha=0.5)

        t_smooth = np.linspace(t.min(), t.max(), 100)
        y_smooth = _exp_decay(t_smooth, row["A"], row["tau"], row["C"])
        ax3.plot(t_smooth, y_smooth, color=color, lw=1.5,
                 label=f"τ={row['tau']:.1f}s  R²={row['r2']:.2f}")

    ax3.set_xlabel("Time since distraction end (s)")
    ax3.set_ylabel("FTD score")
    ax3.set_title("Example Recovery Curves (best R²)")
    ax3.legend(fontsize=7, loc="upper right")

    fig.suptitle("Recovery Curve Analysis", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Recovery curves → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Distraction onset/offset discrimination
# ─────────────────────────────────────────────────────────────────────────────

def compute_onset_discrimination(
    df_te: pd.DataFrame,
    ftd_te: np.ndarray,
    window_s: float = 5.0,
) -> Dict:
    """
    Compare FTD in the pre-distraction baseline vs during-distraction vs
    immediate post-distraction recovery window.

    Uses Mann-Whitney U (non-parametric, appropriate for non-normal scores).
    Effect size: rank-biserial correlation r = 2U / (n1*n2) - 1.
    With scipy's alternative='greater', U counts times a > b, so U near n1*n2
    means a dominates b (r → +1). The correct formula is 2U/(n1*n2) - 1.

    Returns dict with test statistics and effect sizes.
    """
    df = df_te.copy()
    df["ftd"] = ftd_te
    df["_wd"] = df["within_distraction"].fillna(0).astype(int)

    # Post-distraction: first `window_s` seconds after distraction end
    post_mask = (df["_wd"] == 0) & (df["time_since_distraction_end"] <= window_s)
    # During distraction
    during_mask = (df["_wd"] == 1)
    # Baseline: samples far from distraction (time_since > 2*H, or pre-distraction phase)
    # We approximate baseline as post with time_since > 10s (well into recovery)
    baseline_mask = (df["_wd"] == 0) & (df["time_since_distraction_end"] > 10.0)

    scores_post     = df.loc[post_mask,     "ftd"].values
    scores_during   = df.loc[during_mask,   "ftd"].values
    scores_baseline = df.loc[baseline_mask, "ftd"].values

    def _mwu(a, b, label):
        if len(a) < 2 or len(b) < 2:
            return {"n_a": len(a), "n_b": len(b), "note": "insufficient samples"}
        stat, p = mannwhitneyu(a, b, alternative="greater")
        r_eff   = 2.0 * stat / (len(a) * len(b)) - 1.0   # rank-biserial (scipy 'greater')
        return {
            "n_high": len(a),
            "n_low":  len(b),
            "mean_high": round(float(a.mean()), 4),
            "mean_low":  round(float(b.mean()), 4),
            "U_stat":   round(float(stat), 1),
            "p_value":  float(p),
            "effect_r": round(float(r_eff), 4),
            "label":    label,
        }

    results = {
        "during_vs_baseline": _mwu(scores_during,   scores_baseline,
                                   "During > Baseline"),
        "immediate_post_vs_baseline": _mwu(scores_post, scores_baseline,
                                           f"Post-{window_s}s > Baseline"),
        "during_vs_post": _mwu(scores_during, scores_post,
                               "During > Immediate Post"),
    }

    for k, v in results.items():
        if "p_value" in v:
            LOG.info("Onset [%s]  mean_high=%.3f  mean_low=%.3f  "
                     "r=%.3f  p=%.4g",
                     v["label"], v["mean_high"], v["mean_low"],
                     v["effect_r"], v["p_value"])

    return results


def plot_onset_discrimination(
    df_te: pd.DataFrame,
    ftd_te: np.ndarray,
    out: Path,
    window_s: float = 5.0,
) -> None:
    """
    Two-panel figure:
      Left  — mean FTD ± SE as a function of time since distraction end
               (the model's recovery curve, non-parametric)
      Right  — violin plot: during / immediate-post / late-recovery
    """
    df = df_te.copy()
    df["ftd"] = ftd_te
    df["_wd"] = df["within_distraction"].fillna(0).astype(int)

    # --- Safely compute max time_since (handle all-NaN case) ---
    max_t_val = df["time_since_distraction_end"].max()
    if pd.isna(max_t_val):
        LOG.warning("No valid time_since_distraction_end values; skipping onset discrimination plot.")
        # Create an empty placeholder plot with message
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "No valid time_since data", ha='center', va='center')
        fig.suptitle("Distraction Onset / Recovery Discrimination (no data)", fontsize=13)
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    max_t = int(max_t_val)
    bins = np.arange(0, min(max_t + 1, 21))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: mean FTD per second-bin (post-distraction only)
    means, sems, ns = [], [], []
    for b in bins:
        mask = (df["_wd"] == 0) & (df["time_since_distraction_end"] >= b) & \
               (df["time_since_distraction_end"] < b + 1)
        if mask.sum() > 0:
            v = df.loc[mask, "ftd"].values
            means.append(v.mean())
            sems.append(v.std() / np.sqrt(len(v)))
            ns.append(len(v))
        else:
            means.append(np.nan)
            sems.append(np.nan)
            ns.append(0)

    means, sems = np.array(means), np.array(sems)
    valid = ~np.isnan(means)
    ax1.plot(bins[valid], means[valid], "o-", color="#c0392b",
             linewidth=2.5, markersize=6, label="Mean FTD ± SE")
    ax1.fill_between(bins[valid],
                     means[valid] - sems[valid],
                     means[valid] + sems[valid],
                     alpha=0.2, color="#c0392b")

    during_mean = df.loc[df["_wd"] == 1, "ftd"].mean()
    ax1.axhline(during_mean, color="#2c3e50", ls="--", lw=1.5,
                label=f"During-distraction mean ({during_mean:.3f})")

    ax1.set_xlabel("Seconds after distraction end")
    ax1.set_ylabel("Mean FTD score")
    ax1.set_title("Post-Distraction Recovery Curve\n(model output, non-parametric mean)")
    ax1.legend(fontsize=9)

    # Panel 2: violin plot of three conditions
    during_scores   = df.loc[df["_wd"] == 1, "ftd"].values
    post_scores     = df.loc[(df["_wd"] == 0) &
                             (df["time_since_distraction_end"] <= window_s),
                             "ftd"].values
    late_scores     = df.loc[(df["_wd"] == 0) &
                             (df["time_since_distraction_end"] > 10.0),
                             "ftd"].values

    data   = [during_scores, post_scores, late_scores]
    labels = ["During\n(within distraction)",
              f"Early post\n(0–{window_s}s)",
              "Late recovery\n(>10s)"]
    colors = ["#c0392b", "#e67e22", "#2ecc71"]

    parts = ax2.violinplot(data, positions=[1, 2, 3], showmedians=True,
                           showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color); body.set_alpha(0.7)
    parts["cmedians"].set_colors(["black"] * 3)
    parts["cmedians"].set_linewidth(2)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("FTD score")
    ax2.set_title("FTD Score by Recovery Phase")

    for pos, d in zip([1, 2, 3], data):
        ax2.text(pos, np.median(d) + 0.02, f"{np.median(d):.3f}",
                 ha="center", va="bottom", fontsize=9, color="black")

    fig.suptitle("Distraction Onset / Recovery Discrimination", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Onset discrimination → %s", out)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Calibration curve (aggregate FTD decile → error rate)
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration_curve(
    ftd_te: np.ndarray,
    target_te: np.ndarray,
    n_deciles: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Bin test samples by FTD score percentile (decile by default).
    Within each bin, compute the empirical error rate (mean(target)).

    Returns
    -------
    bin_centers  : mean FTD per bin
    error_rates  : mean(target) per bin
    bin_ns       : sample count per bin
    summary      : monotonicity metrics + Spearman r
    """
    percentiles = np.linspace(0, 100, n_deciles + 1)
    edges       = np.percentile(ftd_te, percentiles)
    bin_idx     = np.clip(np.digitize(ftd_te, edges) - 1, 0, n_deciles - 1)

    centers, rates, ns = [], [], []
    for b in range(n_deciles):
        mask = bin_idx == b
        if mask.sum() > 0:
            centers.append(float(ftd_te[mask].mean()))
            rates.append(float(target_te[mask].mean()))
            ns.append(int(mask.sum()))

    centers  = np.array(centers)
    rates    = np.array(rates)
    ns       = np.array(ns)

    # Monotonicity: fraction of adjacent pairs where rate increases with FTD
    mono_pairs = sum(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
    mono_frac  = mono_pairs / max(len(rates) - 1, 1)

    r, p = spearmanr(centers, rates)

    summary = {
        "n_deciles":       n_deciles,
        "spearman_r":      round(float(r), 4),
        "spearman_p":      float(p),
        "monotone_frac":   round(float(mono_frac), 3),
        "mean_error_rate": round(float(target_te.mean()), 4),
        "ftd_range":       [round(float(ftd_te.min()), 4),
                            round(float(ftd_te.max()), 4)],
    }

    LOG.info("Calibration: Spearman r=%.3f (p=%.4g)  monotone_frac=%.2f",
             r, p, mono_frac)

    return centers, rates, ns, summary


def plot_calibration_curve(
    centers: np.ndarray,
    rates: np.ndarray,
    ns: np.ndarray,
    baseline_rate: float,
    out: Path,
) -> None:
    """
    Bar chart of empirical error rate per FTD decile, with baseline overlay.
    Bar width scales with sample count (equal-width by default).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    x    = np.arange(len(centers))
    bars = ax.bar(x, rates, color="#3498db", alpha=0.8, edgecolor="white",
                  linewidth=0.5)

    # Colour bars by relative risk
    for bar, rate in zip(bars, rates):
        rel = rate / max(baseline_rate, 1e-9)
        if rel >= 3:
            bar.set_facecolor("#c0392b")
        elif rel >= 1.5:
            bar.set_facecolor("#e67e22")
        else:
            bar.set_facecolor("#2ecc71")

    ax.axhline(baseline_rate, color="#2c3e50", ls="--", lw=1.5,
               label=f"Baseline rate ({baseline_rate:.4f})")

    # Annotate n per bin
    for i, (bar, n) in enumerate(zip(bars, ns)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + baseline_rate * 0.05,
                f"n={n}", ha="center", va="bottom", fontsize=7)

    r_val = float(spearmanr(centers, rates)[0])
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i+1}\n({c:.2f})" for i, c in enumerate(centers)],
                       fontsize=8)
    ax.set_xlabel("FTD Decile (mean FTD score)")
    ax.set_ylabel("Empirical error rate (mean target)")
    ax.set_title(f"FTD Calibration Curve\n"
                 f"Error rate per FTD decile — Spearman r={r_val:.3f}\n"
                 f"Colours: green < 1.5×baseline, orange < 3×, red ≥ 3×")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Calibration curve → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Concordance index (C-index)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cindex(
    ftd_te: np.ndarray,
    df_te: pd.DataFrame,
    n_sample_pairs: int = 100_000,
    seed: int = 42,
) -> Dict:
    """
    Harrell's C-index adapted for impairment ranking.

    For a pair (i, j) to be "informative", sample i must be more impaired
    than sample j by the known causal structure. We define "more impaired" as:

        (a) i is within_distraction and j is not, OR
        (b) both post-distraction, and i has strictly smaller time_since than j
            (earlier in recovery → more impaired)

    The C-index = P(FTD[i] > FTD[j] | i more impaired than j)

    A random model scores 0.5. A perfect model scores 1.0.

    We also compute Kendall's τ between time_since_distraction_end and FTD
    (for post-distraction samples only) as a complementary ranking metric.
    """
    rng = np.random.default_rng(seed)
    ts  = df_te["time_since_distraction_end"].fillna(999.0).values.astype(float)
    wd  = df_te["within_distraction"].fillna(0).astype(bool).values

    post_idx   = np.where(~wd)[0]
    within_idx = np.where(wd)[0]

    concordant = discordant = tied = 0

    # --- (a) within_distraction vs post-distraction pairs ---
    if len(within_idx) >= 2 and len(post_idx) >= 2:
        n_a = min(n_sample_pairs // 2, len(within_idx) * len(post_idx))
        wi  = rng.choice(within_idx, size=n_a, replace=True)
        pi  = rng.choice(post_idx,   size=n_a, replace=True)
        diff = ftd_te[wi] - ftd_te[pi]
        concordant += int((diff > 0).sum())
        discordant += int((diff < 0).sum())
        tied       += int((diff == 0).sum())

    # --- (b) post-distraction pairs ordered by time_since ---
    if len(post_idx) >= 4:
        n_b      = min(n_sample_pairs // 2, len(post_idx) * (len(post_idx) - 1) // 2)
        idx_a    = rng.choice(post_idx, size=n_b, replace=True)
        idx_b    = rng.choice(post_idx, size=n_b, replace=True)
        valid    = ts[idx_a] < ts[idx_b] - 0.5   # a is earlier (more impaired)
        if valid.sum() > 0:
            diff = ftd_te[idx_a[valid]] - ftd_te[idx_b[valid]]
            concordant += int((diff > 0).sum())
            discordant += int((diff < 0).sum())
            tied       += int((diff == 0).sum())

    total   = concordant + discordant + tied
    cindex  = (concordant + 0.5 * tied) / max(total, 1)

    # Kendall's τ: time_since vs FTD (should be negative: more time → less FTD)
    post_ftd = ftd_te[post_idx]
    post_ts  = ts[post_idx]
    # Sample to keep it tractable
    if len(post_idx) > 5000:
        sample   = rng.choice(len(post_idx), size=5000, replace=False)
        post_ftd = post_ftd[sample]
        post_ts  = post_ts[sample]
    tau_stat, tau_p = kendalltau(post_ts, post_ftd)

    result = {
        "c_index":            round(float(cindex),    4),
        "n_concordant":       concordant,
        "n_discordant":       discordant,
        "n_tied":             tied,
        "n_pairs_total":      total,
        "kendall_tau":        round(float(tau_stat),  4),
        "kendall_tau_p":      float(tau_p),
    }

    LOG.info("C-index = %.4f  (concordant=%d, discordant=%d, tied=%d)",
             cindex, concordant, discordant, tied)
    LOG.info("Kendall τ (time_since vs FTD) = %.4f  p=%.4g  "
             "(negative = FTD decreases as recovery progresses)",
             tau_stat, tau_p)

    return result


def plot_cindex_summary(
    cindex_result: Dict,
    calibration_result: Dict,
    onset_result: Dict,
    recovery_summary: Dict,
    out: Path,
) -> None:
    """
    Single-page summary figure with 4 metric cards.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Construct Validity Summary — Driver Impairment FTD Score",
                 fontsize=13, y=1.02)

    def _card(ax, title, lines, color="#2c3e50"):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.95, title, ha="center", va="top",
                fontsize=11, fontweight="bold", color=color,
                transform=ax.transAxes)
        for i, line in enumerate(lines):
            ax.text(0.5, 0.82 - i * 0.14, line,
                    ha="center", va="top", fontsize=10,
                    transform=ax.transAxes)
        ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96,
                                   fill=False, edgecolor=color, lw=2,
                                   transform=ax.transAxes))

    # Card 1: C-index
    ci = cindex_result
    _card(axes[0], "Concordance Index",
          [f"C-index = {ci['c_index']:.3f}",
           "(0.5 = random, 1.0 = perfect)",
           f"Kendall τ = {ci['kendall_tau']:.3f}",
           f"(p = {ci['kendall_tau_p']:.2e})",
           f"Pairs evaluated: {ci['n_pairs_total']:,}"],
          "#2980b9")

    # Card 2: Calibration
    ca = calibration_result
    _card(axes[1], "Calibration (Decile)",
          [f"Spearman r = {ca['spearman_r']:.3f}",
           f"(p = {ca['spearman_p']:.2e})",
           f"Monotone fraction = {ca['monotone_frac']:.1%}",
           f"Baseline rate = {ca['mean_error_rate']:.4f}",
           f"{ca['n_deciles']} decile bins"],
          "#27ae60")

    # Card 3: Onset discrimination
    od = onset_result.get("during_vs_baseline", {})
    _card(axes[2], "Onset Discrimination",
          [f"During vs Baseline",
           f"Mean during = {od.get('mean_high', float('nan')):.3f}",
           f"Mean baseline = {od.get('mean_low', float('nan')):.3f}",
           f"Effect r = {od.get('effect_r', float('nan')):.3f}",
           f"p = {od.get('p_value', float('nan')):.2e}"],
          "#8e44ad")

    # Card 4: Recovery curve
    rc = recovery_summary
    _card(axes[3], "Recovery Curves",
          [f"Events fitted: {rc.get('n_events_good_fit', '?')}",
           f"Median τ = {rc.get('tau_median', float('nan')):.1f}s",
           f"IQR τ = [{rc.get('tau_iqr', float('nan')):.1f}s]",
           f"Median R² = {rc.get('r2_median', float('nan')):.3f}",
           f"Median A = {rc.get('A_median', float('nan')):.3f}"],
          "#e74c3c")

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Summary card → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point (called from run_pipeline or standalone)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_construct_validity(
    df_te: pd.DataFrame,
    ftd_te: np.ndarray,
    target_te: np.ndarray,
    out_dir: Path,
    H: float = 14.0,
    n_deciles: int = 10,
    onset_window_s: float = 5.0,
    seed: int = 42,
    cal_target_te: Optional[np.ndarray] = None,
) -> Dict:
    """
    Run all four construct-validity evaluations and save plots.

    Parameters
    ----------
    df_te          : test DataFrame with user_id, run_id, sample_ts,
                     time_since_distraction_end, within_distraction,
                     time_in_distraction
    ftd_te         : FTD score array aligned with df_te rows
    target_te      : binary error target (any T) — used only for logging
    cal_target_te  : binary T=1 target for calibration curve.
                     If None, falls back to target_te (correct only when T=1).
                     Always pass T=1 targets here regardless of training T.
    out_dir        : directory for output plots
    H              : hangover horizon in seconds (for reference lines)
    n_deciles      : number of bins for calibration curve
    onset_window_s : seconds after distraction end to define "early post"
    seed           : random seed for C-index pair sampling
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Calibration always uses T=1 target to avoid autocorrelation from T>1
    if cal_target_te is None:
        cal_target_te = target_te
        if target_te.mean() > 0.10:
            LOG.warning(
                "target positive rate is %.1f%% — this suggests T>1 targets "
                "are being used for calibration. Pass cal_target_te built "
                "with T=1 for a clean calibration curve.", 100 * target_te.mean())

    LOG.info("─── Construct Validity Evaluation ──────────────────────────")

    # 1. Recovery curve fitting
    fits_df, recovery_summary = fit_recovery_curves(df_te, ftd_te)
    plot_recovery_curves(fits_df, df_te, ftd_te,
                         out_dir / "recovery_curves.png", H=H)

    # 2. Onset / offset discrimination
    onset_result = compute_onset_discrimination(df_te, ftd_te,
                                                window_s=onset_window_s)
    plot_onset_discrimination(df_te, ftd_te,
                              out_dir / "onset_discrimination.png",
                              window_s=onset_window_s)

    # 3. Calibration curve — always uses T=1 target
    centers, rates, ns, cal_summary = compute_calibration_curve(
        ftd_te, cal_target_te, n_deciles=n_deciles)
    baseline_rate = float(cal_target_te.mean())
    plot_calibration_curve(centers, rates, ns, baseline_rate,
                           out_dir / "calibration_curve.png")

    # 4. C-index
    cindex_result = compute_cindex(ftd_te, df_te, seed=seed)

    # Summary card
    plot_cindex_summary(
        cindex_result, cal_summary, onset_result, recovery_summary,
        out_dir / "construct_validity_summary.png",
    )

    results = {
        "recovery_curves":       recovery_summary,
        "onset_discrimination":  onset_result,
        "calibration":           cal_summary,
        "c_index":               cindex_result,
    }

    with open(out_dir / "construct_validity.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    LOG.info("─── Construct Validity: DONE ────────────────────────────────")
    LOG.info("  C-index:           %.4f  (0.5=random)", cindex_result["c_index"])
    LOG.info("  Kendall τ:         %.4f  (p=%.2e)",
             cindex_result["kendall_tau"], cindex_result["kendall_tau_p"])
    LOG.info("  Calibration r:     %.4f  (p=%.2e)",
             cal_summary["spearman_r"], cal_summary["spearman_p"])
    LOG.info("  Onset effect r:    %.4f  (p=%.2e)",
             onset_result["during_vs_baseline"].get("effect_r", float("nan")),
             onset_result["during_vs_baseline"].get("p_value",  float("nan")))
    LOG.info("  Recovery τ median: %.1f s", recovery_summary.get("tau_median", 0) or 0)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner (loads saved artifact + data, re-generates test set)
# ─────────────────────────────────────────────────────────────────────────────

def build_t1_target(df_te: pd.DataFrame, errs_lookup: dict) -> np.ndarray:
    """
    Build a T=1s binary error target directly from raw error timestamps,
    completely independent of the T used for training.

    Uses the same half-open interval as generate_samples:
        target = 1  iff  any error in [sample_ts, sample_ts + 1s)
    i.e. strictly less-than the right boundary, matching:
        err_ts[j] < ts + horizon

    Parameters
    ----------
    df_te        : test DataFrame (must have sample_ts, user_id, run_id)
    errs_lookup  : dict[(user_id, run_id)] -> List[pd.Timestamp]
                   (the same `errs` dict built by build_lookups)
    """
    from bisect import bisect_left

    horizon = pd.Timedelta(seconds=1)
    target  = np.zeros(len(df_te), dtype=int)

    rows = df_te.reset_index(drop=True)
    for i in range(len(rows)):
        row      = rows.iloc[i]
        key      = (row["user_id"], int(row["run_id"]))
        err_list = errs_lookup.get(key, [])
        if not err_list:
            continue
        ts  = row["sample_ts"]
        end = ts + horizon
        # Find first error >= ts, then check it's strictly < end
        j = bisect_left(err_list, ts)
        if j < len(err_list) and err_list[j] < end:
            target[i] = 1

    return target


def run_standalone(args):
    """Load artifact and regenerate test set to run evaluation standalone."""
    import ftd_hmm_no_time as ftd_hmm    
    import importlib
    importlib.reload(ftd_hmm)   # ensure we have the latest version

    LOG.info("Loading artifact from %s", args.artifact)
    artifact   = joblib.load(args.artifact)
    scaler     = artifact["scaler"]
    feat_cols  = artifact["feature_cols"]
    print(feat_cols)
    # risk_order is stored inside "state_profile"
    risk_order = np.array(artifact["state_profile"]["risk_order"])
    H          = artifact["config"]["H"]
    T_train    = artifact["config"]["T"]

    # Retrieve learned risk weights (mandatory in new script)
    risk_weights = artifact.get("risk_weights")
    risk_intercept = artifact.get("risk_intercept")
    if risk_weights is None or risk_intercept is None:
        raise ValueError("Artifact does not contain learned risk weights. "
                         "Ensure the model was trained with --label-mode pic (or other) that learns weights.")

    # --- Determine the rolling spans used during training from feat_cols ---
    import re
    span_set = set()
    for col in feat_cols:
        # Look for patterns like _ema5, _roll_std10, etc.
        match = re.search(r'_ema(\d+)', col)
        if match:
            span_set.add(int(match.group(1)))
        match = re.search(r'_roll_std(\d+)', col)
        if match:
            span_set.add(int(match.group(1)))
    if span_set:
        required_spans = sorted(span_set)
        LOG.info(f"Detected required rolling spans from artifact: {required_spans}")
        # Override the module's ROLL_SPANS to match training
        ftd_hmm.ROLL_SPANS = required_spans
    else:
        LOG.warning("Could not infer rolling spans from feature columns; using module defaults.")

    # --- Reconstruct HMM from artifact ---
    hmm_dict = artifact["hmm"]
    n_states = hmm_dict["n_states"]
    stored_cov_type = hmm_dict.get("covariance_type", "diag")
    covars_raw = hmm_dict["covars"]

    # Infer actual covariance type from the shape of the stored covars
    cov_array = np.array(covars_raw)
    if cov_array.ndim == 1:
        inferred_cov = "spherical"
    elif cov_array.ndim == 2:
        # Could be diag (n_states, n_features) or tied (n_features, n_features)
        if cov_array.shape[0] == n_states:
            inferred_cov = "diag"
        else:
            inferred_cov = "tied"
    elif cov_array.ndim == 3:
        inferred_cov = "full"
    else:
        inferred_cov = stored_cov_type  # fallback

    if inferred_cov != stored_cov_type:
        LOG.warning(f"Overriding stored covariance type '{stored_cov_type}' with inferred type '{inferred_cov}' based on covars shape.")

    # Create GaussianHMM with the inferred covariance type and set parameters
    hmm_model = GaussianHMM(n_components=n_states, covariance_type=inferred_cov)
    hmm_model.startprob_ = np.array(hmm_dict["startprob"])
    hmm_model.transmat_  = np.array(hmm_dict["transmat"])
    hmm_model.means_     = np.array(hmm_dict["means"])
    hmm_model.covars_    = cov_array   # now in the expected shape

    # --- Prepare test data ---
    d_raw, e_raw, eb_raw, db_raw = ftd_hmm.load_data(args.data_path)
    ftd_hmm.run_integrity_checks(d_raw, e_raw)

    users = sorted(set(d_raw["user_id"].unique()) | set(e_raw["user_id"].unique()))
    _, _, test_users = ftd_hmm.split_users(users, seed=args.seed)
    train_users = [u for u in users if u not in test_users]

    impute_stats = ftd_hmm.fit_train_imputation_stats(train_users, d_raw)
    d = ftd_hmm.apply_imputation_stats(d_raw, impute_stats)

    le_pred, le_emo = ftd_hmm.fit_label_encoders(train_users, d, e_raw)
    pred_enc = ftd_hmm.build_encoding_map(le_pred)
    emo_enc  = ftd_hmm.build_encoding_map(le_emo)
    baselines = ftd_hmm.build_baselines(train_users, eb_raw, db_raw)

    d_train = d[d["user_id"].isin(train_users)]
    gmp = float(d_train[["model_prob_start", "model_prob_end"]].stack().median())
    gep = float(d_train[["emotion_prob_start", "emotion_prob_end"]].stack().median())
    if not np.isfinite(gmp): gmp = 0.5
    if not np.isfinite(gep): gep = 0.5

    wbs, webs, errs = ftd_hmm.build_lookups(d, e_raw, users)

    # Generate test samples using training T (preserves all feature values)
    df_te = ftd_hmm.generate_samples(H, T_train, test_users, wbs, webs, errs,
                                      baselines, pred_enc, emo_enc, gmp, gep)
    df_te = ftd_hmm.add_causal_session_features(df_te)
    df_te = ftd_hmm.apply_feature_postprocess(df_te, artifact["postprocess"])
    df_te = ftd_hmm.sort_for_hmm(df_te)

    X_te = scaler.transform(df_te[feat_cols].astype(float))
    b_te = ftd_hmm.sequence_bounds(df_te)

    # --- Compute gamma using the reconstructed HMM ---
    gamma_te = np.zeros((len(X_te), n_states))
    for s, e in b_te:
        if e > s:
            gamma_te[s:e] = hmm_model.predict_proba(X_te[s:e])

    # --- Compute FTD score using learned weights ---
    ftd_te = ftd_hmm.compute_risk_score(gamma_te, risk_order, risk_weights, risk_intercept)

    # Build T=1 target for calibration (independent of training T)
    LOG.info("Building T=1 calibration target (training T was %d)", T_train)
    target_t1 = build_t1_target(df_te, errs)   # this function is defined in evaluation.py
    LOG.info("T=1 target: %.2f%% positive  (T=%d target was %.2f%% positive)",
             100 * target_t1.mean(), T_train,
             100 * df_te["target"].values.mean())

    evaluate_construct_validity(
        df_te, ftd_te, target_t1,
        out_dir=Path(args.output_dir),
        H=float(H),
    )

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Construct validity evaluation for FTD score")
    p.add_argument("--artifact",    default="result/impairment_hmm.joblib",
                   help="Path to impairment_hmm.joblib")
    p.add_argument("--data-path",   default="data")
    p.add_argument("--output-dir",  default="result/construct_validity/")
    p.add_argument("--seed",        type=int, default=42)
    raise SystemExit(run_standalone(p.parse_args()))