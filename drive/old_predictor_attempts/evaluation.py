#!/usr/bin/env python3
"""
evaluate_impairment.py
======================
Construct-validity evaluation for the driver impairment FTD score.

Goal: FTD is a CONTINUOUS RISK SCORE. Its complement, fitness-to-drive = 1 - FTD,
should be high when the driver is undistracted and low immediately after a
distraction, recovering exponentially. There is no continuous ground-truth
impairment label, so we validate against the known CAUSAL STRUCTURE of
post-distraction recovery:

  1. Recovery curve fitting     — does FTD decay exponentially per distraction?
  2. Onset discrimination       — does FTD spike at distraction onset?
  3. Calibration curve          — does FTD decile monotonically predict error
                                   rate? (secondary validity check only)
  4. Concordance index (C-index)— do higher-risk pairs rank above lower-risk
                                   ones within the same session?
  5. Score range & reliability  — does the score use its full range between
                                   impaired and safe states, and is it stable
                                   when the driver is genuinely undistracted?

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
    min_points: int = 10,
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

    df = df.sort_values(["user_id", "run_id", "sample_ts"]).reset_index(drop=True)
    df["_wd"] = df["within_distraction"].fillna(0).astype(int)
    df["_new_session"] = (
        (df["user_id"] != df["user_id"].shift()) |
        (df["run_id"]  != df["run_id"].shift())
    ).astype(int)
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
        order = np.argsort(t)
        t, y  = t[order], y[order]

        if t.max() - t.min() < 2.0:
            continue

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
            r2_raw = 1.0 - ss_res / max(ss_tot, 1e-10)
            # Adjusted R²: penalise the 3-parameter fit for short sequences.
            # With n=6 and p=3, raw R² is trivially inflated.
            n_pts, p_params = len(t), 3
            if n_pts > p_params + 1:
                r2 = 1.0 - (1.0 - r2_raw) * (n_pts - 1) / (n_pts - p_params - 1)
            else:
                r2 = float("nan")   # not enough dof to report
        except (RuntimeError, ValueError):
            continue

        within   = grp[grp["_wd"] == 1]
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

    # Require adjusted R² >= 0.5 so only genuinely good fits are summarised.
    good = fits_df[fits_df["r2"] >= 0.5]
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
    if fits_df.empty:
        return

    good = fits_df[fits_df["r2"] > 0.0].copy()
    df   = df_te.copy()
    df["ftd"] = ftd_te
    df["_wd"] = df["within_distraction"].fillna(0).astype(int)

    fig = plt.figure(figsize=(17, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

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

    ax2 = fig.add_subplot(gs[1])
    has_dur = good.dropna(subset=["dist_duration"])
    if len(has_dur) > 3:
        ax2.scatter(has_dur["dist_duration"], has_dur["A"],
                    alpha=0.5, s=18, color="#9b59b6")
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

    ax3  = fig.add_subplot(gs[2])
    cmap = plt.cm.tab10
    sample_fits = good.nlargest(min(n_examples, len(good)), "r2")

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
            (df["user_id"]   == row["user_id"]) &
            (df["run_id"]    == row["run_id"])  &
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

    Uses Mann-Whitney U (non-parametric). Effect size: rank-biserial r.
    """
    df = df_te.copy()
    df["ftd"] = ftd_te
    df["_wd"] = df["within_distraction"].fillna(0).astype(int)

    post_mask     = (df["_wd"] == 0) & (df["time_since_distraction_end"] <= window_s)
    during_mask   = (df["_wd"] == 1)
    baseline_mask = (df["_wd"] == 0) & (df["time_since_distraction_end"] > 10.0)

    scores_post     = df.loc[post_mask,     "ftd"].values
    scores_during   = df.loc[during_mask,   "ftd"].values
    scores_baseline = df.loc[baseline_mask, "ftd"].values

    def _mwu(a, b, label):
        if len(a) < 2 or len(b) < 2:
            return {"n_a": len(a), "n_b": len(b), "note": "insufficient samples"}
        stat, p = mannwhitneyu(a, b, alternative="greater")
        r_eff   = 2.0 * stat / (len(a) * len(b)) - 1.0
        return {
            "n_high":    len(a),
            "n_low":     len(b),
            "mean_high": round(float(a.mean()), 4),
            "mean_low":  round(float(b.mean()), 4),
            "U_stat":    round(float(stat), 1),
            "p_value":   float(p),
            "effect_r":  round(float(r_eff), 4),
            "label":     label,
        }

    results = {
        "during_vs_baseline": _mwu(scores_during, scores_baseline,
                                   "During > Baseline"),
        "immediate_post_vs_baseline": _mwu(scores_post, scores_baseline,
                                           f"Post-{window_s}s > Baseline"),
        "during_vs_post": _mwu(scores_during, scores_post,
                               "During > Immediate Post"),
    }

    for k, v in results.items():
        if "p_value" in v:
            LOG.info("Onset [%s]  mean_high=%.3f  mean_low=%.3f  r=%.3f  p=%.4g",
                     v["label"], v["mean_high"], v["mean_low"],
                     v["effect_r"], v["p_value"])

    return results


def plot_onset_discrimination(
    df_te: pd.DataFrame,
    ftd_te: np.ndarray,
    out: Path,
    window_s: float = 5.0,
) -> None:
    df = df_te.copy()
    df["ftd"] = ftd_te
    df["_wd"] = df["within_distraction"].fillna(0).astype(int)

    max_t_val = df["time_since_distraction_end"].max()
    if pd.isna(max_t_val):
        LOG.warning("No valid time_since_distraction_end; skipping onset plot.")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "No valid time_since data", ha="center", va="center")
        fig.suptitle("Distraction Onset / Recovery Discrimination (no data)", fontsize=13)
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    max_t = int(max_t_val)
    bins  = np.arange(0, min(max_t + 1, 21))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    means, sems, ns = [], [], []
    for b in bins:
        mask = (df["_wd"] == 0) & \
               (df["time_since_distraction_end"] >= b) & \
               (df["time_since_distraction_end"] < b + 1)
        if mask.sum() > 0:
            v = df.loc[mask, "ftd"].values
            means.append(v.mean())
            sems.append(v.std() / np.sqrt(len(v)))
            ns.append(len(v))
        else:
            means.append(np.nan); sems.append(np.nan); ns.append(0)

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
    ax1.set_title("Post-Distraction Recovery\n(FTD = risk, lower = safer)")
    ax1.legend(fontsize=9)

    during_scores = df.loc[df["_wd"] == 1, "ftd"].values
    post_scores   = df.loc[(df["_wd"] == 0) &
                           (df["time_since_distraction_end"] <= window_s), "ftd"].values
    late_scores   = df.loc[(df["_wd"] == 0) &
                           (df["time_since_distraction_end"] > 10.0), "ftd"].values

    data   = [during_scores, post_scores, late_scores]
    labels = ["During\n(within distraction)", f"Early post\n(0–{window_s}s)",
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
    ax2.set_ylabel("FTD score  (fitness-to-drive = 1 − FTD)")
    ax2.set_title("FTD Score Distribution by Phase\n"
                  "(good score: during >> safe, small overlap)")

    for pos, d in zip([1, 2, 3], data):
        ax2.text(pos, np.median(d) + 0.02, f"{np.median(d):.3f}",
                 ha="center", va="bottom", fontsize=9, color="black")

    fig.suptitle("Distraction Onset / Recovery Discrimination", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Onset discrimination → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Calibration curve (secondary validity check only)
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration_curve(
    ftd_te: np.ndarray,
    target_te: np.ndarray,
    df_te: pd.DataFrame,
    n_deciles: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Secondary validity check: does FTD decile monotonically predict error rate?

    This is not the primary evaluation metric (we are not predicting errors),
    but monotone ordering of error rates against the risk score is a useful
    downstream sanity check — if higher risk score deciles show lower error
    rates the score is internally inconsistent.

    Aggregated to distraction-event level (mean FTD, max target) to remove
    autocorrelation from per-second samples.
    """
    ev = df_te[["user_id", "run_id", "distraction_idx"]].copy()
    ev["ftd"]    = ftd_te
    ev["target"] = target_te

    event_df = (
        ev.groupby(["user_id", "run_id", "distraction_idx"], sort=False)
        .agg(ftd_mean=("ftd", "mean"), target_max=("target", "max"))
        .reset_index()
    )

    ftd_ev    = event_df["ftd_mean"].values
    target_ev = event_df["target_max"].values.astype(float)

    percentiles = np.linspace(0, 100, n_deciles + 1)
    edges       = np.percentile(ftd_ev, percentiles)
    bin_idx     = np.clip(np.digitize(ftd_ev, edges) - 1, 0, n_deciles - 1)

    centers, rates, ns = [], [], []
    for b in range(n_deciles):
        mask = bin_idx == b
        if mask.sum() > 0:
            centers.append(float(ftd_ev[mask].mean()))
            rates.append(float(target_ev[mask].mean()))
            ns.append(int(mask.sum()))

    centers = np.array(centers)
    rates   = np.array(rates)
    ns      = np.array(ns)

    mono_pairs = sum(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
    mono_frac  = mono_pairs / max(len(rates) - 1, 1)
    r, p       = spearmanr(centers, rates)

    summary = {
        "n_deciles":       n_deciles,
        "n_events":        len(event_df),
        "spearman_r":      round(float(r), 4),
        "spearman_p":      float(p),
        "monotone_frac":   round(float(mono_frac), 3),
        "mean_error_rate": round(float(target_ev.mean()), 4),
        "ftd_range":       [round(float(ftd_ev.min()), 4),
                            round(float(ftd_ev.max()), 4)],
    }

    LOG.info("Calibration (event-level, secondary check): "
             "Spearman r=%.3f (p=%.4g)  monotone_frac=%.2f  n_events=%d",
             r, p, mono_frac, len(event_df))

    return centers, rates, ns, summary


def plot_calibration_curve(
    centers: np.ndarray,
    rates: np.ndarray,
    ns: np.ndarray,
    baseline_rate: float,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    x    = np.arange(len(centers))
    bars = ax.bar(x, rates, color="#3498db", alpha=0.8, edgecolor="white",
                  linewidth=0.5)

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

    for i, (bar, n) in enumerate(zip(bars, ns)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + baseline_rate * 0.05,
                f"n={n}", ha="center", va="bottom", fontsize=7)

    r_val = float(spearmanr(centers, rates)[0])
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i+1}\n({c:.2f})" for i, c in enumerate(centers)],
                       fontsize=8)
    ax.set_xlabel("FTD Decile (mean FTD score, event-level)")
    ax.set_ylabel("Empirical error rate (secondary check only)")
    ax.set_title(f"FTD Calibration — Secondary Validity Check (event-level)\n"
                 f"Monotone ordering of error rates vs score  |  Spearman r={r_val:.3f}")
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
    Concordance and time-correlation metrics for a continuous risk score.

    THREE pair types are evaluated separately so that time-confounded and
    physiology-sensitive pairs are not mixed:

    (a) within_distraction vs post-distraction — tests whether the score is
        higher during than after; ANY monotone-in-time score wins this trivially,
        so it is informative mainly as a sanity check.

    (b) post vs post, same session, ordered by time_since — tests temporal
        decay; again a pure time-model scores 1.0 here by construction.

    (c) [KEY] same-time-bin pairs across different distraction events
        (|Δt| < 1 s, different distraction_idx) — the score can only
        differentiate these through physiology, not time.  This is the
        only pair type that separates the HMM from a time-only model.

    C-index = P(score[more impaired] > score[less impaired]), 0.5=random.

    Kendall τ: session-averaged correlation of time_since vs FTD score.
    Negative = score decays with time as expected.

    partial_tau: Kendall τ of (score - time_only_pred) vs physiology proxy
    (arousal_delta_baseline) within same-second bins — measures how much
    the score varies due to physiology after time is removed.
    """
    rng = np.random.default_rng(seed)

    df = df_te.reset_index(drop=True).copy()
    df["_ftd"] = ftd_te
    df["_wd"]  = df["within_distraction"].fillna(0).astype(bool)
    df["_ts"]  = df["time_since_distraction_end"].fillna(999.0).astype(float)
    df["_sec"] = df["_ts"].astype(int)   # integer-second bin for same-time pairing

    concordant_ab = discordant_ab = tied_ab = 0   # (a)+(b): time-ordered
    concordant_c  = discordant_c  = tied_c  = 0   # (c): same-time, physio-only
    tau_vals: List[float] = []

    sessions          = df.groupby(["user_id", "run_id"], sort=False)
    pairs_per_session = max(1, n_sample_pairs // max(sessions.ngroups, 1))

    # Pre-build post-distraction index grouped by integer-second bin
    # across ALL sessions for cross-event same-time pairing (type c).
    post_df = df[~df["_wd"]].copy()

    for (uid, rid), grp in sessions:
        grp          = grp.reset_index(drop=True)
        within_local = grp.index[grp["_wd"]].tolist()
        post_local   = grp.index[~grp["_wd"]].tolist()

        # ── (a) within vs post ────────────────────────────────────────────────
        if len(within_local) >= 1 and len(post_local) >= 1:
            n_a = min(pairs_per_session // 3,
                      len(within_local), len(post_local))
            if n_a > 0:
                wi   = rng.choice(within_local, size=n_a, replace=False)
                pi   = rng.choice(post_local,   size=n_a, replace=False)
                diff = grp.loc[wi, "_ftd"].values - grp.loc[pi, "_ftd"].values
                concordant_ab += int((diff > 0).sum())
                discordant_ab += int((diff < 0).sum())
                tied_ab       += int((diff == 0).sum())

        # ── (b) post vs post, time-ordered ───────────────────────────────────
        if len(post_local) >= 4:
            n_b = min(pairs_per_session // 3, len(post_local))
            if n_b > 0:
                idx_a = rng.choice(post_local, size=n_b, replace=False)
                idx_b = rng.choice(post_local, size=n_b, replace=False)
                ts_a  = grp.loc[idx_a, "_ts"].values
                ts_b  = grp.loc[idx_b, "_ts"].values
                valid = ts_a < ts_b - 0.5
                if valid.sum() > 0:
                    diff = grp.loc[idx_a[valid], "_ftd"].values - \
                           grp.loc[idx_b[valid], "_ftd"].values
                    concordant_ab += int((diff > 0).sum())
                    discordant_ab += int((diff < 0).sum())
                    tied_ab       += int((diff == 0).sum())

        # ── (c) same-time-bin, different distraction event ───────────────────
        # These pairs have identical time-since-distraction-end (±1 s) so a
        # pure time-model predicts equal scores; any concordance must come from
        # physiology captured by the HMM.
        post_grp = grp[~grp["_wd"]].copy()
        if len(post_grp) >= 2 and "distraction_idx" in post_grp.columns:
            n_c = min(pairs_per_session // 3, len(post_grp))
            if n_c > 0:
                idx_a = rng.choice(post_grp.index.tolist(), size=n_c, replace=False)
                idx_b = rng.choice(post_grp.index.tolist(), size=n_c, replace=False)
                sec_a = grp.loc[idx_a, "_sec"].values
                sec_b = grp.loc[idx_b, "_sec"].values
                did_a = grp.loc[idx_a, "distraction_idx"].values
                did_b = grp.loc[idx_b, "distraction_idx"].values
                # Keep only pairs at same second but different distraction event
                same_t   = np.abs(sec_a - sec_b) == 0
                diff_evt = did_a != did_b
                valid_c  = same_t & diff_evt
                if valid_c.sum() > 0:
                    # "More impaired" = earlier in session (smaller distraction_idx)
                    d = grp.loc[idx_a[valid_c], "_ftd"].values - \
                        grp.loc[idx_b[valid_c], "_ftd"].values
                    earlier = did_a[valid_c] < did_b[valid_c]
                    concordant_c += int(((d > 0) & earlier).sum() +
                                        ((d < 0) & ~earlier).sum())
                    discordant_c += int(((d < 0) & earlier).sum() +
                                        ((d > 0) & ~earlier).sum())
                    tied_c       += int((d == 0).sum())

        # ── Kendall τ (post-distraction, session-level) ───────────────────────
        if len(post_local) >= 5:
            post_ftd = grp.loc[post_local, "_ftd"].values
            post_ts  = grp.loc[post_local, "_ts"].values
            if len(post_local) > 500:
                samp     = rng.choice(len(post_local), size=500, replace=False)
                post_ftd = post_ftd[samp]
                post_ts  = post_ts[samp]
            tau_stat, _ = kendalltau(post_ts, post_ftd)
            if np.isfinite(tau_stat):
                tau_vals.append(float(tau_stat))

    total_ab  = concordant_ab + discordant_ab + tied_ab
    total_c   = concordant_c  + discordant_c  + tied_c
    total_all = total_ab + total_c

    cindex_ab  = (concordant_ab + 0.5 * tied_ab) / max(total_ab,  1)
    cindex_c   = (concordant_c  + 0.5 * tied_c)  / max(total_c,   1)
    cindex_all = ((concordant_ab + concordant_c) +
                  0.5 * (tied_ab + tied_c)) / max(total_all, 1)

    tau_mean = float(np.mean(tau_vals)) if tau_vals else float("nan")

    result = {
        # Overall (time-ordered + physio-only pairs combined)
        "c_index":            round(float(cindex_all), 4),
        # Time-ordered pairs only (a+b) — will be ~1.0 for any time model
        "c_index_time_pairs": round(float(cindex_ab),  4),
        # Same-time, different-event pairs (c) — physiology-sensitive only
        "c_index_physio":     round(float(cindex_c),   4),
        "n_pairs_time":       total_ab,
        "n_pairs_physio":     total_c,
        "n_pairs_total":      total_all,
        "n_concordant":       concordant_ab + concordant_c,
        "n_discordant":       discordant_ab + discordant_c,
        "n_tied":             tied_ab + tied_c,
        "kendall_tau":        round(tau_mean, 4),
        "kendall_tau_p":      float("nan"),
    }

    LOG.info(
        "C-index (all pairs)=%.4f  (time-ordered)=%.4f  "
        "(same-time/physio-only)=%.4f  [%d physio pairs]",
        cindex_all, cindex_ab, cindex_c, total_c,
    )
    LOG.info("Kendall τ (session-avg, time_since vs FTD) = %.4f  "
             "(negative = FTD decays correctly during recovery)", tau_mean)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Score range and stable-period reliability
# ─────────────────────────────────────────────────────────────────────────────

def compute_score_range_reliability(
    ftd_te: np.ndarray,
    df_te: pd.DataFrame,
    safe_threshold_s: float = 10.0,
) -> Dict:
    """
    Two metrics that are primary for a continuous risk / fitness-to-drive score:

    Dynamic range
    -------------
    median(during-distraction) − median(late-recovery FTD).
    A fitness-to-drive indicator is only useful if the score actually separates
    impaired from safe states by a meaningful amount. A large dynamic range
    means the displayed score visibly responds to impairment events.

    Stable-period reliability
    -------------------------
    When the driver has been undistracted for more than `safe_threshold_s`
    seconds, the score should sit consistently near zero (fitness-to-drive ≈ 1).
    We report:
      - stable_mean  : systematic offset of the score in safe periods
                       (false-alarm floor — should be close to 0)
      - stable_std   : moment-to-moment jitter during safe driving
                       (should be small for a smooth display)
      - stable_p95   : 95th percentile in safe periods — a natural threshold
                       above which the score is confidently in the "impaired"
                       regime
    """
    df = df_te.copy()
    df["_ftd"] = ftd_te
    df["_wd"]  = df["within_distraction"].fillna(0).astype(bool)
    df["_ts"]  = df["time_since_distraction_end"].fillna(0.0).astype(float)

    during_scores = df.loc[df["_wd"], "_ftd"].values
    late_scores   = df.loc[~df["_wd"] & (df["_ts"] > safe_threshold_s), "_ftd"].values

    if len(during_scores) > 0 and len(late_scores) > 0:
        peak    = float(np.median(during_scores))
        trough  = float(np.median(late_scores))
        drange  = round(peak - trough, 4)
    else:
        peak = trough = drange = float("nan")

    stable_mean = round(float(np.mean(late_scores)),         4) if len(late_scores) > 0 else float("nan")
    stable_std  = round(float(np.std(late_scores)),          4) if len(late_scores) > 0 else float("nan")
    stable_p95  = round(float(np.percentile(late_scores, 95)), 4) if len(late_scores) > 0 else float("nan")

    result = {
        "dynamic_range":    drange,
        "peak_median":      round(peak,   4) if np.isfinite(peak)   else float("nan"),
        "trough_median":    round(trough, 4) if np.isfinite(trough) else float("nan"),
        "stable_mean":      stable_mean,
        "stable_std":       stable_std,
        "stable_p95":       stable_p95,
        "n_during":         int(len(during_scores)),
        "n_stable":         int(len(late_scores)),
        "safe_threshold_s": safe_threshold_s,
    }

    LOG.info(
        "Score range: dynamic_range=%.4f  (peak=%.4f  trough=%.4f)  "
        "stable_mean=%.4f  stable_std=%.4f  stable_p95=%.4f",
        drange, peak, trough, stable_mean, stable_std, stable_p95,
    )
    return result


def plot_score_range_reliability(
    ftd_te: np.ndarray,
    df_te: pd.DataFrame,
    out: Path,
    safe_threshold_s: float = 10.0,
) -> None:
    """
    Two-panel figure:
      Left  — overlapping histograms of FTD in each phase (during / early-post / safe)
               Good score: during >> safe with minimal overlap
      Right  — CDF of FTD during safe driving with p95 marked as the
               natural fitness-to-drive alarm threshold
    """
    df = df_te.copy()
    df["_ftd"] = ftd_te
    df["_wd"]  = df["within_distraction"].fillna(0).astype(bool)
    df["_ts"]  = df["time_since_distraction_end"].fillna(0.0).astype(float)

    during = df.loc[df["_wd"], "_ftd"].values
    early  = df.loc[~df["_wd"] & (df["_ts"] <= 5.0), "_ftd"].values
    safe   = df.loc[~df["_wd"] & (df["_ts"] > safe_threshold_s), "_ftd"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(0, 1, 31)
    ax1.hist(safe,   bins=bins, alpha=0.6, color="#2ecc71",
             label=f"Safe (>{safe_threshold_s}s)")
    ax1.hist(early,  bins=bins, alpha=0.6, color="#e67e22",
             label="Early post (0–5s)")
    ax1.hist(during, bins=bins, alpha=0.6, color="#c0392b",
             label="During distraction")
    for arr, color in zip([during, early, safe], ["#c0392b", "#e67e22", "#2ecc71"]):
        if len(arr):
            ax1.axvline(np.median(arr), color=color, lw=2, ls="--")
    ax1.set_xlabel("FTD score  (fitness-to-drive = 1 − FTD)")
    ax1.set_ylabel("Count")
    ax1.set_title("Score Distribution by Driving Phase\n"
                  "Ideal: during >> safe, minimal overlap")
    ax1.legend(fontsize=9)

    if len(safe) > 0:
        sorted_safe = np.sort(safe)
        cdf = np.arange(1, len(sorted_safe) + 1) / len(sorted_safe)
        ax2.plot(sorted_safe, cdf, color="#2ecc71", lw=2)
        p95  = np.percentile(safe, 95)
        mean = np.mean(safe)
        ax2.axvline(p95,  color="#c0392b", lw=1.5, ls="--",
                    label=f"p95 = {p95:.3f}  ← alarm threshold")
        ax2.axvline(mean, color="#3498db", lw=1.5, ls=":",
                    label=f"mean = {mean:.3f}  ← false-alarm floor")
        ax2.set_xlabel("FTD score")
        ax2.set_ylabel("Cumulative fraction")
        ax2.set_title(f"Safe-Period Score CDF\n"
                      f"(n={len(safe)} samples, >{safe_threshold_s}s post-distraction)")
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No safe-period samples", ha="center", va="center")
        ax2.set_title("CDF in safe periods (no data)")

    fig.suptitle("Score Range & Stable-Period Reliability", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Score range / reliability → %s", out)


def plot_cindex_summary(
    cindex_result: Dict,
    calibration_result: Dict,
    onset_result: Dict,
    recovery_summary: Dict,
    score_range_result: Dict,
    out: Path,
) -> None:
    """
    Five-card summary covering all continuous risk score validity dimensions.
    Cards 1–4 = construct validity. Card 5 = practical usability (range + stability).
    """
    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    fig.suptitle("Construct Validity Summary — FTD Risk Score  "
                 "(fitness-to-drive = 1 − FTD)",
                 fontsize=13, y=1.02)

    def _card(ax, title, lines, color="#2c3e50"):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.95, title, ha="center", va="top",
                fontsize=10, fontweight="bold", color=color,
                transform=ax.transAxes)
        for i, line in enumerate(lines):
            ax.text(0.5, 0.82 - i * 0.14, line,
                    ha="center", va="top", fontsize=9,
                    transform=ax.transAxes)
        ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96,
                                   fill=False, edgecolor=color, lw=2,
                                   transform=ax.transAxes))

    ci = cindex_result
    _card(axes[0], "Concordance Index",
          [f"C-index = {ci['c_index']:.3f}",
           "(0.5=random, 1.0=perfect)",
           f"Kendall τ = {ci['kendall_tau']:.3f}",
           "(session-avg, −1 = perfect decay)",
           f"Pairs: {ci['n_pairs_total']:,}"],
          "#2980b9")

    od = onset_result.get("during_vs_baseline", {})
    _card(axes[1], "Onset Discrimination",
          [f"During vs Baseline",
           f"Mean during = {od.get('mean_high', float('nan')):.3f}",
           f"Mean baseline = {od.get('mean_low', float('nan')):.3f}",
           f"Effect r = {od.get('effect_r', float('nan')):.3f}",
           f"p = {od.get('p_value', float('nan')):.2e}"],
          "#8e44ad")

    rc = recovery_summary
    _card(axes[2], "Recovery Curves",
          [f"Events fitted: {rc.get('n_events_good_fit', '?')}",
           f"Median τ = {rc.get('tau_median', float('nan')):.1f}s",
           f"IQR τ = {rc.get('tau_iqr', float('nan')):.1f}s",
           f"Median R² = {rc.get('r2_median', float('nan')):.3f}",
           f"Median A = {rc.get('A_median', float('nan')):.3f}"],
          "#e74c3c")

    sr = score_range_result
    _card(axes[3], "Score Range",
          [f"Dynamic range = {sr.get('dynamic_range', float('nan')):.3f}",
           f"Peak (during) = {sr.get('peak_median', float('nan')):.3f}",
           f"Trough (safe) = {sr.get('trough_median', float('nan')):.3f}",
           "(larger = more usable display)",
           f"n_during={sr.get('n_during','?')}  n_safe={sr.get('n_stable','?')}"],
          "#16a085")

    _card(axes[4], "Safe-Period Reliability",
          [f"Mean = {sr.get('stable_mean', float('nan')):.3f}  (false-alarm floor)",
           f"Std  = {sr.get('stable_std', float('nan')):.3f}  (jitter)",
           f"p95  = {sr.get('stable_p95', float('nan')):.3f}  (alarm threshold)",
           f"(lower mean/std = more reliable)",
           f"threshold > {sr.get('safe_threshold_s','?')}s post-distraction"],
          "#d35400")

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Summary card → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
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
    Run all five construct-validity evaluations and save plots.

    Parameters
    ----------
    df_te          : test DataFrame — must contain user_id, run_id, sample_ts,
                     time_since_distraction_end, within_distraction,
                     time_in_distraction, distraction_idx
    ftd_te         : FTD risk score array aligned with df_te rows
    target_te      : binary T=1 error target — used only for calibration
                     secondary check (not the primary evaluation criterion)
    cal_target_te  : override calibration target (pass T=1 target if training T>1)
    out_dir        : directory for output plots
    H              : hangover horizon in seconds (for reference lines)
    n_deciles      : number of bins for calibration curve
    onset_window_s : seconds after distraction end to define "early post"
    seed           : random seed for C-index pair sampling
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cal_target_te is None:
        cal_target_te = target_te
        if target_te.mean() > 0.03:
            LOG.warning(
                "target positive rate is %.1f%% — may indicate T>1 targets "
                "are being used for calibration. Pass cal_target_te built "
                "with T=1 for an unbiased calibration curve.",
                100 * target_te.mean(),
            )

    LOG.info("─── Construct Validity Evaluation ──────────────────────────")

    # 1. Recovery curve fitting
    fits_df, recovery_summary = fit_recovery_curves(df_te, ftd_te)
    plot_recovery_curves(fits_df, df_te, ftd_te,
                         out_dir / "recovery_curves.png", H=H)

    # 2. Onset discrimination
    onset_result = compute_onset_discrimination(df_te, ftd_te,
                                                window_s=onset_window_s)
    plot_onset_discrimination(df_te, ftd_te,
                              out_dir / "onset_discrimination.png",
                              window_s=onset_window_s)

    # 3. Calibration curve (secondary validity check, event-level, T=1 target)
    centers, rates, ns, cal_summary = compute_calibration_curve(
        ftd_te, cal_target_te, df_te, n_deciles=n_deciles)
    baseline_rate = float(cal_target_te.mean())
    plot_calibration_curve(centers, rates, ns, baseline_rate,
                           out_dir / "calibration_curve.png")

    # 4. C-index (within-session pairs only)
    cindex_result = compute_cindex(ftd_te, df_te, seed=seed)

    # 5. Score range and stable-period reliability (primary usability metrics)
    score_range_result = compute_score_range_reliability(ftd_te, df_te)
    plot_score_range_reliability(ftd_te, df_te,
                                 out_dir / "score_range_reliability.png")

    # Summary card (5 panels)
    plot_cindex_summary(
        cindex_result, cal_summary, onset_result, recovery_summary,
        score_range_result,
        out_dir / "construct_validity_summary.png",
    )

    results = {
        "recovery_curves":      recovery_summary,
        "onset_discrimination": onset_result,
        "calibration":          cal_summary,
        "c_index":              cindex_result,
        "score_range":          score_range_result,
    }

    with open(out_dir / "construct_validity.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    LOG.info("─── Construct Validity: DONE ────────────────────────────────")
    LOG.info("  C-index:           %.4f  (0.5=random)", cindex_result["c_index"])
    LOG.info("  Kendall τ:         %.4f  (session-avg, negative=correct decay)",
             cindex_result["kendall_tau"])
    LOG.info("  Onset effect r:    %.4f  (p=%.2e)",
             onset_result["during_vs_baseline"].get("effect_r", float("nan")),
             onset_result["during_vs_baseline"].get("p_value",  float("nan")))
    LOG.info("  Recovery τ median: %.1f s", recovery_summary.get("tau_median", 0) or 0)
    LOG.info("  Dynamic range:     %.4f  (peak %.4f  trough %.4f)",
             score_range_result["dynamic_range"],
             score_range_result["peak_median"],
             score_range_result["trough_median"])
    LOG.info("  Safe-period:       mean=%.4f  std=%.4f  p95=%.4f",
             score_range_result["stable_mean"],
             score_range_result["stable_std"],
             score_range_result["stable_p95"])
    LOG.info("  Calibration r:     %.4f  (secondary check, p=%.2e)",
             cal_summary.get("spearman_r", float("nan")),
             cal_summary.get("spearman_p", float("nan")))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

def build_t1_target(df_te: pd.DataFrame, errs_lookup: dict) -> np.ndarray:
    """Build a T=1s binary error target from raw error timestamps."""
    from bisect import bisect_left
    horizon = pd.Timedelta(seconds=1)
    target  = np.zeros(len(df_te), dtype=int)
    rows    = df_te.reset_index(drop=True)
    for i in range(len(rows)):
        row      = rows.iloc[i]
        key      = (row["user_id"], int(row["run_id"]))
        err_list = errs_lookup.get(key, [])
        if not err_list:
            continue
        ts = row["sample_ts"]
        j  = bisect_left(err_list, ts)
        if j < len(err_list) and err_list[j] < ts + horizon:
            target[i] = 1
    return target


def run_standalone(args):
    """Load artifact and regenerate test set to run evaluation standalone."""
    import ftd_hmm as ftd_hmm_mod

    LOG.info("Loading artifact from %s", args.artifact)
    artifact       = joblib.load(args.artifact)
    scaler         = artifact["scaler"]
    feat_cols      = artifact["feature_cols"]
    risk_order     = np.array(artifact["state_profile"]["risk_order"])
    H              = artifact["config"]["H"]
    T_train        = artifact["config"]["T"]
    risk_weights   = artifact.get("risk_weights")
    risk_intercept = artifact.get("risk_intercept")
    if risk_weights is None or risk_intercept is None:
        raise ValueError("Artifact does not contain learned risk weights.")

    import re
    span_set = set()
    for col in feat_cols:
        m = re.search(r'_ema(\d+)', col)
        if m: span_set.add(int(m.group(1)))
        m = re.search(r'_roll_std(\d+)', col)
        if m: span_set.add(int(m.group(1)))
    if span_set:
        ftd_hmm_mod.ROLL_SPANS = sorted(span_set)

    hmm_dict  = artifact["hmm"]
    n_states  = hmm_dict["n_states"]
    cov_array = np.array(hmm_dict["covars"])
    if cov_array.ndim == 1:
        cov_type = "spherical"
    elif cov_array.ndim == 2:
        cov_type = "diag" if cov_array.shape[0] == n_states else "tied"
    elif cov_array.ndim == 3:
        cov_type = "full"
    else:
        cov_type = hmm_dict.get("covariance_type", "diag")

    hmm_model = GaussianHMM(n_components=n_states, covariance_type=cov_type)
    hmm_model.startprob_ = np.array(hmm_dict["startprob"])
    hmm_model.transmat_  = np.array(hmm_dict["transmat"])
    hmm_model.means_     = np.array(hmm_dict["means"])
    hmm_model.covars_    = cov_array

    d_raw, e_raw, eb_raw, db_raw = ftd_hmm_mod.load_data(args.data_path)
    ftd_hmm_mod.run_integrity_checks(d_raw, e_raw)

    users = sorted(set(d_raw["user_id"].unique()) | set(e_raw["user_id"].unique()))
    train_users, cal_users, test_users = ftd_hmm_mod.split_users(users, seed=args.seed)

    impute_stats = ftd_hmm_mod.fit_train_imputation_stats(train_users, d_raw)
    d = ftd_hmm_mod.apply_imputation_stats(d_raw, impute_stats)

    le_pred, le_emo = ftd_hmm_mod.fit_label_encoders(train_users, d, e_raw)
    pred_enc  = ftd_hmm_mod.build_encoding_map(le_pred)
    emo_enc   = ftd_hmm_mod.build_encoding_map(le_emo)
    baselines = ftd_hmm_mod.build_baselines(train_users, eb_raw, db_raw)

    d_train = d[d["user_id"].isin(train_users)]
    gmp = float(d_train[["model_prob_start", "model_prob_end"]].stack().median())
    gep = float(d_train[["emotion_prob_start", "emotion_prob_end"]].stack().median())
    if not np.isfinite(gmp): gmp = 0.5
    if not np.isfinite(gep): gep = 0.5

    wbs, webs, errs = ftd_hmm_mod.build_lookups(d_raw, e_raw, users)

    df_te = ftd_hmm_mod.generate_samples(H, T_train, test_users, wbs, webs, errs,
                                          baselines, pred_enc, emo_enc, gmp, gep)
    df_te = ftd_hmm_mod.add_causal_session_features(df_te)
    df_te = ftd_hmm_mod.apply_feature_postprocess(df_te, artifact["postprocess"])
    df_te = ftd_hmm_mod.sort_for_hmm(df_te)

    X_te     = scaler.transform(df_te[feat_cols].astype(float))
    b_te     = ftd_hmm_mod.sequence_bounds(df_te)
    gamma_te = np.zeros((len(X_te), n_states))
    for s, e in b_te:
        if e > s:
            gamma_te[s:e] = hmm_model.predict_proba(X_te[s:e])

    ftd_te    = ftd_hmm_mod.compute_risk_score(gamma_te, risk_order,
                                                risk_weights, risk_intercept)
    target_t1 = build_t1_target(df_te, errs)

    LOG.info("T=1 target: %.2f%% positive  (training T=%d target was %.2f%% positive)",
             100 * target_t1.mean(), T_train, 100 * df_te["target"].values.mean())

    evaluate_construct_validity(
        df_te, ftd_te, target_t1,
        out_dir=Path(args.output_dir),
        H=float(H),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Construct validity evaluation for FTD continuous risk score"
    )
    p.add_argument("--artifact",   default="result/impairment_hmm.joblib")
    p.add_argument("--data-path",  default="data")
    p.add_argument("--output-dir", default="result/construct_validity/")
    p.add_argument("--seed",       type=int, default=42)
    raise SystemExit(run_standalone(p.parse_args()))