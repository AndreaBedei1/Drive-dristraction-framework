#!/usr/bin/env python3
"""
full_evaluation.py
==================
Comparison of two HMM‑based FTD scores (full physiology vs. distraction-only)
against a time-only exponential baseline for the purpose of evaluating a
CONTINUOUS RISK SCORE (fitness-to-drive = 1 - FTD).

We are NOT predicting discrete error events. Binary classification metrics
(AUC-ROC, Average Precision, ROC/PR curves) are therefore inappropriate and
have been removed. The comparison is entirely construct-validity based:

  Primary metrics (does the score track impairment state?):
    - C-index             : ordinal consistency within sessions
    - Kendall τ           : signed correlation between time-since and FTD
    - Onset effect r      : Mann-Whitney r during vs. baseline
    - Dynamic range       : median(during) - median(safe) — score usability
    - Stable-period mean  : false-alarm floor in safe driving
    - Stable-period std   : jitter in safe driving
    - Recovery τ median   : time constant of exponential decay

  Secondary validity check:
    - Calibration r       : Spearman r between FTD decile and error rate —
                            checks internal consistency only, not the main goal

Baseline: a time-only model that outputs exp(-time_since / tau_fixed).
This is the right baseline for a risk score: it asks "does knowing physiology
add value beyond just knowing how long ago the distraction ended?"

Outputs
-------
  full_comparison.json    — all metrics per model
  score_distributions.png — phase-separated score histograms per model
  delta_construct.png     — Δ construct metrics vs time-only baseline
  recovery_comparison.png — overlaid non-parametric recovery curves
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import curve_fit

# Import the two HMM modules
import ftd_hmm2 as notime_module
import ftd_hmm_ablation as ablation_module

# Import all construct-validity functions from the fixed evaluation script
from evaluation import (
    compute_cindex,
    compute_onset_discrimination,
    fit_recovery_curves,
    compute_score_range_reliability,
    build_t1_target,
)


LOG = logging.getLogger("full_evaluation")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

COLORS = {
    "HMM (full)":     "#2ecc71",
    "HMM (ablation)": "#e67e22",
    "Time-only exp":  "#3498db",
}

# ─────────────────────────────────────────────────────────────────────────────
# Time-only exponential baseline
# ─────────────────────────────────────────────────────────────────────────────

def _exp_decay(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    return A * np.exp(-t / np.clip(tau, 0.1, 300.0)) + C


def fit_time_only_baseline(df_ca: pd.DataFrame, target_ca: np.ndarray) -> dict:
    """
    Fit a simple A·exp(-t/τ)+C curve to the EMPIRICAL ERROR RATE as a function
    of time_since_distraction_end on the calibration set.

    This is the correct, non-circular baseline: it uses only the time axis and
    ground-truth error labels — NOT the HMM's own output.  Comparing this against
    the HMM answers "does knowing physiology add value beyond just knowing how
    long ago the distraction ended?"

    We bin post-distraction samples by integer second, compute the empirical
    error rate per bin, then fit the exponential curve to those bin rates.
    """
    df = df_ca.copy()
    df["_target"] = target_ca.astype(float)
    df["_wd"]  = df["within_distraction"].fillna(0).astype(bool)
    df["_ts"]  = df["time_since_distraction_end"].fillna(0.0).astype(float)

    post = df[~df["_wd"] & df["_ts"].between(0.0, 60.0)].copy()
    if len(post) < 20:
        LOG.warning("Too few post-distraction calibration samples; "
                    "using defaults A=0.3, tau=10, C=0.1")
        return {"A": 0.3, "tau": 10.0, "C": 0.1}

    # Aggregate to second-level bins for a stable fit target
    post["_sec"] = post["_ts"].astype(int)
    bin_stats = post.groupby("_sec")["_target"].agg(["mean", "count"])
    bin_stats = bin_stats[bin_stats["count"] >= 3]   # skip under-populated bins

    if len(bin_stats) < 5:
        LOG.warning("Too few populated second-bins in calibration set; "
                    "using defaults A=0.3, tau=10, C=0.1")
        return {"A": 0.3, "tau": 10.0, "C": 0.1}

    t = bin_stats.index.values.astype(float)
    y = bin_stats["mean"].values

    try:
        popt, _ = curve_fit(
            _exp_decay, t, y,
            p0=[y.max() - y.min(), 10.0, y.min()],
            bounds=([0, 0.5, 0.0], [1.0, 120.0, 1.0]),
            maxfev=5000,
        )
        A, tau, C = popt
        LOG.info("Time-only baseline fitted to EMPIRICAL ERROR RATE: "
                 "A=%.3f  tau=%.1fs  C=%.3f  (from %d second-bins)",
                 A, tau, C, len(bin_stats))
        return {"A": float(A), "tau": float(tau), "C": float(C)}
    except (RuntimeError, ValueError) as exc:
        LOG.warning("Time-only baseline fit failed (%s); using defaults.", exc)
        return {"A": 0.3, "tau": 10.0, "C": 0.1}


def apply_time_only_baseline(df_te: pd.DataFrame, params: dict) -> np.ndarray:
    """
    Produce a time-only risk score for the test set:
      score = A * exp(-time_since / tau) + C  (post-distraction)
      score = A + C                            (during distraction, t=0)
    The score is clipped to [0, 1].
    """
    ts   = df_te["time_since_distraction_end"].fillna(0.0).values.astype(float)
    wd   = df_te["within_distraction"].fillna(0).astype(bool).values
    ts   = np.where(wd, 0.0, ts)   # during distraction treat as t=0
    score = _exp_decay(ts, params["A"], params["tau"], params["C"])
    return np.clip(score, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Construct-validity metrics for a single model
# ─────────────────────────────────────────────────────────────────────────────

def compute_construct_metrics(score: np.ndarray, df: pd.DataFrame,
                               name: str) -> dict:
    """
    Compute all construct-validity metrics for a continuous risk score.
    No binary classification metrics are used.
    """
    LOG.info("Computing construct-validity metrics for %s", name)

    cidx_res = compute_cindex(score, df, seed=42)
    onset_res = compute_onset_discrimination(df, score, window_s=5.0)
    onset_effect = onset_res.get("during_vs_baseline", {}).get("effect_r",
                                                               float("nan"))

    fits_df, rec_summary = fit_recovery_curves(df, score, min_points=10)
    rec_tau    = rec_summary.get("tau_median",        float("nan"))
    rec_r2     = rec_summary.get("r2_median",         float("nan"))
    n_fits     = rec_summary.get("n_events_good_fit", 0)

    sr = compute_score_range_reliability(score, df)

    return {
        "c_index":             round(cidx_res["c_index"],                       4),
        "c_index_physio":      round(cidx_res.get("c_index_physio",  float("nan")), 4),
        "c_index_time_pairs":  round(cidx_res.get("c_index_time_pairs", float("nan")), 4),
        "n_pairs_physio":      cidx_res.get("n_pairs_physio", 0),
        "kendall_tau":         round(cidx_res["kendall_tau"],     4),
        "onset_effect_r":      round(float(onset_effect), 4) if np.isfinite(onset_effect) else None,
        "dynamic_range":       sr["dynamic_range"],
        "stable_mean":         sr["stable_mean"],
        "stable_std":          sr["stable_std"],
        "stable_p95":          sr["stable_p95"],
        "recovery_tau_median": round(rec_tau, 2) if not np.isnan(rec_tau) else None,
        "recovery_r2_median":  round(rec_r2, 3) if not np.isnan(rec_r2)  else None,
        "recovery_n_events":   n_fits,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_distributions(models: list, out: Path) -> None:
    """
    For each model: overlapping histograms of FTD score during / early-post
    / safe phases. A good risk score has clearly separated distributions.
    """
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    bins = np.linspace(0, 1, 31)

    for ax, m in zip(axes, models):
        df     = m["df"]
        score  = m["score"]
        wd     = df["within_distraction"].fillna(0).astype(bool).values
        ts     = df["time_since_distraction_end"].fillna(0.0).values.astype(float)

        during = score[wd]
        early  = score[~wd & (ts <= 5.0)]
        safe   = score[~wd & (ts > 10.0)]

        ax.hist(safe,   bins=bins, alpha=0.6, color="#2ecc71",
                label=f"Safe >10s  (med={np.median(safe):.3f})" if len(safe) else "Safe >10s")
        ax.hist(early,  bins=bins, alpha=0.6, color="#e67e22",
                label=f"Early post 0-5s  (med={np.median(early):.3f})" if len(early) else "Early post")
        ax.hist(during, bins=bins, alpha=0.6, color="#c0392b",
                label=f"During  (med={np.median(during):.3f})" if len(during) else "During")

        for arr, color in zip([during, early, safe], ["#c0392b", "#e67e22", "#2ecc71"]):
            if len(arr):
                ax.axvline(np.median(arr), color=color, lw=2, ls="--")

        dr = m.get("dynamic_range")
        ax.set_title(f"{m['name']}\ndynamic_range={dr:.3f}" if dr is not None else m["name"],
                     fontsize=10)
        ax.set_xlabel("FTD score  (fitness-to-drive = 1 − FTD)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle("Score Distribution by Driving Phase\n"
                 "(ideal: during >> safe, minimal overlap)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Score distributions → %s", out)


def plot_recovery_comparison(models: list, out: Path) -> None:
    """
    Overlaid non-parametric recovery curves (mean FTD ± SE per second bin,
    post-distraction only) for all models. This directly visualises how
    quickly each model's score returns to baseline after a distraction.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(0, 21)

    for m in models:
        df    = m["df"]
        score = m["score"]
        wd    = df["within_distraction"].fillna(0).astype(bool).values
        ts    = df["time_since_distraction_end"].fillna(0.0).values.astype(float)

        means, sems = [], []
        for b in bins:
            mask = ~wd & (ts >= b) & (ts < b + 1)
            if mask.sum() > 1:
                v = score[mask]
                means.append(v.mean())
                sems.append(v.std() / np.sqrt(len(v)))
            else:
                means.append(np.nan); sems.append(np.nan)

        means = np.array(means); sems = np.array(sems)
        valid = ~np.isnan(means)
        color = COLORS.get(m["name"], "#7f8c8d")
        ax.plot(bins[valid], means[valid], "o-", lw=2, color=color,
                label=m["name"])
        ax.fill_between(bins[valid],
                        means[valid] - sems[valid],
                        means[valid] + sems[valid],
                        alpha=0.15, color=color)

    ax.set_xlabel("Seconds after distraction end")
    ax.set_ylabel("Mean FTD score  (lower = safer)")
    ax.set_title("Non-Parametric Recovery Curves\n"
                 "(faster decay = score clears impairment sooner)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Recovery comparison → %s", out)


def plot_delta_construct(models: list, out: Path) -> None:
    """
    Bar chart showing Δ construct-validity metrics vs time-only baseline.
    Metrics: C-index, onset_effect_r, dynamic_range (all: higher = better).
    Stable_mean shows as negative delta (lower false-alarm floor = better).
    """
    b0 = next(m for m in models if m["name"] == "Time-only exp")
    others = [m for m in models if m["name"] != "Time-only exp"]
    if not others:
        LOG.warning("No non-baseline models to plot delta construct")
        return

    metric_keys   = ["c_index", "onset_effect_r", "dynamic_range", "stable_mean"]
    metric_labels = ["ΔC-index", "ΔOnset r", "ΔDynamic range",
                     "ΔStable mean\n(−= less false alarms)"]
    # For stable_mean: lower is better, so we negate the delta
    signs = [1, 1, 1, -1]

    n_metrics = len(metric_keys)
    n_models  = len(others)
    x = np.arange(n_metrics)
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, m in enumerate(others):
        deltas = []
        for key, sign in zip(metric_keys, signs):
            v_model    = m.get(key)
            v_baseline = b0.get(key)
            if v_model is None or v_baseline is None or \
               not np.isfinite(float(v_model)) or not np.isfinite(float(v_baseline)):
                deltas.append(0.0)
            else:
                deltas.append(sign * (float(v_model) - float(v_baseline)))

        offset = (i - (n_models - 1) / 2) * w
        bars   = ax.bar(x + offset, deltas, w,
                        color=COLORS.get(m["name"], "#7f8c8d"),
                        alpha=0.85, label=m["name"])
        for bar, val in zip(bars, deltas):
            ypos = bar.get_height() + (0.002 if val >= 0 else -0.005)
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:+.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Delta vs Time-only baseline")
    ax.set_title("Lift Over Time-only Baseline\n"
                 "(positive = better continuous risk score)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Delta construct → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Data generation helpers
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_hmm(artifact: dict):
    hmm_dict = artifact["hmm"]
    n_states = hmm_dict["n_states"]
    cov_arr  = np.array(hmm_dict["covars"])
    if cov_arr.ndim == 1:
        cov_type = "spherical"
    elif cov_arr.ndim == 2:
        cov_type = "diag" if cov_arr.shape[0] == n_states else "tied"
    elif cov_arr.ndim == 3:
        cov_type = "full"
    else:
        cov_type = hmm_dict.get("covariance_type", "diag")

    hmm_model = GaussianHMM(n_components=n_states, covariance_type=cov_type)
    hmm_model.startprob_ = np.array(hmm_dict["startprob"])
    hmm_model.transmat_  = np.array(hmm_dict["transmat"])
    hmm_model.means_     = np.array(hmm_dict["means"])
    hmm_model.covars_    = cov_arr
    return hmm_model, n_states


def generate_test_data(artifact: dict, module, data_path: str, seed: int):
    d_raw, e_raw, eb_raw, db_raw = module.load_data(data_path)
    module.run_integrity_checks(d_raw, e_raw)

    users = sorted(set(d_raw["user_id"].unique()) | set(e_raw["user_id"].unique()))
    train_users, cal_users, test_users = module.split_users(users, seed=seed)

    impute_stats = module.fit_train_imputation_stats(train_users, d_raw)
    d = module.apply_imputation_stats(d_raw, impute_stats)

    le_pred, le_emo  = module.fit_label_encoders(train_users, d, e_raw)
    pred_enc         = module.build_encoding_map(le_pred)
    emo_enc          = module.build_encoding_map(le_emo)
    baselines_bl     = module.build_baselines(train_users, eb_raw, db_raw)

    d_train = d[d["user_id"].isin(train_users)]
    gmp = float(d_train[["model_prob_start", "model_prob_end"]].stack().median())
    gep = float(d_train[["emotion_prob_start", "emotion_prob_end"]].stack().median())
    if not np.isfinite(gmp): gmp = 0.5
    if not np.isfinite(gep): gep = 0.5

    # Use d_raw for lookups — no imputation leakage into test metadata
    wbs, webs, errs = module.build_lookups(d_raw, e_raw, users)

    H = artifact["config"]["H"]
    T = artifact["config"]["T"]

    df_te = module.generate_samples(
        H, T, test_users, wbs, webs, errs,
        baselines_bl, pred_enc, emo_enc, gmp, gep,
    )
    df_te = module.add_causal_session_features(df_te)
    # Use this artifact's own postprocess stats — not another model's
    df_te = module.apply_feature_postprocess(df_te, artifact["postprocess"])
    df_te = module.sort_for_hmm(df_te)

    target_t1 = build_t1_target(df_te, errs)
    return df_te, target_t1, errs, test_users


def get_cal_data(artifact: dict, module, data_path: str, seed: int):
    """
    Generate calibration-set samples and T=1 binary targets.

    Returns (df_ca, target_ca, errs, cal_users) — the same shape as
    generate_test_data so callers can treat both symmetrically.
    The target_ca array is built from raw error timestamps (T=1 horizon),
    giving an unbiased error-rate signal for fitting the time-only baseline.
    """
    d_raw, e_raw, eb_raw, db_raw = module.load_data(data_path)
    module.run_integrity_checks(d_raw, e_raw)

    users = sorted(set(d_raw["user_id"].unique()) | set(e_raw["user_id"].unique()))
    train_users, cal_users, test_users = module.split_users(users, seed=seed)

    impute_stats = module.fit_train_imputation_stats(train_users, d_raw)
    d = module.apply_imputation_stats(d_raw, impute_stats)

    le_pred, le_emo  = module.fit_label_encoders(train_users, d, e_raw)
    pred_enc         = module.build_encoding_map(le_pred)
    emo_enc          = module.build_encoding_map(le_emo)
    baselines_bl     = module.build_baselines(train_users, eb_raw, db_raw)

    d_train = d[d["user_id"].isin(train_users)]
    gmp = float(d_train[["model_prob_start", "model_prob_end"]].stack().median())
    gep = float(d_train[["emotion_prob_start", "emotion_prob_end"]].stack().median())
    if not np.isfinite(gmp): gmp = 0.5
    if not np.isfinite(gep): gep = 0.5

    wbs, webs, errs = module.build_lookups(d_raw, e_raw, users)

    H = artifact["config"]["H"]
    T = artifact["config"]["T"]

    df_ca = module.generate_samples(
        H, T, cal_users, wbs, webs, errs,
        baselines_bl, pred_enc, emo_enc, gmp, gep,
    )
    df_ca = module.add_causal_session_features(df_ca)
    # Use this artifact's own postprocess stats
    df_ca = module.apply_feature_postprocess(df_ca, artifact["postprocess"])
    df_ca = module.sort_for_hmm(df_ca)

    # Build T=1 binary targets from raw error timestamps (unbiased ground truth)
    target_ca = build_t1_target(df_ca, errs)
    return df_ca, target_ca, errs, cal_users


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    LOG.info("Loading full physiology artifact from %s", args.artifact_full)
    art_full = joblib.load(args.artifact_full)
    LOG.info("Loading distraction-only artifact from %s", args.artifact_ablation)
    art_ablation = joblib.load(args.artifact_ablation)

    # Reconstruct HMMs
    hmm_full,     n_states_full     = reconstruct_hmm(art_full)
    hmm_ablation, n_states_ablation = reconstruct_hmm(art_ablation)

    # Generate test data (each model uses its own module and artifact)
    LOG.info("Generating test data for full physiology HMM...")
    df_te_full, target_full, errs_full, test_users_full = generate_test_data(
        art_full, notime_module, args.data_path, args.seed)

    LOG.info("Generating test data for distraction-only HMM...")
    df_te_ablation, target_ablation, errs_ablation, test_users_ablation = generate_test_data(
        art_ablation, ablation_module, args.data_path, args.seed)

    assert set(test_users_full) == set(test_users_ablation), "Test user mismatch!"
    LOG.info("Test set: %d samples", len(df_te_full))

    # Compute FTD scores — full HMM
    scaler_full    = art_full["scaler"]
    feat_cols_full = art_full["feature_cols"]
    X_te_full      = scaler_full.transform(df_te_full[feat_cols_full].astype(float))
    b_te_full      = notime_module.sequence_bounds(df_te_full)

    gamma_te_full = np.zeros((len(df_te_full), n_states_full))
    for s, e in b_te_full:
        if e > s:
            gamma_te_full[s:e] = hmm_full.predict_proba(X_te_full[s:e])

    risk_order_full = np.array(art_full["state_profile"]["risk_order"])
    weights_full    = np.array(art_full["risk_weights"])
    intercept_full  = art_full["risk_intercept"]
    ftd_full = notime_module.compute_risk_score(
        gamma_te_full, risk_order_full, weights_full, intercept_full)

    # Compute FTD scores — ablation HMM
    scaler_ablation    = art_ablation["scaler"]
    feat_cols_ablation = art_ablation["feature_cols"]
    X_te_ablation      = scaler_ablation.transform(
        df_te_ablation[feat_cols_ablation].astype(float))
    b_te_ablation = ablation_module.sequence_bounds(df_te_ablation)

    gamma_te_ablation = np.zeros((len(df_te_ablation), n_states_ablation))
    for s, e in b_te_ablation:
        if e > s:
            gamma_te_ablation[s:e] = hmm_ablation.predict_proba(X_te_ablation[s:e])

    risk_order_ablation = np.array(art_ablation["state_profile"]["risk_order"])
    weights_ablation    = np.array(art_ablation["risk_weights"])
    intercept_ablation  = art_ablation["risk_intercept"]
    ftd_ablation = ablation_module.compute_risk_score(
        gamma_te_ablation, risk_order_ablation, weights_ablation, intercept_ablation)

    # Time-only exponential baseline
    # Fit to EMPIRICAL ERROR RATE on the calibration set — NOT to the HMM's
    # own output.  This is the only non-circular comparison: it answers
    # "does knowing physiology add value beyond just knowing how long ago the
    # distraction ended?"
    LOG.info("Fitting time-only exponential baseline on calibration set "
             "(target = empirical error rate, not HMM score)...")
    df_ca_full, target_ca, _errs_ca, _cal_users = get_cal_data(
        art_full, notime_module, args.data_path, args.seed)

    time_params  = fit_time_only_baseline(df_ca_full, target_ca)
    ftd_time_te  = apply_time_only_baseline(df_te_full, time_params)

    # Bundle all models
    model_bundles = [
        ("HMM (full)",     ftd_full,     df_te_full),
        ("HMM (ablation)", ftd_ablation, df_te_ablation),
        ("Time-only exp",  ftd_time_te,  df_te_full),
    ]

    # Compute construct-validity metrics for each
    models = []
    for name, score, df in model_bundles:
        m = {"name": name, "score": score, "df": df}
        m.update(compute_construct_metrics(score, df, name))
        models.append(m)

    # Plots
    plot_score_distributions(models, out_dir / "score_distributions.png")
    plot_recovery_comparison(models, out_dir / "recovery_comparison.png")
    plot_delta_construct(models, out_dir / "delta_construct.png")

    # Serialise (drop bulky arrays)
    output_rows = []
    for m in models:
        row = {k: v for k, v in m.items() if k not in ("score", "df")}
        output_rows.append(row)

    # Delta columns vs time-only baseline
    b0 = next(m for m in output_rows if m["name"] == "Time-only exp")
    for m in output_rows:
        for key in ("c_index", "c_index_physio", "onset_effect_r", "dynamic_range",
                    "stable_mean", "recovery_tau_median"):
            v  = m.get(key)
            v0 = b0.get(key)
            if v is not None and v0 is not None and \
               np.isfinite(float(v)) and np.isfinite(float(v0)):
                m[f"delta_{key}_vs_time"] = round(float(v) - float(v0), 4)
            else:
                m[f"delta_{key}_vs_time"] = None

    with open(out_dir / "full_comparison.json", "w") as f:
        json.dump(output_rows, f, indent=2)

    # Summary table
    LOG.info("─── Full Comparison (continuous risk score) ─────────────────")
    header = (f"{'Model':<22}  {'C-idx':>7}  {'C-physio':>9}  {'Kend tau':>8}  "
              f"{'Onset r':>8}  {'Dyn rng':>8}  {'Stbl mean':>10}  {'Stbl std':>9}  "
              f"{'tau_rec':>8}  {'adj-R2':>7}")
    LOG.info(header)
    LOG.info("  NOTE: C-physio = concordance on same-time/diff-event pairs ")
    LOG.info("        This is the only physiology-sensitive metric; ")
    LOG.info("        C-idx ~1.0 for ANY model that decays with time.")
    for m in output_rows:
        LOG.info(
            "%-22s  %7.4f  %9.4f  %7.4f  %8.4f  %8.4f  %10.4f  %9.4f  %8.2f  %7.3f",
            m["name"],
            m.get("c_index")             or 0,
            m.get("c_index_physio")      or 0,
            m.get("kendall_tau")         or 0,
            m.get("onset_effect_r")      or 0,
            m.get("dynamic_range")       or 0,
            m.get("stable_mean")         or 0,
            m.get("stable_std")          or 0,
            m.get("recovery_tau_median") or 0,
            m.get("recovery_r2_median")  or 0,
        )

    LOG.info("All outputs saved to %s", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Full evaluation: two HMM risk scores vs time-only baseline "
                    "(continuous risk score evaluation, no binary classification metrics)"
    )
    p.add_argument("--artifact-full",
                   default="result/impairment_hmm.joblib")
    p.add_argument("--artifact-ablation",
                   default="result_ablation/impairment_hmm_ablation.joblib")
    p.add_argument("--data-path",   default="data")
    p.add_argument("--output-dir",  default="results/full_comparison/")
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()
    run(args)