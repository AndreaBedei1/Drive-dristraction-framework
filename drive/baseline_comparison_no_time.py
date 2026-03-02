import argparse
import json
import logging
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler

import ftd_hmm_no_time as ftd_hmm

LOG = logging.getLogger("baseline_comparison")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_t1_target(df: pd.DataFrame, errs_lookup: dict) -> np.ndarray:
    """Binary T=1 target, independent of training T."""
    from bisect import bisect_left
    horizon = pd.Timedelta(seconds=1)
    target  = np.zeros(len(df), dtype=int)
    rows    = df.reset_index(drop=True)
    for i in range(len(rows)):
        row      = rows.iloc[i]
        key      = (row["user_id"], int(row["run_id"]))
        err_list = errs_lookup.get(key, [])
        if not err_list:
            continue
        ts  = row["sample_ts"]
        j   = bisect_left(err_list, ts)
        if j < len(err_list) and err_list[j] < ts + horizon:
            target[i] = 1
    return target


def calibration_spearman(score: np.ndarray, target: np.ndarray,
                          n_deciles: int = 10) -> float:
    """
    Bin score into n_deciles, compute empirical error rate per bin,
    return Spearman r between bin-mean score and bin error rate.
    This mirrors evaluate.py's compute_calibration_curve.
    """
    edges   = np.percentile(score, np.linspace(0, 100, n_deciles + 1))
    bin_idx = np.clip(np.digitize(score, edges) - 1, 0, n_deciles - 1)
    centers, rates = [], []
    for b in range(n_deciles):
        mask = bin_idx == b
        if mask.sum() > 0:
            centers.append(float(score[mask].mean()))
            rates.append(float(target[mask].mean()))
    r, _ = spearmanr(centers, rates)
    return float(r)


def fit_lr_baseline(X_ca: np.ndarray, y_ca: np.ndarray,
                    seed: int = 42) -> LogisticRegression:
    """Logistic regression fitted on calibration set."""
    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    )
    lr.fit(X_ca, y_ca)
    return lr


def score_model(name: str, proba: np.ndarray, target: np.ndarray) -> dict:
    auc  = roc_auc_score(target, proba)
    ap   = average_precision_score(target, proba)
    cal  = calibration_spearman(proba, target)
    LOG.info("%-22s  AUC-ROC=%.4f  AP=%.4f  Cal-r=%.4f", name, auc, ap, cal)
    return {"name": name, "auc_roc": auc, "avg_precision": ap, "calibration_r": cal}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "HMM FTD":       "#e74c3c",
    "B0 Time-only":  "#3498db",
    "B1 Physio-only":"#2ecc71",
    "B2 Full-LR":    "#9b59b6",
}


def plot_roc(models: list, target: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in models:
        fpr, tpr, _ = roc_curve(target, m["proba"])
        ax.plot(fpr, tpr, lw=2, color=COLORS[m["name"]],
                label=f"{m['name']}  (AUC={m['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — HMM vs Baselines")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOG.info("ROC curves → %s", out)


def plot_pr(models: list, target: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    baseline_rate = target.mean()
    ax.axhline(baseline_rate, color="gray", ls="--", lw=1,
               label=f"Baseline rate ({baseline_rate:.4f})")
    for m in models:
        prec, rec, _ = precision_recall_curve(target, m["proba"])
        ax.plot(rec, prec, lw=2, color=COLORS[m["name"]],
                label=f"{m['name']}  (AP={m['avg_precision']:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — HMM vs Baselines")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOG.info("PR curves → %s", out)


def plot_calibration_curves(models: list, target: np.ndarray, out: Path,
                             n_deciles: int = 10) -> None:
    """
    Side-by-side decile calibration bars for all four models,
    plus a line showing the perfect-calibration diagonal.
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    baseline_rate = target.mean()

    for ax, m in zip(axes, models):
        score = m["proba"]
        edges = np.percentile(score, np.linspace(0, 100, n_deciles + 1))
        bin_idx = np.clip(np.digitize(score, edges) - 1, 0, n_deciles - 1)
        centers, rates, ns = [], [], []
        for b in range(n_deciles):
            mask = bin_idx == b
            if mask.sum() > 0:
                centers.append(float(score[mask].mean()))
                rates.append(float(target[mask].mean()))
                ns.append(int(mask.sum()))

        centers = np.array(centers)
        rates   = np.array(rates)
        x       = np.arange(len(centers))

        bars = ax.bar(x, rates, color=COLORS[m["name"]], alpha=0.75,
                      edgecolor="white")
        ax.axhline(baseline_rate, color="black", ls="--", lw=1.2,
                   label=f"Baseline ({baseline_rate:.4f})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:.2f}" for c in centers],
                           rotation=45, fontsize=7)
        ax.set_xlabel("Mean score in decile")
        ax.set_title(f"{m['name']}\nCal-r={m['calibration_r']:.3f}", fontsize=10)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Empirical error rate")
    fig.suptitle("Calibration: FTD Decile → Observed Error Rate", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Calibration curves → %s", out)


def plot_delta_bar(models: list, out: Path) -> None:
    """
    Bar chart of ΔAUC-ROC and ΔAP relative to B0 (time-only).
    Shows what the HMM and LR baselines add beyond knowing elapsed time alone.
    """
    b0_auc = next(m for m in models if m["name"] == "B0 Time-only")["auc_roc"]
    b0_ap  = next(m for m in models if m["name"] == "B0 Time-only")["avg_precision"]

    non_b0 = [m for m in models if m["name"] != "B0 Time-only"]
    names  = [m["name"] for m in non_b0]
    d_auc  = [m["auc_roc"]       - b0_auc for m in non_b0]
    d_ap   = [m["avg_precision"]  - b0_ap  for m in non_b0]

    x  = np.arange(len(names))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_auc = ax.bar(x - w / 2, d_auc, w,
                      color=[COLORS[n] for n in names], alpha=0.85,
                      label="ΔAUC-ROC")
    bars_ap  = ax.bar(x + w / 2, d_ap,  w,
                      color=[COLORS[n] for n in names], alpha=0.45,
                      edgecolor=[COLORS[n] for n in names], linewidth=1.5,
                      label="ΔAvg Precision")

    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Delta vs B0 Time-only")
    ax.set_title("Lift Over Time-only Baseline\n"
                 "(positive = model adds value beyond elapsed time)")
    ax.legend()

    for bar, val in list(zip(bars_auc, d_auc)) + list(zip(bars_ap, d_ap)):
        ypos = bar.get_height() + (0.001 if val >= 0 else -0.003)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:+.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Delta-AUC bar → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load artifact ──────────────────────────────────────────────────────
    LOG.info("Loading artifact from %s", args.artifact)
    artifact       = joblib.load(args.artifact)
    scaler_tr      = artifact["scaler"]          # fitted on train users only
    feat_cols      = artifact["feature_cols"]
    risk_order     = np.array(artifact["state_profile"]["risk_order"])
    risk_weights   = np.array(artifact["risk_weights"])
    risk_intercept = float(artifact["risk_intercept"])
    H              = artifact["config"]["H"]
    T_train        = artifact["config"]["T"]

    # Infer rolling spans used during training
    span_set = set()
    for col in feat_cols:
        for pat in [r'_ema(\d+)', r'_roll_std(\d+)']:
            m = re.search(pat, col)
            if m:
                span_set.add(int(m.group(1)))
    if span_set:
        ftd_hmm.ROLL_SPANS = sorted(span_set)

    # Reconstruct GaussianHMM
    hmm_dict  = artifact["hmm"]
    n_states  = hmm_dict["n_states"]
    cov_arr   = np.array(hmm_dict["covars"])
    cov_type  = ("full" if cov_arr.ndim == 3 else
                 "spherical" if cov_arr.ndim == 1 else
                 "diag" if cov_arr.shape[0] == n_states else "tied")
    hmm_model = GaussianHMM(n_components=n_states, covariance_type=cov_type)
    hmm_model.startprob_ = np.array(hmm_dict["startprob"])
    hmm_model.transmat_  = np.array(hmm_dict["transmat"])
    hmm_model.means_     = np.array(hmm_dict["means"])
    hmm_model.covars_    = cov_arr

    # ── 2. Rebuild the exact same train / cal / test split ────────────────────
    d_raw, e_raw, eb_raw, db_raw = ftd_hmm.load_data(args.data_path)
    ftd_hmm.run_integrity_checks(d_raw, e_raw)

    users = sorted(set(d_raw["user_id"].unique()) | set(e_raw["user_id"].unique()))
    # ── FIX applied here: proper three-way split (not two-way) ───────────────
    train_users, cal_users, test_users = ftd_hmm.split_users(users, seed=args.seed)
    LOG.info("Split: %d train  %d cal  %d test users",
             len(train_users), len(cal_users), len(test_users))

    impute_stats = ftd_hmm.fit_train_imputation_stats(train_users, d_raw)
    d = ftd_hmm.apply_imputation_stats(d_raw, impute_stats)

    le_pred, le_emo = ftd_hmm.fit_label_encoders(train_users, d, e_raw)
    pred_enc = ftd_hmm.build_encoding_map(le_pred)
    emo_enc  = ftd_hmm.build_encoding_map(le_emo)
    baselines_bl = ftd_hmm.build_baselines(train_users, eb_raw, db_raw)

    d_train = d[d["user_id"].isin(train_users)]
    gmp = float(d_train[["model_prob_start", "model_prob_end"]].stack().median())
    gep = float(d_train[["emotion_prob_start", "emotion_prob_end"]].stack().median())
    if not np.isfinite(gmp): gmp = 0.5
    if not np.isfinite(gep): gep = 0.5

    wbs, webs, errs = ftd_hmm.build_lookups(d, e_raw, users)

    def gen(user_set):
        df = ftd_hmm.generate_samples(
            H, T_train, user_set, wbs, webs, errs,
            baselines_bl, pred_enc, emo_enc, gmp, gep,
        )
        df = ftd_hmm.add_causal_session_features(df)
        df = ftd_hmm.apply_feature_postprocess(df, artifact["postprocess"])
        return ftd_hmm.sort_for_hmm(df)

    LOG.info("Generating samples…")
    df_ca = gen(cal_users)
    df_te = gen(test_users)

    # ── 3. Build targets (always T=1 for fair comparison) ────────────────────
    target_ca = build_t1_target(df_ca, errs)
    target_te = build_t1_target(df_te, errs)
    LOG.info("Cal  positive rate: %.2f%%  (%d samples)",
             100 * target_ca.mean(), len(target_ca))
    LOG.info("Test positive rate: %.2f%%  (%d samples)",
             100 * target_te.mean(), len(target_te))

        # ── 4. Feature matrices ───────────────────────────────────────────────────
    # Production features as used by the HMM (already excludes or includes
    # time_since depending on label_mode at training time)
    X_ca_full = scaler_tr.transform(df_ca[feat_cols].astype(float))
    X_te_full = scaler_tr.transform(df_te[feat_cols].astype(float))

    # Time-only index within feat_cols (may be absent if label_mode==pic)
    TIME_COL = "time_since_distraction_end"
    if TIME_COL in feat_cols:
        t_idx = feat_cols.index(TIME_COL)
        X_ca_time = X_ca_full[:, [t_idx]]
        X_te_time = X_te_full[:, [t_idx]]
    else:
        LOG.warning(
            "'%s' not in feat_cols (model was trained without it). "
            "Using raw column for B0.", TIME_COL
        )
        # --- FIX: impute NaN with 0 and do not scale ---
        time_ca = df_ca[[TIME_COL]].astype(float).fillna(0.0).values
        time_te = df_te[[TIME_COL]].astype(float).fillna(0.0).values
        X_ca_time = time_ca
        X_te_time = time_te
        LOG.info("Time feature after imputation: %d samples, %.2f%% zeros in cal",
                 len(time_ca), 100 * (time_ca.ravel() == 0).mean())

    # Physio-only: every production feature except time_since
    physio_idx = [i for i, c in enumerate(feat_cols) if c != TIME_COL]
    X_ca_physio = X_ca_full[:, physio_idx]
    X_te_physio = X_te_full[:, physio_idx]

    LOG.info("Feature counts — time: %d  physio: %d  full: %d",
             X_ca_time.shape[1], X_ca_physio.shape[1], X_ca_full.shape[1])
    # ── 5. HMM FTD score on test (using reconstructed model) ─────────────────
    b_ca = ftd_hmm.sequence_bounds(df_ca)
    b_te = ftd_hmm.sequence_bounds(df_te)

    gamma_ca = np.zeros((len(df_ca), n_states))
    gamma_te = np.zeros((len(df_te), n_states))
    for s, e in b_ca:
        if e > s:
            gamma_ca[s:e] = hmm_model.predict_proba(X_ca_full[s:e])
    for s, e in b_te:
        if e > s:
            gamma_te[s:e] = hmm_model.predict_proba(X_te_full[s:e])

    ftd_te = ftd_hmm.compute_risk_score(
        gamma_te, risk_order, risk_weights, risk_intercept)

    # ── 6. Fit baselines on CAL, score on TEST ────────────────────────────────
    # B0 — Time-only
    LOG.info("Fitting B0 (time-only)…")
    lr_b0 = fit_lr_baseline(X_ca_time, target_ca, seed=args.seed)
    b0_te = lr_b0.predict_proba(X_te_time)[:, 1]

    # B1 — Physio-only (no time_since)
    LOG.info("Fitting B1 (physio-only)…")
    lr_b1 = fit_lr_baseline(X_ca_physio, target_ca, seed=args.seed)
    b1_te = lr_b1.predict_proba(X_te_physio)[:, 1]

    # B2 — Full features (time + physio, no HMM)
    LOG.info("Fitting B2 (full-LR)…")
    lr_b2 = fit_lr_baseline(X_ca_full, target_ca, seed=args.seed)
    b2_te = lr_b2.predict_proba(X_te_full)[:, 1]

    # ── 7. Evaluate all models ────────────────────────────────────────────────
    LOG.info("─── Test-set metrics ───────────────────────────────────────────")
    results = [
        {**score_model("HMM FTD",       ftd_te, target_te), "proba": ftd_te},
        {**score_model("B0 Time-only",  b0_te,  target_te), "proba": b0_te},
        {**score_model("B1 Physio-only",b1_te,  target_te), "proba": b1_te},
        {**score_model("B2 Full-LR",    b2_te,  target_te), "proba": b2_te},
    ]

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    plot_roc(results, target_te,               out_dir / "roc_curves.png")
    plot_pr( results, target_te,               out_dir / "pr_curves.png")
    plot_calibration_curves(results, target_te, out_dir / "calibration_curves.png")
    plot_delta_bar(results,                    out_dir / "delta_auc_bar.png")

    # ── 9. Save comparison table ──────────────────────────────────────────────
    table = [
        {k: v for k, v in m.items() if k != "proba"}
        for m in results
    ]

    # Compute incremental lift columns
    b0_auc = next(m for m in table if m["name"] == "B0 Time-only")["auc_roc"]
    b0_ap  = next(m for m in table if m["name"] == "B0 Time-only")["avg_precision"]
    for m in table:
        m["delta_auc_vs_time"] = round(m["auc_roc"]      - b0_auc, 4)
        m["delta_ap_vs_time"]  = round(m["avg_precision"] - b0_ap,  4)

    out_json = out_dir / "comparison_table.json"
    with open(out_json, "w") as f:
        json.dump(table, f, indent=2)
    LOG.info("Comparison table → %s", out_json)

    # ── 10. Pretty-print summary ──────────────────────────────────────────────
    LOG.info("─── Summary (ΔAUC vs B0 time-only) ────────────────────────────")
    header = f"{'Model':<22}  {'AUC-ROC':>8}  {'Avg-P':>8}  {'Cal-r':>7}  {'ΔAUC':>7}  {'ΔAP':>7}"
    LOG.info(header)
    for m in table:
        LOG.info(
            "%-22s  %8.4f  %8.4f  %7.4f  %+7.4f  %+7.4f",
            m["name"], m["auc_roc"], m["avg_precision"],
            m["calibration_r"], m["delta_auc_vs_time"], m["delta_ap_vs_time"],
        )

    return table


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="HMM vs time-only baseline comparison")
    p.add_argument("--artifact",    default="result_no_time/impairment_hmm.joblib",
                   help="Path to impairment_hmm.joblib")
    p.add_argument("--data-path",  default="data")
    p.add_argument("--output-dir", default="result_no_time/baseline_comparison/")
    p.add_argument("--seed",       type=int, default=42)
    raise SystemExit(run(p.parse_args()))