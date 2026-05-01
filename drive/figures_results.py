"""
figures_results.py — Main results figures for the paper.

Figures produced (saved to impairment_results/figures/):
  fig_main_results.pdf   — AUROC + ECE bar chart vs baselines
  fig_roc.pdf            — pooled ROC curves for all models
  fig_per_driver.pdf     — per-driver AUROC scatter (KinTCN+Cal vs XGBoost)
                           with Wilcoxon annotation
  fig_perm_importance.pdf — permutation feature importance (mean ΔAUROC)
  fig_audit.pdf          — audit driver FPR at threshold 0.5
"""

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "impairment_results"
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    11,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
})

# Colorblind-friendly palette (Wong 2011)
C_LR       = "#E69F00"   # orange          — LSTM baseline
C_XGB      = "#56B4E9"   # sky blue        — XGBoost baseline
C_TCN      = "#009E73"   # green           — KinTCN raw
C_NOATT    = "#D55E00"   # vermillion      — KinTCN no-attention raw
C_NOATT_CAL= "#CC79A7"   # reddish purple  — KinTCN no-attention + Cal
C_CAL      = "#0072B2"   # blue            — KinTCN+Cal

SIGNAL_LABELS = {
    "speed.x":            "Speed",
    "steeringWheelAngle": "Steering angle",
    "steeringTorq":       "Steering torque",
    "acceleration.y":     "Lateral acceleration",
}

# ── Load best config ───────────────────────────────────────────────────────────
BEST_L, BEST_H = 45, 10
best_path = RESULTS_DIR / f"kin_tcn_L{BEST_L}_H{BEST_H}_results.json"
with open(best_path) as f:
    best = json.load(f)

print(f"Best config: L={BEST_L} s, H={BEST_H} s")
print(f"  KinTCN+Cal AUROC : {best['pooled']['kin_cal']['auroc_mean']:.4f} "
      f"[{best['pooled']['kin_cal']['auroc_ci'][0]:.4f}, "
      f"{best['pooled']['kin_cal']['auroc_ci'][1]:.4f}]")
print(f"  XGBoost    AUROC : {best['pooled']['xgb']['auroc_mean']:.4f}")
print(f"  LSTM       AUROC : {best['pooled']['lstm']['auroc_mean']:.4f}")
print(f"  Eval drivers     : {len(best['per_driver'])}")
print(f"  Audit drivers    : {len(best['safe_driver_audit'])}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Main results: AUROC and ECE vs baselines
# ══════════════════════════════════════════════════════════════════════════════
def fig_main_results():
    p      = best["pooled"]
    models = ["LSTM", "XGBoost", "KinTCN", "KinTCN\n(no attn)", "KinTCN\n(no attn)+Cal", "KinTCN+Cal"]
    keys   = ["lstm", "xgb", "kin_tcn", "kin_noatt", "kin_noatt_cal", "kin_cal"]
    colors = [C_LR, C_XGB, C_TCN, C_NOATT, C_NOATT_CAL, C_CAL]
    x      = np.arange(len(models))
    width  = 0.55

    auroc_means = [p[k]["auroc_mean"]          for k in keys]
    auroc_lo    = [p[k]["auroc_mean"] - p[k]["auroc_ci"][0] for k in keys]
    auroc_hi    = [p[k]["auroc_ci"][1] - p[k]["auroc_mean"] for k in keys]
    ece_vals    = [p[k]["ece"]                 for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # AUROC
    ax = axes[0]
    ax.bar(x, auroc_means, width, color=colors, zorder=3)
    ax.errorbar(x, auroc_means, yerr=[auroc_lo, auroc_hi],
                fmt="none", capsize=4, capthick=1.2, elinewidth=1.2,
                ecolor="black", zorder=4)
    ax.axhline(0.5, color="grey", lw=1, ls="--", zorder=2, label="Random chance")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("(a) Discrimination", pad=6)
    ax.legend(fontsize=9)

    # ECE
    ax = axes[1]
    ax.bar(x, ece_vals, width, color=colors, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("ECE  (lower is better)")
    ax.set_title("(b) Calibration", pad=6)

    fig.suptitle(f"Model comparison  ($L={BEST_L}$~s, $H={BEST_H}$~s, $G=5$~s)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "fig_main_results.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


# ── Shared helper: mean curve across drivers ───────────────────────────────────

def _mean_roc(preds, score_key, grid_size=300):
    """Per-driver ROC curves interpolated onto a common FPR grid, then averaged."""
    fpr_grid = np.linspace(0, 1, grid_size)
    tprs = []
    aucs = []
    for d in preds:
        y = np.array(d["y_true"], dtype=float)
        s = np.array(d[score_key], dtype=float)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        tprs.append(np.interp(fpr_grid, fpr, tpr))
        aucs.append(auc(fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
    return fpr_grid, mean_tpr, float(np.mean(aucs))


def _mean_pr(preds, score_key, grid_size=300):
    """Per-driver PR curves interpolated onto a common recall grid, then averaged."""
    rec_grid = np.linspace(0, 1, grid_size)
    precs = []
    aps   = []
    baselines = []
    for d in preds:
        y = np.array(d["y_true"], dtype=float)
        s = np.array(d[score_key], dtype=float)
        if len(np.unique(y)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y, s)
        # precision_recall_curve returns decreasing recall; flip for interp
        precs.append(np.interp(rec_grid, rec[::-1], prec[::-1]))
        aps.append(average_precision_score(y, s))
        baselines.append(float(y.mean()))
    mean_prec = np.mean(precs, axis=0)
    return rec_grid, mean_prec, float(np.mean(aps)), float(np.mean(baselines))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Mean ROC curves (averaged across drivers)
# ══════════════════════════════════════════════════════════════════════════════
def fig_roc():
    if "predictions" not in best:
        print("Skipping fig_roc: 'predictions' key missing.")
        return

    preds   = best["predictions"]
    keys    = {
        "LSTM":              "lstm",
        "XGBoost":           "xgb",
        "KinTCN":            "kin_tcn",
        "KinTCN (no attn)":  "kin_noatt",
        "KinTCN (no attn)+Cal": "kin_noatt_cal",
        "KinTCN+Cal":        "kin_cal",
    }
    colors  = {
        "LSTM":                 C_LR,
        "XGBoost":              C_XGB,
        "KinTCN":               C_TCN,
        "KinTCN (no attn)":     C_NOATT,
        "KinTCN (no attn)+Cal": C_NOATT_CAL,
        "KinTCN+Cal":           C_CAL,
    }
    ls_map  = {
        "LSTM":                 ":",
        "XGBoost":              "--",
        "KinTCN":               "-.",
        "KinTCN (no attn)":     (0, (3, 1, 1, 1)),
        "KinTCN (no attn)+Cal": (0, (5, 2)),
        "KinTCN+Cal":           "-",
    }

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color="lightgrey", lw=1, ls="--", zorder=1)

    for name, key in keys.items():
        fpr_grid, mean_tpr, mean_auc = _mean_roc(preds, key)
        ax.plot(fpr_grid, mean_tpr, color=colors[name], lw=2, ls=ls_map[name], zorder=3,
                label=f"{name}  (AUC = {mean_auc:.3f})")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Mean ROC curves  ($L={BEST_L}$~s, $H={BEST_H}$~s)", pad=6)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    fig.tight_layout()
    out = FIG_DIR / "fig_roc.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Mean Precision-Recall curves (averaged across drivers)
# ══════════════════════════════════════════════════════════════════════════════
def fig_pr():
    if "predictions" not in best:
        print("Skipping fig_pr: 'predictions' key missing.")
        return

    preds   = best["predictions"]
    keys    = {"LSTM": "lstm", "XGBoost": "xgb", "KinTCN": "kin_tcn", "KinTCN+Cal": "kin_cal"}
    colors  = {"LSTM": C_LR, "XGBoost": C_XGB, "KinTCN": C_TCN, "KinTCN+Cal": C_CAL}
    ls_map  = {"LSTM": ":", "XGBoost": "--", "KinTCN": "-.", "KinTCN+Cal": "-"}

    # compute baseline from first model (same for all)
    _, _, _, baseline = _mean_pr(preds, "kin_cal")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axhline(baseline, color="lightgrey", lw=1, ls="--", zorder=1,
               label=f"Random  (AP = {baseline:.3f})")

    for name, key in keys.items():
        rec_grid, mean_prec, mean_ap, _ = _mean_pr(preds, key)
        ax.plot(rec_grid, mean_prec, color=colors[name], lw=2, ls=ls_map[name], zorder=3,
                label=f"{name}  (AP = {mean_ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Mean Precision-Recall curves  ($L={BEST_L}$~s, $H={BEST_H}$~s)", pad=6)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    fig.tight_layout()
    out = FIG_DIR / "fig_pr.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Per-driver AUROC scatter with Wilcoxon annotation
# ══════════════════════════════════════════════════════════════════════════════
def fig_per_driver():
    drivers   = best["per_driver"]
    lstm_aucs = np.array([d["auc_lstm"] for d in drivers])
    cal_aucs  = np.array([d["auc_cal"]  for d in drivers])
    pos_rates = np.array([d["pos_rate"] for d in drivers])

    # Wilcoxon
    diffs = cal_aucs - lstm_aucs
    if np.any(diffs != 0):
        stat, pval = wilcoxon(cal_aucs.tolist(), lstm_aucs.tolist(), alternative="two-sided")
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s."))
    else:
        stat, pval, sig = 0.0, 1.0, "n.s."

    n_wins = int((cal_aucs > lstm_aucs).sum())

    fig, ax = plt.subplots(figsize=(5, 5))
    lims = (0.3, 1.02)
    ax.plot(lims, lims, color="grey", lw=1, ls="--", zorder=1)

    sc = ax.scatter(lstm_aucs, cal_aucs, c=pos_rates, cmap="YlOrRd",
                    s=60, zorder=3, edgecolors="white", linewidths=0.5,
                    vmin=0, vmax=0.25)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Positive rate", fontsize=10)

    # Wilcoxon annotation
    ax.text(0.04, 0.97,
            f"KinTCN+Cal $>$ LSTM: {n_wins}/{len(cal_aucs)} drivers\n"
            f"Wilcoxon  $W={stat:.0f}$,  $p={pval:.3f}$  ({sig})",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgrey", alpha=0.9))

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("LSTM  AUROC")
    ax.set_ylabel("KinTCN+Cal  AUROC")
    ax.set_title(f"Per-driver AUROC  ($L={BEST_L}$~s, $H={BEST_H}$~s)", pad=6)

    fig.tight_layout()
    out = FIG_DIR / "fig_per_driver.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out}")
    print(f"  Wilcoxon: W={stat:.1f}  p={pval:.4f}  {sig}  "
          f"(mean gain {diffs.mean():+.4f} ± {diffs.std():.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Permutation feature importance
# ══════════════════════════════════════════════════════════════════════════════
def fig_perm_importance():
    if "perm_importance" not in best:
        print("Skipping fig_perm_importance: 'perm_importance' key missing.")
        return

    pi     = best["perm_importance"]
    labels = [SIGNAL_LABELS.get(k, k) for k in pi]
    means  = [pi[k]["mean_delta_auroc"] for k in pi]
    stds   = [pi[k]["std_delta_auroc"]  for k in pi]

    order  = np.argsort(means)
    labels = [labels[i] for i in order]
    means  = [means[i]  for i in order]
    stds   = [stds[i]   for i in order]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    y = np.arange(len(labels))
    ax.barh(y, means, color=C_CAL, zorder=3,
            xerr=stds, error_kw=dict(capsize=4, capthick=1.2, elinewidth=1.2, ecolor="black"))
    ax.axvline(0, color="grey", lw=0.8, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Mean $\\Delta$AUROC when permuted")
    ax.set_title("Permutation feature importance  (KinTCN+Cal)", pad=6)

    fig.tight_layout()
    out = FIG_DIR / "fig_perm_importance.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Audit driver FPR
# ══════════════════════════════════════════════════════════════════════════════
def fig_audit():
    audit = best["safe_driver_audit"]
    if not audit:
        print("Skipping fig_audit: no audit drivers.")
        return

    drivers = [d["driver"]           for d in audit]
    probs   = [d["mean_prob_neg_cal"] for d in audit]
    n_pos   = [d["n_pos"]            for d in audit]

    order   = np.argsort(probs)[::-1]
    drivers = [drivers[i] for i in order]
    probs   = [probs[i]   for i in order]
    n_pos   = [n_pos[i]   for i in order]

    x      = np.arange(len(drivers))
    colors = ["#CC79A7" if n > 0 else C_CAL for n in n_pos]

    fig, ax = plt.subplots(figsize=(0.45 * len(drivers) + 1.5, 4.5))
    ax.bar(x, probs, color=colors, width=0.6, zorder=3)
    ax.axhline(0.5, color="#D55E00", lw=1.5, ls="--", zorder=4, label="Alert threshold (0.5)")

    ax.set_xticks(x); ax.set_xticklabels(drivers, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean calibrated impairment probability\n(negative windows)")
    ax.set_ylim(0, 0.55)
    ax.set_title("Audit drivers: false alarm profile", pad=6)

    patch_zero = mpatches.Patch(color=C_CAL,    label="0 positive windows")
    patch_few  = mpatches.Patch(color="#CC79A7", label="1–4 positive windows")
    ax.legend(handles=[patch_zero, patch_few,
                        mpatches.Patch(color="#D55E00", label="Alert threshold (0.5)")],
              fontsize=9, loc="upper right")

    fig.tight_layout()
    out = FIG_DIR / "fig_audit.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig_main_results()
    fig_roc()
    fig_pr()
    fig_per_driver()
    fig_perm_importance()
    fig_audit()
    print(f"\nAll figures saved to {FIG_DIR}")
