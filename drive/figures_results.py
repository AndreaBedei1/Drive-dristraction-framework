"""
figures_results.py — Generate all figures for the Results section.

Figures produced (saved to impairment_results/figures/):
  fig_main_results.pdf    — AUROC + ECE + Brier comparison across models (best config)
  fig_roc.pdf             — ROC curves: LR / XGBoost / KinTCN / KinTCN+Cal (best config)
  fig_per_driver.pdf      — per-driver scatter: KinTCN+Cal vs XGBoost AUROC
  fig_perm_importance.pdf — permutation feature importance (mean ΔAUROC per signal)
  fig_ablation.pdf        — L × H ablation line plot (AUROC + ECE)
  fig_audit.pdf           — audit driver calibrated probabilities vs alert threshold

Requires JSON files produced by tcnKin_30/45/60.py with the updated save block
(fields: predictions, perm_importance). Re-run the training scripts to
regenerate the JSON files with these fields before running this script.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve, auc

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
C_LR  = "#E69F00"   # orange  — LR baseline
C_XGB = "#56B4E9"   # sky blue — XGBoost baseline
C_TCN = "#009E73"   # green   — KinTCN raw
C_CAL = "#0072B2"   # blue    — KinTCN+Cal

# ── Load all result files ──────────────────────────────────────────────────────
def load_all():
    results = {}
    for f in sorted(RESULTS_DIR.glob("kin_tcn_L*_H*_results.json")):
        m = re.search(r"L(\d+)_H(\d+)", f.name)
        if m:
            L, H = int(m.group(1)), int(m.group(2))
            with open(f) as fh:
                results[(L, H)] = json.load(fh)
    return results

all_results = load_all()

# Best config: highest KinTCN+Cal AUROC across the ablation grid
BEST_L, BEST_H = max(
    all_results,
    key=lambda k: all_results[k]["pooled"]["kin_cal"]["auroc_mean"]
)
best = all_results[(BEST_L, BEST_H)]
print(f"Auto-selected best config: L={BEST_L} s, H={BEST_H} s  "
      f"(AUROC={best['pooled']['kin_cal']['auroc_mean']:.4f})")

# ── Wilcoxon: KinTCN+Cal vs XGBoost ───────────────────────────────────────────
xgb_aucs = [r["auc_xgb"] for r in best["per_driver"]
            if r["auc_xgb"] is not None and r["auc_cal"] is not None]
cal_aucs  = [r["auc_cal"] for r in best["per_driver"]
            if r["auc_xgb"] is not None and r["auc_cal"] is not None]
diffs = np.array(cal_aucs) - np.array(xgb_aucs)
if np.any(diffs != 0):
    stat, pval = wilcoxon(cal_aucs, xgb_aucs, alternative="two-sided")
    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
    print(f"\nWilcoxon KinTCN+Cal vs XGBoost:")
    print(f"  W={stat:.1f}  p={pval:.4f}  {sig}")
    print(f"  Mean gain: {diffs.mean():+.4f} ± {diffs.std():.4f}")
    print(f"  KinTCN+Cal > XGBoost: {(diffs>0).sum()}/{len(diffs)} drivers")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Main results: AUROC and ECE across models
# ══════════════════════════════════════════════════════════════════════════════
def fig_main_results():
    p = best["pooled"]
    models = ["LR", "XGBoost", "KinTCN", "KinTCN+Cal"]
    keys   = ["lr", "xgb", "kin_tcn", "kin_cal"]
    colors = [C_LR, C_XGB, C_TCN, C_CAL]

    auroc_means = [p[k]["auroc_mean"] for k in keys]
    auroc_lo    = [p[k]["auroc_mean"] - p[k]["auroc_ci"][0] for k in keys]
    auroc_hi    = [p[k]["auroc_ci"][1] - p[k]["auroc_mean"] for k in keys]
    ece_vals    = [p[k]["ece"]         for k in keys]
    brier_vals  = [p[k]["brier"]       for k in keys]

    x      = np.arange(len(models))
    width  = 0.55
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=False)

    # AUROC
    ax = axes[0]
    bars = ax.bar(x, auroc_means, width, color=colors, zorder=3,
                  error_kw=dict(capsize=4, capthick=1.2, elinewidth=1.2, ecolor="black"))
    ax.errorbar(x, auroc_means,
                yerr=[auroc_lo, auroc_hi],
                fmt="none", capsize=4, capthick=1.2, elinewidth=1.2, ecolor="black", zorder=4)
    ax.axhline(0.5, color="grey", lw=1, ls="--", zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("(a) Discrimination", pad=6)

    # ECE
    ax = axes[1]
    ax.bar(x, ece_vals, width, color=colors, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("ECE  (lower is better)")
    ax.set_title("(b) Calibration — ECE", pad=6)

    # Brier
    ax = axes[2]
    ax.bar(x, brier_vals, width, color=colors, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Brier score  (lower is better)")
    ax.set_title("(c) Calibration — Brier", pad=6)

    fig.suptitle(f"Model comparison  (L={BEST_L} s, H={BEST_H} s, G=5 s)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "fig_main_results.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — ROC curves (pooled across drivers, best config)
# ══════════════════════════════════════════════════════════════════════════════
def fig_roc():
    if "predictions" not in best:
        print("Skipping fig_roc: 'predictions' key missing — re-run training scripts.")
        return

    preds   = best["predictions"]
    y_true  = np.concatenate([np.array(d["y_true"])  for d in preds])
    scores  = {
        "LR":         np.concatenate([np.array(d["lr"])      for d in preds]),
        "XGBoost":    np.concatenate([np.array(d["xgb"])     for d in preds]),
        "KinTCN":     np.concatenate([np.array(d["kin_tcn"]) for d in preds]),
        "KinTCN+Cal": np.concatenate([np.array(d["kin_cal"]) for d in preds]),
    }
    colors  = {"LR": C_LR, "XGBoost": C_XGB, "KinTCN": C_TCN, "KinTCN+Cal": C_CAL}
    ls_map  = {"LR": ":", "XGBoost": "--", "KinTCN": "-.", "KinTCN+Cal": "-"}

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="lightgrey", lw=1, ls="--", zorder=1)

    for name, score in scores.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[name], lw=2,
                ls=ls_map[name], zorder=3,
                label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves  (L={BEST_L} s, H={BEST_H} s)", pad=6)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    fig.tight_layout()
    out = FIG_DIR / "fig_roc.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Per-driver AUROC scatter: KinTCN+Cal vs XGBoost
# ══════════════════════════════════════════════════════════════════════════════
def fig_per_driver():
    drivers = best["per_driver"]
    xgb = np.array([d["auc_xgb"] for d in drivers])
    cal = np.array([d["auc_cal"] for d in drivers])
    pos = np.array([d["pos_rate"] for d in drivers])

    fig, ax = plt.subplots(figsize=(5, 5))
    lims = (0.3, 1.02)
    ax.plot(lims, lims, color="grey", lw=1, ls="--", zorder=1, label="Equal performance")

    sc = ax.scatter(xgb, cal, c=pos, cmap="YlOrRd", s=60, zorder=3,
                    edgecolors="white", linewidths=0.5, vmin=0, vmax=0.22)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Positive rate", fontsize=10)

    n_wins = int((cal > xgb).sum())
    ax.text(0.04, 0.97, f"KinTCN+Cal > XGBoost: {n_wins}/{len(cal)} drivers",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgrey", alpha=0.8))

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("XGBoost  AUROC")
    ax.set_ylabel("KinTCN+Cal  AUROC")
    ax.set_title(f"Per-driver AUROC  (L={BEST_L} s, H={BEST_H} s)", pad=6)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    out = FIG_DIR / "fig_per_driver.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Permutation feature importance
# ══════════════════════════════════════════════════════════════════════════════
SIGNAL_LABELS = {
    "speed.x":             "Speed",
    "steeringWheelAngle":  "Steering angle",
    "steeringTorq":        "Steering torque",
    "acceleration.y":      "Lateral acceleration",
}

def fig_perm_importance():
    if "perm_importance" not in best:
        print("Skipping fig_perm_importance: 'perm_importance' key missing — re-run training scripts.")
        return

    pi     = best["perm_importance"]
    labels = [SIGNAL_LABELS.get(k, k) for k in pi]
    means  = [pi[k]["mean_delta_auroc"] for k in pi]
    stds   = [pi[k]["std_delta_auroc"]  for k in pi]

    order  = np.argsort(means)[::-1]
    labels = [labels[i] for i in order]
    means  = [means[i]  for i in order]
    stds   = [stds[i]   for i in order]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(labels))
    ax.bar(x, means, color=C_CAL, zorder=3,
           yerr=stds, error_kw=dict(capsize=4, capthick=1.2, elinewidth=1.2, ecolor="black"))
    ax.axhline(0, color="grey", lw=0.8, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean $\\Delta$AUROC when permuted")
    ax.set_title("Permutation feature importance  (KinTCN+Cal)", pad=6)

    fig.tight_layout()
    out = FIG_DIR / "fig_perm_importance.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Ablation: L × H → AUROC (KinTCN+Cal vs XGBoost) + ECE + ΔAUROC
# ══════════════════════════════════════════════════════════════════════════════
def fig_ablation():
    Ls       = sorted({k[0] for k in all_results})
    Hs       = sorted({k[1] for k in all_results})
    colors_H = ["#0072B2", "#E69F00", "#009E73"]
    ls_xgb   = "--"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=False)

    # ── Panel (a): AUROC — KinTCN+Cal and XGBoost, with std error bars ─────────
    ax = axes[0]
    for H, col in zip(Hs, colors_H):
        Ls_av  = [L for L in Ls if (L, H) in all_results]
        cal_m  = [all_results[(L,H)]["pooled"]["kin_cal"]["auroc_mean"] for L in Ls_av]
        cal_s  = [all_results[(L,H)]["pooled"]["kin_cal"]["auroc_std"]  for L in Ls_av]
        xgb_m  = [all_results[(L,H)]["pooled"]["xgb"]["auroc_mean"]     for L in Ls_av]
        xgb_s  = [all_results[(L,H)]["pooled"]["xgb"]["auroc_std"]      for L in Ls_av]
        ax.errorbar(Ls_av, cal_m, yerr=cal_s, marker="o", color=col, lw=2, ms=6,
                    capsize=3, label=f"KinTCN+Cal  $H$={H} s")
        ax.errorbar(Ls_av, xgb_m, yerr=xgb_s, marker="s", color=col, lw=1.5,
                    ms=5, capsize=3, ls=ls_xgb, alpha=0.6,
                    label=f"XGBoost  $H$={H} s")
        if BEST_H == H and BEST_L in Ls_av:
            idx = Ls_av.index(BEST_L)
            ax.plot(BEST_L, cal_m[idx], marker="*", color="#D55E00", ms=14, zorder=5)

    ax.set_xticks(Ls); ax.set_xticklabels([f"{l} s" for l in Ls])
    ax.set_xlabel("Lookback duration  $L$")
    ax.set_ylabel("AUROC  (mean ± std)")
    ax.set_title("(a) Discrimination", pad=6)

    # ── Panel (b): ΔAUROC (KinTCN+Cal − XGBoost) ───────────────────────────────
    ax = axes[1]
    for H, col in zip(Hs, colors_H):
        Ls_av = [L for L in Ls if (L, H) in all_results]
        delta = [all_results[(L,H)]["pooled"]["kin_cal"]["auroc_mean"] -
                 all_results[(L,H)]["pooled"]["xgb"]["auroc_mean"]
                 for L in Ls_av]
        ax.plot(Ls_av, delta, marker="o", color=col, lw=2, ms=6,
                label=f"$H$ = {H} s")
        if BEST_H == H and BEST_L in Ls_av:
            idx = Ls_av.index(BEST_L)
            ax.plot(BEST_L, delta[idx], marker="*", color="#D55E00", ms=14, zorder=5)

    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(Ls); ax.set_xticklabels([f"{l} s" for l in Ls])
    ax.set_xlabel("Lookback duration  $L$")
    ax.set_ylabel("$\\Delta$AUROC  (KinTCN+Cal $-$ XGBoost)")
    ax.set_title("(b) Gain over XGBoost", pad=6)

    # ── Panel (c): ECE — KinTCN+Cal only ───────────────────────────────────────
    ax = axes[2]
    for H, col in zip(Hs, colors_H):
        Ls_av = [L for L in Ls if (L, H) in all_results]
        ece   = [all_results[(L,H)]["pooled"]["kin_cal"]["ece"] for L in Ls_av]
        ax.plot(Ls_av, ece, marker="o", color=col, lw=2, ms=6, label=f"$H$ = {H} s")
        if BEST_H == H and BEST_L in Ls_av:
            idx = Ls_av.index(BEST_L)
            ax.plot(BEST_L, ece[idx], marker="*", color="#D55E00", ms=14, zorder=5)

    ax.set_xticks(Ls); ax.set_xticklabels([f"{l} s" for l in Ls])
    ax.set_xlabel("Lookback duration  $L$")
    ax.set_ylabel("ECE  (lower is better)")
    ax.set_title("(c) Calibration", pad=6)
    ax.legend(fontsize=8.5, loc="upper right")

    # shared legend for panels (a) and (b)
    cal_line = plt.Line2D([0],[0], color="grey", lw=2,  marker="o", ms=5, label="KinTCN+Cal (solid)")
    xgb_line = plt.Line2D([0],[0], color="grey", lw=1.5, marker="s", ms=4, ls="--", alpha=0.6, label="XGBoost (dashed)")
    star_p   = plt.Line2D([0],[0], color="#D55E00", marker="*", ms=10, lw=0, label="Selected config")
    axes[0].legend(fontsize=7.5, ncol=2, loc="lower right")
    axes[1].legend(handles=[cal_line, xgb_line, star_p], fontsize=8.5)

    fig.suptitle("Ablation: lookback duration $L$ and labelling horizon $H$",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "fig_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Audit driver false alarm profile
# ══════════════════════════════════════════════════════════════════════════════
def fig_audit():
    audit = best["safe_driver_audit"]
    drivers  = [d["driver"] for d in audit]
    probs    = [d["mean_prob_neg_cal"] for d in audit]
    n_pos    = [d["n_pos"] for d in audit]

    # Sort by mean probability descending for readability
    order   = np.argsort(probs)[::-1]
    drivers = [drivers[i] for i in order]
    probs   = [probs[i]   for i in order]
    n_pos   = [n_pos[i]   for i in order]

    y = np.arange(len(drivers))
    fig, ax = plt.subplots(figsize=(5.5, 0.38 * len(drivers) + 1.2))

    colors_d = ["#CC79A7" if n > 0 else C_CAL for n in n_pos]
    ax.barh(y, probs, color=colors_d, height=0.6, zorder=3)
    ax.axvline(0.5, color="#D55E00", lw=1.5, ls="--", zorder=4, label="Alert threshold (0.5)")

    ax.set_yticks(y); ax.set_yticklabels(drivers, fontsize=9)
    ax.set_xlabel("Mean calibrated impairment probability (negative windows)")
    ax.set_title("Audit drivers: false alarm profile", pad=6)
    ax.set_xlim(0, 0.55)

    patch_zero = mpatches.Patch(color=C_CAL,    label="0 positive windows")
    patch_few  = mpatches.Patch(color="#CC79A7", label="1–4 positive windows")
    ax.legend(handles=[patch_zero, patch_few,
                        mpatches.Patch(color="#D55E00", label="Alert threshold")],
              fontsize=9, loc="lower right")

    fig.tight_layout()
    out = FIG_DIR / "fig_audit.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Best config: L={BEST_L} s, H={BEST_H} s")
    print(f"  KinTCN+Cal AUROC: {best['pooled']['kin_cal']['auroc_mean']:.4f} "
          f"[{best['pooled']['kin_cal']['auroc_ci'][0]:.4f}, "
          f"{best['pooled']['kin_cal']['auroc_ci'][1]:.4f}]")
    print(f"  XGBoost    AUROC: {best['pooled']['xgb']['auroc_mean']:.4f}")
    print(f"  LR         AUROC: {best['pooled']['lr']['auroc_mean']:.4f}")
    print(f"  Evaluable drivers: {len(best['per_driver'])}")
    print(f"  Audit drivers    : {len(best['safe_driver_audit'])}")

    fig_main_results()
    fig_roc()
    fig_per_driver()
    fig_perm_importance()
    fig_ablation()
    fig_audit()

    print(f"\nAll figures saved to {FIG_DIR}")
