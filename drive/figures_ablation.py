"""
figures_ablation.py — KinTCN hyperparameter ablation: L × H grid.

Metrics are recomputed from raw per-driver predictions restricted to the
maximum common driver subset across all (L, H) cells, so every cell is
evaluated on the same cohort.

Saves to impairment_results/figures/fig_ablation.pdf / .png
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

RESULTS_DIR = Path(__file__).parent / "impairment_results"
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEED       = 42
N_BOOTSTRAP = 2000

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi":      150,
})

# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0; n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo)
        if not mask.any(): continue
        ece += mask.sum() / n * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def bootstrap_auroc_ci(driver_aucs, n_boot=N_BOOTSTRAP, seed=SEED):
    aucs = np.array(driver_aucs); n = len(aucs)
    if n < 2: return float("nan"), float("nan")
    rng  = np.random.default_rng(seed)
    boot = [rng.choice(aucs, n, replace=True).mean() for _ in range(n_boot)]
    return tuple(np.percentile(boot, [2.5, 97.5]))


def recompute_metrics(raw, common_drivers):
    """
    Given a loaded JSON dict and a set of driver IDs, filter predictions to
    common_drivers and return recomputed auroc_mean, auroc_std, auprc, ece.
    """
    pred_map = {p["driver"]: p for p in raw["predictions"]}
    driver_aucs, all_y, all_s = [], [], []

    for d in sorted(common_drivers):
        p = pred_map[d]
        y = np.array(p["y_true"], dtype=float)
        s = np.array(p["kin_cal"], dtype=float)
        all_y.append(y); all_s.append(s)
        if len(np.unique(y)) == 2:
            driver_aucs.append(roc_auc_score(y, s))

    all_y = np.concatenate(all_y)
    all_s = np.concatenate(all_s)

    auroc_mean = float(np.mean(driver_aucs)) if driver_aucs else float("nan")
    auroc_std  = float(np.std(driver_aucs))  if driver_aucs else float("nan")
    auprc      = float(average_precision_score(all_y, all_s)) if len(np.unique(all_y)) == 2 else float("nan")
    ece        = compute_ece(all_y, all_s)

    return auroc_mean, auroc_std, auprc, ece


# ── Load all results ───────────────────────────────────────────────────────────

raw_data = {}
for f in RESULTS_DIR.glob("kin_tcn_L*_H*_results.json"):
    m = re.search(r"L(\d+)_H(\d+)", f.name)
    if m:
        L, H = int(m.group(1)), int(m.group(2))
        raw_data[(L, H)] = json.load(open(f))

Ls = sorted({k[0] for k in raw_data})
Hs = sorted({k[1] for k in raw_data})

# ── Maximum common driver subset ───────────────────────────────────────────────

driver_sets = [
    {p["driver"] for p in raw_data[(L, H)]["predictions"]}
    for L in Ls for H in Hs
]
common_drivers = set.intersection(*driver_sets)
print(f"Common driver subset: {len(common_drivers)} drivers")
print(f"  {sorted(common_drivers)}")

# ── Recompute metrics for each cell ───────────────────────────────────────────

cell = {}
for L in Ls:
    for H in Hs:
        auroc_mean, auroc_std, auprc, ece = recompute_metrics(raw_data[(L, H)], common_drivers)
        cell[(L, H)] = {
            "auroc_mean": auroc_mean,
            "auroc_std":  auroc_std,
            "auprc":      auprc,
            "ece":        ece,
        }
        print(f"  L={L:2d} H={H:2d}  AUROC={auroc_mean:.4f}±{auroc_std:.4f}  "
              f"AUPRC={auprc:.4f}  ECE={ece:.4f}")

# ── Best config by AUROC ───────────────────────────────────────────────────────

auroc_grid = np.array([[cell[(L, H)]["auroc_mean"] for H in Hs] for L in Ls])
best_i, best_j = np.unravel_index(np.argmax(auroc_grid), auroc_grid.shape)
best_L, best_H = Ls[best_i], Hs[best_j]
print(f"\nBest config: L={best_L} s, H={best_H} s  →  AUROC={auroc_grid[best_i, best_j]:.4f}")

# ── Figure ────────────────────────────────────────────────────────────────────

grid  = np.array([[cell[(L, H)]["auroc_mean"] for H in Hs] for L in Ls])
sgrid = np.array([[cell[(L, H)]["auroc_std"]  for H in Hs] for L in Ls])

fig, ax = plt.subplots(figsize=(5, 4.0))

vmin, vmax = grid.min() - 0.005, grid.max() + 0.005
im   = ax.imshow(grid, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("AUROC", fontsize=10)

for i in range(len(Ls)):
    for j in range(len(Hs)):
        brightness = (grid[i, j] - vmin) / (vmax - vmin)
        txt_color  = "white" if brightness > 0.6 else "black"
        ax.text(j, i, f"{grid[i, j]:.2f}\n±{sgrid[i, j]:.2f}",
                ha="center", va="center", fontsize=9, color=txt_color, linespacing=1.4)

mi, mj = np.unravel_index(np.argmax(grid), grid.shape)
ax.add_patch(mpatches.FancyBboxPatch(
    (mj - 0.48, mi - 0.48), 0.96, 0.96,
    boxstyle="square,pad=0", fill=False,
    edgecolor="#D55E00", linewidth=2.5, zorder=7))

ax.set_xticks(range(len(Hs))); ax.set_xticklabels([f"{h} s" for h in Hs])
ax.set_yticks(range(len(Ls))); ax.set_yticklabels([f"{l} s" for l in Ls])
ax.set_xlabel("Labelling horizon  $H$")
ax.set_ylabel("Lookback duration  $L$")
ax.set_title(f"AUROC — common cohort: {len(common_drivers)} drivers", pad=8)

fig.tight_layout()

out = FIG_DIR / "fig_ablation.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved {out}")
