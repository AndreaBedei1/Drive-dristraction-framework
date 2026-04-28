"""
figures_ablation.py — KinTCN hyperparameter ablation: L × H grid.
Saves to impairment_results/figures/fig_ablation.pdf
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent / "impairment_results"
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi":      150,
})

# ── Load ───────────────────────────────────────────────────────────────────────
data = {}
for f in RESULTS_DIR.glob("kin_tcn_L*_H*_results.json"):
    m = re.search(r"L(\d+)_H(\d+)", f.name)
    if m:
        L, H = int(m.group(1)), int(m.group(2))
        data[(L, H)] = json.load(open(f))

Ls = sorted({k[0] for k in data})   # rows
Hs = sorted({k[1] for k in data})   # cols

metrics = [
    ("auroc_mean", "auroc_std", "AUROC (higher is better)",  "Blues",   True),
    ("auprc",      None,        "AUPRC (higher is better)",  "Greens",  True),
    ("ece",        None,        "ECE (lower is better)",     "Oranges", False),
]

# Best config by AUROC (for reference)
auroc_grid = np.array([[data[(L,H)]["pooled"]["kin_cal"]["auroc_mean"] for H in Hs] for L in Ls])
best_i, best_j = np.unravel_index(np.argmax(auroc_grid), auroc_grid.shape)
best_L, best_H = Ls[best_i], Hs[best_j]
print(f"Best config by AUROC: L={best_L} s, H={best_H} s  →  AUROC={auroc_grid[best_i,best_j]:.4f}")

# ── Figure: three heatmaps side by side ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

for ax, (mean_key, std_key, label, cmap, higher_better) in zip(axes, metrics):
    grid = np.array([[data[(L,H)]["pooled"]["kin_cal"][mean_key] for H in Hs] for L in Ls])
    if std_key:
        sgrid = np.array([[data[(L,H)]["pooled"]["kin_cal"][std_key] for H in Hs] for L in Ls])
    else:
        sgrid = None

    vmin, vmax = grid.min() - 0.005, grid.max() + 0.005
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label.split(" (")[0], fontsize=10)

    for i in range(len(Ls)):
        for j in range(len(Hs)):
            brightness = (grid[i, j] - vmin) / (vmax - vmin)
            txt_color  = "white" if brightness > 0.6 else "black"
            val_str = f"{grid[i,j]:.2f}"
            if sgrid is not None:
                val_str += f"\n±{sgrid[i,j]:.2f}"
            ax.text(j, i, val_str, ha="center", va="center",
                    fontsize=9, color=txt_color, linespacing=1.4)

    # Best config marker: argmax or argmin depending on metric direction
    fn = np.argmax if higher_better else np.argmin
    mi, mj = np.unravel_index(fn(grid), grid.shape)
    ax.add_patch(mpatches.FancyBboxPatch(
        (mj - 0.48, mi - 0.48), 0.96, 0.96,
        boxstyle="square,pad=0", fill=False,
        edgecolor="#D55E00", linewidth=2.5, zorder=7))

    ax.set_xticks(range(len(Hs))); ax.set_xticklabels([f"{h} s" for h in Hs])
    ax.set_yticks(range(len(Ls))); ax.set_yticklabels([f"{l} s" for l in Ls])
    ax.set_xlabel("Labelling horizon  $H$")
    ax.set_ylabel("Lookback duration  $L$")
    ax.set_title(label, pad=8)


fig.suptitle("KinTCN ablation: lookback duration $L$ × labelling horizon $H$",
             fontsize=12, y=1.01)
fig.tight_layout()
out = FIG_DIR / "fig_ablation.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved {out}")
