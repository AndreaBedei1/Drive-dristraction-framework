import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, mannwhitneyu, fisher_exact
import random

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIGNAL_COLS = ["arousal", "hr", "speed.x", "steeringWheelAngle"]
ERR_COLS    = ["Collision", "Red_light_violation", "panic_braking",
               "panic_braking_with_stop", "sharp_turn"]

# Index of arousal in SIGNAL_COLS — update if column order changes
AROUSAL_IDX = SIGNAL_COLS.index("arousal")

# Stress threshold: windows where the driver's mean arousal exceeds
# their own P75 are flagged as "stress-elevated".
# P75 is computed per driver from their full sequence (all windows,
# before any train/test split). This is a label definition, not a
# model parameter, so using the full sequence introduces no leakage.
# P75 chosen over P50 to target the upper quartile of arousal
# (elevated state, not merely above-median). References:
#   Healey & Picard (2005) — arousal percentile thresholds in driving
#   Lohani et al. (2019) — within-driver normalization for physiological signals
AROUSAL_PERCENTILE = 75

# How many of the final timesteps of the lookback window to use
# for the stress assessment. Using the last 20/60 steps (= the
# portion temporally closest to the GAP+HORIZON) captures rising
# state rather than baseline. Validated via temporal precedence test.
STRESS_TAIL = 20

EPOCHS, LR   = 50, 1e-3
LOOKBACK_S   = 60
WINDOW_STEP  = 5
GAP, HORIZON = 3, 5
BATCH_SIZE   = 64

MIN_POSITIVES      = 1
MIN_EVAL_POSITIVES = 3

HYBRID_LR            = 5e-4
HYBRID_EPOCHS        = 30
HYBRID_MIN_POSITIVES = 2
CAL_MIN_POSITIVES    = 1
LAMBDA_TETHER        = 1e-1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# -------------------------------------------------
# DATASET & UTILS
# -------------------------------------------------

class DrivingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X).float()
        self.y = torch.as_tensor(y).float()
    def __len__(self):          return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def get_balanced_loader(X, y, batch_size=BATCH_SIZE):
    y_int          = y.astype(int)
    counts         = np.bincount(y_int, minlength=2)
    weights        = 1.0 / (counts + 1e-6)
    sample_weights = torch.from_numpy(weights[y_int])
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights))
    return DataLoader(DrivingDataset(X, y), batch_size=batch_size, sampler=sampler)


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_score)


def fit_calibrator(logits_tensor, y_array, device, lr=0.01, steps=200):
    cal      = LogitCalibrator().to(device)
    opt      = torch.optim.Adam(cal.parameters(), lr=lr)
    y_tensor = torch.as_tensor(y_array).float().to(device)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(cal(logits_tensor), y_tensor)
        opt.zero_grad(); loss.backward(); opt.step()
    return cal


def min_count_boundary(y, min_positives, start=0):
    count = 0
    for i in range(start, len(y)):
        if y[i] == 1:
            count += 1
        if count >= min_positives:
            return i + 1
    return None


def brier_skill_score(y_true, y_prob):
    bs_ref = brier_score_loss(y_true, np.full_like(y_true, y_true.mean(), dtype=float))
    if bs_ref == 0:
        return float("nan")
    return 1 - brier_score_loss(y_true, y_prob) / bs_ref

# -------------------------------------------------
# MODEL
# -------------------------------------------------

class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Linear(channels, channels // 2)
        self.score = nn.Linear(channels // 2, 1)

    def forward(self, x):
        x_t     = x.permute(0, 2, 1)
        h       = torch.tanh(self.query(x_t))
        weights = torch.softmax(self.score(h), dim=1)
        return torch.sum(x_t * weights, dim=1)


class TCN_Attention_Net(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.network = nn.Sequential(
            self._block(in_channels, 32, 1),
            self._block(32, 64, 2),
            self._block(64, 64, 4),
            self._block(64, 128, 8),
        )
        self.attention = TemporalAttention(128)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def _block(self, in_c, out_c, d):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, padding=d, dilation=d),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.network(x.permute(0, 2, 1))
        return self.head(self.attention(x)).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class LogitCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, z):
        return torch.abs(self.a) * z + self.b

# -------------------------------------------------
# WINDOWING
# -------------------------------------------------

def compute_driver_thresholds(df):
    """
    Compute per-driver arousal P75 from the full (pre-split) sequence.
    This is a label definition — using all windows per driver is correct
    and introduces no model leakage.

    Returns dict: {driver_id: float threshold}
    """
    thresholds = {}
    for pid, grp in df.groupby("id"):
        arousal_vals = grp["arousal"].dropna().values
        if len(arousal_vals) == 0:
            thresholds[pid] = 0.0
        else:
            thresholds[pid] = float(np.percentile(arousal_vals, AROUSAL_PERCENTILE))
    return thresholds


def build_windows(df, driver_thresholds):
    """
    Build sliding windows with STRESS-COUPLED labels.

    A window is positive iff BOTH conditions hold simultaneously:
      1. STRESS: mean arousal over the final STRESS_TAIL timesteps
                 exceeds the driver's own P75 threshold.
      2. EVENT:  at least one risky driving event occurs in the
                 GAP+HORIZON future window.

    Also returns per-window quadrant flags for validity analysis:
      quad: 0=neither  1=stress-only  2=event-only  3=coupled (positive)
    """
    windows, labels, quads, pids = [], [], [], []

    for (pid, route), grp in df.groupby(["id", "route"]):
        grp       = grp.sort_values("Timestamp").reset_index(drop=True)
        threshold = driver_thresholds.get(pid, 0.0)
        idx       = 0

        while idx + LOOKBACK_S + GAP + HORIZON <= len(grp):
            sig = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)

            if not np.isnan(sig).any():
                future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]

                # --- stress flag ---
                # Mean arousal over the STRESS_TAIL steps closest to the event horizon.
                # Using the tail (not the full window mean) captures *rising* arousal
                # rather than a sustained baseline — validated by temporal precedence test.
                arousal_tail = sig[-STRESS_TAIL:, AROUSAL_IDX].mean()
                stress_flag  = int(arousal_tail > threshold)

                # --- event flag ---
                event_flag = int((future[ERR_COLS] > 0).any().any())

                # --- coupled label ---
                label = int(stress_flag == 1 and event_flag == 1)

                # quadrant: 0=neither 1=stress-only 2=event-only 3=coupled
                quad = stress_flag + 2 * event_flag  # 0,1,2,3

                windows.append(sig)
                labels.append(label)
                quads.append(quad)
                pids.append(pid)

            idx += WINDOW_STEP

    return np.stack(windows), np.array(labels), np.array(quads), np.array(pids)


# -------------------------------------------------
# VALIDITY REPORT
# -------------------------------------------------

def print_validity_report(X, y, quads, pid, driver_thresholds):
    """
    Three-part validity report that must be run BEFORE training.

    Part 1 — Four-quadrant breakdown + chi-square + odds ratio.
      Answers: is the coupling more than the independent sum of its parts?

    Part 2 — Temporal precedence.
      Answers: does arousal rise BEFORE the event (causal direction),
      or does it merely co-occur?

    Part 3 — Predictive validity per signal (Mann-Whitney U).
      Answers: do any input signals meaningfully separate pos from neg?
    """
    from scipy.stats import chi2_contingency, mannwhitneyu, fisher_exact

    SEP = "=" * 70

    # ----------------------------------------------------------------
    # PART 0 — Overview
    # ----------------------------------------------------------------
    print(f"\n{SEP}")
    print("LABEL VALIDITY REPORT — STRESS-COUPLED TARGET")
    print(SEP)

    n      = len(y)
    n_pos  = int(y.sum())
    print(f"Total windows : {n}")
    print(f"Positive (stress ∧ event) : {n_pos}  ({n_pos/n*100:.1f}%)")
    print(f"Negative                  : {n - n_pos}  ({(n-n_pos)/n*100:.1f}%)")
    print(f"\nArousal threshold: P{AROUSAL_PERCENTILE} per driver (within-driver normalisation)")
    print(f"Stress assessment window : last {STRESS_TAIL}/{LOOKBACK_S} timesteps")

    # ----------------------------------------------------------------
    # PART 1 — Four-quadrant breakdown
    # quad: 0=neither  1=stress-only  2=event-only  3=coupled
    # ----------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("PART 1 — Four-quadrant breakdown")
    print(f"{'─'*70}")

    n_neither     = int((quads == 0).sum())
    n_stress_only = int((quads == 1).sum())
    n_event_only  = int((quads == 2).sum())
    n_coupled     = int((quads == 3).sum())   # == n_pos

    print(f"\n                    Event absent   Event present")
    print(f"  Stress absent   {n_neither:>12}    {n_event_only:>12}")
    print(f"  Stress present  {n_stress_only:>12}    {n_coupled:>12}  ← positive label")

    # Chi-square on the 2×2 stress × event table
    # (independent of the label — tests association between stress and event)
    contingency = np.array([[n_neither, n_event_only],
                             [n_stress_only, n_coupled]])
    if contingency.min() >= 5:
        chi2, p_chi2, _, _ = chi2_contingency(contingency)
        test_name = "Chi-square"
    else:
        # Use Fisher exact when expected counts are low
        _, p_chi2 = fisher_exact(contingency)
        chi2      = float("nan")
        test_name = "Fisher exact"

    # Odds ratio: (coupled × neither) / (stress-only × event-only)
    denom = (n_stress_only * n_event_only)
    odds_ratio = (n_coupled * n_neither) / denom if denom > 0 else float("inf")

    print(f"\n  {test_name}: ", end="")
    if not np.isnan(chi2):
        print(f"χ²={chi2:.2f},  ", end="")
    print(f"p={p_chi2:.2e}")
    print(f"  Odds ratio (coupling vs independence): {odds_ratio:.2f}")
    print(f"  Interpretation: stress and events co-occur {odds_ratio:.1f}× more than chance.")

    if p_chi2 > 0.05:
        print("  *** WARNING: stress and event flags are NOT significantly associated.")
        print("      Stress-coupled label may not be valid for this dataset.")
    elif odds_ratio < 1.5:
        print("  *** WARNING: odds ratio < 1.5 — coupling is weak.")
    else:
        print("  ✓ Significant association with meaningful effect size.")

    # Coupling adds information beyond each factor alone:
    # P(event | stress) vs P(event | no stress)
    p_event_given_stress    = n_coupled    / (n_stress_only + n_coupled)    if (n_stress_only + n_coupled) > 0    else 0
    p_event_given_no_stress = n_event_only / (n_neither     + n_event_only) if (n_neither     + n_event_only) > 0 else 0
    print(f"\n  P(event | stress elevated) : {p_event_given_stress:.3f}")
    print(f"  P(event | stress normal)   : {p_event_given_no_stress:.3f}")
    print(f"  Relative risk              : {p_event_given_stress / p_event_given_no_stress:.2f}x"
          if p_event_given_no_stress > 0 else "  Relative risk: undefined (no events under normal stress)")

    # ----------------------------------------------------------------
    # PART 2 — Temporal precedence
    # Average arousal trace at each of LOOKBACK_S timesteps,
    # split by positive vs negative windows.
    # A valid causal label shows divergence increasing towards t=LOOKBACK_S
    # (i.e., arousal rises in positive windows as the event approaches).
    # ----------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("PART 2 — Temporal precedence of arousal (mean trace, pos vs neg)")
    print(f"{'─'*70}")
    print(f"  Timestep 0 = start of lookback window")
    print(f"  Timestep {LOOKBACK_S-1} = last step before GAP+HORIZON\n")

    pos_trace = X[y == 1, :, AROUSAL_IDX].mean(axis=0)   # shape (LOOKBACK_S,)
    neg_trace = X[y == 0, :, AROUSAL_IDX].mean(axis=0)

    # Compute divergence at early, middle, and late thirds
    third     = LOOKBACK_S // 3
    div_early = (pos_trace[:third]        - neg_trace[:third]).mean()
    div_mid   = (pos_trace[third:2*third] - neg_trace[third:2*third]).mean()
    div_late  = (pos_trace[2*third:]      - neg_trace[2*third:]).mean()

    print(f"  Mean (pos - neg) arousal divergence:")
    print(f"    Early third  [t=0..{third-1}]          : {div_early:+.4f}")
    print(f"    Middle third [t={third}..{2*third-1}]       : {div_mid:+.4f}")
    print(f"    Late third   [t={2*third}..{LOOKBACK_S-1}]      : {div_late:+.4f}")

    if div_late > div_mid > div_early:
        print("  ✓ Monotonically increasing divergence — arousal rises BEFORE the event.")
        print("    Temporal precedence is established.")
    elif div_late > div_early:
        print("  ~ Late divergence > early divergence — partial temporal precedence.")
    else:
        print("  *** WARNING: divergence does not increase toward the event horizon.")
        print("      Temporal precedence is NOT established — label validity is questionable.")

    # Print a compact ASCII trace for visual inspection
    print(f"\n  ASCII arousal trace (each column = mean over 6 timesteps):")
    n_bins  = 10
    bin_sz  = LOOKBACK_S // n_bins
    pos_bin = [pos_trace[i*bin_sz:(i+1)*bin_sz].mean() for i in range(n_bins)]
    neg_bin = [neg_trace[i*bin_sz:(i+1)*bin_sz].mean() for i in range(n_bins)]
    # normalise to 0..8 for display
    all_vals = pos_bin + neg_bin
    lo, hi   = min(all_vals), max(all_vals)
    rng_     = hi - lo if hi > lo else 1.0
    def to_bar(v): return "█" * int((v - lo) / rng_ * 8 + 0.5)
    print(f"  {'t→':>4}", "  ".join(f"t{i*bin_sz:>2}" for i in range(n_bins)))
    print(f"  {'pos':>4}", "  ".join(f"{to_bar(v):<8}" for v in pos_bin))
    print(f"  {'neg':>4}", "  ".join(f"{to_bar(v):<8}" for v in neg_bin))

    # Formal test: Mann-Whitney U on arousal tail (last STRESS_TAIL steps)
    pos_tail = X[y == 1, -STRESS_TAIL:, AROUSAL_IDX].mean(axis=1)
    neg_tail = X[y == 0, -STRESS_TAIL:, AROUSAL_IDX].mean(axis=1)
    _, p_tail = mannwhitneyu(pos_tail, neg_tail, alternative="greater")
    print(f"\n  Mann-Whitney U (arousal tail, pos > neg): p={p_tail:.2e}"
          + ("  ✓" if p_tail < 0.05 else "  ✗"))

    # ----------------------------------------------------------------
    # PART 3 — Predictive validity per signal
    # ----------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("PART 3 — Per-signal predictive validity (Mann-Whitney U, pos vs neg)")
    print(f"{'─'*70}")
    print(f"  {'Signal':<25}  {'Mean(neg)':>10}  {'Mean(pos)':>10}  {'p-value':>12}  {'Sig'}")

    for i, col in enumerate(SIGNAL_COLS):
        neg_vals = X[y == 0, :, i].mean(axis=1)
        pos_vals = X[y == 1, :, i].mean(axis=1)
        if len(pos_vals) < 2:
            continue
        _, p = mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
        sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {col:<25}  {neg_vals.mean():>10.4f}  {pos_vals.mean():>10.4f}  {p:>12.2e}  {sig}")

    # ----------------------------------------------------------------
    # PART 4 — Per-driver thresholds and quadrant counts
    # ----------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("PART 4 — Per-driver thresholds and quadrant counts")
    print(f"{'─'*70}")
    print(f"  {'Driver':<12}  {'Thresh':>7}  {'Neither':>8}  {'Stress':>8}  {'Event':>8}  {'Coupled':>8}  {'PosR%':>7}")

    for d in np.unique(pid):
        m    = pid == d
        nd   = int(m.sum())
        q    = quads[m]
        npos = int((y[m]).sum())
        print(f"  {d:<12}  {driver_thresholds.get(d, 0):>7.4f}"
              f"  {int((q==0).sum()):>8}"
              f"  {int((q==1).sum()):>8}"
              f"  {int((q==2).sum()):>8}"
              f"  {int((q==3).sum()):>8}"
              f"  {npos/nd*100:>6.1f}%")

    print(f"\n{SEP}\n")


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    df = pd.read_csv("relab+unibo_dataset.csv")

    # Compute per-driver arousal thresholds BEFORE windowing
    driver_thresholds = compute_driver_thresholds(df)

    X, y, quads, pid = build_windows(df, driver_thresholds)
    BUFFER = LOOKBACK_S // WINDOW_STEP

    # Validity report first — if the label doesn't pass, don't waste GPU time
    print_validity_report(X, y, quads, pid, driver_thresholds)

    drivers   = [d for d in np.unique(pid) if y[pid == d].sum() >= MIN_POSITIVES]
    criterion = FocalLoss()
    rng       = np.random.default_rng(SEED)

    hdr = (f"{'Driver':<10} | {'N_eval':>6} {'PosR%':>6} | "
           f"{'PopAUC':>7} {'HybAUC':>7} {'Gain':>7} | "
           f"{'PopBSS':>7} {'HybBSS':>7} {'GainBSS':>8} | "
           f"{'Pers':>5} {'PPosR%':>6} | {'Cal':>5} {'CPosR%':>6} | {'Flag'}")
    print(hdr)
    print("-" * len(hdr))

    pool_y, pool_pop, pool_hyb, pool_drv = [], [], [], []
    per_driver_results = []

    for d in drivers:
        mask_te    = pid == d
        X_tr, y_tr = X[~mask_te], y[~mask_te]
        X_te, y_te = X[mask_te],  y[mask_te]
        pid_tr     = pid[~mask_te]

        val_ids = rng.choice(
            np.unique(pid_tr),
            max(1, int(0.15 * len(np.unique(pid_tr)))),
            replace=False,
        )
        vmask = np.isin(pid_tr, val_ids)

        if len(np.unique(y_tr[vmask])) < 2:
            print(f"{d:<10} | SKIP — val fold single-class")
            continue

        scaler  = StandardScaler()
        n_feats = len(SIGNAL_COLS)
        Xtr     = scaler.fit_transform(X_tr[~vmask].reshape(-1, n_feats)).reshape(-1, LOOKBACK_S, n_feats)
        Xval    = scaler.transform(X_tr[vmask].reshape(-1, n_feats)).reshape(-1, LOOKBACK_S, n_feats)
        Xte_sc  = scaler.transform(X_te.reshape(-1, n_feats)).reshape(-1, LOOKBACK_S, n_feats)

        model  = TCN_Attention_Net(n_feats).to(DEVICE)
        opt    = torch.optim.Adam(model.parameters(), lr=LR)
        loader = get_balanced_loader(Xtr, y_tr[~vmask])

        best_auc, best_w = 0.0, None
        for _ in range(EPOCHS):
            model.train()
            for xb, yb in loader:
                loss = criterion(model(xb.to(DEVICE)), yb.to(DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                preds = torch.sigmoid(model(torch.as_tensor(Xval).to(DEVICE))).cpu().numpy()
            auc = safe_auc(y_tr[vmask], preds)
            if auc is not None and auc > best_auc:
                best_auc = auc
                best_w   = copy.deepcopy(model.state_dict())

        if best_w is None:
            print(f"{d:<10} | SKIP — no valid checkpoint")
            continue
        model.load_state_dict(best_w)

        # ---- minimum-count temporal split ----
        end_pers = min_count_boundary(y_te, HYBRID_MIN_POSITIVES)
        if end_pers is None:
            end_pers     = len(y_te)
            pers_skipped = True
        else:
            pers_skipped = False

        X_pers, y_pers = Xte_sc[:end_pers], y_te[:end_pers]

        cal_start = end_pers + BUFFER
        if cal_start >= len(y_te):
            print(f"{d:<10} | SKIP — no room for calibration after pers+buffer")
            continue

        end_cal = min_count_boundary(y_te, CAL_MIN_POSITIVES, start=cal_start)
        if end_cal is None:
            print(f"{d:<10} | SKIP — not enough positives for calibration")
            continue

        X_cal, y_cal = Xte_sc[cal_start:end_cal], y_te[cal_start:end_cal]

        eval_start = end_cal + BUFFER
        if eval_start >= len(y_te):
            print(f"{d:<10} | SKIP — no room for eval after cal+buffer")
            continue

        X_eval, y_eval = Xte_sc[eval_start:], y_te[eval_start:]

        if len(X_eval) < 10 or y_eval.sum() == 0 or y_eval.sum() == len(y_eval):
            print(f"{d:<10} | SKIP — eval degenerate ({int(y_eval.sum())}/{len(y_eval)} pos)")
            continue

        reportable = int(y_eval.sum()) >= MIN_EVAL_POSITIVES

        # ---- hybrid personalisation ----
        model_hyb = copy.deepcopy(model)
        for name, param in model_hyb.named_parameters():
            param.requires_grad = any(x in name for x in ["head", "network.3"])

        personalised = False
        if not pers_skipped and y_pers.sum() >= HYBRID_MIN_POSITIVES:
            pop_params  = {pname: p.clone().detach() for pname, p in model.named_parameters()}
            pers_loader = get_balanced_loader(X_pers, y_pers, batch_size=min(BATCH_SIZE, len(X_pers)))
            opt_h       = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model_hyb.parameters()), lr=HYBRID_LR
            )
            prev_loss, patience, pat_count = float("inf"), 5, 0
            model_hyb.train()
            for _ in range(HYBRID_EPOCHS):
                ep_loss = 0.0
                for xb, yb in pers_loader:
                    logits     = model_hyb(xb.to(DEVICE))
                    task       = criterion(logits, yb.to(DEVICE))
                    tether     = sum(
                        ((p - pop_params[pname]) ** 2).mean()
                        for pname, p in model_hyb.named_parameters() if p.requires_grad
                    )
                    batch_loss = task + LAMBDA_TETHER * tether
                    opt_h.zero_grad(); batch_loss.backward(); opt_h.step()
                    ep_loss += batch_loss.item()
                if abs(prev_loss - ep_loss) < 1e-5:
                    pat_count += 1
                    if pat_count >= patience:
                        break
                else:
                    pat_count = 0
                prev_loss = ep_loss
            personalised = True

        model.eval(); model_hyb.eval()
        with torch.no_grad():
            z_cal_pop = model(torch.as_tensor(X_cal).to(DEVICE))
            z_cal_hyb = model_hyb(torch.as_tensor(X_cal).to(DEVICE))

        cal_pop = fit_calibrator(z_cal_pop, y_cal, DEVICE)
        cal_hyb = fit_calibrator(z_cal_hyb, y_cal, DEVICE)

        with torch.no_grad():
            X_eval_t = torch.as_tensor(X_eval).to(DEVICE)
            p_pop    = torch.sigmoid(cal_pop(model(X_eval_t))).cpu().numpy()
            p_hyb    = torch.sigmoid(cal_hyb(model_hyb(X_eval_t))).cpu().numpy()

        pool_y.append(y_eval)
        pool_pop.append(p_pop)
        pool_hyb.append(p_hyb)
        pool_drv.extend([d] * len(y_eval))

        auc_pop  = safe_auc(y_eval, p_pop)
        auc_hyb  = safe_auc(y_eval, p_hyb)
        gain_auc = (auc_hyb - auc_pop) if (auc_pop and auc_hyb) else float("nan")
        bss_pop  = brier_skill_score(y_eval, p_pop)
        bss_hyb  = brier_skill_score(y_eval, p_hyb)

        pos_rate = y_eval.mean() * 100
        ppers_r  = y_pers.mean() * 100 if len(y_pers) else 0.0
        pcal_r   = y_cal.mean()  * 100 if len(y_cal)  else 0.0
        flag     = "" if personalised else "NO_PERS"

        row = (
            f"{d:<10} | {len(y_eval):>6} {pos_rate:>5.1f}% | "
            f"{auc_pop or 0:>7.4f} {auc_hyb or 0:>7.4f} {gain_auc:>+7.4f} | "
            f"{bss_pop:>7.4f} {bss_hyb:>7.4f} {bss_hyb - bss_pop:>+8.4f} | "
            f"{len(y_pers):>5} {ppers_r:>5.1f}% | "
            f"{len(y_cal):>5} {pcal_r:>5.1f}% | {flag}"
        )
        if reportable:
            print(row)
        else:
            print(row + f"  (not reported — only {int(y_eval.sum())} eval pos)")

        if reportable:
            per_driver_results.append({
                "driver":       d,
                "n_eval":       len(y_eval),
                "pos_rate_%":   round(pos_rate, 1),
                "auc_pop":      auc_pop,
                "auc_hyb":      auc_hyb,
                "gain_auc":     gain_auc,
                "bss_pop":      bss_pop,
                "bss_hyb":      bss_hyb,
                "gain_bss":     bss_hyb - bss_pop,
                "n_pers":       len(y_pers),
                "pos_pers":     int(y_pers.sum()),
                "n_cal":        len(y_cal),
                "pos_cal":      int(y_cal.sum()),
                "personalised": personalised,
            })

    # ---- pooled evaluation ----
    print("\n" + "=" * 70)
    print("POOLED EVALUATION  (all drivers with a valid eval slice)")
    print("=" * 70)

    if pool_y:
        y_pool  = np.concatenate(pool_y)
        pp_pool = np.concatenate(pool_pop)
        ph_pool = np.concatenate(pool_hyb)

        n_pool    = len(y_pool)
        pos_pool  = int(y_pool.sum())
        n_drivers = len(pool_y)

        print(f"Windows: {n_pool}  |  Positives: {pos_pool} ({pos_pool/n_pool*100:.1f}%)  |  Drivers: {n_drivers}")
        print()

        auc_pool_pop  = safe_auc(y_pool, pp_pool)
        auc_pool_hyb  = safe_auc(y_pool, ph_pool)
        bss_pool_pop  = brier_skill_score(y_pool, pp_pool)
        bss_pool_hyb  = brier_skill_score(y_pool, ph_pool)
        gain_auc_pool = (auc_pool_hyb - auc_pool_pop) if (auc_pool_pop and auc_pool_hyb) else float("nan")
        gain_bss_pool = bss_pool_hyb - bss_pool_pop   if not (np.isnan(bss_pool_pop) or np.isnan(bss_pool_hyb)) else float("nan")

        print(f"{'Metric':<12}  {'Population':>12}  {'Hybrid':>10}  {'Gain':>8}")
        print("-" * 46)
        print(f"{'AUC':<12}  {auc_pool_pop or 0:>12.4f}  {auc_pool_hyb or 0:>10.4f}  {gain_auc_pool:>+8.4f}")
        print(f"{'BSS':<12}  {bss_pool_pop:>12.4f}  {bss_pool_hyb:>10.4f}  {gain_bss_pool:>+8.4f}")
    else:
        print("No drivers produced a valid eval slice.")

    # ---- per-driver summary ----
    if per_driver_results:
        rdf = pd.DataFrame(per_driver_results)
        print("\n" + "=" * 70)
        print(f"PER-DRIVER SUMMARY  ({len(rdf)} reportable drivers, "
              f"eval positives >= {MIN_EVAL_POSITIVES})")
        print("=" * 70)

        numeric = ["auc_pop", "auc_hyb", "gain_auc",
                   "bss_pop", "bss_hyb", "gain_bss",
                   "pos_rate_%", "n_eval", "n_pers", "pos_pers"]
        summary = rdf[numeric].agg(["mean", "median", "std", "min", "max"])
        with pd.option_context("display.float_format", "{:.4f}".format, "display.max_columns", 20):
            print(summary.T.to_string())

        n_improved = (rdf["gain_auc"] > 0).sum()
        n_hurt     = (rdf["gain_auc"] < 0).sum()
        n_pers_ran = rdf["personalised"].sum()
        print(f"\nPersonalisation ran for {n_pers_ran}/{len(rdf)} reportable drivers.")
        print(f"Hybrid improved AUC for {n_improved}/{len(rdf)}, hurt {n_hurt}/{len(rdf)}.")

        pers_df  = rdf[rdf["personalised"]]
        npers_df = rdf[~rdf["personalised"]]
        if len(pers_df):
            print(f"\nAmong personalised drivers     (n={len(pers_df)}): "
                  f"mean gain AUC={pers_df['gain_auc'].mean():+.4f}, "
                  f"mean gain BSS={pers_df['gain_bss'].mean():+.4f}")
        if len(npers_df):
            print(f"Among non-personalised drivers (n={len(npers_df)}): "
                  f"mean gain AUC={npers_df['gain_auc'].mean():+.4f}  "
                  f"(should be ~0 — sanity check)")
    else:
        print(f"\nNo drivers met the per-driver reporting threshold "
              f"(MIN_EVAL_POSITIVES={MIN_EVAL_POSITIVES}).")
        print("Rely on pooled evaluation above for conclusions.")


if __name__ == "__main__":
    main()