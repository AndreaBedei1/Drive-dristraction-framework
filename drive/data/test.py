#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

LOOKBACK_S = 60
WINDOW_STEP = 5
AROUSAL_DELTA_THRESH = 0.02

DATA = "relab+unibo_dataset.csv"

SIGNAL_COLS = [
    "arousal",
    "hr",
    "0.speed",
    "1.speed",
    "2.speed",
    "3.speed",
    "steeringWheelAngle"
]

ERR_COLS = [
    "Collision",
    "Red_light_violation",
    "panic_braking",
    "panic_braking_with_stop",
    "sharp_turn",
]

MIN_POSITIVES = 8
MIN_WINDOWS   = 30

# Personalization: fraction of each driver's data used for calibration
CALIB_FRAC    = 0.3
CALIB_EPOCHS  = 10
CALIB_LR      = 1e-4

# ----------------------------
# Window extraction
# ----------------------------

def build_windows(df):
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df = df.sort_values(["id", "route", "Timestamp"])

    windows, labels, pids = [], [], []

    for (pid, route), grp in df.groupby(["id", "route"]):
        grp = grp.reset_index(drop=True)
        arousal = grp["arousal"].values.astype(np.float32)

        delta = np.diff(arousal, prepend=arousal[0])
        trend_mask = np.abs(delta) > AROUSAL_DELTA_THRESH

        idx = 0
        while idx + LOOKBACK_S < len(grp):
            window_indices = np.arange(idx, min(idx + LOOKBACK_S, len(grp)))
            if not trend_mask[window_indices].any():
                idx += WINDOW_STEP
                continue

            win = grp.iloc[window_indices]
            sig = win[SIGNAL_COLS].values.astype(np.float32)
            if np.isnan(sig).any():
                idx += WINDOW_STEP
                continue

            future = grp.iloc[window_indices[-1]+1 : window_indices[-1]+6]
            label  = int((future[ERR_COLS] > 0).any(axis=1).any())

            windows.append(sig)
            labels.append(label)
            pids.append(pid)
            idx += WINDOW_STEP

    return np.stack(windows), np.array(labels), np.array(pids)

# ----------------------------
# Eligibility filtering
# ----------------------------

def driver_eligibility_report(X, y, pid, verbose=True):
    if verbose:
        print(f"{'Driver':<12} {'Windows':>8} {'Positives':>10} "
              f"{'Pos Rate':>10} {'Eligible':>10}")
        print("-" * 55)

    eligible, ineligible = [], []

    for p in np.unique(pid):
        mask    = pid == p
        y_p     = y[mask]
        n_win   = len(y_p)
        n_pos   = int(y_p.sum())
        pos_rate = y_p.mean()

        enough_pos  = n_pos   >= MIN_POSITIVES
        enough_win  = n_win   >= MIN_WINDOWS
        not_trivial = pos_rate <  0.5
        ok = enough_pos and enough_win and not_trivial

        if verbose:
            flag    = "✓" if ok else "✗"
            reasons = []
            if not enough_pos:  reasons.append(f"only {n_pos} positives")
            if not enough_win:  reasons.append(f"only {n_win} windows")
            if not not_trivial: reasons.append("pos_rate >= 0.5")
            print(f"{p:<12} {n_win:>8} {n_pos:>10} {pos_rate:>10.3f} "
                  f"{flag:>8}  {', '.join(reasons)}")

        (eligible if ok else ineligible).append(p)

    if verbose:
        print(f"\nEligible:   {len(eligible)} drivers: {eligible}")
        print(f"Ineligible: {len(ineligible)} drivers: {ineligible}\n")

    return eligible, ineligible

# ----------------------------
# Per-driver normalization
# ----------------------------

def normalize_per_driver(X, pid):
    Xn = np.zeros_like(X)
    for p in np.unique(pid):
        mask = pid == p
        data = X[mask]
        mean = data.mean(axis=(0, 1), keepdims=True)
        std  = data.std (axis=(0, 1), keepdims=True) + 1e-6
        Xn[mask] = (data - mean) / std
    return Xn

# ----------------------------
# Augmentation
# ----------------------------

def augment_positives(X, y, multiplier=2):
    X_pos, y_pos = X[y == 1], y[y == 1]
    aug_X, aug_y = [X], [y]

    for _ in range(multiplier):
        Xa  = X_pos.copy()
        Xa += np.random.normal(0, 0.05, Xa.shape)
        Xa  = np.roll(Xa, np.random.randint(-5, 5), axis=1)
        Xa *= np.random.uniform(0.9, 1.1)
        aug_X.append(Xa)
        aug_y.append(y_pos)

    return np.concatenate(aug_X), np.concatenate(aug_y)

# ----------------------------
# Dataset
# ----------------------------

class DrivingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()

    def __len__(self): return len(self.X)

    def __getitem__(self, i): return self.X[i], self.y[i]

# ----------------------------
# Focal Loss
# ----------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt    = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

# ----------------------------
# Temporal CNN
# ----------------------------

class TemporalCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        z = self.net(x).squeeze(-1)
        return self.head(z).squeeze(-1)

# ----------------------------
# Training (population model)
# ----------------------------

def train_model(X_tr, y_tr):
    X_tr, y_tr = augment_positives(X_tr, y_tr, multiplier=2)

    loader    = DataLoader(DrivingDataset(X_tr, y_tr),
                           batch_size=64, shuffle=True, num_workers=0)
    model     = TemporalCNN(len(SIGNAL_COLS)).to(DEVICE)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    opt       = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

    for epoch in range(30):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
        for x, y in pbar:
            x, y  = x.to(DEVICE), y.to(DEVICE)
            loss  = criterion(model(x), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

    return model

# ----------------------------
# Personalization (fine-tune head only)
# ----------------------------

def personalize_model(population_model, X_calib, y_calib):
    """
    Takes the trained population model and fine-tunes only the
    classification head on a small calibration set from the target driver.
    The CNN body is frozen — only the 2-layer head is updated.
    """
    import copy
    model = copy.deepcopy(population_model)

    # Freeze the CNN body, unfreeze only the head
    for param in model.net.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    if len(X_calib) == 0 or y_calib.sum() == 0:
        return model  # not enough data to calibrate, return as-is

    X_calib, y_calib = augment_positives(X_calib, y_calib, multiplier=2)

    loader    = DataLoader(DrivingDataset(X_calib, y_calib),
                           batch_size=min(32, len(X_calib)), shuffle=True, num_workers=0)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    opt       = torch.optim.Adam(model.head.parameters(),
                                 lr=CALIB_LR, weight_decay=1e-4)

    for epoch in range(CALIB_EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(x), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    return model

# ----------------------------
# Evaluation
# ----------------------------

def evaluate(model, X_te):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te), 512):
            batch = torch.tensor(X_te[i:i+512]).float().to(DEVICE)
            preds.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(preds)

# ----------------------------
# Baseline LOPO
# ----------------------------

def build_stat_features(X):
    feats = []
    for win in X:
        f = []
        for c in range(win.shape[1]):
            sig = win[:, c]
            f.extend([sig.mean(), sig.std(), sig.min(), sig.max()])
        feats.append(f)
    return np.array(feats)

def lopo_baseline(X, y, pid):
    Xf = build_stat_features(X)
    results = {}
    for p in np.unique(pid):
        tr, te      = pid != p, pid == p
        X_tr, y_tr  = Xf[tr], y[tr]
        X_te, y_te  = Xf[te], y[te]
        if y_te.sum() == 0: continue

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)
        clf    = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X_tr, y_tr)
        proba  = clf.predict_proba(X_te)[:, 1]

        results[p] = {
            "auc": roc_auc_score(y_te, proba),
            "ap":  average_precision_score(y_te, proba)
        }

    return results

# ----------------------------
# CNN LOPO + personalization
# ----------------------------

def lopo_cnn(X, y, pid):
    """
    For each driver:
      1. Train population model on all other drivers
      2. Evaluate population model on test driver  → population score
      3. Split test driver into calib (first 30%) + eval (last 70%)
      4. Fine-tune head on calib set               → personalized score
    """
    pop_results  = {}
    pers_results = {}

    for p in np.unique(pid):
        tr, te      = pid != p, pid == p
        X_tr, y_tr  = X[tr], y[tr]
        X_te, y_te  = X[te], y[te]
        if y_te.sum() == 0: continue

        # --- population model ---
        pop_model = train_model(X_tr, y_tr)
        proba_pop = evaluate(pop_model, X_te)
        pop_results[p] = {
            "auc": roc_auc_score(y_te, proba_pop),
            "ap":  average_precision_score(y_te, proba_pop)
        }

        # --- personalization ---
        n_calib      = max(1, int(len(X_te) * CALIB_FRAC))
        X_calib, y_calib = X_te[:n_calib], y_te[:n_calib]
        X_eval,  y_eval  = X_te[n_calib:], y_te[n_calib:]

        if y_eval.sum() == 0:
            # not enough positives in eval portion — skip personalization
            pers_results[p] = pop_results[p]
        else:
            pers_model  = personalize_model(pop_model, X_calib, y_calib)
            proba_pers  = evaluate(pers_model, X_eval)
            pers_results[p] = {
                "auc": roc_auc_score(y_eval, proba_pers),
                "ap":  average_precision_score(y_eval, proba_pers)
            }

    return pop_results, pers_results

# ----------------------------
# Unified comparison table
# ----------------------------

def compare_results(baseline, population, personalized):
    print(f"\n{'Driver':<12} {'Base AUC':>10} {'Pop AUC':>10} {'Pers AUC':>10} "
          f"{'Base AP':>10} {'Pop AP':>10} {'Pers AP':>10}")
    print("-" * 74)

    drivers = sorted(baseline.keys())
    for p in drivers:
        b  = baseline.get(p,      {"auc": float("nan"), "ap": float("nan")})
        po = population.get(p,    {"auc": float("nan"), "ap": float("nan")})
        pe = personalized.get(p,  {"auc": float("nan"), "ap": float("nan")})
        print(f"{p:<12} {b['auc']:>10.3f} {po['auc']:>10.3f} {pe['auc']:>10.3f} "
              f"{b['ap']:>10.3f} {po['ap']:>10.3f} {pe['ap']:>10.3f}")

    print("-" * 74)

    def mean_std(res, key):
        vals = [v[key] for v in res.values() if not np.isnan(v[key])]
        return np.mean(vals), np.std(vals)

    for label, res in [("Baseline",     baseline),
                       ("Population",   population),
                       ("Personalized", personalized)]:
        m_auc, s_auc = mean_std(res, "auc")
        m_ap,  s_ap  = mean_std(res, "ap")
        print(f"{label:<12} {'':>10} "
              f"AUC {m_auc:.3f} ± {s_auc:.3f}    "
              f"AP  {m_ap:.3f} ± {s_ap:.3f}")

# ----------------------------
# Main
# ----------------------------

def main():
    df = pd.read_csv(DATA)
    X, y, pid = build_windows(df)
    print(f"All windows: {X.shape}, positive rate: {y.mean():.3f}\n")

    eligible, _ = driver_eligibility_report(X, y, pid)

    mask       = np.isin(pid, eligible)
    X, y, pid  = X[mask], y[mask], pid[mask]
    print(f"After filtering: {X.shape}, {len(np.unique(pid))} drivers, "
          f"positive rate: {y.mean():.3f}\n")

    # Baseline
    print("Running baseline...")
    baseline_res = lopo_baseline(X, y, pid)

    # CNN population + personalization (on normalized data)
    Xn = normalize_per_driver(X, pid)
    print("\nRunning CNN (population + personalization)...")
    pop_res, pers_res = lopo_cnn(Xn, y, pid)

    # Print unified table
    compare_results(baseline_res, pop_res, pers_res)

if __name__ == "__main__":
    main()