#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Threading constraints for stability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOOKBACK_S = 15    # 15 seconds of context is plenty for HR and Steering
WINDOW_STEP = 1    # Slide 1 second at a time to create a dense dataset
AROUSAL_DELTA_THRESH = 0.01 # Slightly lower to catch more subtle stress shifts
DATA = "relab+unibo_dataset.csv"

# Signals and Errors
SIGNAL_COLS = ["arousal", "hr", "speed.x", "steeringWheelAngle"]
ERR_COLS = ["Collision", "Red_light_violation", "panic_braking", "panic_braking_with_stop", "sharp_turn"]

MIN_POSITIVES = 8
MIN_WINDOWS = 30
CALIB_FRAC = 0.3
CALIB_EPOCHS = 10
CALIB_LR = 1e-4

# ----------------------------
# Dataset & Loss
# ----------------------------
class DrivingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, targets):
        # Ensure targets are float for BCE
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

# ----------------------------
# Models
# ----------------------------
class DrivingAutoencoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, in_channels, 5, padding=2),
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        z = self.encoder(x)
        return self.decoder(z).permute(0, 2, 1)

class HybridTemporalCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.gru = nn.GRU(128, 64, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        z = self.net(x).permute(0, 2, 1)
        _, hidden = self.gru(z)
        return self.head(hidden[-1]).squeeze(-1)

# ----------------------------
# Core Logic
# ----------------------------
def build_windows(df):
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df = df.sort_values(["id", "route", "Timestamp"])
    windows, labels, pids = [], [], []
    for (pid, route), grp in df.groupby(["id", "route"]):
        grp = grp.reset_index(drop=True)
        arousal = grp["arousal"].values.astype(np.float32)
        trend_mask = np.abs(np.diff(arousal, prepend=arousal[0])) > AROUSAL_DELTA_THRESH
        idx = 0
        while idx + LOOKBACK_S < len(grp):
            win_idx = np.arange(idx, idx + LOOKBACK_S)
            if not trend_mask[win_idx].any(): 
                idx += WINDOW_STEP; continue
            sig = grp.iloc[win_idx][SIGNAL_COLS].values.astype(np.float32)
            if np.isnan(sig).any(): 
                idx += WINDOW_STEP; continue
            future = grp.iloc[idx + LOOKBACK_S : idx + LOOKBACK_S + 5]
            label = int((future[ERR_COLS] > 0).any().any())
            windows.append(sig); labels.append(label)
            pids.append(pid); idx += WINDOW_STEP
    return np.stack(windows), np.array(labels), np.array(pids)

def augment_positives(X, y, multiplier=2):
    X_pos, y_pos = X[y == 1], y[y == 1]
    if len(X_pos) == 0: return X, y
    aug_X, aug_y = [X], [y]
    for _ in range(multiplier):
        Xa = X_pos.copy() + np.random.normal(0, 0.02, X_pos.shape)
        aug_X.append(Xa); aug_y.append(y_pos)
    return np.concatenate(aug_X), np.concatenate(aug_y)

def train_ae(X_train_normal):
    model = DrivingAutoencoder(len(SIGNAL_COLS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(DrivingDataset(X_train_normal, np.zeros(len(X_train_normal))), batch_size=64, shuffle=True)
    for _ in range(15):
        model.train()
        for x, _ in loader:
            x = x.to(DEVICE)
            loss = F.mse_loss(model(x), x)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model

def train_hybrid_cnn(X_train, y_train):
    X_aug, y_aug = augment_positives(X_train, y_train)
    model = HybridTemporalCNN(in_channels=X_train.shape[2]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = FocalLoss()
    loader = DataLoader(DrivingDataset(X_aug, y_aug), batch_size=64, shuffle=True)
    for _ in range(20):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model

# ----------------------------
# Main Pipeline
# ----------------------------
def main():
    df = pd.read_csv(DATA)
    X_all, y_all, pid_all = build_windows(df)
    
    eligible = []
    for p in np.unique(pid_all):
        mask = pid_all == p
        if y_all[mask].sum() >= MIN_POSITIVES and len(y_all[mask]) >= MIN_WINDOWS:
            eligible.append(p)

    results = []
    for p in eligible:
        print(f"\n>>> Driver {p}")
        tr_m, te_m = pid_all != p, pid_all == p
        X_tr, y_tr = X_all[tr_m], y_all[tr_m]
        X_te, y_te = X_all[te_m], y_all[te_m]

        # Normalization
        scaler = StandardScaler()
        X_tr_f = scaler.fit_transform(X_tr.reshape(-1, len(SIGNAL_COLS))).reshape(X_tr.shape)
        X_te_f = scaler.transform(X_te.reshape(-1, len(SIGNAL_COLS))).reshape(X_te.shape)

        # 1. AE Feature Generation
        ae = train_ae(X_tr_f[y_tr == 0])
        def add_ae(data, model):
            model.eval()
            with torch.no_grad():
                t = torch.tensor(data).to(DEVICE)
                err = torch.mean((t - model(t))**2, dim=2, keepdim=True)
                return torch.cat([t, err], dim=2).cpu().numpy()

        X_tr_h = add_ae(X_tr_f, ae)
        X_te_h = add_ae(X_te_f, ae)

        # 2. Hybrid CNN
        model = train_hybrid_cnn(X_tr_h, y_tr)
        
        # 3. Personalization Split
        n_cal = int(len(X_te_h) * CALIB_FRAC)
        X_cal, y_cal = X_te_h[:n_cal], y_te[:n_cal]
        X_ev, y_ev = X_te_h[n_cal:], y_te[n_cal:]
        
        if y_cal.sum() > 0 and y_ev.sum() > 0:
            # Freeze body, tune head
            for param in model.net.parameters(): param.requires_grad = False
            opt = torch.optim.Adam(model.head.parameters(), lr=CALIB_LR)
            crit = FocalLoss()
            
            for _ in range(CALIB_EPOCHS):
                model.train()
                lx = torch.tensor(X_cal).to(DEVICE)
                ly = torch.tensor(y_cal).float().to(DEVICE) # Ensure float here
                loss = crit(model(lx), ly)
                opt.zero_grad(); loss.backward(); opt.step()
            
            model.eval()
            with torch.no_grad():
                preds = torch.sigmoid(model(torch.tensor(X_ev).to(DEVICE))).cpu().numpy()
                auc = roc_auc_score(y_ev, preds)
                ap = average_precision_score(y_ev, preds)
                print(f"   [RESULT] AUC: {auc:.3f}, AP: {ap:.3f}")
                results.append((auc, ap))

    if results:
        m_auc = np.mean([r[0] for r in results])
        m_ap = np.mean([r[1] for r in results])
        print(f"\nFINAL HYBRID LOPO: AUC={m_auc:.3f}, AP={m_ap:.3f}")

if __name__ == "__main__":
    main()