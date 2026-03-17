#!/usr/bin/env python3
#V4
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

# ----------------------------
# CONFIGURATION
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOOKBACK_S = 60 
WINDOW_STEP = 5
DATA = "relab+unibo_dataset.csv"

SIGNAL_COLS = ["arousal", "hr", "speed.x", "steeringWheelAngle"]
ERR_COLS = ["Collision", "Red_light_violation", "panic_braking", "panic_braking_with_stop", "sharp_turn"]

MIN_POSITIVES = 5
CALIB_FRAC = 0.3 
CALIB_EPOCHS = 15
CALIB_LR = 1e-4

# ----------------------------
# CORE CLASSES
# ----------------------------
class DrivingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X).float()
        self.y = torch.as_tensor(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def get_balanced_loader(X, y, batch_size=64):
    y_int = y.astype(int)
    counts = np.bincount(y_int)
    weights = 1. / (counts + 1e-6)
    samples_weights = torch.from_numpy(weights[y_int])
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
    return DataLoader(DrivingDataset(X, y), batch_size=batch_size, sampler=sampler)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

# ----------------------------
# ARCHITECTURE
# ----------------------------
class DrivingAutoencoder(nn.Module):
    def __init__(self, in_channels=len(SIGNAL_COLS)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, in_channels, 3, padding=1),
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.decoder(self.encoder(x)).permute(0, 2, 1)

class HybridTemporalCNN(nn.Module):
    def __init__(self, in_channels=len(SIGNAL_COLS) + 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        z = self.net(x).squeeze(-1)
        return self.head(z).squeeze(-1)

# ----------------------------
# PROCESSING
# ----------------------------
def build_windows(df):
    windows, labels, pids = [], [], []
    for (pid, route), grp in df.groupby(["id", "route"]):
        grp = grp.sort_values("Timestamp").reset_index(drop=True)
        idx = 0
        while idx + LOOKBACK_S < len(grp):
            win_idx = np.arange(idx, idx + LOOKBACK_S)
            sig = grp.iloc[win_idx][SIGNAL_COLS].values.astype(np.float32)
            if np.isnan(sig).any(): 
                idx += WINDOW_STEP; continue
            future = grp.iloc[idx + LOOKBACK_S : idx + LOOKBACK_S + 5]
            labels.append(int((future[ERR_COLS] > 0).any().any()))
            windows.append(sig); pids.append(pid); idx += WINDOW_STEP
    return np.stack(windows), np.array(labels), np.array(pids)

def main():
    df = pd.read_csv(DATA)
    X_all, y_all, pid_all = build_windows(df)
    eligible = [p for p in np.unique(pid_all) if y_all[pid_all==p].sum() >= MIN_POSITIVES]

    print(f"{'Driver':<10} | {'Pop AUC':<8} {'Pers AUC':<8} | {'Pop AP':<8} {'Pers AP':<8}")
    print("-" * 75)

    for p in eligible:
        X_tr_raw, y_tr = X_all[pid_all != p], y_all[pid_all != p]
        X_te_raw, y_te = X_all[pid_all == p], y_all[pid_all == p]

        scaler_raw = StandardScaler()
        n_feats = len(SIGNAL_COLS)
        X_tr_s = scaler_raw.fit_transform(X_tr_raw.reshape(-1, n_feats)).reshape(X_tr_raw.shape)
        X_te_s = scaler_raw.transform(X_te_raw.reshape(-1, n_feats)).reshape(X_te_raw.shape)

        ae = DrivingAutoencoder(in_channels=n_feats).to(DEVICE)
        ae_opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
        ae_loader = DataLoader(DrivingDataset(X_tr_s[y_tr==0], X_tr_s[y_tr==0]), batch_size=64, shuffle=True)
        
        ae.train()
        for _ in range(10):
            for x, _ in ae_loader:
                loss = F.mse_loss(ae(x.to(DEVICE)), x.to(DEVICE))
                ae_opt.zero_grad(); loss.backward(); ae_opt.step()

        def get_err(data_s):
            ae.eval()
            with torch.no_grad():
                t = torch.as_tensor(data_s).to(DEVICE)
                return torch.mean((t - ae(t))**2, dim=2, keepdim=True).cpu().numpy()

        err_tr, err_te = get_err(X_tr_s), get_err(X_te_s)
        err_scaler = StandardScaler()
        err_scaler.fit(err_tr[y_tr == 0].reshape(-1, 1))
        err_tr_s = err_scaler.transform(err_tr.reshape(-1, 1)).reshape(err_tr.shape)
        err_te_s = err_scaler.transform(err_te.reshape(-1, 1)).reshape(err_te.shape)

        X_tr_h = np.concatenate([X_tr_s, err_tr_s], axis=2)
        X_te_h = np.concatenate([X_te_s, err_te_s], axis=2)

        cnn_pop = HybridTemporalCNN(in_channels=n_feats+1).to(DEVICE)
        cnn_opt = torch.optim.Adam(cnn_pop.parameters(), lr=1e-3)
        loader = get_balanced_loader(X_tr_h, y_tr)
        
        cnn_pop.train()
        for _ in range(15):
            for x, y in loader:
                loss = FocalLoss()(cnn_pop(x.to(DEVICE)), y.to(DEVICE))
                cnn_opt.zero_grad(); loss.backward(); cnn_opt.step()

        split_idx = int(len(X_te_h) * CALIB_FRAC)
        buffer_size = LOOKBACK_S // WINDOW_STEP 
        X_cal, y_cal = X_te_h[:split_idx], y_te[:split_idx]
        X_ev, y_ev   = X_te_h[split_idx + buffer_size:], y_te[split_idx + buffer_size:]

        if len(X_ev) < 5 or y_cal.sum() == 0 or y_ev.sum() == 0:
            print(f"{p:<10} | --- Skipping (Insufficient Eval Data) ---")
            continue

        cnn_pop.eval()
        with torch.no_grad():
            pop_preds = torch.sigmoid(cnn_pop(torch.as_tensor(X_ev).to(DEVICE))).cpu().numpy()
            pop_auc, pop_ap = roc_auc_score(y_ev, pop_preds), average_precision_score(y_ev, pop_preds)

        # --- REFINED PERSONALIZATION ---
        cnn_pers = copy.deepcopy(cnn_pop)
        
        # 1. Parameter Freezing & Affine BN Adaptation
        for name, param in cnn_pers.named_parameters():
            is_head      = "head" in name
            is_last_conv = "net.3" in name          # last Conv1d weights
            is_bn_affine = ("net.1" in name or "net.4" in name)  # BN gamma/beta only
            param.requires_grad = is_head or is_last_conv or is_bn_affine

        # 2. Strict BN Evaluation Mode (Freeze Running Stats)
        for module in cnn_pers.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval() # Use Population mean/var, only adapt gamma/beta

        # 3. L2 Regularization (Weight Decay) to prevent overfitting on Cal set
        p_opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, cnn_pers.parameters()), 
            lr=CALIB_LR,
            weight_decay=1e-4 
        )
        
        cal_x, cal_y = torch.as_tensor(X_cal).to(DEVICE), torch.as_tensor(y_cal).to(DEVICE)
        
        for _ in range(CALIB_EPOCHS):
            # Model stays in eval() to keep BN stats frozen
            logits = cnn_pers(cal_x)
            loss = FocalLoss()(logits, cal_y)
            p_opt.zero_grad(); loss.backward(); p_opt.step()

        cnn_pers.eval()
        with torch.no_grad():
            pers_preds = torch.sigmoid(cnn_pers(torch.as_tensor(X_ev).to(DEVICE))).cpu().numpy()
            pers_auc, pers_ap = roc_auc_score(y_ev, pers_preds), average_precision_score(y_ev, pers_preds)
            print(f"{p:<10} | {pop_auc:<8.3f} {pers_auc:<8.3f} | {pop_ap:<8.3f} {pers_ap:<8.3f}")

if __name__ == "__main__":
    main()