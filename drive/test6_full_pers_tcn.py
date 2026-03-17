import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
import random

# -------------------------------------------------
# CONFIG & MODELS (Reusing TCN_Attention_Net from previous)
# -------------------------------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIGNAL_COLS = ["arousal", "hr", "speed.x", "steeringWheelAngle"]
ERR_COLS = ["Collision", "Red_light_violation", "panic_braking", "panic_braking_with_stop", "sharp_turn"]

# Hyperparameters
EPOCHS, LR = 50, 1e-3
LOOKBACK_S, WINDOW_STEP = 60, 5
GAP, HORIZON = 3, 5
CALIB_FRAC = 0.3
MIN_POSITIVES = 5
BATCH_SIZE = 64

# Hybrid specific
HYBRID_LR = 5e-4
HYBRID_EPOCHS = 15
LAMBDA_TETHER = 0.1 

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# -------------------------------------------------
# DATASET & MODEL
# -------------------------------------------------

class DrivingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X).float()
        self.y = torch.as_tensor(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def get_balanced_loader(X, y):
    y_int = y.astype(int)
    counts = np.bincount(y_int)
    weights = 1.0 / (counts + 1e-6)
    sample_weights = torch.from_numpy(weights[y_int])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return DataLoader(DrivingDataset(X, y), batch_size=BATCH_SIZE, sampler=sampler)

class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Linear(channels, channels // 2)
        self.score = nn.Linear(channels // 2, 1)
    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        h = torch.tanh(self.query(x_t))
        weights = torch.softmax(self.score(h), dim=1)
        return torch.sum(x_t * weights, dim=1)

class TCN_Attention_Net(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.network = nn.Sequential(
            self._block(in_channels, 32, 1), self._block(32, 64, 2),
            self._block(64, 64, 4), self._block(64, 128, 8)
        )
        self.attention = TemporalAttention(128)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def _block(self, in_c, out_c, d):
        return nn.Sequential(nn.Conv1d(in_c, out_c, 3, padding=d, dilation=d), nn.BatchNorm1d(out_c), nn.ReLU())
    def forward(self, x):
        x = self.network(x.permute(0, 2, 1))
        return self.head(self.attention(x)).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt)**self.gamma * bce).mean()

class LogitCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
    def forward(self, z):
        return torch.abs(self.a) * z + self.b

# -------------------------------------------------
# DATA UTILS (With Corrected Buffer)
# -------------------------------------------------

def build_windows(df):
    windows, labels, pids = [], [], []
    for (pid, route), grp in df.groupby(["id", "route"]):
        grp = grp.sort_values("Timestamp").reset_index(drop=True)
        idx = 0
        while idx + LOOKBACK_S + GAP + HORIZON <= len(grp):
            sig = grp.iloc[idx:idx+LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
            if not np.isnan(sig).any():
                future = grp.iloc[idx+LOOKBACK_S+GAP : idx+LOOKBACK_S+GAP+HORIZON]
                windows.append(sig)
                labels.append(int((future[ERR_COLS] > 0).any().any()))
                pids.append(pid)
            idx += WINDOW_STEP
    return np.stack(windows), np.array(labels), np.array(pids)

# -------------------------------------------------
# MAIN HYBRID LOOP
# -------------------------------------------------

def main():
    df = pd.read_csv("relab+unibo_dataset.csv")
    X_all, y_all, pid_all = build_windows(df)
    drivers = [p for p in np.unique(pid_all) if y_all[pid_all==p].sum() >= MIN_POSITIVES]
    criterion = FocalLoss()

    # Informative Header
    header = f"{'Driver':<10} | {'Pop AUC':<8} {'Hyb AUC':<8} {'AUC Gain':<8} | {'Pop BSS':<8} {'Hyb BSS':<8} {'BSS Gain':<8}"
    print(header)
    print("-" * len(header))

    for p in drivers:
        # LODO SPLIT
        X_tr, y_tr = X_all[pid_all != p], y_all[pid_all != p]
        X_te, y_te = X_all[pid_all == p], y_all[pid_all == p]
        pid_tr = pid_all[pid_all != p]

        # Driver-level validation split for Early Stopping
        unique_pids = np.unique(pid_tr)
        val_pids = np.random.default_rng(SEED).choice(unique_pids, max(1, int(len(unique_pids)*0.15)), replace=False)
        v_mask = np.isin(pid_tr, val_pids)
        
        # GUARD: Ensure val set is usable for Early Stopping
        if y_tr[v_mask].sum() == 0: 
            continue 

        # --- SCALING (Defined here so it exists for all subsequent steps) ---
        scaler = StandardScaler()
        n_sig = len(SIGNAL_COLS)
        X_tr_s = scaler.fit_transform(X_tr[~v_mask].reshape(-1, n_sig)).reshape(-1, LOOKBACK_S, n_sig)
        X_val_s = scaler.transform(X_tr[v_mask].reshape(-1, n_sig)).reshape(-1, LOOKBACK_S, n_sig)
        X_te_s = scaler.transform(X_te.reshape(-1, n_sig)).reshape(X_te.shape)

        # 1. POPULATION MODEL + SCHEDULER + EARLY STOPPING
        model_pop = TCN_Attention_Net(n_sig).to(DEVICE)
        optimizer = torch.optim.Adam(model_pop.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        loader = get_balanced_loader(X_tr_s, y_tr[~v_mask])
        
        best_auc, best_w, patience = 0, None, 8
        for epoch in range(EPOCHS):
            model_pop.train()
            for xb, yb in loader:
                l = criterion(model_pop(xb.to(DEVICE)), yb.to(DEVICE))
                optimizer.zero_grad(); l.backward(); optimizer.step()
            
            scheduler.step()
            model_pop.eval()
            with torch.no_grad():
                v_preds = torch.sigmoid(model_pop(torch.as_tensor(X_val_s).to(DEVICE))).cpu().numpy()
                auc = roc_auc_score(y_tr[v_mask], v_preds)
                if auc > best_auc:
                    best_auc, best_w, patience = auc, copy.deepcopy(model_pop.state_dict()), 8
                else:
                    patience -= 1
                    if patience == 0: break

        if best_w is None: 
            continue
        model_pop.load_state_dict(best_w)

        # 2. HYBRID PERSONALIZATION (Double-Split Calibration)
        split_idx = int(len(X_te_s) * CALIB_FRAC)
        buffer_len = (LOOKBACK_S + GAP + HORIZON) // WINDOW_STEP + 1
        X_cal_full, y_cal_full = X_te_s[:split_idx], y_te[:split_idx]
        X_ev, y_ev = X_te_s[split_idx + buffer_len:], y_te[split_idx + buffer_len:]

        # GUARD: Ensure evaluation set has both classes
        if len(y_ev) < 5 or len(np.unique(y_ev)) < 2: 
            continue

        # Baseline Brier for Skill Score (Climatology)
        bs_ref = brier_score_loss(y_ev, np.full_like(y_ev, y_ev.mean()))

        # Personalization Split (Fine-tuning vs Logit Calibration)
        pos_idx, neg_idx = np.where(y_cal_full == 1)[0], np.where(y_cal_full == 0)[0]

        # Give Stage A at least 2 positives before splitting off any for Stage B
        if len(pos_idx) >= 4:
            n_logit = max(2, len(pos_idx) // 3)
            logit_idx = np.concatenate([pos_idx[-n_logit:], neg_idx[-n_logit*4:]])
            fine_idx = np.array([i for i in range(len(y_cal_full)) if i not in set(logit_idx)])
        else:
            # Too few positives to split — give everything to Stage A, skip Stage B
            fine_idx = np.arange(len(y_cal_full))
            logit_idx = np.array([], dtype=int)
        
        X_cal_fine, y_cal_fine = X_cal_full[fine_idx], y_cal_full[fine_idx]
        X_cal_logit, y_cal_logit = X_cal_full[logit_idx], y_cal_full[logit_idx]

        # Stage A: Tethered Fine-Tuning
        model_hyb = copy.deepcopy(model_pop)
        if y_cal_fine.sum() >= 1:
            # Hold out last 20% of cal_fine for Stage A validation
            val_cut = max(1, int(len(X_cal_fine) * 0.2))
            X_fine_tr, y_fine_tr = X_cal_fine[:-val_cut], y_cal_fine[:-val_cut]
            X_fine_val, y_fine_val = X_cal_fine[-val_cut:], y_cal_fine[-val_cut:]

            pop_params = {n: p.detach().clone() for n, p in model_pop.named_parameters()}
            for n, param in model_hyb.named_parameters():
                param.requires_grad = any(x in n for x in ["attention", "head"])
            h_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model_hyb.parameters()), lr=HYBRID_LR)
            for _ in range(HYBRID_EPOCHS):
                model_hyb.train()
                logits = model_hyb(torch.as_tensor(X_fine_tr).to(DEVICE))
                loss_f = criterion(logits, torch.as_tensor(y_fine_tr).to(DEVICE))
                tether = sum(((p - pop_params[n])**2).mean() for n, p in model_hyb.named_parameters() if p.requires_grad)
                h_opt.zero_grad(); (loss_f + LAMBDA_TETHER * tether).backward(); h_opt.step()

            # Only keep fine-tuned model if it improves on the held-out slice
            model_hyb.eval()
            model_pop.eval()
            with torch.no_grad():
                hyb_out = torch.sigmoid(model_hyb(torch.as_tensor(X_fine_val).to(DEVICE))).cpu().numpy()
                pop_out = torch.sigmoid(model_pop(torch.as_tensor(X_fine_val).to(DEVICE))).cpu().numpy()
            if y_fine_val.sum() == 0 or roc_auc_score(y_fine_val, hyb_out) <= roc_auc_score(y_fine_val, pop_out):
                model_hyb = copy.deepcopy(model_pop)  # revert

        # Stage B: Logit Calibration
        cal_mod = LogitCalibrator().to(DEVICE)
        if len(logit_idx) > 0 and y_cal_logit.sum() >= 2:
            model_pop.eval()
            with torch.no_grad(): z_logit_in = model_pop(torch.as_tensor(X_cal_logit).to(DEVICE))
            c_opt = torch.optim.Adam(cal_mod.parameters(), lr=0.01)
            for _ in range(100):
                l = F.binary_cross_entropy_with_logits(cal_mod(z_logit_in), torch.tensor(y_cal_logit).float().to(DEVICE))
                c_opt.zero_grad(); l.backward(); c_opt.step()

        # 3. EVALUATE & INFORM
        with torch.no_grad():
            p_pop = torch.sigmoid(model_pop(torch.as_tensor(X_ev).to(DEVICE))).cpu().numpy()
            p_hyb = torch.sigmoid(cal_mod(model_hyb(torch.as_tensor(X_ev).to(DEVICE)))).cpu().numpy()

        # Metrics & Skill Scores
        pop_auc, hyb_auc = roc_auc_score(y_ev, p_pop), roc_auc_score(y_ev, p_hyb)
        pop_bss = 1 - (brier_score_loss(y_ev, p_pop) / bs_ref)
        hyb_bss = 1 - (brier_score_loss(y_ev, p_hyb) / bs_ref)

        print(f"{p:<10} | {pop_auc:<8.3f} {hyb_auc:<8.3f} {hyb_auc-pop_auc:<+8.3f} | "
              f"{pop_bss:<8.3f} {hyb_bss:<8.3f} {hyb_bss-pop_bss:<+8.3f}")

if __name__ == "__main__":
    main()