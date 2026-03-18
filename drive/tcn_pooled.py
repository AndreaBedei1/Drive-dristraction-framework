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
# CONFIG
# -------------------------------------------------
SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIGNAL_COLS = ["arousal", "hr", "speed.x", "steeringWheelAngle"]
ERR_COLS    = ["Collision", "Red_light_violation", "panic_braking",
               "panic_braking_with_stop", "sharp_turn"]

EPOCHS, LR   = 50, 1e-3
LOOKBACK_S   = 60
WINDOW_STEP  = 5
GAP, HORIZON = 3, 5
BATCH_SIZE   = 64

MIN_POSITIVES      = 1   # to enter the LODO loop at all
MIN_EVAL_POSITIVES = 3   # to count toward per-driver reported metrics

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
    """
    First exclusive index i such that y[start:i] contains
    at least `min_positives` ones. Returns None if exhausted.
    """
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

def build_windows(df):
    windows, labels, pids = [], [], []
    for (pid, route), grp in df.groupby(["id", "route"]):
        grp = grp.sort_values("Timestamp").reset_index(drop=True)
        idx = 0
        while idx + LOOKBACK_S + GAP + HORIZON <= len(grp):
            sig = grp.iloc[idx: idx + LOOKBACK_S][SIGNAL_COLS].values.astype(np.float32)
            if not np.isnan(sig).any():
                future = grp.iloc[idx + LOOKBACK_S + GAP: idx + LOOKBACK_S + GAP + HORIZON]
                windows.append(sig)
                labels.append(int((future[ERR_COLS] > 0).any().any()))
                pids.append(pid)
            idx += WINDOW_STEP
    return np.stack(windows), np.array(labels), np.array(pids)

# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    df        = pd.read_csv("relab+unibo_dataset.csv")
    X, y, pid = build_windows(df)
    BUFFER    = LOOKBACK_S // WINDOW_STEP

    drivers   = [d for d in np.unique(pid) if y[pid == d].sum() >= MIN_POSITIVES]
    criterion = FocalLoss()
    rng       = np.random.default_rng(SEED)

    hdr = (f"{'Driver':<10} | {'N_eval':>6} {'PosR%':>6} | "
           f"{'PopAUC':>7} {'HybAUC':>7} {'Gain':>7} | "
           f"{'PopBSS':>7} {'HybBSS':>7} {'GainBSS':>8} | "
           f"{'Pers':>5} {'PPosR%':>6} | {'Cal':>5} {'CPosR%':>6} | {'Flag'}")
    print(hdr)
    print("-" * len(hdr))

    # --- pooled collection: every driver contributes regardless of reportability ---
    pool_y    = []   # true labels
    pool_pop  = []   # population model probabilities
    pool_hyb  = []   # hybrid model probabilities
    pool_drv  = []   # driver id per window (for stratified analysis if needed)

    per_driver_results = []

    for d in drivers:
        # ---- LODO split ----
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

        Xtr    = scaler.fit_transform(X_tr[~vmask].reshape(-1, n_feats)).reshape(-1, LOOKBACK_S, n_feats)
        Xval   = scaler.transform(X_tr[vmask].reshape(-1, n_feats)).reshape(-1, LOOKBACK_S, n_feats)
        Xte_sc = scaler.transform(X_te.reshape(-1, n_feats)).reshape(-1, LOOKBACK_S, n_feats)

        # ---- train base population model ----
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

        # ---- calibration ----
        model.eval(); model_hyb.eval()
        with torch.no_grad():
            z_cal_pop = model(torch.as_tensor(X_cal).to(DEVICE))
            z_cal_hyb = model_hyb(torch.as_tensor(X_cal).to(DEVICE))

        cal_pop = fit_calibrator(z_cal_pop, y_cal, DEVICE)
        cal_hyb = fit_calibrator(z_cal_hyb, y_cal, DEVICE)

        # ---- per-driver predictions ----
        with torch.no_grad():
            X_eval_t = torch.as_tensor(X_eval).to(DEVICE)
            p_pop    = torch.sigmoid(cal_pop(model(X_eval_t))).cpu().numpy()
            p_hyb    = torch.sigmoid(cal_hyb(model_hyb(X_eval_t))).cpu().numpy()

        # --- accumulate into pool (every driver with a valid eval slice) ---
        pool_y.append(y_eval)
        pool_pop.append(p_pop)
        pool_hyb.append(p_hyb)
        pool_drv.extend([d] * len(y_eval))

        # ---- per-driver metrics ----
        auc_pop  = safe_auc(y_eval, p_pop)
        auc_hyb  = safe_auc(y_eval, p_hyb)
        gain_auc = (auc_hyb - auc_pop) if (auc_pop and auc_hyb) else float("nan")
        bss_pop  = brier_skill_score(y_eval, p_pop)
        bss_hyb  = brier_skill_score(y_eval, p_hyb)

        pos_rate = y_eval.mean() * 100
        ppers_r  = y_pers.mean() * 100 if len(y_pers) else 0.0
        pcal_r   = y_cal.mean()  * 100 if len(y_cal)  else 0.0
        flag     = "" if personalised else "NO_PERS"
        reportable = int(y_eval.sum()) >= MIN_EVAL_POSITIVES

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

    # =================================================================
    # POOLED EVALUATION
    # Each driver's predictions come from a model that never trained on
    # that driver (LODO guarantee). We are only aggregating the
    # measurement — not mixing training and test data.
    # =================================================================
    print("\n" + "=" * 70)
    print("POOLED EVALUATION  (all drivers with a valid eval slice)")
    print("=" * 70)

    if pool_y:
        y_pool   = np.concatenate(pool_y)
        pp_pool  = np.concatenate(pool_pop)
        ph_pool  = np.concatenate(pool_hyb)

        n_pool       = len(y_pool)
        pos_pool     = int(y_pool.sum())
        n_drivers    = len(pool_y)

        print(f"Windows: {n_pool}  |  Positives: {pos_pool} ({pos_pool/n_pool*100:.1f}%)  |  Drivers: {n_drivers}")
        print()

        auc_pool_pop = safe_auc(y_pool, pp_pool)
        auc_pool_hyb = safe_auc(y_pool, ph_pool)
        bss_pool_pop = brier_skill_score(y_pool, pp_pool)
        bss_pool_hyb = brier_skill_score(y_pool, ph_pool)

        gain_auc_pool = (auc_pool_hyb - auc_pool_pop) if (auc_pool_pop and auc_pool_hyb) else float("nan")
        gain_bss_pool = bss_pool_hyb - bss_pool_pop if not (np.isnan(bss_pool_pop) or np.isnan(bss_pool_hyb)) else float("nan")

        print(f"{'Metric':<12}  {'Population':>12}  {'Hybrid':>10}  {'Gain':>8}")
        print(f"{'-'*46}")
        print(f"{'AUC':<12}  {auc_pool_pop or 0:>12.4f}  {auc_pool_hyb or 0:>10.4f}  {gain_auc_pool:>+8.4f}")
        print(f"{'BSS':<12}  {bss_pool_pop:>12.4f}  {bss_pool_hyb:>10.4f}  {gain_bss_pool:>+8.4f}")
    else:
        print("No drivers produced a valid eval slice.")

    # =================================================================
    # PER-DRIVER SUMMARY  (reportable drivers only)
    # =================================================================
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

        low_rate = rdf[rdf["pos_rate_%"] < 10]
        if len(low_rate):
            print(f"\nDrivers with eval positive rate < 10% (metrics less reliable):")
            print(low_rate[["driver", "n_eval", "pos_rate_%", "auc_pop", "auc_hyb", "gain_auc"]]
                  .to_string(index=False))
    else:
        print(f"\nNo drivers met the per-driver reporting threshold "
              f"(MIN_EVAL_POSITIVES={MIN_EVAL_POSITIVES}).")
        print("Rely on pooled evaluation above for conclusions.")


if __name__ == "__main__":
    main()