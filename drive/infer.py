"""
Inference script for the deploy_final impairment detection model.

Input : numpy array of shape (45, 4) — 45 seconds @ 1 Hz of:
        [speed.x, steeringWheelAngle, steeringTorq, acceleration.y]
Output: float in [0, 1] — probability that a driving error occurs
        in the 5–15 s window after the observation ends.

Usage
-----
    from drive.infer import ImpairmentPredictor
    model = ImpairmentPredictor()
    prob  = model.predict(window_45s)          # single window
    probs = model.predict_batch(windows)       # (N, 45, 4) → (N,)
"""

import json
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path

DEPLOY_DIR = Path(__file__).parent / "impairment_results" / "deploy_final"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── model architecture (mirrors tcnKin_45.py) ────────────────────────────────

def _gn_groups(out_c: int) -> int:
    g = min(out_c, 8)
    while g > 1 and out_c % g != 0:
        g //= 2
    return max(g, 1)


class _ResBlock(nn.Module):
    def __init__(self, in_c, out_c, d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConstantPad1d((2 * d, 0), 0.0),
            nn.Conv1d(in_c, out_c, 3, padding=0, dilation=d),
            nn.GroupNorm(_gn_groups(out_c), out_c),
            nn.ReLU(),
            nn.Dropout1d(0.1),
        )
        self.res = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.res(x)


class _TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Linear(channels, channels // 2)
        self.score = nn.Linear(channels // 2, 1)

    def forward(self, x):
        xt      = x.permute(0, 2, 1)
        h       = torch.tanh(self.query(xt))
        weights = torch.softmax(self.score(h), dim=1)
        return (xt * weights).sum(dim=1)


class _KinTCN(nn.Module):
    KIN_D = 64

    def __init__(self, n_kin_feats: int, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.kin_branch = nn.Sequential(
            _ResBlock(n_kin_feats,        self.KIN_D // 2, 1),
            _ResBlock(self.KIN_D // 2,    self.KIN_D,      2),
            _ResBlock(self.KIN_D,         self.KIN_D,      4),
            _ResBlock(self.KIN_D,         self.KIN_D,      8),
            _ResBlock(self.KIN_D,         self.KIN_D,     16),
        )
        self.attn = _TemporalAttention(self.KIN_D)
        self.head = nn.Sequential(
            nn.Linear(self.KIN_D, 48), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 1),
        )

    def forward(self, x_kin):
        feat = self.kin_branch(x_kin.permute(0, 2, 1))
        pool = self.attn(feat) if self.use_attention else feat.mean(dim=-1)
        return self.head(pool).squeeze(-1)


# ── feature engineering (mirrors tcnKin_45.py) ───────────────────────────────

def _engineer_features_batch(windows: np.ndarray, roll_scales=(5, 10, 25)) -> np.ndarray:
    """windows: (N, T, C) float32 → (N, T, n_feat) float32"""
    diff1  = np.diff(windows, axis=1, prepend=windows[:, :1, :])
    w_mean = windows.mean(axis=1, keepdims=True)
    w_std  = windows.std(axis=1,  keepdims=True) + 1e-6
    z      = (windows - w_mean) / w_std
    parts  = [windows, diff1, z]
    cs     = np.cumsum(windows,        axis=1)
    cs_sq  = np.cumsum(windows ** 2,   axis=1)
    T      = windows.shape[1]
    for k in roll_scales:
        cs_lag    = np.zeros_like(cs)
        cs_sq_lag = np.zeros_like(cs_sq)
        if k < T:
            cs_lag[:, k:, :]    = cs[:, :T - k, :]
            cs_sq_lag[:, k:, :] = cs_sq[:, :T - k, :]
        win_len   = np.minimum(np.arange(1, T + 1)[None, :, None], k)
        roll_mean = (cs - cs_lag) / win_len
        roll_var  = (cs_sq - cs_sq_lag) / win_len - roll_mean ** 2
        roll_std  = np.sqrt(np.maximum(roll_var, 0.0)) + 1e-6
        parts.extend([roll_mean, roll_std])
    return np.concatenate(parts, axis=2).astype(np.float32)


# ── predictor ─────────────────────────────────────────────────────────────────

class ImpairmentPredictor:
    """
    Load once, call predict() many times.

    Parameters
    ----------
    deploy_dir : path to the deploy_final/ folder (default: auto-detected)
    device     : 'cpu' or 'cuda' (default: auto)
    """

    def __init__(self, deploy_dir: Path = DEPLOY_DIR, device: str = DEVICE):
        self.device = device
        meta_path   = Path(deploy_dir) / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        self.kin_cols    = meta["kin_cols"]       # ['speed.x', 'steeringWheelAngle', ...]
        self.lookback_s  = meta["lookback_s"]     # 45
        self.roll_scales = tuple(meta["roll_scales"])
        n_kin_feat       = meta["n_kin_feat"]     # 36
        use_attention    = meta["use_attention"]
        ensemble_k       = meta["ensemble_k"]     # 10

        self.scaler   = joblib.load(Path(deploy_dir) / "scaler_k.joblib")
        self.platt_lr = joblib.load(Path(deploy_dir) / "platt_lr.joblib")

        self.models = []
        for i in range(ensemble_k):
            m = _KinTCN(n_kin_feat, use_attention).to(device)
            state = torch.load(
                Path(deploy_dir) / f"tcn_{i:02d}.pt",
                map_location=device,
                weights_only=True,
            )
            m.load_state_dict(state)
            m.eval()
            self.models.append(m)

        self._n_feat = n_kin_feat
        print(f"ImpairmentPredictor ready: {ensemble_k} models, device={device}")

    def _preprocess(self, windows: np.ndarray) -> torch.Tensor:
        """
        windows : (N, T, 4) float32
        returns : (N, T, n_feat) tensor ready for the TCN
        """
        N, T, _ = windows.shape
        feats   = _engineer_features_batch(windows, self.roll_scales)  # (N, T, n_feat)
        scaled  = self.scaler.transform(
            feats.reshape(-1, self._n_feat)
        ).reshape(N, T, self._n_feat).astype(np.float32)
        return torch.from_numpy(scaled).to(self.device)

    def predict_batch(self, windows: np.ndarray) -> np.ndarray:
        """
        windows : (N, 45, 4) — columns in order: speed.x, steeringWheelAngle,
                               steeringTorq, acceleration.y
        returns : (N,) float32 — impairment probability per window
        """
        windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim == 2:
            windows = windows[None]                             # (1, T, 4)

        x = self._preprocess(windows)
        with torch.no_grad():
            scores = torch.stack(
                [torch.sigmoid(m(x)) for m in self.models], dim=0
            ).mean(dim=0).cpu().numpy()                        # (N,)

        probs = self.platt_lr.predict_proba(
            scores.reshape(-1, 1)
        )[:, 1].astype(np.float32)
        return probs

    def predict(self, window: np.ndarray) -> float:
        """
        window : (45, 4) — single observation window
        returns: float — impairment probability
        """
        return float(self.predict_batch(window[None])[0])


# ── demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    predictor = ImpairmentPredictor()

    # --- Option A: run on a real CSV slice ---
    csv_path = Path(__file__).parent / "relab+unibo_dataset.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        pid   = df["id"].iloc[0]
        route = df["route"].iloc[df["id"].eq(pid).values.nonzero()[0][0]]
        grp   = df[(df["id"] == pid) & (df["route"] == route)].sort_values("Timestamp")
        window_df = grp[predictor.kin_cols].iloc[:predictor.lookback_s].values
        if window_df.shape == (predictor.lookback_s, 4):
            prob = predictor.predict(window_df)
            print(f"[real data] pid={pid}  impairment prob = {prob:.3f}")
        else:
            print("Not enough rows for a demo window from first participant/route.")
    else:
        print(f"CSV not found at {csv_path}, using random noise as demo.")

    # --- Option B: random noise (sanity check only) ---
    rng      = np.random.default_rng(0)
    fake_win = rng.normal(0, 1, (predictor.lookback_s, 4)).astype(np.float32)
    prob     = predictor.predict(fake_win)
    print(f"[random noise] impairment prob = {prob:.3f}")

    # --- Option C: batch inference ---
    batch = rng.normal(0, 1, (5, predictor.lookback_s, 4)).astype(np.float32)
    probs = predictor.predict_batch(batch)
    print(f"[batch of 5]   impairment probs = {np.round(probs, 3)}")

    # --- Feature engineering self-test: batched == per-window loop ---
    def _engineer_features_ref(window, roll_scales=(5, 10, 25)):
        diff1  = np.diff(window, axis=0, prepend=window[:1])
        w_mean = window.mean(axis=0, keepdims=True)
        w_std  = window.std(axis=0,  keepdims=True) + 1e-6
        z      = (window - w_mean) / w_std
        parts  = [window, diff1, z]
        cs     = np.cumsum(window,       axis=0)
        cs_sq  = np.cumsum(window ** 2,  axis=0)
        T      = len(window)
        for k in roll_scales:
            cs_lag    = np.zeros_like(cs)
            cs_sq_lag = np.zeros_like(cs_sq)
            if k < T:
                cs_lag[k:]    = cs[:T - k]
                cs_sq_lag[k:] = cs_sq[:T - k]
            win_len   = np.minimum(np.arange(1, T + 1)[:, None], k)
            roll_mean = (cs - cs_lag) / win_len
            roll_var  = (cs_sq - cs_sq_lag) / win_len - roll_mean ** 2
            roll_std  = np.sqrt(np.maximum(roll_var, 0.0)) + 1e-6
            parts.extend([roll_mean, roll_std])
        return np.concatenate(parts, axis=1).astype(np.float32)

    test_wins = rng.normal(0, 1, (8, predictor.lookback_s, 4)).astype(np.float32)
    ref  = np.stack([_engineer_features_ref(test_wins[i], predictor.roll_scales) for i in range(8)])
    fast = _engineer_features_batch(test_wins, predictor.roll_scales)
    assert np.allclose(ref, fast, atol=1e-6), "Feature engineering mismatch!"
    print("[self-test] batched feature engineering matches per-window loop ✓")
