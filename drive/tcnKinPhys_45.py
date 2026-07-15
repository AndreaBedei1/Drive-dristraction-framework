"""Physiology-enhanced 45 s KinTCN experiment.

This is deliberately a thin experimental layer over ``tcnKin_45.py`` so the
original, validated pipeline remains untouched.  The target is unchanged:
safety-critical events in the future [t + 5 s, t + 15 s] weakly supervise the
current fitness-to-drive state.  HR is an input modality, not ground truth.

Only heart rate is used.  The legacy ``arousal`` field, emotion fields and
individual distraction-type fields are never read.  Consequently the model
does not learn to distinguish distraction kinds.  The experimental condition
may still be audited as safe-driving versus distracted outside this script,
but it is not substituted for the fitness-to-drive target.

Run exactly as the original script::

    python tcnKinPhys_45.py

Results and deployment artefacts go to ``physio_impairment_results`` so they
cannot overwrite the kinematics-only experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import tcnKin_45 as core


KINEMATIC_COLS = [
    "speed.x",
    "steeringWheelAngle",
    "steeringTorq",
    "acceleration.y",
]
PHYSIOLOGY_COLS = ["hr"]
ALL_SIGNAL_COLS = KINEMATIC_COLS + PHYSIOLOGY_COLS

# engineer_features emits nine transform-major groups.  With five raw
# channels the HR feature is the final element in every group.
N_TRANSFORMS = 3 + 2 * len(core.ROLL_SCALES)
N_RAW_CHANNELS = len(ALL_SIGNAL_COLS)
HR_FEATURE_IDX = [
    transform * N_RAW_CHANNELS + len(KINEMATIC_COLS)
    for transform in range(N_TRANSFORMS)
]
KIN_FEATURE_IDX = [
    idx
    for idx in range(N_TRANSFORMS * N_RAW_CHANNELS)
    if idx not in HR_FEATURE_IDX
]

HR_BASELINE_SECONDS = core.LOOKBACK_S + core.GAP
HR_MIN_SCALE_BPM = 1.0
PHYS_MODALITY_DROPOUT = 0.20


def _robust_hr_normalise(values):
    """Causally impute and robustly normalise one session's HR.

    The reference is restricted to the same initial calibration segment used
    by KinTCN.  Later gaps use last-observation-carried-forward.  A session
    with no sensor data maps to zero, the neutral value after normalisation.
    """
    s = values.astype(float).copy()
    finite = np.isfinite(s)
    if not finite.any():
        return np.zeros(len(s), dtype=np.float64)

    prefix = s[:HR_BASELINE_SECONDS]
    prefix = prefix[np.isfinite(prefix)]
    if prefix.size == 0:
        # This is needed only when the sensor starts after calibration.  It
        # uses the first available HR as a neutral anchor without using labels.
        baseline = float(s[finite][0])
        scale = HR_MIN_SCALE_BPM
    else:
        baseline = float(np.median(prefix))
        mad = float(np.median(np.abs(prefix - baseline)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale < HR_MIN_SCALE_BPM:
            std = float(np.std(prefix))
            scale = std if np.isfinite(std) and std >= HR_MIN_SCALE_BPM else HR_MIN_SCALE_BPM

    # Causal forward fill.  Leading missing samples are neutral rather than
    # back-filled from a future measurement.
    last = baseline
    for i in range(len(s)):
        if np.isfinite(s[i]):
            last = s[i]
        else:
            s[i] = last
    return np.clip((s - baseline) / scale, -core.RENORM_CLIP, core.RENORM_CLIP)


def normalize_signals_with_hr(df):
    """Original kinematic normalisation plus robust personal HR normalisation."""
    out = df.copy()
    prefix = core.LOOKBACK_S + core.GAP
    for _, grp in out.groupby(["id", "route"]):
        idx = grp.index
        for col in KINEMATIC_COLS:
            mu = grp.iloc[:prefix][col].mean()
            sig = grp.iloc[:prefix][col].std() + 1e-6
            out.loc[idx, col] = (grp[col] - mu) / sig
        out.loc[idx, "hr"] = _robust_hr_normalise(grp["hr"].to_numpy())
    return out


class PhysioKinDataset(core.KinDataset):
    """Existing augmentation with whole-modality HR dropout.

    Dropping all nine HR-derived channels together teaches the model that a
    missing wearable must reduce to the kinematics-only prediction.
    """

    def __getitem__(self, idx):
        xk, y, weight = super().__getitem__(idx)
        if self.aug and torch.rand(1).item() < PHYS_MODALITY_DROPOUT:
            xk[:, HR_FEATURE_IDX] = 0.0
        return xk, y, weight


class PhysioKinTCN(nn.Module):
    """Separate HR encoder with gated residual late fusion.

    ``phys_alpha`` starts at zero, making every new instance exactly the
    kinematics-only scorer at initialisation.  Training can learn a positive
    or negative correction, or leave HR disabled when it does not generalise.
    """

    KIN_D = 64
    PHYS_D = 16

    def __init__(self, n_kin_feats: int, use_attention: bool = True):
        super().__init__()
        expected = len(KIN_FEATURE_IDX) + len(HR_FEATURE_IDX)
        if n_kin_feats != expected:
            raise ValueError(f"Expected {expected} engineered inputs, got {n_kin_feats}")
        self.use_attention = use_attention

        self.kin_branch = nn.Sequential(
            core.ResBlock(len(KIN_FEATURE_IDX), 32, 1),
            core.ResBlock(32, self.KIN_D, 2),
            core.ResBlock(self.KIN_D, self.KIN_D, 4),
            core.ResBlock(self.KIN_D, self.KIN_D, 8),
            core.ResBlock(self.KIN_D, self.KIN_D, 16),
        )
        self.phys_branch = nn.Sequential(
            core.ResBlock(len(HR_FEATURE_IDX), self.PHYS_D, 1),
            core.ResBlock(self.PHYS_D, self.PHYS_D, 2),
            core.ResBlock(self.PHYS_D, self.PHYS_D, 4),
            core.ResBlock(self.PHYS_D, self.PHYS_D, 8),
        )
        self.kin_attn = core.TemporalAttention(self.KIN_D)
        self.phys_attn = core.TemporalAttention(self.PHYS_D)
        # Alias retained because the original ablation code accesses .attn.
        self.attn = self.kin_attn

        self.kin_head = nn.Sequential(
            nn.Linear(self.KIN_D, 48), nn.ReLU(), nn.Dropout(0.2), nn.Linear(48, 1)
        )
        self.phys_head = nn.Sequential(
            nn.Linear(self.PHYS_D, 12), nn.ReLU(), nn.Dropout(0.2), nn.Linear(12, 1)
        )
        self.phys_alpha = nn.Parameter(torch.zeros(()))

    def _pool(self, sequence, attention):
        return attention(sequence) if self.use_attention else sequence.mean(dim=-1)

    def forward(self, x_all):
        kin = x_all[:, :, KIN_FEATURE_IDX].permute(0, 2, 1)
        phys = x_all[:, :, HR_FEATURE_IDX].permute(0, 2, 1)
        kin_pool = self._pool(self.kin_branch(kin), self.kin_attn)
        phys_pool = self._pool(self.phys_branch(phys), self.phys_attn)
        kin_logit = self.kin_head(kin_pool).squeeze(-1)
        phys_logit = self.phys_head(phys_pool).squeeze(-1)
        return kin_logit + torch.tanh(self.phys_alpha) * phys_logit


_base_save_deployment_bundle = core.save_deployment_bundle


def save_physio_deployment_bundle(*args, **kwargs):
    bundle_dir = _base_save_deployment_bundle(*args, **kwargs)
    meta_path = Path(bundle_dir) / "meta.json"
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    meta.update(
        {
            "architecture": "PhysioKinTCN",
            "kinematic_cols": KINEMATIC_COLS,
            "physiology_cols": PHYSIOLOGY_COLS,
            "hr_normalisation": "initial_50s_median_mad_causal_ffill",
            "physiology_modality_dropout": PHYS_MODALITY_DROPOUT,
            "legacy_arousal_used": False,
            "route_population_renormalisation": core.USE_ROUTE_RENORM,
            "target": "unchanged_future_safety_event_weak_supervision",
        }
    )
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return bundle_dir


def configure_experiment():
    # Five raw inputs are engineered together, but PhysioKinTCN separates the
    # resulting 36 kinematic and 9 HR features before either encoder sees them.
    core.SIGNAL_COLS = ALL_SIGNAL_COLS
    core.KIN_COLS = ALL_SIGNAL_COLS
    core.KIN_IDX = list(range(len(ALL_SIGNAL_COLS)))

    # Keep HR out of the route-level cross-participant renormalisation.  HR is
    # personal physiology and is normalised causally within session above.
    core.VEHICLE_COLS = KINEMATIC_COLS
    core.normalize_signals = normalize_signals_with_hr
    core.KinDataset = PhysioKinDataset
    core.KinTCN = PhysioKinTCN
    core.save_deployment_bundle = save_physio_deployment_bundle
    core.OUT_DIR = Path(__file__).parent / "physio_impairment_results"
    core.OUT_DIR.mkdir(exist_ok=True)


if __name__ == "__main__":
    configure_experiment()
    print("\nPhysioKinTCN experiment")
    print("  Target     : unchanged KinTCN safety-event weak supervision")
    print("  Inputs     : four kinematic channels + HR")
    print("  Excluded   : arousal, emotions, distraction types")
    print("  HR fusion  : separate causal branch, zero-initialised residual gate")
    print(f"  Output dir : {core.OUT_DIR}\n")
    core.main()
