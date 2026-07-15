"""Test for residual impairment after a distraction has ended.

The analysis keeps KinTCN's fitness-to-drive outcome: safety-critical events
in [t + 5 s, t + 15 s].  Distraction is used only to locate recovery periods;
individual distraction types and emotions are never analysed.

Outputs are written to ``post_distraction_recovery_results``:

* ``recovery_windows.csv``: one row per eligible 5-second endpoint;
* ``recovery_summary.csv``: outcome/physiology summaries by recovery band;
* ``participant_contrasts.csv``: paired recovery-minus-control contrasts;
* ``bootstrap_results.csv``: participant-bootstrap confidence intervals.

Run from the repository root with the project's Python environment:

    python drive/analyze_post_distraction_recovery.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KIN_LOOKBACK_S = 45
GAP_S = 5
HORIZON_S = 10
STEP_S = 5
BASELINE_S = KIN_LOOKBACK_S + GAP_S
MAX_LABEL_HOLD_S = 5
CONTROL_AFTER_S = 60
N_BOOTSTRAP = 2000
SEED = 42

ACTIVITY_COLS = [
    "safe_driving",
    "drinking",
    "brushing_hair",
    "talking_phone",
    "texting_phone",
]
DISTRACTION_COLS = [c for c in ACTIVITY_COLS if c != "safe_driving"]
EVENT_WEIGHTS = {
    "Collision": 5,
    "Red_light_violation": 3,
    "panic_braking_with_stop": 2,
    "panic_braking": 1,
}


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=here / "relab+unibo_dataset.csv")
    parser.add_argument(
        "--out", type=Path, default=here / "post_distraction_recovery_results"
    )
    parser.add_argument("--max-label-hold", type=int, default=MAX_LABEL_HOLD_S)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    return parser.parse_args()


def mark_event_onsets(group):
    out = group.copy()
    for col in EVENT_WEIGHTS:
        values = out[col].fillna(0).to_numpy() > 0
        out[col] = (values & ~np.r_[False, values[:-1]]).astype(np.int8)
    return out


def reconstruct_binary_distraction(group, max_hold_s):
    """Convert sparse activity probabilities into safe/distracted/unknown.

    Individual distraction classes are collapsed immediately.  They are used
    only to establish whether a valid activity prediction exists and whether
    its combined probability exceeds ``safe_driving``.  An all-zero row is
    unknown, not distracted.  Sparse predictions are held for at most
    ``max_hold_s`` seconds, preventing unlimited propagation through gaps.
    """
    scores = group[ACTIVITY_COLS].fillna(0.0).clip(lower=0.0)
    total = scores.sum(axis=1)
    safe_score = scores["safe_driving"]
    distracted_score = scores[DISTRACTION_COLS].sum(axis=1)

    observed = total > 1e-6
    state = pd.Series(pd.NA, index=group.index, dtype="Float64")
    state.loc[observed] = (
        distracted_score.loc[observed] > safe_score.loc[observed]
    ).astype(float)

    # At 1 Hz, limit=N means the last observed prediction covers at most N
    # subsequent samples.  Unknown rows beyond that remain excluded.
    state = state.ffill(limit=max_hold_s)
    return state.astype(float)


def causal_hr_features(hr):
    """Report-derived, fully causal HR features for every session sample."""
    raw = pd.Series(hr, dtype=float)
    available = raw.notna().astype(float)
    prefix = raw.iloc[:BASELINE_S].dropna()
    baseline = float(prefix.median()) if not prefix.empty else np.nan

    if np.isfinite(baseline):
        filled = raw.ffill().fillna(baseline)
    else:
        filled = raw.copy()

    recent10 = filled.rolling(10, min_periods=5).mean()
    previous10 = filled.shift(10).rolling(10, min_periods=5).mean()
    delta10 = recent10 - previous10  # causal E1 analogue

    def slope(values):
        y = np.asarray(values, dtype=float)
        ok = np.isfinite(y)
        if ok.sum() < 5:
            return np.nan
        x = np.arange(len(y), dtype=float)[ok]
        return float(np.polyfit(x, y[ok], 1)[0])

    slope10 = filled.rolling(10, min_periods=5).apply(slope, raw=True)
    previous_slope10 = slope10.shift(10)

    # Runtime-safe session-local reference: expanding median of previously
    # observed HR.  Shift(1) prevents the current measurement entering its own
    # baseline.  The initial calibration median is the fallback.
    local_baseline = raw.shift(1).expanding(min_periods=10).median()
    if np.isfinite(baseline):
        local_baseline = local_baseline.fillna(baseline)
    above = (filled > local_baseline + 3.0).astype(float)

    return pd.DataFrame(
        {
            "hr": raw,
            "hr_available": available,
            "hr_delta_10s": delta10,
            "hr_slope_10s": slope10,
            "hr_slope_change_10s": slope10 - previous_slope10,
            "hr_burden_plus3_recent10": above.rolling(10, min_periods=5).mean(),
            "hr_deviation_local": filled - local_baseline,
        },
        index=raw.index,
    )


def recovery_band(seconds):
    if not np.isfinite(seconds):
        return None
    if seconds <= 10:
        return "R00_10"
    if seconds <= 20:
        return "R10_20"
    if seconds <= 30:
        return "R20_30"
    if seconds <= CONTROL_AFTER_S:
        return "R30_60"
    return "CONTROL_GT60"


def session_windows(group, max_hold_s):
    group = group.sort_values("Timestamp").reset_index(drop=True)
    group = mark_event_onsets(group)
    state = reconstruct_binary_distraction(group, max_hold_s)
    hr_features = causal_hr_features(group["hr"])

    # Detect 1 -> 0 only when both states are observed/reconstructed.  Unknown
    # periods cannot create a synthetic distraction end.
    previous = state.shift(1)
    ended = previous.eq(1.0) & state.eq(0.0)
    last_end = np.nan
    seconds_since = np.full(len(group), np.nan)
    for i in range(len(group)):
        if ended.iloc[i]:
            last_end = float(i)
        if state.iloc[i] == 0.0 and np.isfinite(last_end):
            seconds_since[i] = i - last_end

    rows = []
    minimum_endpoint = max(KIN_LOOKBACK_S - 1, 20)
    maximum_endpoint = len(group) - GAP_S - HORIZON_S - 1
    for end in range(minimum_endpoint, maximum_endpoint + 1, STEP_S):
        # We test residual impairment only while the driver is presently
        # classified as non-distracted.  Unknown and active-distraction rows
        # are excluded rather than forced into either class.
        if state.iloc[end] != 0.0:
            continue
        band = recovery_band(seconds_since[end])
        if band is None:
            # A valid safe sample before any distraction is also an excellent
            # within-session control, provided it is not near a future onset.
            band = "CONTROL_PRE"

        future = group.iloc[end + GAP_S : end + GAP_S + HORIZON_S]
        score = sum(
            weight * int((future[col] > 0).any())
            for col, weight in EVENT_WEIGHTS.items()
        )
        row = {
            "id": group.at[end, "id"],
            "route": group.at[end, "route"],
            "Timestamp": group.at[end, "Timestamp"],
            "recovery_band": band,
            "seconds_since_distraction_end": seconds_since[end],
            "future_event": int(score >= 1),
            "future_risk_score": float(score),
        }
        row.update(hr_features.iloc[end].to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def build_windows(df, max_hold_s):
    pieces = []
    for _, group in df.groupby(["id", "route"], sort=False):
        piece = session_windows(group, max_hold_s)
        if not piece.empty:
            pieces.append(piece)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def summarize(windows):
    metrics = [
        "future_event",
        "future_risk_score",
        "hr_delta_10s",
        "hr_slope_10s",
        "hr_slope_change_10s",
        "hr_burden_plus3_recent10",
        "hr_deviation_local",
        "hr_available",
    ]
    return (
        windows.groupby("recovery_band", observed=True)[metrics]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )


def participant_contrasts(windows):
    """Paired participant estimates reduce between-driver HR confounding."""
    recovery = windows[windows["recovery_band"].isin(["R00_10", "R10_20", "R20_30"])]
    controls = windows[windows["recovery_band"].isin(["CONTROL_PRE", "CONTROL_GT60"])]
    metrics = [
        "future_event",
        "future_risk_score",
        "hr_delta_10s",
        "hr_slope_10s",
        "hr_slope_change_10s",
        "hr_burden_plus3_recent10",
        "hr_deviation_local",
    ]
    rec = recovery.groupby("id")[metrics].mean().add_suffix("_recovery")
    ctl = controls.groupby("id")[metrics].mean().add_suffix("_control")
    paired = rec.join(ctl, how="inner")
    for metric in metrics:
        paired[f"{metric}_delta"] = paired[f"{metric}_recovery"] - paired[f"{metric}_control"]
    paired["n_recovery"] = recovery.groupby("id").size()
    paired["n_control"] = controls.groupby("id").size()
    return paired.reset_index()


def participant_bootstrap(contrasts, n_bootstrap):
    delta_cols = [c for c in contrasts if c.endswith("_delta")]
    rng = np.random.default_rng(SEED)
    rows = []
    for col in delta_cols:
        values = contrasts[col].dropna().to_numpy(float)
        if len(values) == 0:
            continue
        boot = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            boot[i] = rng.choice(values, size=len(values), replace=True).mean()
        rows.append(
            {
                "metric": col.removesuffix("_delta"),
                "n_participants": len(values),
                "mean_recovery_minus_control": values.mean(),
                "median_recovery_minus_control": np.median(values),
                "ci95_low": np.percentile(boot, 2.5),
                "ci95_high": np.percentile(boot, 97.5),
                "fraction_positive": np.mean(values > 0),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    required = [
        "id",
        "route",
        "Timestamp",
        "hr",
        *ACTIVITY_COLS,
        *EVENT_WEIGHTS,
    ]
    df = pd.read_csv(args.csv, usecols=required)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).copy()

    args.out.mkdir(parents=True, exist_ok=True)
    windows = build_windows(df, args.max_label_hold)
    if windows.empty:
        raise RuntimeError(
            "No eligible recovery windows were reconstructed. Try inspecting "
            "the activity scores or increasing --max-label-hold conservatively."
        )

    summary = summarize(windows)
    contrasts = participant_contrasts(windows)
    bootstrap = participant_bootstrap(contrasts, args.bootstrap)

    windows.to_csv(args.out / "recovery_windows.csv", index=False)
    summary.to_csv(args.out / "recovery_summary.csv", index=False)
    contrasts.to_csv(args.out / "participant_contrasts.csv", index=False)
    bootstrap.to_csv(args.out / "bootstrap_results.csv", index=False)

    print("\nPOST-DISTRACTION RECOVERY ANALYSIS")
    print(f"Eligible endpoints : {len(windows)}")
    print(f"Participants       : {windows['id'].nunique()}")
    print("\nEndpoints by band:")
    print(windows["recovery_band"].value_counts().sort_index().to_string())
    print("\nParticipant-bootstrap paired results:")
    if bootstrap.empty:
        print("No participants had both recovery and control windows.")
    else:
        print(bootstrap.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to: {args.out}")
    print(
        "\nInterpretation: a useful residual signal should show a positive "
        "future_event contrast and a CI that does not depend on only a few "
        "participants. HR effects without increased event risk are activation, "
        "not evidence of fitness-to-drive impairment."
    )


if __name__ == "__main__":
    main()
