"""Test clean post-SURT residual impairment using actual task timestamps.

Each distraction episode ends at the response to its final SURT phase.  The
clean recovery interval is censored at the next trial start, because a fixed
10--30 second post window would usually contain another distraction.  The
control is the same participant's non-distracted route at matched fractional
session progress.

The primary outcomes preserve the fitness-to-drive interpretation:

* safety-event occurrence after a 5 s gap and before the next distraction;
* severity-weighted safety-event score in that same clean interval.

HR pre/post features are secondary mechanistic outcomes.  Individual
distraction types and emotion are not used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_WEIGHTS = {
    "Collision": 5,
    "Red_light_violation": 3,
    "panic_braking_with_stop": 2,
    "panic_braking": 1,
}
SEED = 42
MIN_CLEAN_RECOVERY_S = 7.0
MAX_CLEAN_RECOVERY_S = 15.0
ALIGN_TOLERANCE_S = 2.0
N_BOOTSTRAP = 2000


def args_parser():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drive-csv", type=Path, default=here / "relab+unibo_dataset.csv")
    p.add_argument(
        "--surt-csv", type=Path,
        default=here / "surt_logs_enriched_with_offline_activation.csv",
    )
    p.add_argument("--out", type=Path, default=here / "surt_recovery_results")
    p.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    return p.parse_args()


def event_onsets(group):
    group = group.copy()
    for col in EVENT_WEIGHTS:
        active = group[col].fillna(0).to_numpy() > 0
        group[col] = (active & ~np.r_[False, active[:-1]]).astype(np.int8)
    return group


def final_phase_episodes(surt):
    """One end timestamp per complete SURT trial, not one per phase."""
    surt = surt.copy()
    surt["Phase"] = pd.to_numeric(surt["Phase"], errors="coerce")
    surt["Trial"] = pd.to_numeric(surt["Trial"], errors="coerce")
    surt["trial_start"] = pd.to_datetime(surt["TrialStart_dt"], utc=True, errors="coerce")
    surt["response"] = pd.to_datetime(surt["Response_dt"], utc=True, errors="coerce")
    surt = surt.dropna(subset=["canonical_subject_id", "route", "trial_start", "response"])

    keys = ["canonical_subject_id", "session_id", "route", "Trial"]
    idx = surt.groupby(keys, observed=True)["Phase"].idxmax()
    episodes = surt.loc[idx].sort_values(["session_id", "trial_start"]).copy()
    episodes["next_trial_start"] = episodes.groupby("session_id")["trial_start"].shift(-1)
    episodes["clean_recovery_s"] = (
        episodes["next_trial_start"] - episodes["response"]
    ).dt.total_seconds()
    return episodes


def nearest_index(timestamps, target):
    position = int(np.searchsorted(timestamps, target))
    candidates = [i for i in (position - 1, position) if 0 <= i < len(timestamps)]
    if not candidates:
        return None
    return min(candidates, key=lambda i: abs(timestamps[i] - target))


def slope(values):
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)[ok]
    return float(np.polyfit(x, y[ok], 1)[0])


def interval_features(group, anchor_idx, clean_seconds):
    """Causal/pre-post HR features and clean/fixed future event outcomes."""
    clean_n = max(1, min(int(np.floor(clean_seconds)), int(MAX_CLEAN_RECOVERY_S)))
    pre = group.iloc[max(0, anchor_idx - 10) : anchor_idx]
    post5 = group.iloc[anchor_idx : min(len(group), anchor_idx + min(5, clean_n))]
    risk_end = min(len(group), anchor_idx + clean_n)
    clean_future = group.iloc[min(anchor_idx + 5, risk_end) : risk_end]
    fixed_future = group.iloc[anchor_idx + 5 : min(len(group), anchor_idx + 15)]

    baseline_prefix = group["hr"].iloc[:50].dropna()
    baseline = float(baseline_prefix.median()) if not baseline_prefix.empty else np.nan
    pre_hr = pre["hr"].dropna()
    post_hr = post5["hr"].dropna()

    def risk(frame):
        score = sum(
            weight * int((frame[col] > 0).any())
            for col, weight in EVENT_WEIGHTS.items()
        )
        return int(score > 0), float(score)

    clean_event, clean_score = risk(clean_future)
    fixed_event, fixed_score = risk(fixed_future)
    return {
        "clean_observation_s": max(0, clean_n - 5),
        "future_event_clean": clean_event,
        "future_risk_score_clean": clean_score,
        # Diagnostic only: usually contaminated by the next SURT task.
        "future_event_fixed_5_15": fixed_event,
        "future_risk_score_fixed_5_15": fixed_score,
        "hr_pre10": pre_hr.mean(),
        "hr_post5": post_hr.mean(),
        "hr_delta_post5_pre10": post_hr.mean() - pre_hr.mean(),
        "hr_slope_pre10": slope(pre["hr"]),
        "hr_slope_post5": slope(post5["hr"]),
        "hr_slope_change": slope(post5["hr"]) - slope(pre["hr"]),
        "hr_burden_plus3_post5": (
            float((post_hr > baseline + 3.0).mean())
            if len(post_hr) and np.isfinite(baseline) else np.nan
        ),
        "hr_coverage_post5": len(post_hr) / max(len(post5), 1),
    }


def build_pairs(drive, episodes):
    sessions = {}
    for (pid, route), group in drive.groupby(["id", "route"], sort=False):
        group = event_onsets(group.sort_values("Timestamp").reset_index(drop=True))
        sessions[(str(pid), str(route))] = group

    rows = []
    for _, episode in episodes.iterrows():
        pid, route = str(episode["canonical_subject_id"]), str(episode["route"])
        clean_s = float(episode["clean_recovery_s"])
        if not np.isfinite(clean_s) or clean_s < MIN_CLEAN_RECOVERY_S:
            continue
        distracted = sessions.get((pid, route))
        if distracted is None or len(distracted) < 30:
            continue

        # The other route is the participant's ND session.  P1/P2 themselves
        # are never interpreted as conditions.
        alternatives = [g for (p, r), g in sessions.items() if p == pid and r != route]
        if len(alternatives) != 1:
            continue
        control = alternatives[0]

        d_ts = distracted["Timestamp"].astype("int64").to_numpy() / 1e9
        anchor_s = episode["response"].value / 1e9
        d_idx = nearest_index(d_ts, anchor_s)
        if d_idx is None or abs(d_ts[d_idx] - anchor_s) > ALIGN_TOLERANCE_S:
            continue
        progress = d_idx / max(len(distracted) - 1, 1)
        c_idx = int(round(progress * (len(control) - 1)))

        d_feat = interval_features(distracted, d_idx, clean_s)
        c_feat = interval_features(control, c_idx, clean_s)
        row = {
            "id": pid,
            "distracted_route": route,
            "trial": episode["Trial"],
            "response_time": episode["response"],
            "clean_recovery_s": clean_s,
            "matched_progress": progress,
            "order_subgroup": episode.get("order_subgroup", ""),
        }
        for key in d_feat:
            row[f"{key}_recovery"] = d_feat[key]
            row[f"{key}_control"] = c_feat[key]
            row[f"{key}_delta"] = d_feat[key] - c_feat[key]
        rows.append(row)
    return pd.DataFrame(rows)


def participant_results(pairs):
    delta_cols = [c for c in pairs if c.endswith("_delta")]
    counts = pairs.groupby("id").size().rename("n_pairs")
    return pairs.groupby("id")[delta_cols].mean().join(counts).reset_index()


def bootstrap_participants(participants, n_bootstrap):
    rng = np.random.default_rng(SEED)
    rows = []
    for col in [c for c in participants if c.endswith("_delta")]:
        values = participants[col].dropna().to_numpy(float)
        if not len(values):
            continue
        boot = np.array([
            rng.choice(values, len(values), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        rows.append({
            "metric": col.removesuffix("_delta"),
            "n_participants": len(values),
            "mean_recovery_minus_control": values.mean(),
            "median_recovery_minus_control": np.median(values),
            "ci95_low": np.percentile(boot, 2.5),
            "ci95_high": np.percentile(boot, 97.5),
            "fraction_positive": np.mean(values > 0),
        })
    return pd.DataFrame(rows)


def main():
    args = args_parser()
    drive_cols = ["id", "route", "Timestamp", "hr", *EVENT_WEIGHTS]
    drive = pd.read_csv(args.drive_csv, usecols=drive_cols)
    drive["Timestamp"] = pd.to_datetime(drive["Timestamp"], utc=True, errors="coerce")
    drive = drive.dropna(subset=["Timestamp"])
    surt = pd.read_csv(args.surt_csv)
    episodes = final_phase_episodes(surt)

    gap_audit = pd.DataFrame({
        "threshold_s": [5, 7, 10, 15, 20],
        "n_episodes": [int((episodes["clean_recovery_s"] >= x).sum()) for x in [5, 7, 10, 15, 20]],
    })
    pairs = build_pairs(drive, episodes)
    if pairs.empty:
        raise RuntimeError("No SURT episodes could be aligned to paired D/ND driving sessions.")
    participants = participant_results(pairs)
    bootstrap = bootstrap_participants(participants, args.bootstrap)

    args.out.mkdir(parents=True, exist_ok=True)
    gap_audit.to_csv(args.out / "gap_audit.csv", index=False)
    pairs.to_csv(args.out / "matched_events.csv", index=False)
    participants.to_csv(args.out / "participant_contrasts.csv", index=False)
    bootstrap.to_csv(args.out / "bootstrap_results.csv", index=False)

    primary = bootstrap[bootstrap["metric"].isin([
        "future_event_clean", "future_risk_score_clean",
        "hr_delta_post5_pre10", "hr_slope_change", "hr_burden_plus3_post5",
    ])]
    print("\nSURT POST-DISTRACTION RECOVERY TEST")
    print(f"Final-phase episodes : {len(episodes)}")
    print(f"Matched clean pairs  : {len(pairs)}")
    print(f"Participants         : {participants['id'].nunique()}")
    print("\nClean-gap availability:")
    print(gap_audit.to_string(index=False))
    print("\nPrimary participant-bootstrap contrasts (recovery - ND control):")
    print(primary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to: {args.out}")
    print(
        "\nUse future_event_clean as the primary decision. The fixed 5-15 s "
        "outcome is diagnostic only because it commonly overlaps the next task."
    )


if __name__ == "__main__":
    main()
