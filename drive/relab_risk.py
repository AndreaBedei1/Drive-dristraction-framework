#!/usr/bin/env python3
"""
Arousal & distraction influence on driving errors — relab+unibo dataset.

Dataset: 44 participants × up to 2 sessions (route P1/P2).
Each row is a timestamped observation (~1.8 s sampling interval).

Analysis:
1. Event-anchored lookback (5/10/15 s) → compare arousal/HR before error vs non-error
2. Arousal bins → error rate per bin
3. Distraction flag (safe_driving < 0.5) → combined arousal×distraction breakdown
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr, chi2_contingency

DATA = Path(__file__).parent / "relab+unibo_dataset.csv"

ERR_COLS = [
    "Collision", "Red_light_violation", "center_line_crossing",
    "panic_braking", "panic_braking_with_stop", "sharp_turn",
]
LOOKBACKS = [25]   # seconds


# ── load ──────────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["id", "route", "Timestamp"]).reset_index(drop=True)
    df["has_error"]     = (df[ERR_COLS] > 0).any(axis=1).astype(int)
    df["is_distracted"] = (df["safe_driving"] < 0.5).astype(int)
    return df


# ── feature extraction ────────────────────────────────────────────────────────

def extract_lookback(df: pd.DataFrame, lookback_s: int) -> pd.DataFrame:
    """
    For every row compute mean arousal/HR and distraction fraction
    in the [t - lookback_s, t) window within the same (id, route) session.
    Rows with no valid lookback data are dropped.
    """
    records = []
    lb = pd.Timedelta(seconds=lookback_s)

    for (pid, route), grp in df.groupby(["id", "route"]):
        grp = grp.sort_values("Timestamp").reset_index(drop=True)
        ts  = grp["Timestamp"].values
        ar  = grp["arousal"].values
        hr  = grp["hr"].values
        dst = grp["is_distracted"].values
        err = grp["has_error"].values

        for i in range(len(grp)):
            t0   = grp["Timestamp"].iloc[i]
            mask = (grp["Timestamp"] >= t0 - lb) & (grp["Timestamp"] < t0)
            win  = grp[mask]
            if len(win) < 2:
                continue
            ar_vals = win["arousal"].dropna().values
            hr_vals = win["hr"].dropna().values
            if len(ar_vals) < 2:
                continue
            records.append({
                "id":           pid,
                "route":        route,
                "timestamp":    t0,
                "has_error":    err[i],
                "is_distracted": dst[i],
                "ar_mean":      float(np.mean(ar_vals)),
                "ar_slope":     float(np.polyfit(range(len(ar_vals)), ar_vals, 1)[0]),
                "hr_mean":      float(np.mean(hr_vals)) if len(hr_vals) >= 2 else np.nan,
                "dist_frac":    float(win["is_distracted"].mean()),
            })

    feat = pd.DataFrame(records)

    # ── per-participant z-score normalization ─────────────────────────────────
    # Compute mean/std of ar_mean per participant across all their windows,
    # then normalize: ar_z = (ar_mean - participant_mean) / participant_std
    stats = feat.groupby("id")["ar_mean"].agg(["mean", "std"]).rename(
        columns={"mean": "pid_ar_mean", "std": "pid_ar_std"})
    feat = feat.join(stats, on="id")
    feat["pid_ar_std"] = feat["pid_ar_std"].fillna(1.0).clip(lower=1e-6)
    feat["ar_z"] = (feat["ar_mean"] - feat["pid_ar_mean"]) / feat["pid_ar_std"]

    return feat


# ── analysis helpers ──────────────────────────────────────────────────────────

def mann_whitney(feat: pd.DataFrame, col: str) -> tuple:
    err  = feat.loc[feat["has_error"] == 1, col].dropna()
    noerr = feat.loc[feat["has_error"] == 0, col].dropna()
    stat, p = mannwhitneyu(err, noerr, alternative="two-sided")
    delta = err.mean() - noerr.mean()
    return float(delta), float(p), float(err.mean()), float(noerr.mean())


def bin_analysis(feat: pd.DataFrame, col: str, n_bins: int = 5) -> pd.DataFrame:
    feat = feat.dropna(subset=[col]).copy()
    feat["bin"] = pd.qcut(feat[col], n_bins, labels=False, duplicates="drop")
    tbl = (feat.groupby("bin")["has_error"]
               .agg(["mean", "sum", "count"])
               .rename(columns={"mean": "error_rate", "sum": "n_errors", "count": "n_total"}))
    # edges for display
    edges = feat.groupby("bin")[col].agg(["min", "max"])
    tbl   = pd.concat([edges, tbl], axis=1)

    rho, p_rho = spearmanr(feat[col], feat["has_error"])
    chi2, p_chi, *_ = chi2_contingency(
        pd.crosstab(feat["bin"], feat["has_error"])
    )
    return tbl, float(rho), float(p_rho), float(chi2), float(p_chi)


# ── main analyses ─────────────────────────────────────────────────────────────

def arousal_influence(feat: pd.DataFrame) -> None:
    print("\n=== AROUSAL INFLUENCE ON ERRORS (absolute) ===")
    delta, p, mu_err, mu_ok = mann_whitney(feat, "ar_mean")
    print(f"  ar_mean before error : {mu_err:.4f}")
    print(f"  ar_mean before no-err: {mu_ok:.4f}")
    print(f"  Δ = {delta:+.4f}   Mann-Whitney p = {p:.3e}")

    tbl, rho, p_rho, chi2, p_chi = bin_analysis(feat, "ar_mean")
    print(f"\n  Spearman ρ = {rho:+.4f}  p = {p_rho:.3e}")
    print(f"  Chi-squared across bins: χ² = {chi2:.2f}  p = {p_chi:.3e}")
    print("\n  Error rate by arousal quintile (absolute):")
    print(tbl[["min","max","error_rate","n_errors","n_total"]].to_string(float_format="%.3f"))

    print("\n--- PERSONALIZED (per-participant z-score) ---")
    delta_z, p_z, mu_err_z, mu_ok_z = mann_whitney(feat, "ar_z")
    print(f"  ar_z before error : {mu_err_z:.4f}")
    print(f"  ar_z before no-err: {mu_ok_z:.4f}")
    print(f"  Δ = {delta_z:+.4f}   Mann-Whitney p = {p_z:.3e}")

    tbl_z, rho_z, p_rho_z, chi2_z, p_chi_z = bin_analysis(feat, "ar_z")
    print(f"\n  Spearman ρ = {rho_z:+.4f}  p = {p_rho_z:.3e}")
    print(f"  Chi-squared across bins: χ² = {chi2_z:.2f}  p = {p_chi_z:.3e}")
    print("\n  Error rate by ar_z quintile (personalized):")
    print(tbl_z[["min","max","error_rate","n_errors","n_total"]].to_string(float_format="%.3f"))


def hr_influence(feat: pd.DataFrame) -> None:
    print("\n=== HR INFLUENCE ON ERRORS ===")
    feat_hr = feat.dropna(subset=["hr_mean"])
    delta, p, mu_err, mu_ok = mann_whitney(feat_hr, "hr_mean")
    print(f"  hr_mean before error : {mu_err:.2f}")
    print(f"  hr_mean before no-err: {mu_ok:.2f}")
    print(f"  Δ = {delta:+.2f}   Mann-Whitney p = {p:.3e}")
    rho, p_rho = spearmanr(feat_hr["hr_mean"], feat_hr["has_error"])
    print(f"  Spearman ρ = {rho:+.4f}  p = {p_rho:.3e}")


def distraction_influence(feat: pd.DataFrame) -> None:
    print("\n=== DISTRACTION × AROUSAL INFLUENCE ON ERRORS ===")
    for dist_val, label in [(0, "Not distracted"), (1, "Distracted")]:
        sub = feat[feat["is_distracted"] == dist_val]
        rate = sub["has_error"].mean()
        print(f"  {label:20s}: error_rate = {rate:.4f}  (n={len(sub)})")

    # Combined: distraction × arousal bin
    print("\n  Error rate by distraction × arousal quintile:")
    feat2 = feat.dropna(subset=["ar_mean"]).copy()
    feat2["ar_bin"] = pd.qcut(feat2["ar_mean"], 5, labels=False, duplicates="drop")
    tbl = (feat2.groupby(["is_distracted", "ar_bin"])["has_error"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "error_rate", "count": "n"}))
    print(tbl.to_string(float_format="%.3f"))

    # Spearman ρ within distracted-only
    dist_sub = feat2[feat2["is_distracted"] == 1].dropna(subset=["ar_mean"])
    if len(dist_sub) > 10:
        rho, p = spearmanr(dist_sub["ar_mean"], dist_sub["has_error"])
        print(f"\n  Spearman ρ (distracted only): {rho:+.4f}  p = {p:.3e}  (n={len(dist_sub)})")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    df = load()
    print(f"Dataset: {df['id'].nunique()} participants, "
          f"{df.groupby(['id','route']).ngroups} sessions, "
          f"{len(df)} rows")
    print(f"Error rows: {df['has_error'].sum()} ({df['has_error'].mean()*100:.1f}%)")
    print(f"Distracted rows: {df['is_distracted'].sum()} ({df['is_distracted'].mean()*100:.1f}%)")

    for lb in LOOKBACKS:
        print(f"\n{'='*60}")
        print(f"LOOKBACK = {lb} s")
        print(f"{'='*60}")
        feat = extract_lookback(df, lb)
        print(f"  Feature rows: {len(feat)}  "
              f"(error={feat['has_error'].sum()}, no-error={( feat['has_error']==0).sum()})")
        arousal_influence(feat)
        hr_influence(feat)
        distraction_influence(feat)


if __name__ == "__main__":
    main()
