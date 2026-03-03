from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ERR_COLS = [
    "user_id",
    "run_id",
    "weather",
    "map_name",
    "error_type",
    "model_pred",
    "model_prob",
    "emotion_label",
    "emotion_prob",
    "speed_kmh",
    "steer_angle_deg",
    "timestamp",
    "frame",
    "sim_time_seconds",
    "x",
    "y",
    "z",
    "road_id",
    "lane_id",
    "details",
]

BASE_ERR_COLS = [
    "user_id",
    "run_id",
    "weather",
    "map_name",
    "error_type",
    "model_pred",
    "model_prob",
    "emotion_label",
    "emotion_prob",
    "speed_kmh",
    "timestamp",
    "frame",
    "sim_time_seconds",
    "x",
    "y",
    "z",
    "road_id",
    "lane_id",
    "details",
    "steer_angle_deg",
]

MODEL_LABELS = ["pick_floor", "phone_call", "reach_back", "drink", "eat"]
EMOTION_LABELS = ["fear", "angry", "sad", "neutral"]


def _load_old_datasets(data_dir: Path):
    distractions = pd.read_csv(data_dir / "Dataset Distractions_distraction_old.csv")
    errors_base = pd.read_csv(data_dir / "Dataset Errors_baseline_old.csv")
    driving_base = pd.read_csv(data_dir / "Dataset Driving Time_baseline_old.csv")

    distractions["timestamp_start"] = pd.to_datetime(distractions["timestamp_start"], format="ISO8601", errors="coerce")
    distractions["timestamp_end"] = pd.to_datetime(distractions["timestamp_end"], format="ISO8601", errors="coerce")
    errors_base["timestamp"] = pd.to_datetime(errors_base["timestamp"], format="ISO8601", errors="coerce")
    driving_base["timestamp"] = pd.to_datetime(driving_base["timestamp"], format="ISO8601", errors="coerce")

    distractions = distractions.dropna(subset=["timestamp_start", "timestamp_end"]).copy()
    errors_base = errors_base.dropna(subset=["timestamp"]).copy()
    driving_base = driving_base.dropna(subset=["timestamp"]).copy()
    for col in ["model_pred_start", "model_pred_end", "emotion_label_start", "emotion_label_end"]:
        distractions[col] = distractions[col].astype(object)
    return distractions, errors_base, driving_base


def _baseline_map(driving_base: pd.DataFrame):
    frame = (
        driving_base.groupby("user_id")
        .agg(hr_baseline=("hr_baseline", "median"), arousal_baseline=("arousal_baseline", "median"))
        .reset_index()
    )
    return {
        row.user_id: (float(row.hr_baseline), float(row.arousal_baseline))
        for row in frame.itertuples(index=False)
    }


def _synthesize_distractions(distractions: pd.DataFrame, base_map, user_risk) -> pd.DataFrame:
    out = distractions.copy()
    severity = []
    for idx, row in out.iterrows():
        uid = row["user_id"]
        hr0, ar0 = base_map.get(uid, (70.0, 0.35))
        duration_s = max(3.0, float((row["timestamp_end"] - row["timestamp_start"]).total_seconds()))

        latent = (
            0.40 * user_risk[uid]
            + 0.30 * float(np.clip(np.random.normal(0.65, 0.18), 0.05, 0.98))
            + 0.30 * float(np.clip((duration_s - 5.0) / 25.0, 0.0, 1.0))
        )
        sev = float(np.clip(latent, 0.10, 0.98))
        severity.append(sev)

        hr_delta_start = 5 + 18 * sev + np.random.normal(0, 0.9)
        hr_delta_end = hr_delta_start + 10 + 18 * sev + np.random.normal(0, 0.9)
        ar_delta_start = 0.04 + 0.16 * sev + np.random.normal(0, 0.006)
        ar_delta_end = ar_delta_start + 0.08 + 0.16 * sev + np.random.normal(0, 0.006)
        model_prob_start = np.clip(0.48 + 0.16 * sev + np.random.normal(0, 0.035), 0.15, 0.90)
        model_prob_end = np.clip(model_prob_start + 0.04 + 0.08 * sev + np.random.normal(0, 0.03), 0.18, 0.95)
        emotion_prob_start = np.clip(0.30 + 0.55 * sev + np.random.normal(0, 0.02), 0.08, 0.98)
        emotion_prob_end = np.clip(emotion_prob_start + 0.12 + 0.18 * sev + np.random.normal(0, 0.015), 0.12, 0.995)

        out.at[idx, "model_pred_start"] = random.choice(MODEL_LABELS)
        out.at[idx, "model_pred_end"] = out.at[idx, "model_pred_start"]
        out.at[idx, "emotion_label_start"] = "neutral" if sev < 0.25 else random.choice(["sad", "surprise"])
        out.at[idx, "emotion_label_end"] = random.choice(["fear", "angry", "sad"]) if sev > 0.35 else random.choice(["neutral", "sad"])
        out.at[idx, "model_prob_start"] = round(float(model_prob_start), 3)
        out.at[idx, "model_prob_end"] = round(float(model_prob_end), 3)
        out.at[idx, "emotion_prob_start"] = round(float(emotion_prob_start), 3)
        out.at[idx, "emotion_prob_end"] = round(float(emotion_prob_end), 3)
        out.at[idx, "hr_bpm_start"] = round(float(np.clip(hr0 + hr_delta_start, 45, 180)), 1)
        out.at[idx, "hr_bpm_end"] = round(float(np.clip(hr0 + hr_delta_end, 50, 200)), 1)
        out.at[idx, "arousal_start"] = round(float(np.clip(ar0 + ar_delta_start, 0.05, 0.92)), 3)
        out.at[idx, "arousal_end"] = round(float(np.clip(ar0 + ar_delta_end, 0.10, 0.98)), 3)

    out["severity"] = severity
    return out


def _synthesize_distraction_errors(distractions: pd.DataFrame, base_map) -> pd.DataFrame:
    rows = []
    for _, row in distractions.iterrows():
        sev = float(row["severity"])
        uid = row["user_id"]
        hr0, ar0 = base_map.get(uid, (70.0, 0.35))
        duration_s = max((row["timestamp_end"] - row["timestamp_start"]).total_seconds(), 1.0)

        hr_score = np.clip((float(row["hr_bpm_end"]) - hr0) / 35.0, 0.0, 1.5)
        ar_score = np.clip((float(row["arousal_end"]) - ar0) / 0.35, 0.0, 1.5)
        model_score = float(row["model_prob_end"])
        emotion_score = float(row["emotion_prob_end"])
        risk = (
            0.14 * model_score
            + 0.28 * emotion_score
            + 0.24 * np.clip(hr_score, 0, 1)
            + 0.18 * np.clip(ar_score, 0, 1)
            + 0.10 * (emotion_score * np.clip(hr_score, 0, 1))
            + 0.06 * (emotion_score * np.clip(ar_score, 0, 1))
        )

        if risk > 0.78:
            n_errors = 3
        elif risk > 0.60:
            n_errors = 2
        elif risk > 0.45:
            n_errors = 1
        else:
            n_errors = 0

        for j in range(n_errors):
            if j == 0:
                offset = np.random.uniform(max(0.0, duration_s - 1.0), duration_s + 0.5)
            else:
                offset = duration_s + np.random.uniform(0.2, min(5.0, 1.0 + 3 * sev))
            rows.append(
                {
                    "user_id": row["user_id"],
                    "run_id": int(row["run_id"]),
                    "weather": row["weather"],
                    "map_name": row["map_name"],
                    "error_type": random.choice(["Stop sign violation", "Red light violation", "Harsh braking", "Solid line crossing"]),
                    "model_pred": row["model_pred_end"],
                    "model_prob": round(float(np.clip(row["model_prob_end"] - np.random.uniform(0, 0.02), 0.3, 0.999)), 3),
                    "emotion_label": row["emotion_label_end"],
                    "emotion_prob": round(float(np.clip(row["emotion_prob_end"] - np.random.uniform(0, 0.03), 0.2, 0.995)), 3),
                    "speed_kmh": round(float(np.clip(np.random.normal(25 + 15 * sev, 4), 5, 90)), 3),
                    "steer_angle_deg": round(float(np.random.normal(0, 8 + 8 * sev)), 3),
                    "timestamp": row["timestamp_start"] + pd.Timedelta(seconds=float(offset)),
                    "frame": 1000 + j,
                    "sim_time_seconds": round(float(offset), 3),
                    "x": round(float(np.random.normal(0, 50)), 6),
                    "y": round(float(np.random.normal(0, 50)), 6),
                    "z": 0.0,
                    "road_id": 1,
                    "lane_id": 1,
                    "details": f"synth_sev={sev:.3f};risk={float(risk):.3f}",
                }
            )
    return pd.DataFrame(rows, columns=ERR_COLS).sort_values(["user_id", "run_id", "timestamp"]).reset_index(drop=True)


def _synthesize_driving_time(driving_base: pd.DataFrame, user_risk) -> pd.DataFrame:
    out = driving_base.copy().sort_values(["user_id", "run_id"]).reset_index(drop=True)
    for idx, row in out.iterrows():
        uid = row["user_id"]
        new_secs = max(600.0, float(row["run_duration_seconds"]) + np.random.normal(40, 15))
        out.at[idx, "run_duration_seconds"] = round(new_secs, 3)
        out.at[idx, "run_duration_minutes"] = round(new_secs / 60.0, 3)
        out.at[idx, "hr_baseline"] = round(float(np.clip(float(row["hr_baseline"]) + np.random.normal(0, 1.0), 50, 90)), 1)
        out.at[idx, "arousal_baseline"] = round(float(np.clip(float(row["arousal_baseline"]) + np.random.normal(0, 0.008), 0.15, 0.50)), 3)
    for uid, idxs in out.groupby("user_id").groups.items():
        total = 0.0
        for idx in sorted(list(idxs)):
            total += float(out.at[idx, "run_duration_seconds"])
            out.at[idx, "total_duration_seconds"] = round(total, 3)
            out.at[idx, "total_duration_minutes"] = round(total / 60.0, 3)
    return out


def _synthesize_baseline_errors(driving_base: pd.DataFrame, user_risk) -> pd.DataFrame:
    rows = []
    for _, row in driving_base.iterrows():
        uid = row["user_id"]
        risk = user_risk.get(uid, 0.5)
        run_secs = float(row["run_duration_seconds"])
        n_errors = max(0, int(round((0.10 + 0.45 * risk) * run_secs / 2200.0)))
        for j in range(n_errors):
            tsec = float(np.random.uniform(20, max(30, run_secs - 20)))
            rows.append(
                {
                    "user_id": uid,
                    "run_id": int(row["run_id"]),
                    "weather": row["weather"],
                    "map_name": row["map_name"],
                    "error_type": random.choice(["Harsh braking", "Stop sign violation"]),
                    "model_pred": "None",
                    "model_prob": round(float(np.clip(0.08 + 0.08 * risk + np.random.normal(0, 0.02), 0.02, 0.30)), 3),
                    "emotion_label": "neutral",
                    "emotion_prob": round(float(np.clip(0.08 + 0.08 * risk + np.random.normal(0, 0.02), 0.02, 0.30)), 3),
                    "speed_kmh": round(float(np.clip(np.random.normal(20 + 5 * risk, 4), 0, 60)), 3),
                    "timestamp": pd.Timestamp(row["timestamp"]) - pd.Timedelta(seconds=(run_secs - tsec)),
                    "frame": 2000 + j,
                    "sim_time_seconds": round(tsec, 3),
                    "x": round(float(np.random.normal(0, 50)), 6),
                    "y": round(float(np.random.normal(0, 50)), 6),
                    "z": 0.0,
                    "road_id": 1,
                    "lane_id": 1,
                    "details": f"baseline_user_risk={risk:.3f}",
                    "steer_angle_deg": round(float(np.random.normal(0, 3)), 3),
                }
            )
    return pd.DataFrame(rows, columns=BASE_ERR_COLS).sort_values(["user_id", "run_id", "timestamp"]).reset_index(drop=True)


def _save_iso(df: pd.DataFrame, timestamp_cols, path: Path):
    out = df.copy()
    for col in timestamp_cols:
        out[col] = pd.to_datetime(out[col], format="ISO8601", errors="coerce")
        fmt = "%Y-%m-%dT%H:%M:%S.%f" if "Driving Time" in path.name else "%Y-%m-%d %H:%M:%S.%f"
        out[col] = out[col].dt.strftime(fmt)
    out.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Create a synthetic benchmark dataset for ftd_hmm.py")
    parser.add_argument("--data-dir", default="drive/data")
    parser.add_argument("--output-dir", default="drive/data_hmm_synth_v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    distractions, errors_base, driving_base = _load_old_datasets(data_dir)
    base_map = _baseline_map(driving_base)
    users = sorted(distractions["user_id"].unique())
    user_risk = {u: float(np.clip(np.random.beta(2.2, 2.2), 0.20, 0.9)) for u in users}

    distractions_syn = _synthesize_distractions(distractions, base_map, user_risk)
    errors_dist_syn = _synthesize_distraction_errors(distractions_syn, base_map)
    driving_base_syn = _synthesize_driving_time(driving_base, user_risk)
    errors_base_syn = _synthesize_baseline_errors(driving_base_syn, user_risk)

    distractions_save = distractions_syn.drop(columns=["severity"]).copy()
    _save_iso(distractions_save, ["timestamp_start", "timestamp_end"], output_dir / "Dataset Distractions_distraction.csv")
    _save_iso(errors_dist_syn, ["timestamp"], output_dir / "Dataset Errors_distraction.csv")
    _save_iso(errors_base_syn, ["timestamp"], output_dir / "Dataset Errors_baseline.csv")
    _save_iso(driving_base_syn, ["timestamp"], output_dir / "Dataset Driving Time_baseline.csv")

    print(f"Saved synthetic benchmark dataset to {output_dir}")


if __name__ == "__main__":
    main()
