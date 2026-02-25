import pandas as pd
import numpy as np
from datetime import timedelta
import random

# ----------------------------------------------------------------------
# Helper: compute frame rate and offset from original errors
# ----------------------------------------------------------------------
def compute_frame_statistics(df_err):
    df = df_err.sort_values('timestamp').copy()
    df['delta_frame'] = df['frame'].diff()
    df['delta_sim'] = df['sim_time_seconds'].diff()
    valid = (df['delta_sim'] > 0) & (df['delta_frame'] > 0)
    rates = df.loc[valid, 'delta_frame'] / df.loc[valid, 'delta_sim']
    rate_median = rates.median()
    intercepts = df['frame'] - rate_median * df['sim_time_seconds']
    intercept_median = intercepts.median()
    residuals = intercepts - intercept_median
    noise_std = residuals.std()
    return rate_median, intercept_median, noise_std

# ----------------------------------------------------------------------
# Helper: load and prepare distraction windows
# ----------------------------------------------------------------------
def load_distractions(distractions_file):
    df = pd.read_csv(distractions_file)
    df['timestamp_start'] = pd.to_datetime(df['timestamp_start'], format='ISO8601')
    df['timestamp_end']   = pd.to_datetime(df['timestamp_end'], format='ISO8601')
    return df

# ----------------------------------------------------------------------
# Main augmentation function
# ----------------------------------------------------------------------
def augment_errors_distraction(
    errors_file,
    distractions_file,
    output_file,
    new_users=None,
    new_runs=None,
    noise_scale=0.05
):
    if new_users is None:
        new_users = [f'participant_{i:02d}' for i in range(19, 26)]
    if new_runs is None:
        new_runs = [1, 2, 3]

    # Load original errors
    df_err = pd.read_csv(errors_file)
    df_err['timestamp'] = pd.to_datetime(df_err['timestamp'], format='ISO8601')

    # Filter to rows with valid model_pred and steer_angle_deg
    valid_err = df_err[
        (df_err['model_pred'].notna()) & 
        (df_err['model_pred'] != 'None') & 
        (df_err['steer_angle_deg'].notna())
    ].copy()
    if valid_err.empty:
        valid_err = df_err
        print("Warning: No rows with valid model_pred and steer_angle_deg. Using full dataset.")

    # Load augmented distractions
    df_dist = load_distractions(distractions_file)

    # ------------------------------------------------------------------
    # Compute empirical distributions from original errors
    # ------------------------------------------------------------------
    error_counts = df_err.groupby(['user_id', 'run_id']).size().tolist()
    if not error_counts:
        error_counts = [1]

    frame_rate, frame_intercept, frame_noise_std = compute_frame_statistics(df_err)

    continuous_cols = [
        'model_prob', 'emotion_prob', 'speed_kmh', 'steer_angle_deg',
        'x', 'y', 'z'
    ]
    stds = {col: df_err[col].std() for col in continuous_cols}

    categorical_cols = [
        'weather', 'map_name', 'error_type', 'model_pred',
        'emotion_label', 'road_id', 'lane_id', 'details'
    ]

    # Delay probabilities (normalized)
    raw_probs = {
        0: 0.121212, 1: 0.078788, 2: 0.055758, 3: 0.061453,
        4: 0.045895, 5: 0.049383, 6: 0.032403, 7: 0.030505,
        8: 0.026000, 9: 0.026344, 10: 0.020045, 11: 0.022619,
        12: 0.016291, 13: 0.017333, 14: 0.014327, 15: 0.012579,
        16: 0.005181, 17: 0.005976, 18: 0.009174, 19: 0.005391
    }
    total = sum(raw_probs.values())
    delay_probs = [raw_probs[i] / total for i in range(20)]

    # Build distraction window map AND compute run boundaries
    dist_map = {}
    run_end_map = {}
    for (user, run), group in df_dist.groupby(['user_id', 'run_id']):
        windows = []
        # Run ends at the latest window end (all windows are inside the same run)
        run_end = group['timestamp_end'].max()
        run_end_map[(user, run)] = run_end
        for _, row in group.iterrows():
            start_ts = row['timestamp_start']
            end_ts = row['timestamp_end']
            duration = (end_ts - start_ts).total_seconds()
            start_sim = row['sim_time_start']
            end_sim = row['sim_time_end']
            windows.append({
                'start_ts': start_ts,
                'end_ts': end_ts,
                'duration': duration,
                'start_sim': start_sim,
                'end_sim': end_sim
            })
        dist_map[(user, run)] = windows

    # ------------------------------------------------------------------
    # Generate synthetic error rows
    # ------------------------------------------------------------------
    synthetic_rows = []

    for user in new_users:
        for run in new_runs:
            key = (user, run)
            if key not in dist_map:
                print(f"Warning: No distraction windows for {user}, run {run}. Skipping.")
                continue
            windows = dist_map[key]
            run_end = run_end_map[key]

            n_errors = np.random.choice(error_counts)

            for _ in range(n_errors):
                win = random.choice(windows)
                delay = np.random.choice(20, p=delay_probs)

                if delay == 0:
                    offset = random.uniform(0, win['duration'])   # inside window
                else:
                    offset = delay   # could be after window (if delay > duration)

                error_ts = win['start_ts'] + timedelta(seconds=offset)

                # Skip if the error would occur after the run ends
                if error_ts > run_end:
                    continue

                error_sim = win['start_sim'] + offset

                frame_pred = frame_rate * error_sim + frame_intercept
                frame_noise = np.random.normal(0, frame_noise_std)
                error_frame = int(round(frame_pred + frame_noise))

                # Bootstrap from valid templates only
                template = valid_err.sample(1).iloc[0].to_dict()

                new_row = {}
                for col in categorical_cols:
                    new_row[col] = template[col]
                new_row['user_id'] = user
                new_row['run_id']  = run
                for col in continuous_cols:
                    val = template[col]
                    noise = np.random.normal(0, noise_scale * stds[col])
                    new_row[col] = val + noise

                new_row['timestamp'] = error_ts.isoformat()
                new_row['sim_time_seconds'] = error_sim
                new_row['frame'] = error_frame

                synthetic_rows.append(new_row)

    # ------------------------------------------------------------------
    # Combine and save
    # ------------------------------------------------------------------
    df_synthetic = pd.DataFrame(synthetic_rows)
    df_augmented = pd.concat([df_err, df_synthetic], ignore_index=True)
    df_augmented.to_csv(output_file, index=False)

    print(f"Augmented errors-distraction dataset saved to {output_file}")
    print(f"Added {len(synthetic_rows)} synthetic rows. Total rows: {len(df_augmented)}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    augment_errors_distraction(
        errors_file='Dataset Errors_distraction.csv',
        distractions_file='Dataset Distractions_distraction.csv',
        output_file='Dataset Errors_distraction.csv',
        new_users=[f'participant_{i:02d}' for i in range(19, 26)],
        new_runs=[1, 2, 3, 4],
        noise_scale=0.05
    )