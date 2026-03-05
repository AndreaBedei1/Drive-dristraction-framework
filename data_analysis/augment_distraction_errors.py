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
        new_runs = [1, 2, 3, 4]

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

    # Load augmented distractions (already contains original + synthetic)
    df_dist = load_distractions(distractions_file)

    # ------------------------------------------------------------------
    # Compute empirical distributions from original errors
    # ------------------------------------------------------------------
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
        0: 0.151212, 1: 0.10788, 2: 0.105758, 3: 0.101453,
        4: 0.0575895, 5: 0.0669383, 6: 0.062403, 7: 0.040505,
        8: 0.036000, 9: 0.029344, 10: 0.026045, 11: 0.022619,
        12: 0.056291, 13: 0.057333, 14: 0.054327, 15: 0.052579,
        16: 0.055181, 17: 0.035976, 18: 0.019174, 19: 0.015391
    }
    total = sum(raw_probs.values())
    delay_probs = [raw_probs[i] / total for i in range(20)]

    # Compute run end for each (user, run) from distraction windows
    run_end_map = {}
    for (user, run), group in df_dist.groupby(['user_id', 'run_id']):
        run_end_map[(user, run)] = group['timestamp_end'].max()

    # ------------------------------------------------------------------
    # Generate one error per distraction event
    # ------------------------------------------------------------------
    synthetic_rows = []

    # Iterate over all distraction events (including synthetic ones)
    for idx, row in df_dist.iterrows():
        user = row['user_id']
        run = row['run_id']
        win_start = row['timestamp_start']
        win_end   = row['timestamp_end']
        win_duration = (win_end - win_start).total_seconds()
        win_sim_start = row['sim_time_start']

        # Only generate for new users/runs
        if user not in new_users or run not in new_runs:
            continue

        # Sample delay according to distribution
        delay = np.random.choice(20, p=delay_probs)

        if delay == 0:
            offset = random.uniform(0, win_duration)   # inside window
        else:
            offset = delay   # after window start (could be after window ends)

        error_ts = win_start + timedelta(seconds=offset)

        # Ensure error does not exceed the run end
        run_end = run_end_map[(user, run)]
        if error_ts > run_end:
            # Skip this error (preserves distribution of feasible delays)
            continue

        error_sim = win_sim_start + offset

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

        # -------------------- Apply constraints --------------------
        # Ensure probabilities are within [0, 1]
        new_row['model_prob']   = np.clip(new_row['model_prob'], 0.0, 1.0)
        new_row['emotion_prob'] = np.clip(new_row['emotion_prob'], 0.0, 1.0)
        # ----------------------------------------------------------

        new_row['timestamp'] = error_ts.isoformat()
        new_row['sim_time_seconds'] = error_sim
        new_row['frame'] = abs(error_frame)

        synthetic_rows.append(new_row)

    # ------------------------------------------------------------------
    # Combine and save
    # ------------------------------------------------------------------
    df_synthetic = pd.DataFrame(synthetic_rows)
    df_augmented = pd.concat([df_err, df_synthetic], ignore_index=True)
    df_augmented.to_csv(output_file, index=False)

    print(f"Augmented errors-distraction dataset saved to {output_file}")
    print(f"Added {len(synthetic_rows)} synthetic rows. Total rows: {len(df_augmented)}")


