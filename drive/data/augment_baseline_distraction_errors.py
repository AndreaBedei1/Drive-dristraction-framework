import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
# Helper: generate synthetic driving time rows for missing (user, run)
# ----------------------------------------------------------------------
def generate_synthetic_driving_time(driving_time_df, target_users, target_runs, n_mixup_samples=1):
    """
    Generate synthetic driving time rows for each (user, run) not already present,
    using mixup of existing rows.

    Parameters:
        driving_time_df (pd.DataFrame): Original driving time data.
        target_users (list): List of user_id strings (e.g., ['participant_19', ...]).
        target_runs (list): List of run_id integers.
        n_mixup_samples (int): Number of synthetic rows to generate per missing (user, run).
                                Usually 1 because each run is one row.

    Returns:
        pd.DataFrame: Synthetic driving time rows.
    """
    numerical_cols = [
        'run_duration_seconds', 'run_duration_minutes',
        'total_duration_seconds', 'total_duration_minutes',
        'hr_baseline', 'arousal_baseline'
    ]
    data = driving_time_df[numerical_cols].values
    n_rows = data.shape[0]
    synthetic_rows = []

    # We'll also need to assign cumulative totals sequentially for each user.
    # Since we generate per run, we'll handle cumulative totals after generation.
    # First generate raw run durations.
    for user in target_users:
        cum_seconds = 0
        cum_minutes = 0
        for run in sorted(target_runs):
            # Check if already exists in original
            if ((driving_time_df['user_id'] == user) & (driving_time_df['run_id'] == run)).any():
                # Skip, already present (should not happen for new users)
                continue

            # Generate one synthetic run using mixup
            idx1, idx2 = np.random.choice(n_rows, size=2, replace=False)
            row1 = data[idx1]
            row2 = data[idx2]
            alpha = np.random.beta(0.2, 0.2)
            mixed = alpha * row1 + (1 - alpha) * row2

            # Build row
            new_row = {
                'user_id': user,
                'run_id': run,
                'weather': 'day',
                'map_name': 'Town10HD',
                'timestamp': None,  # will set later
                'run_duration_seconds': mixed[0],
                'run_duration_minutes': mixed[1],
                'total_duration_seconds': cum_seconds + mixed[0],
                'total_duration_minutes': cum_minutes + mixed[1],
                'hr_baseline': mixed[4],
                'arousal_baseline': mixed[5]
            }
            # Assign a plausible timestamp: base on last timestamp in original + random offset
            # We'll set later, after we have cumulative times.
            synthetic_rows.append(new_row)

            cum_seconds += mixed[0]
            cum_minutes += mixed[1]

    # Now assign timestamps: we need to decide a base time for each user.
    # Use the latest timestamp from original driving time as reference.
    last_ts = pd.to_datetime(driving_time_df['timestamp']).max()
    # For each user, shift by a random number of days (1-10) to separate them.
    user_base = {}
    for user in target_users:
        days_offset = random.randint(1, 4)
        user_base[user] = last_ts + timedelta(days=days_offset)

    # For each synthetic row, compute timestamp based on cumulative total_duration_seconds from start of user's first run.
    # We'll need to know the cumulative for each run. We have total_duration_seconds already.
    # But we need the start time of the first run for each user.
    for row in synthetic_rows:
        user = row['user_id']
        run = row['run_id']
        # Find the total_duration_seconds at the end of this run
        total_end = row['total_duration_seconds']
        # Find the total_duration_seconds at the start of this run (previous run's total)
        prev_total = 0
        # We need to look at other rows for same user with lower run_id
        # Since we generated in order, we can accumulate in a dict, but easier: compute from stored list
        # Instead, we can compute start time as user_base[user] + timedelta(seconds=total_end - run_duration_seconds)
        run_dur = row['run_duration_seconds']
        start_offset = total_end - run_dur
        row['timestamp'] = (user_base[user] + timedelta(seconds=start_offset)).isoformat()

    return pd.DataFrame(synthetic_rows)

# ----------------------------------------------------------------------
# Main augmentation function for errors
# ----------------------------------------------------------------------
def augment_errors_with_new_users(
    errors_file,
    driving_time_file,
    output_file,
    new_users=None,
    new_runs=None,
    noise_scale=0.05
):
    """
    Augment errors dataset by adding synthetic rows for new users and runs.

    Parameters:
        errors_file (str): Path to original errors CSV.
        driving_time_file (str): Path to original driving time CSV.
        output_file (str): Path to save augmented errors CSV.
        new_users (list): List of user_id strings (e.g., ['participant_19', ...]).
        new_runs (list): List of run_id integers.
        noise_scale (float): Relative scale of Gaussian noise added to continuous columns.
    """
    if new_users is None:
        new_users = [f'participant_{i:02d}' for i in range(19, 26)]
    if new_runs is None:
        new_runs = [1, 2, 3]

    # Load datasets
    df_err = pd.read_csv(errors_file)
    df_drive = pd.read_csv(driving_time_file)

    # Ensure timestamp columns are datetime
    df_err['timestamp'] = pd.to_datetime(df_err['timestamp'], format="ISO8601")
    df_drive['timestamp'] = pd.to_datetime(df_drive['timestamp'], format="ISO8601")

    # ------------------------------------------------------------------
    # Step 1: Ensure driving time contains entries for all new (user, run)
    # ------------------------------------------------------------------
    # Combine original and synthetic driving time
    drive_synth = generate_synthetic_driving_time(df_drive, new_users, new_runs, n_mixup_samples=1)
    df_drive_aug = pd.concat([df_drive, drive_synth], ignore_index=True)

    # Build a map from (user, run) to start info:
    #   start_timestamp, run_duration_seconds, total_duration_seconds (end of run)
    # Also need total_duration_seconds at beginning of run (prev_total)
    run_info = {}
    for (user, run), group in df_drive_aug.groupby(['user_id', 'run_id']):
        # There should be exactly one row per run
        row = group.iloc[0]
        total_end = row['total_duration_seconds']
        run_dur = row['run_duration_seconds']
        total_start = total_end - run_dur
        start_ts = row['timestamp'] - timedelta(seconds=run_dur)  # timestamp is at end of run
        run_info[(user, run)] = {
            'start_timestamp': start_ts,
            'run_duration': run_dur,
            'total_start': total_start,
            'total_end': total_end
        }

    # ------------------------------------------------------------------
    # Step 2: Prepare for error generation
    # ------------------------------------------------------------------
    # Identify continuous columns to add noise to
    continuous_cols = [
        'model_prob', 'emotion_prob', 'speed_kmh',
        'x', 'y', 'z', 'steer_angle_deg'
    ]
    # Columns we will generate explicitly from time offset
    generated_cols = ['timestamp', 'sim_time_seconds', 'frame']
    # Categorical columns copied from bootstrap
    categorical_cols = [
        'user_id', 'run_id', 'weather', 'map_name', 'error_type',
        'model_pred', 'emotion_label', 'road_id', 'lane_id', 'details'
    ]

    # Precompute global standard deviations for continuous columns
    stds = {col: df_err[col].std() for col in continuous_cols}

    # Compute global frame rate (frames per second) from original data
    # Use median of (frame / sim_time_seconds) to be robust
    df_err_clean = df_err[(df_err['sim_time_seconds'] > 0) & (df_err['frame'] > 0)]
    frame_rates = df_err_clean['frame'] / df_err_clean['sim_time_seconds']
    global_frame_rate = frame_rates.median()  # frames per second

    synthetic_rows = []

    # ------------------------------------------------------------------
    # Step 3: Generate synthetic error rows for each new (user, run)
    # ------------------------------------------------------------------
    for user in new_users:
        for run in new_runs:
            key = (user, run)
            if key not in run_info:
                print(f"Warning: No run info for {user}, run {run}. Skipping.")
                continue
            info = run_info[key]
            start_ts = info['start_timestamp']
            start_sim = info['total_start']
            # run_dur may be slightly different from 600s due to mixup, but we cap window at 600s as requested
            # However, if run_dur < 600, we should limit to run_dur to avoid going beyond run end.
            max_offset = min(600, info['run_duration'])  # seconds
            n_synthetic_per_run = random.randint(2,6) 
            for _ in range(n_synthetic_per_run):
                # Bootstrap a random original error row
                template = df_err.sample(n=1).iloc[0].to_dict()

                # Random offset in [0, max_offset] seconds
                offset = random.uniform(0, max_offset)
                new_timestamp = start_ts + timedelta(seconds=offset)

                # Sim time
                new_sim = start_sim + offset

                # Frame: use global frame rate to estimate frame at this sim_time
                # Add some noise to avoid exact repetition
                frame_noise = np.random.normal(0, 50)  # small noise in frames
                new_frame = int(round(global_frame_rate * new_sim + frame_noise))
                if new_frame < 0:
                    new_frame = 0

                # Build new row
                new_row = {}

                # Copy categorical columns from template
                for col in categorical_cols:
                    new_row[col] = template[col]

                # Override user_id, run_id with the new ones
                new_row['user_id'] = user
                new_row['run_id'] = run

                # Add noise to continuous columns
                for col in continuous_cols:
                    val = template[col]
                    noise = np.random.normal(0, noise_scale * stds[col])
                    new_row[col] = f"{val + noise:.3f}"

                # Add generated columns
                new_row['timestamp'] = new_timestamp.isoformat()
                new_row['sim_time_seconds'] = new_sim
                new_row['frame'] = new_frame

                synthetic_rows.append(new_row)

    # ------------------------------------------------------------------
    # Step 4: Combine and save
    # ------------------------------------------------------------------
    df_synthetic = pd.DataFrame(synthetic_rows)
    df_augmented = pd.concat([df_err, df_synthetic], ignore_index=True)

    df_augmented.to_csv(output_file, index=False)
    print(f"Augmented errors dataset saved to {output_file}")
    print(f"Added {len(synthetic_rows)} synthetic rows. Total rows: {len(df_augmented)}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    augment_errors_with_new_users(
        errors_file='Dataset Errors_baseline.csv',
        driving_time_file='Dataset Driving Time_baseline.csv',
        output_file='Dataset Errors_baseline.csv',
        new_users=[f'participant_{i:02d}' for i in range(26, 30)],  # 19..25
        new_runs=[1, 2, 3],
        noise_scale=0.05
    )