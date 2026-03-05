import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def augment_mixup(df, numerical_cols, n_samples, user_id, run_id, timestamp):
    """
    Generate synthetic rows using mixup, all with the same constant columns.
    (This function remains unchanged and is used internally.)
    """
    data = df[numerical_cols].values
    n_rows = data.shape[0]
    synthetic_rows = []
    
    for _ in range(n_samples):
        idx1, idx2 = np.random.choice(n_rows, size=2, replace=False)
        row1 = data[idx1]
        row2 = data[idx2]
        alpha = np.random.beta(0.2, 0.2)
        mixed = alpha * row1 + (1 - alpha) * row2
        
        new_row = {
            'user_id': user_id,
            'run_id': run_id,
            'weather': 'day',
            'map_name': 'Town10HD',
            'timestamp': timestamp,
        }
        for col, val in zip(numerical_cols, mixed):
            new_row[col] = f"{val:.3f}"
        synthetic_rows.append(new_row)
    
    return synthetic_rows


def augment_driving_time(
    input_file: str,
    output_file: str,
    base_timestamp: str = '2026-02-25T09:24:42.117241',
    new_users: list = None,
    new_runs: list = None,
    gap_range: tuple = (15, 25),        # minutes
    n_samples_per_run: int = 1
) -> None:
    """
    Augment the driving time dataset by adding synthetic rows for new users and runs.
    
    Parameters:
        input_file (str): Path to the original driving time CSV.
        output_file (str): Path where the augmented CSV will be saved.
        base_timestamp (str): ISO timestamp for the start of the first new run.
        new_users (list): List of user_id strings (e.g., ['participant_22', ...]).
        new_runs (list): List of run_id integers.
        gap_range (tuple): (min, max) gap in minutes between consecutive runs.
        n_samples_per_run (int): Number of synthetic rows per (user, run). Usually 1.
    """
    if new_users is None:
        new_users = [f'participant_{i:02d}' for i in range(22, 30)]
    if new_runs is None:
        new_runs = [1, 2, 3]

    # Load original data
    df_original = pd.read_csv(input_file)
    
    # Columns we will mix (except total durations – we compute them cumulatively)
    mix_cols = [
        'run_duration_seconds', 'run_duration_minutes',
        'hr_baseline', 'arousal_baseline'
    ]
    # Ensure they exist
    missing = [c for c in mix_cols if c not in df_original.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # Prepare data for mixup
    data = df_original[mix_cols].values
    n_rows = data.shape[0]
    all_synthetic_rows = []

    # Convert base timestamp
    current_time = datetime.fromisoformat(base_timestamp)  # this is the start of the first run

    for user in new_users:
        cum_seconds = 0.0
        cum_minutes = 0.0
        for run in sorted(new_runs):
            # Mix a single row for this run
            idx1, idx2 = np.random.choice(n_rows, size=2, replace=False)
            row1, row2 = data[idx1], data[idx2]
            alpha = np.random.beta(0.2, 0.2)
            mixed = alpha * row1 + (1 - alpha) * row2

            run_dur_sec = mixed[0]          # run_duration_seconds from mixup
            run_dur_min = mixed[1]          # run_duration_minutes
            hr = mixed[2]
            aro = mixed[3]

            # Update cumulative totals
            cum_seconds += run_dur_sec
            cum_minutes += run_dur_min

            # End time of this run
            end_time = current_time + timedelta(seconds=run_dur_sec)

            # Build row
            new_row = {
                'user_id': user,
                'run_id': run,
                'weather': 'day',
                'map_name': 'Town10HD',
                'timestamp': end_time.isoformat(),
                'run_duration_seconds': f"{run_dur_sec:.3f}",
                'run_duration_minutes': f"{run_dur_min:.3f}",
                'total_duration_seconds': f"{cum_seconds:.3f}",
                'total_duration_minutes': f"{cum_minutes:.3f}",
                'hr_baseline': f"{hr:.3f}",
                'arousal_baseline': f"{aro:.3f}",
            }
            all_synthetic_rows.append(new_row)

            # Prepare start time for next run: end_time + random gap
            if run != new_runs[-1]:
                gap_seconds = random.uniform(gap_range[0]*60, gap_range[1]*60)
                # Add random seconds and milliseconds for extra variability
                extra_seconds = random.uniform(0, 1)   # up to 1 sec
                current_time = end_time + timedelta(seconds=gap_seconds + extra_seconds)
            else:
                # After last run of this user, we will add a gap before next user,
                # but that gap is set before starting the next user.
                pass

        # After finishing all runs for this user, add a gap before the next user
        if user != new_users[-1]:
            gap_seconds = random.uniform(gap_range[0]*60, gap_range[1]*60)
            extra_seconds = random.uniform(0, 1)
            current_time = end_time + timedelta(seconds=gap_seconds + extra_seconds)

    # Create DataFrame from synthetic rows and concatenate
    df_synthetic = pd.DataFrame(all_synthetic_rows)
    df_augmented = pd.concat([df_original, df_synthetic], ignore_index=True)
    df_augmented.to_csv(output_file, index=False)

    print(f"Augmented driving time dataset saved to {output_file}")
    print(f"Added {len(all_synthetic_rows)} synthetic rows. Total rows: {len(df_augmented)}")

