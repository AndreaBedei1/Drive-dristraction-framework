import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import random

# ----------------------------------------------------------------------
# Helper: fill NaN in a bootstrapped template
# ----------------------------------------------------------------------
def fill_template_nans(template, df_original, categorical_cols, continuous_cols, 
                       cont_means, cont_stds, cat_value_pools):
    """
    If any value in template is NaN or empty string, replace it with a plausible value.
    For continuous: draw from normal distribution (mean, std).
    For categorical: draw uniformly from the pool of observed non-null values.
    """
    # Continuous columns
    for col in continuous_cols:
        if pd.isna(template[col]):
            val = np.random.normal(cont_means[col], cont_stds[col])
            template[col] = val

    # Categorical columns
    for col in categorical_cols:
        # Treat empty string as missing
        if pd.isna(template[col]) or template[col] == '':
            pool = cat_value_pools[col]
            if len(pool) == 0:
                # Fallback: use a placeholder (should never happen with clean data)
                template[col] = random.randint(10000,9999999)
            else:
                template[col] = random.choice(pool)
    return template

# ----------------------------------------------------------------------
# Main augmentation function
# ----------------------------------------------------------------------
def augment_distractions(
    distractions_file,
    output_file,
    first_run_timestamp='2026-02-18T15:52:21.858625',
    new_users=None,
    new_runs=None,
    noise_scale=0.05,
    min_gap_seconds=5,
    max_gap_seconds=20
):
    if new_users is None:
        new_users = [f'participant_{i:02d}' for i in range(19, 26)]
    if new_runs is None:
        new_runs = [1, 2, 3, 4]

    # Load original distractions
    df_dist = pd.read_csv(distractions_file)

    # ------------------------------------------------------------------
    # Preprocess: identify columns and clean missing values
    # ------------------------------------------------------------------
    continuous_cols = [
        'arousal_start', 'arousal_end',
        'hr_bpm_start', 'hr_bpm_end',
        'model_prob_start', 'model_prob_end',
        'emotion_prob_start', 'emotion_prob_end',
        'speed_kmh_start', 'speed_kmh_end',
        'steer_angle_deg_start', 'steer_angle_deg_end',
        'start_x', 'start_y', 'start_z',
        'end_x', 'end_y', 'end_z'
    ]
    # Ensure all continuous columns exist
    for col in continuous_cols:
        if col not in df_dist.columns:
            df_dist[col] = np.nan
        else:
            df_dist[col] = pd.to_numeric(df_dist[col], errors='coerce')

    categorical_cols = [
        'weather', 'map_name',
        'model_pred_start', 'model_pred_end',
        'emotion_label_start', 'emotion_label_end',
        'details'
    ]
    for col in categorical_cols:
        if col not in df_dist.columns:
            df_dist[col] = ''
        else:
            df_dist[col] = df_dist[col].replace('', np.nan)

    # Convert timestamp columns
    df_dist['timestamp_start'] = pd.to_datetime(df_dist['timestamp_start'])
    df_dist['timestamp_end']   = pd.to_datetime(df_dist['timestamp_end'])

    # ------------------------------------------------------------------
    # Prepare statistics for imputation
    # ------------------------------------------------------------------
    cont_means = {col: df_dist[col].mean() for col in continuous_cols}
    cont_stds  = {col: df_dist[col].std() for col in continuous_cols}

    cat_value_pools = {}
    for col in categorical_cols:
        vals = df_dist[col].dropna().unique()
        vals = [str(v) for v in vals if pd.notna(v)]
        cat_value_pools[col] = vals

    # ------------------------------------------------------------------
    # Compute empirical distributions from original data
    # ------------------------------------------------------------------
    # 1. Number of events per run
    event_counts = df_dist.groupby(['user_id', 'run_id']).size().tolist()
    if not event_counts:
        event_counts = [1]

    # 2. Event durations (seconds)
    durations = (df_dist['timestamp_end'] - df_dist['timestamp_start']).dt.total_seconds().values
    durations = durations[durations > 0]

    # 3. Frame rate
    delta_frame = df_dist['frame_end'] - df_dist['frame_start']
    delta_sim   = df_dist['sim_time_end'] - df_dist['sim_time_start']
    valid = (delta_sim > 0) & (delta_frame > 0)
    rates = delta_frame[valid] / delta_sim[valid]
    frame_rate_median = np.median(rates) if len(rates) > 0 else 1.0
    residuals = df_dist['frame_start'] - frame_rate_median * df_dist['sim_time_start']
    frame_noise_std = residuals.std() if len(residuals) > 0 else 0.0

    # ------------------------------------------------------------------
    # Generate run start times
    # ------------------------------------------------------------------
    current_real = datetime.fromisoformat(first_run_timestamp)
    current_sim = 0.0
    run_starts = {}
    for user in new_users:
        for run in new_runs:
            run_starts[(user, run)] = (current_real, current_sim)
            current_real += timedelta(seconds=600)
            current_sim += 600
            if run != new_runs[-1]:
                gap_seconds = random.uniform(15*60, 20*60)
                current_real += timedelta(seconds=gap_seconds)
                current_sim += gap_seconds
        if user != new_users[-1]:
            gap_seconds = random.uniform(15*60, 20*60)
            current_real += timedelta(seconds=gap_seconds)
            current_sim += gap_seconds

    # ------------------------------------------------------------------
    # Generate synthetic rows for each new (user, run)
    # ------------------------------------------------------------------
    synthetic_rows = []
    run_dur = 600.0

    for user in new_users:
        for run in new_runs:
            run_real_start, run_sim_start = run_starts[(user, run)]

            # Sample number of events for this run
            n_events = np.random.choice(event_counts)

            # Sequential generation of events
            events = []
            current_time = 0.0
            for _ in range(n_events):
                # For the first event, gap = 0; otherwise uniform gap in [min_gap, max_gap]
                if current_time == 0:
                    gap = 0
                else:
                    gap = random.uniform(min_gap_seconds, max_gap_seconds)

                # Sample duration from empirical distribution
                duration = np.random.choice(durations)

                start = current_time + gap
                # If event would exceed the run end, stop adding more events
                if start + duration > run_dur:
                    break

                events.append((start, duration))
                current_time = start + duration

            # Generate each event in the sequence
            for offset, duration in events:
                new_start_ts = run_real_start + timedelta(seconds=offset)
                new_end_ts   = new_start_ts + timedelta(seconds=duration)

                new_sim_start = run_sim_start + offset
                new_sim_end   = new_sim_start + duration

                frame_noise = np.random.normal(0, frame_noise_std)
                new_frame_start = int(round(frame_rate_median * new_sim_start + frame_noise))
                new_frame_end   = new_frame_start + int(round(frame_rate_median * duration))

                # Bootstrap template (any row, may contain NaNs)
                template = df_dist.sample(1).iloc[0].to_dict()

                # Fill any missing values in the template
                template = fill_template_nans(
                    template, df_dist, 
                    categorical_cols, continuous_cols,
                    cont_means, cont_stds, cat_value_pools
                )

                # Build new row
                new_row = {}
                for col in categorical_cols:
                    new_row[col] = template[col]
                new_row['user_id'] = user
                new_row['run_id']  = run
                for col in continuous_cols:
                    val = template[col]
                    noise = np.random.normal(0, noise_scale * cont_stds[col])
                    new_row[col] = val + noise

                # Add generated time columns
                new_row['timestamp_start'] = new_start_ts.isoformat()
                new_row['timestamp_end']   = new_end_ts.isoformat()
                new_row['sim_time_start']  = new_sim_start
                new_row['sim_time_end']    = new_sim_end
                new_row['frame_start']     = new_frame_start
                new_row['frame_end']       = new_frame_end

                synthetic_rows.append(new_row)

    # ------------------------------------------------------------------
    # Combine and save
    # ------------------------------------------------------------------
    df_synthetic = pd.DataFrame(synthetic_rows)
    df_augmented = pd.concat([df_dist, df_synthetic], ignore_index=True)
    df_augmented.to_csv(output_file, index=False)

    print(f"Augmented distractions dataset saved to {output_file}")
    print(f"Added {len(synthetic_rows)} synthetic rows. Total rows: {len(df_augmented)}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    augment_distractions(
        distractions_file='Dataset Distractions_distraction.csv',
        output_file='Dataset Distractions_distraction.csv',
        first_run_timestamp='2026-02-21T08:52:21.858625',
        new_users=[f'participant_{i:02d}' for i in range(9, 15)],
        new_runs=[1, 2, 3, 4],
        noise_scale=0.05
    )