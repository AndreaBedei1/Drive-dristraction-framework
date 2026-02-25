import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def augment_mixup(df, numerical_cols, n_samples, user_id, run_id, timestamp):
    """
    Generate synthetic rows using mixup, all with the same constant columns.
    
    Parameters:
        df (pd.DataFrame): Original dataframe (used only for numerical data).
        numerical_cols (list): List of numerical column names to mix.
        n_samples (int): How many synthetic rows to create for this (user, run).
        user_id (str): Constant value for 'user_id'.
        run_id (int): Constant value for 'run_id'.
        timestamp (str): Constant value for 'timestamp' (ISO format).
    
    Returns:
        list of dicts: Synthetic rows.
    """
    data = df[numerical_cols].values
    n_rows = data.shape[0]
    synthetic_rows = []
    
    for _ in range(n_samples):
        idx1, idx2 = np.random.choice(n_rows, size=2, replace=False)
        row1 = data[idx1]
        row2 = data[idx2]
        alpha = np.random.beta(0.2, 0.2)          # mixup coefficient
        mixed = alpha * row1 + (1 - alpha) * row2
        
        new_row = {
            'user_id': user_id,
            'run_id': run_id,
            'weather': 'day',                     # constant
            'map_name': 'Town10HD',                # constant
            'timestamp': timestamp,
        }
        for col, val in zip(numerical_cols, mixed):
            new_row[col] = f"{val:.3f}"
        synthetic_rows.append(new_row)
    
    return synthetic_rows

if __name__ == "__main__":
    # Load original data
    input_file = 'Dataset Driving Time_baseline.csv'
    output_file = 'Dataset Driving Time_baseline.csv'
    df_original = pd.read_csv(input_file)
    
    numerical_cols = [
        'run_duration_seconds', 'run_duration_minutes',
        'total_duration_seconds', 'total_duration_minutes',
        'hr_baseline', 'arousal_baseline'
    ]
    
    # Base timestamp – we'll use the last timestamp in the file as reference
    last_timestamp_str = '2026-02-19T10:24:42.517342' #df_original['timestamp'].iloc[-1]
    base_time = datetime.fromisoformat(last_timestamp_str)
    
    # Define the synthetic users and runs
    user_ids = [f'participant_{i:02d}' for i in range(9, 15)]   # 20..25
    run_range = range(1, 4)                                      # 1,2,3
    
    all_synthetic_rows = []
    
    for user in user_ids:
        for run in run_range:
            # Random offset between 15 and 25 minutes (inclusive)
            offset_minutes = random.randint(15, 25)
            offset_seconds = random.randint(0, 59)  # add some random seconds for more variability
            new_time = base_time + timedelta(minutes=offset_minutes, seconds=offset_seconds, milliseconds=random.randint(0, 999))
            base_time = new_time
            timestamp_str = new_time.isoformat()
            
            # Generate ONE synthetic row for this user/run/timestamp
            rows = augment_mixup(df_original, numerical_cols,
                                 n_samples=1,
                                 user_id=user,
                                 run_id=run,
                                 timestamp=timestamp_str)
            all_synthetic_rows.extend(rows)
    
    # Create DataFrame from synthetic rows and concatenate
    df_synthetic = pd.DataFrame(all_synthetic_rows)
    df_augmented = pd.concat([df_original, df_synthetic], ignore_index=True)
    
    # Save
    df_augmented.to_csv(output_file, index=False)
    print(f"Augmented dataset saved to {output_file}")
    print(f"Added {len(all_synthetic_rows)} synthetic rows. Total rows: {len(df_augmented)}")