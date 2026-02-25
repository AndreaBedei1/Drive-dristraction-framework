from augment_baseline_distraction_errors import *
from augment_baseline_driving import *
from augment_distraction_events import *
from augment_distraction_errors import *



users = [f'participant_{i:02d}' for i in range(23,31)]
runs_baseline = [1, 2, 3]
runs_errors = [1, 2, 3, 4]
noise=0.06

base_timestamp='2026-02-23T09:00:42.023752'
error_timestamp = '2026-02-24T10:24:42.128109'

augment_driving_time(
        input_file='Dataset Driving Time_baseline.csv',
        output_file='Dataset Driving Time_baseline.csv',
        base_timestamp=base_timestamp,
        new_users=users,
        new_runs=runs_baseline,
        gap_range=(15, 25),        
        n_samples_per_run=1
)

augment_errors_with_new_users(
        errors_file='Dataset Errors_baseline.csv',
        driving_time_file='Dataset Driving Time_baseline.csv',
        output_file='Dataset Errors_baseline.csv',
        new_users=users,
        new_runs=runs_baseline,
        noise_scale=noise
)

augment_distractions(
        distractions_file='Dataset Distractions_distraction.csv',
        output_file='Dataset Distractions_distraction.csv',
        first_run_timestamp=error_timestamp,
        new_users=users,
        new_runs=runs_errors,
        noise_scale=noise
)


augment_errors_distraction(
        errors_file='Dataset Errors_distraction.csv',
        distractions_file='Dataset Distractions_distraction.csv',
        output_file='Dataset Errors_distraction.csv',
        new_users=users,
        new_runs=runs_errors,
        noise_scale=noise
)