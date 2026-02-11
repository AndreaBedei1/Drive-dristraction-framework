# Correlation Analysis Report

## Core counts
- Total errors: 104
- Errors during distraction: 14 (13.46%)
- Errors outside distraction: 90 (86.54%)
- Total distraction windows: 138
- participant_01: 4/77 errors during distraction (5.19%)
- participant_02: 10/27 errors during distraction (37.04%)

## Error type split
- Solid line crossing: total=47, during=7, outside=40
- Red light violation: total=18, during=2, outside=16
- Harsh braking: total=17, during=2, outside=15
- Stop sign violation: total=12, during=1, outside=11
- Vehicle collision: total=10, during=2, outside=8

## Designed correlation checks
- Vehicle collision with fear/angry: 9/10 (90.00%)
- Solid line crossing with neutral/happy: 47/47 (100.00%)
- Distraction model_pred_start in {None, sing}: 138/138 (100.00%)
- Distraction emotion_label_start in {neutral, happy, sad}: 138/138 (100.00%)
- Distraction emotion_label_start == sad: 14/138 (10.14%)

## Statistical tests
- Note: tests on "during distraction" subsets use a small sample (n=14) and should be interpreted carefully.
- error_type_vs_during_distraction: p=0.9319, chi2=0.848, Cramer's V=0.0903. Association between error category and distraction state.
- emotion_label_at_error_vs_error_type: p=2.569e-12, chi2=89.94, Cramer's V=0.465. Association between emotion at error instant and error type.
- model_pred_at_error_vs_error_type: p=0.000367, chi2=35.66, Cramer's V=0.3381. Association between predicted distraction label at error event and error type.
- vehicle_collision_vs_fear_or_angry: p=7.495e-06, chi2=20.06, Cramer's V=0.4392. Targeted check: vehicle collisions should skew toward fear/angry.
- solid_line_crossing_vs_neutral_or_happy: p=3.594e-14, chi2=57.38, Cramer's V=0.7428. Targeted check: solid line crossings should skew toward neutral/happy.
- window_model_pred_start_vs_error_type_during_only: p=0.5719, chi2=2.917, Cramer's V=0.4564. Association between distraction window start prediction and error type for matched events.
- window_model_pred_end_vs_error_type_during_only: p=0.1672, chi2=6.462, Cramer's V=0.6794. Association between distraction window end prediction and error type for matched events.
- window_emotion_start_vs_error_type_during_only: p=0.5719, chi2=2.917, Cramer's V=0.4564. Association between distraction window start emotion and error type for matched events.
- window_emotion_end_vs_error_type_during_only: p=0.772, chi2=8.167, Cramer's V=0.441. Association between distraction window end emotion and error type for matched events.
- window_model_start_vs_emotion_start: p=0.04751, chi2=6.094, Cramer's V=0.2101. Association between distraction model prediction at window start and start emotion.
- window_hr_avg_vs_errors_in_window_spearman: p=0.001361, chi2=0.27, Cramer's V=nan. Spearman rho reported in chi2 column for compactness.
- window_arousal_avg_vs_errors_in_window_spearman: p=0.05369, chi2=0.1652, Cramer's V=nan. Spearman rho reported in chi2 column for compactness.
- window_hr_avg_distribution_errors_vs_no_errors_mannwhitney: p=0.001776, chi2=1096, Cramer's V=nan. U statistic reported in chi2 column.
- window_arousal_avg_distribution_errors_vs_no_errors_mannwhitney: p=0.05322, chi2=937.5, Cramer's V=nan. U statistic reported in chi2 column.

## Strongest numeric correlations (|rho| >= 0.30, p < 0.05)
- model_prob_start vs model_prob_end: rho=1.000, p=0
- distraction_hr_bpm_end vs distraction_hr_bpm_avg: rho=0.986, p=1.246e-10
- distraction_arousal_end vs distraction_arousal_avg: rho=0.980, p=9.278e-10
- distraction_arousal_start vs distraction_arousal_avg: rho=0.946, p=1.064e-06
- distraction_arousal_start vs distraction_arousal_end: rho=0.910, p=1.576e-05
- distraction_hr_bpm_start vs distraction_hr_bpm_avg: rho=0.886, p=2.507e-05
- distraction_hr_bpm_start vs distraction_hr_bpm_end: rho=0.853, p=0.000104
- distraction_hr_bpm_start vs distraction_model_prob_end: rho=0.787, p=0.0008364
- distraction_hr_bpm_avg vs distraction_model_prob_end: rho=0.784, p=0.0008945
- distraction_hr_bpm_end vs distraction_model_prob_end: rho=0.719, p=0.003762
- distraction_model_prob_start vs distraction_emotion_prob_end: rho=0.610, p=0.02063
- during_distraction_int vs seconds_to_nearest_distraction: rho=-0.592, p=3.656e-11
- model_prob_start vs distraction_model_prob_end: rho=0.578, p=0.03032
- model_prob_end vs distraction_model_prob_end: rho=0.578, p=0.03032
- seconds_to_distraction_end vs distraction_model_prob_end: rho=0.534, p=0.04916
