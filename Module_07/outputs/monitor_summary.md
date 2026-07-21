# Module 7 Monitoring Summary

## Dataset split

- Reference rows: 2,217,905
- Current rows: 552,504
- Drifted features: 20 / 37

## Latest rolling metrics

|   window_start_step |   window_end_step |   n_transactions |    n_fraud |   n_flagged |   precision |   recall |     f1 |   average_precision |   fraud_rate |
|--------------------:|------------------:|-----------------:|-----------:|------------:|------------:|---------:|-------:|--------------------:|-------------:|
|            576.0000 |          743.0000 |      68,117.0000 | 1,854.0000 | 68,064.0000 |      0.0272 |   1.0000 | 0.0530 |              0.8918 |       0.0272 |

## Top drifted features

| feature                 | method      |   score |   p_value |   reference_mean |   current_mean | drifted   |
|:------------------------|:------------|--------:|----------:|-----------------:|---------------:|:----------|
| step                    | ks_numeric  |  1.0000 |    0.0000 |         193.4754 |       436.8317 | True      |
| dest_steps_since_last   | ks_numeric  |  0.3106 |    0.0000 |          20.1848 |        69.5753 | True      |
| hour_cos                | ks_numeric  |  0.1118 |    0.0000 |          -0.3689 |        -0.2389 | True      |
| hour_sin                | ks_numeric  |  0.0580 |    0.0000 |          -0.4509 |        -0.5138 | True      |
| is_transfer             | chi2_binary |  0.0127 |    0.0000 |           0.1898 |         0.2025 | True      |
| dest_prior_count        | ks_numeric  |  0.0080 |    0.0000 |           5.2674 |         5.3870 | True      |
| dest_first_seen         | chi2_binary |  0.0071 |    0.0000 |           0.1825 |         0.1896 | True      |
| is_night                | chi2_binary |  0.0052 |    0.0000 |           0.0091 |         0.0143 | True      |
| failed_payment_attempts | ks_numeric  |  0.0035 |    0.0000 |           0.3035 |         0.3155 | True      |
| ip_billing_distance_km  | ks_numeric  |  0.0034 |    0.0001 |         291.2367 |       299.4037 | True      |

## Retraining decision

- Should retrain: True
- drift_fraction: value=0.5405, threshold=0.3000, breached=True
- precision_drop: value=0.0272, threshold=0.0090, breached=False
- recall_drop: value=1.0000, threshold=0.8488, breached=False
- scheduled_retrain: review monthly even without drift, because fraud patterns evolve
