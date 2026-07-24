# Module 7 Monitoring Summary

## Dataset split

- Reference rows: 2,217,905
- Current rows: 47
- Drifted features: 13 / 37

## Latest rolling metrics

|   window_start_step |   window_end_step |   n_transactions |   n_fraud |   n_flagged |   precision |   recall |     f1 |   average_precision |   fraud_rate |
|--------------------:|------------------:|-----------------:|----------:|------------:|------------:|---------:|-------:|--------------------:|-------------:|
|            634.0000 |          801.0000 |           2.0000 |    1.0000 |      2.0000 |      0.5000 |   1.0000 | 0.6667 |              1.0000 |       0.5000 |

## Top drifted features

| feature                 | method      |   score |   p_value |   reference_mean |   current_mean | drifted   |
|:------------------------|:------------|--------:|----------:|-----------------:|---------------:|:----------|
| step                    | ks_numeric  |  1.0000 |    0.0000 |         193.4754 |       754.3617 | True      |
| dest_prior_count        | ks_numeric  |  0.7324 |    0.0000 |           5.2674 |         0.0851 | True      |
| dest_steps_since_last   | ks_numeric  |  0.7324 |    0.0000 |          20.1848 |        -0.4255 | True      |
| dest_first_seen         | chi2_binary |  0.7324 |    0.0000 |           0.1825 |         0.9149 | True      |
| hour_sin                | ks_numeric  |  0.5593 |    0.0000 |          -0.4509 |         0.3714 | True      |
| is_night                | chi2_binary |  0.3526 |    0.0000 |           0.0091 |         0.3617 | True      |
| hour_cos                | ks_numeric  |  0.2775 |    0.0011 |          -0.3689 |        -0.1403 | True      |
| account_age_days        | ks_numeric  |  0.2467 |    0.0053 |         359.2482 |       261.4681 | True      |
| failed_payment_attempts | ks_numeric  |  0.2080 |    0.0292 |           0.3035 |         0.7660 | True      |
| young_account           | chi2_binary |  0.1418 |    0.0019 |           0.0922 |         0.2340 | True      |

## Retraining decision

- Should retrain: True
- drift_fraction: value=0.3514, threshold=0.3000, breached=True
- precision_drop: value=0.5000, threshold=0.0090, breached=False
- recall_drop: value=1.0000, threshold=0.8488, breached=False
- scheduled_retrain: review monthly even without drift, because fraud patterns evolve
