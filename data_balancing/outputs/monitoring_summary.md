# Monitoring Drift Summary

## Status Counts

- alert: 0
- warning: 0
- ok: 21

## Recommended Realtime Monitoring

- Log every scored transaction with timestamp, model version, score, threshold and final action.
- Track rolling fraud-score distribution by hour/day and alert when PSI >= 0.25.
- Track high-signal features: amount ratios, balance errors, failed attempts, new device, IP distance, country mismatch and transaction type.
- Monitor prediction volume: review queue size, auto-block count, approval count and score percentiles.
- When labels arrive later, monitor delayed precision/recall and fraud value captured.
- Add graph features later: account-device-IP-country bipartite links, shared device count, shared destination count and connected component risk.

## Top Drift Features

                 feature metric_type  reference_mean  current_mean  reference_std   current_std  psi  ks_statistic  ks_pvalue  category_distribution_shift status  score
        account_age_days     numeric      362.106200    362.106200     262.078253    262.078253  0.0           0.0        1.0                          NaN     ok    0.0
                  amount     numeric   181784.215984 181784.215984  508245.922152 508245.922152  0.0           0.0        1.0                          NaN     ok    0.0
  amount_to_oldOrg_ratio     numeric    70632.375482  70632.375482  375537.704918 375537.704918  0.0           0.0        1.0                          NaN     ok    0.0
amount_vs_prev_mean_orig     numeric       74.884819     74.884819    3715.279201   3715.279201  0.0           0.0        1.0                          NaN     ok    0.0
        errorBalanceDest     numeric    66424.869990  66424.869990  241149.058024 241149.058024  0.0           0.0        1.0                          NaN     ok    0.0
        errorBalanceOrig     numeric   202836.864168 202836.864168  503109.624041 503109.624041  0.0           0.0        1.0                          NaN     ok    0.0
 failed_payment_attempts     numeric        0.292800      0.292800       0.549116      0.549116  0.0           0.0        1.0                          NaN     ok    0.0
    home_billing_country categorical             NaN           NaN            NaN           NaN  NaN           NaN        NaN                          0.0     ok    0.0
             hour_of_day     numeric       15.311200     15.311200       4.240183      4.240183  0.0           0.0        1.0                          NaN     ok    0.0
  ip_billing_distance_km     numeric      275.995360    275.995360     862.886158    862.886158  0.0           0.0        1.0                          NaN     ok    0.0