# Discovery Summary

Total elapsed: 3.06s
Total findings: 10

## Method Results

- **residual_slice_tree**: 5 findings in 3.05s
- **label_noise_scan**: 100 findings in 0.00s

## Top Findings

### Finding 1 [ERROR] — residual_slice_tree_004
- Kind: residual_slice
- Residual lift: 2.0482
- Priority: 2.0482
- Coverage: 0.0952
- Slice: `actioned_last_10 > 1.5 AND actioned_last_10 > 2.5 AND actioned_last_10 > 3.5`

### Finding 2 [ERROR] — residual_slice_tree_003
- Kind: residual_slice
- Residual lift: 1.6745
- Priority: 1.6745
- Coverage: 0.0368
- Slice: `actioned_last_10 > 1.5 AND actioned_last_10 > 2.5 AND actioned_last_10 <= 3.5`

### Finding 3 [ERROR] — residual_slice_tree_002
- Kind: residual_slice
- Residual lift: 1.4251
- Priority: 1.4251
- Coverage: 0.0306
- Slice: `actioned_last_10 > 1.5 AND actioned_last_10 <= 2.5 AND lifetime_actions_prior > 24.5`

### Finding 4 [ERROR] — residual_slice_tree_001
- Kind: residual_slice
- Residual lift: 0.7450
- Priority: 0.7450
- Coverage: 0.0285
- Slice: `actioned_last_10 > 1.5 AND actioned_last_10 <= 2.5 AND lifetime_actions_prior <= 24.5`

### Finding 5 [ERROR] — residual_slice_tree_000
- Kind: residual_slice
- Residual lift: 0.5697
- Priority: 0.5697
- Coverage: 0.1089
- Slice: `actioned_last_10 <= 1.5 AND sends_since_last_action <= 16.5 AND lifetime_actions_prior > 9.5`

### Finding 6 [WARNING] — label_noise_scan_0000
- Kind: label_noise
- Residual lift: 0.9966
- Priority: 0.9966
- Coverage: 0.0000
- Slice: `{"type": "label_noise_candidate", "user_id": 23775335, "email_id": 2534972, "date_sent": "2024-11-19 16:05:34+00:00", "s`

### Finding 7 [WARNING] — label_noise_scan_0001
- Kind: label_noise
- Residual lift: 0.9965
- Priority: 0.9965
- Coverage: 0.0000
- Slice: `{"type": "label_noise_candidate", "user_id": 545531, "email_id": 2577927, "date_sent": "2025-01-09 21:04:12+00:00", "sco`

### Finding 8 [WARNING] — label_noise_scan_0002
- Kind: label_noise
- Residual lift: 0.9963
- Priority: 0.9963
- Coverage: 0.0000
- Slice: `{"type": "label_noise_candidate", "user_id": 74408614, "email_id": 2647783, "date_sent": "2025-03-08 14:34:26+00:00", "s`

### Finding 9 [WARNING] — label_noise_scan_0003
- Kind: label_noise
- Residual lift: 0.8234
- Priority: 0.8234
- Coverage: 0.0000
- Slice: `{"type": "label_noise_candidate", "user_id": 8250, "email_id": 2663645, "date_sent": "2025-03-20 13:32:12+00:00", "score`

### Finding 10 [WARNING] — label_noise_scan_0004
- Kind: label_noise
- Residual lift: 0.8234
- Priority: 0.8234
- Coverage: 0.0000
- Slice: `{"type": "label_noise_candidate", "user_id": 845519, "email_id": 2671252, "date_sent": "2025-03-26 13:34:56+00:00", "sco`
