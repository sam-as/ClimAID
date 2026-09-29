# ClimAID v2 — Feedback-to-Implementation Map

How reviewer concerns are addressed in ClimAID 0.4.0 (the first release with v2). The evidence for each
fix is in [Claude-Testing_FINDINGS_2026-09.md](Claude-Testing_FINDINGS_2026-09.md); the full list of changes is in
[CHANGELOG.md](CHANGELOG.md).

| Reviewer concern | ClimAID v2 response |
|---|---|
| Scope broader than evidence | v2 separates software capability from empirical forecast evaluation and makes hindcasts explicit. |
| No simple benchmarks | Seasonal-naive baseline is built into v2. |
| No autoregressive/statistical comparison | Statistical and legacy ML families are available to the v2 forecast branch. |
| Calibration claimed without calibration evidence | v2 reports WIS, interval coverage and calibration-related diagnostics rather than relying only on RMSE/R², and calibrates its intervals from rolling hindcasts (split-conformal, per lead time). |
| Residual target may be in-sample | v2 residual learning uses temporal out-of-fold base predictions. |
| Train/validation/test unclear | v2 records forecast origin, horizon and training-observation counts in the validation contract. In v1, lag and model selection now use a validation split inside the training period; the test set is used once. |
| 2020 excluded without sensitivity | The COVID-19 period is a user choice (`exclude_period`): `"2020"` (default), a custom period (`"YYYY-MM:YYYY-MM"`) or `"none"`, applied the same way in v1 and v2, so results can be compared with and without it. In v2, excluded months are never used as scoring targets. |
| Dual-baseline flag called probability | v2 explicitly distinguishes threshold/ensemble agreement from a calibrated outbreak probability. |
| No outbreak hindcast | v2 includes historical hindcast evaluation infrastructure. |
| Disease provenance insufficient | v2 report metadata exposes disease and forecast configuration; future work can extend this to formal data cards. |
| Climate leakage | Climate feature statistics are fitted at the forecast origin, and future climate is explicitly sourced as observed or projected. |
| Broader geographic/disease claims | Multi-district/multi-disease evaluation can now be performed using a common forecast interface without changing the engine. |
