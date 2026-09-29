# ClimAID v2 — Feedback-to-Implementation Map

| Reviewer concern | ClimAID v2 response |
|---|---|
| Scope broader than evidence | v2 separates software capability from empirical forecast evaluation and makes hindcasts explicit. |
| No simple benchmarks | Seasonal-naive baseline is built into v2. |
| No autoregressive/statistical comparison | Statistical and legacy ML families are available to the v2 forecast branch. |
| Calibration claimed without calibration evidence | v2 reports WIS, interval coverage and calibration-related diagnostics rather than relying only on RMSE/R². |
| Residual target may be in-sample | v2 residual learning uses temporal out-of-fold base predictions. |
| Train/validation/test unclear | v2 records forecast origin, horizon and training-observation counts in the validation contract. |
| 2020 excluded without sensitivity | v2 no longer hard-codes exclusion of 2020; origins can include disruption periods. |
| Dual-baseline flag called probability | v2 explicitly distinguishes threshold/ensemble agreement from a calibrated outbreak probability. |
| No outbreak hindcast | v2 includes historical hindcast evaluation infrastructure. |
| Disease provenance insufficient | v2 report metadata exposes disease and forecast configuration; future work can extend this to formal data cards. |
| Climate leakage | Climate feature statistics are fitted at the forecast origin, and future climate is explicitly sourced as observed or projected. |
| Broader geographic/disease claims | Multi-district/multi-disease evaluation can now be performed using a common forecast interface without changing the engine. |
