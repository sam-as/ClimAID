| dataset              | model             |   RMSE |   MAE |   WIS |   coverage_80 |   coverage_95 |   WIS_vs_seasonal_naive |
|:---------------------|:------------------|-------:|------:|------:|--------------:|--------------:|------------------------:|
| non-seasonal (clean) | ensemble          |   8.58 |  7.31 |  3.53 |          0.92 |          1    |                    0.86 |
| non-seasonal (clean) | gradient_boosting |   9.57 |  8.29 |  4.48 |          0.69 |          1    |                    1.09 |
| non-seasonal (clean) | poisson           |  11.37 |  9.68 |  4.26 |          0.83 |          1    |                    1.04 |
| non-seasonal (clean) | random_forest     |   9.12 |  7.94 |  5.02 |          0.58 |          1    |                    1.23 |
| non-seasonal (clean) | sarimax           |   5.1  |  4.21 |  2.36 |          0.94 |          1    |                    0.58 |
| non-seasonal (clean) | seasonal_naive    |   8.77 |  7.21 |  4.1  |          1    |          1    |                    1    |
| realistic (hard)     | ensemble          |  39.42 | 21.99 | 13.44 |          0.85 |          1    |                    0.63 |
| realistic (hard)     | gradient_boosting |  42.5  | 24.98 | 16.06 |          0.79 |          1    |                    0.75 |
| realistic (hard)     | poisson           |  37.36 | 19.91 | 13.54 |          0.87 |          0.98 |                    0.63 |
| realistic (hard)     | random_forest     |  37.05 | 21.93 | 13.64 |          0.87 |          0.96 |                    0.64 |
| realistic (hard)     | sarimax           |  57.88 | 33.99 | 17.95 |          0.72 |          0.98 |                    0.84 |
| realistic (hard)     | seasonal_naive    |  59.63 | 32.37 | 21.44 |          0.87 |          0.98 |                    1    |
| seasonal (clean)     | ensemble          |   9.05 |  7.39 |  4.02 |          0.85 |          1    |                    0.6  |
| seasonal (clean)     | gradient_boosting |  11.73 |  9.81 |  4.99 |          0.9  |          1    |                    0.74 |
| seasonal (clean)     | poisson           |  17.18 | 15.06 |  6.01 |          0.92 |          1    |                    0.89 |
| seasonal (clean)     | random_forest     |  10.45 |  8.55 |  4.23 |          0.9  |          1    |                    0.63 |
| seasonal (clean)     | sarimax           |   9.47 |  7.36 |  4.24 |          0.98 |          1    |                    0.63 |
| seasonal (clean)     | seasonal_naive    |  15.96 | 11.73 |  6.71 |          0.85 |          1    |                    1    |