| dataset              | model             |   RMSE |   MAE |   WIS |   coverage_80 |   coverage_95 |   WIS_vs_seasonal_naive |
|:---------------------|:------------------|-------:|------:|------:|--------------:|--------------:|------------------------:|
| non-seasonal (clean) | ensemble          |   8.59 |  7.27 |  3.6  |          0.88 |          1    |                    0.88 |
| non-seasonal (clean) | gradient_boosting |   9.57 |  8.29 |  4.48 |          0.69 |          1    |                    1.09 |
| non-seasonal (clean) | poisson           |  10.39 |  9.42 |  3.49 |          0.88 |          1    |                    0.85 |
| non-seasonal (clean) | random_forest     |   9.12 |  7.94 |  5.02 |          0.58 |          1    |                    1.23 |
| non-seasonal (clean) | seasonal_naive    |   8.77 |  7.21 |  4.1  |          1    |          1    |                    1    |
| realistic (hard)     | ensemble          |  38.67 | 22.12 | 13.59 |          0.85 |          1    |                    0.63 |
| realistic (hard)     | gradient_boosting |  42.5  | 24.98 | 16.06 |          0.79 |          1    |                    0.75 |
| realistic (hard)     | poisson           |  38.49 | 21.84 | 13.85 |          0.79 |          0.98 |                    0.65 |
| realistic (hard)     | random_forest     |  37.05 | 21.93 | 13.64 |          0.87 |          0.96 |                    0.64 |
| realistic (hard)     | seasonal_naive    |  59.63 | 32.37 | 21.44 |          0.87 |          0.98 |                    1    |
| seasonal (clean)     | ensemble          |   9.8  |  8    |  4.23 |          0.9  |          1    |                    0.63 |
| seasonal (clean)     | gradient_boosting |  11.73 |  9.81 |  4.99 |          0.9  |          1    |                    0.74 |
| seasonal (clean)     | poisson           |  17.95 | 15.39 |  6.29 |          0.85 |          0.94 |                    0.94 |
| seasonal (clean)     | random_forest     |  10.45 |  8.55 |  4.23 |          0.9  |          1    |                    0.63 |
| seasonal (clean)     | seasonal_naive    |  15.96 | 11.73 |  6.71 |          0.85 |          1    |                    1    |