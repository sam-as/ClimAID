"""Probabilistic and point-forecast metrics for ClimAID v2."""

from __future__ import annotations

import numpy as np


def weighted_interval_score(
    y: np.ndarray,
    quantiles: np.ndarray,
    probs: np.ndarray,
) -> float:
    """Compute the mean weighted interval score.

    ``quantiles`` has shape (n_observations, n_quantiles). ``probs`` must be
    sorted and include the median probability 0.5.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    q = np.asarray(quantiles, dtype=float)
    p = np.asarray(probs, dtype=float).reshape(-1)
    if q.ndim != 2 or q.shape[0] != y.size or q.shape[1] != p.size:
        raise ValueError("quantiles must have shape (n_observations, n_quantiles)")
    if not np.all(np.diff(p) > 0):
        raise ValueError("probs must be strictly increasing")
    median_idx = int(np.argmin(np.abs(p - 0.5)))
    if not np.isclose(p[median_idx], 0.5):
        raise ValueError("probs must contain the 0.5 quantile")

    # Canonical WIS (Bracher et al. 2021): a weighted average of the median
    # absolute error (weight w0 = 1/2) and K central-interval scores (weight
    # wk = alpha_k/2 each), normalised by (K + 1/2). Crucially, this
    # normalising constant depends only on the number of intervals K, not on
    # the specific alpha values -- WIS is equal to 2x the mean pinball loss
    # across the 2K+1 quantile levels used. A normaliser that depends on the
    # alpha values (as opposed to just K) does not recover this identity and
    # is not the metric described in the literature.
    score = 0.5 * np.abs(y - q[:, median_idx])
    n_intervals = 0
    for j, prob in enumerate(p):
        if prob >= 0.5:
            continue
        upper_idx = int(np.where(np.isclose(p, 1 - prob))[0][0]) if np.any(np.isclose(p, 1 - prob)) else None
        if upper_idx is None:
            continue
        alpha = 2 * prob
        lower = q[:, j]
        upper = q[:, upper_idx]
        interval = upper - lower
        under = np.maximum(lower - y, 0.0)
        over = np.maximum(y - upper, 0.0)
        interval_score = interval + (2.0 / alpha) * under + (2.0 / alpha) * over
        weight = alpha / 2.0
        score += weight * interval_score
        n_intervals += 1
    norm = n_intervals + 0.5
    return float(np.mean(score / norm))


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((y >= lower) & (y <= upper)))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(y - pred)))

def pinball_loss(y, q, p):
    y = np.asarray(y, float); q = np.asarray(q, float)
    return float(np.mean(np.where(y >= q, p*(y-q), (1-p)*(q-y))))
