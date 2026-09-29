"""Distributed lag effects (DLM) for ClimAID v2.

Instead of choosing one lag per climate variable (v1) or giving every lag its own free
coefficient (v2's lag 0-3 features), a distributed lag model lets a variable's effect be spread
over a range of lags 0..L, with the lag coefficients constrained to a smooth curve:

    effect(lag l) = sum_k gamma_k * B_k(l),      k = 0..degree

where B_k are Legendre polynomials over the scaled lag index. The model sees, per variable,
degree+1 "cross-basis" features  sum_l x(t-l) * B_k(l)  instead of L+1 raw lags, so long lags
(e.g. ENSO up to 12 months) cost only a few features, and the **cumulative effect** of a
sustained change (sum over lags, what matters under warming) is estimated directly.

This is the linear-exposure special case of distributed lag non-linear models (DLNM;
Gasparrini et al.): the lag response is smooth, the exposure response is linear on the
model's scale. Non-linear exposure (e.g. a thermal optimum) is handled separately by the
temperature curve / quadratic temperature options.
"""
from __future__ import annotations

import numpy as np

DEFAULT_MAX_LAG = {"temperature": 6, "rainfall": 6, "humidity": 6, "enso": 12}
DEFAULT_DEGREE = 2


def lag_basis(max_lag, degree=DEFAULT_DEGREE):
    """(max_lag+1) x (degree+1) Legendre basis over lags 0..max_lag."""
    lags = np.arange(max_lag + 1)
    u = 2 * lags / max(max_lag, 1) - 1.0
    return np.column_stack([np.polynomial.legendre.Legendre.basis(k)(u) for k in range(degree + 1)])


def cross_basis(x, max_lag, degree=DEFAULT_DEGREE):
    """T x (degree+1) features: sum over lags of x(t-l) * B_k(l). NaN until max_lag of history exists."""
    x = np.asarray(x, dtype=float)
    B = lag_basis(max_lag, degree)
    T = len(x)
    L = np.full((T, max_lag + 1), np.nan)
    for l in range(max_lag + 1):
        L[l:, l] = x[: T - l]
    return L @ B


def lag_curve(gamma, max_lag, degree=DEFAULT_DEGREE):
    """Coefficients per lag and the cumulative effect, from cross-basis coefficients."""
    beta = lag_basis(max_lag, degree) @ np.asarray(gamma, dtype=float)
    return {"by_lag": [float(b) for b in beta], "cumulative": float(beta.sum()),
            "peak_lag": int(np.argmax(np.abs(beta)))}
