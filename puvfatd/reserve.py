"""Stage 3: uncertainty-aware reserve (Eqs. 6-8)."""
from __future__ import annotations

import numpy as np

from .config import Params


def normalised_errors(d: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """e = (d - mu) / sigma, Eq. (6)."""
    return (d - mu) / np.maximum(sigma, 1e-9)


def calibrate_q0(d: np.ndarray, mu: np.ndarray, sigma: np.ndarray, alpha: float) -> np.ndarray:
    """q_i^0: (1 - alpha) empirical quantile of e for each ingress point on a calibration segment."""
    e = normalised_errors(d, mu, sigma)                       # (T, I, K)
    return np.quantile(e.transpose(1, 0, 2).reshape(e.shape[1], -1), 1 - alpha, axis=1)


def plan_series(d_obs: np.ndarray, mu: np.ndarray, sigma: np.ndarray, q0: np.ndarray, p: Params,
                mode: str = "adaptive") -> tuple[np.ndarray, np.ndarray]:
    """Planning demand D_hat[t] for every interval (it does not depend on the actions taken).

    adaptive: Eq. (7) online multiplier + Eq. (8) planning demand
    fixed:    ablation, fixed margin on the forecast mean
    none:     ablation without forecasting, plan on the last measured demand only
    """
    T, I, K = d_obs.shape
    D = np.empty_like(d_obs)
    q = np.zeros((T, I))
    q[0] = q0
    D[0] = d_obs[0]
    for t in range(1, T):
        last = d_obs[t - 1]
        if mode == "adaptive":
            miss = (d_obs[t - 1] > D[t - 1]).mean(axis=1)                       # (I,)
            q[t] = q[t - 1] + p.eta_q * (miss - p.alpha)                         # Eq. (7)
            D[t] = np.maximum(mu[t] + np.maximum(q[t], 0)[:, None] * sigma[t], last)   # Eq. (8)
        elif mode == "fixed":
            q[t] = q[t - 1]
            D[t] = np.maximum(mu[t] * (1 + p.fixed_reserve), last)
        elif mode == "none":
            D[t] = last
        else:
            raise ValueError(mode)
    return D, q
