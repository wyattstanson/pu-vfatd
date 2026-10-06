"""Stage 2: short-term demand forecasting with uncertainty (Eq. 4).

This build uses statistical forecasters only (no trained neural model). The main one is a seasonal-
ratio forecaster: the last measurement is carried forward with yesterday's change at the same time
of day, and its spread sigma is proportional to the forecast level with a per-series relative error
estimated on history. Persistence and seasonal-naive forecasters are kept as baselines. Eq. (5) is
used only as an evaluation score (the Gaussian NLL of the forecasts); nothing is trained with it.
"""
from __future__ import annotations

import numpy as np


class PersistenceForecaster:
    name = "persistence"

    def mean(self, obs: np.ndarray, season: int) -> np.ndarray:
        mu = np.empty_like(obs)
        mu[0] = obs[0]
        mu[1:] = obs[:-1]
        return mu

    def estimate_spread(self, history: list[np.ndarray], season: int) -> "PersistenceForecaster":
        rel = np.concatenate([((h - self.mean(h, season)) / np.maximum(self.mean(h, season), 1e-6))[2 * season:]
                              for h in history])
        self.rel_sd = rel.std(axis=0) + 1e-4     # (I, K)
        return self

    def predict(self, obs: np.ndarray, hours: np.ndarray, season: int):
        mu = self.mean(obs, season)
        return mu, self.rel_sd[None] * np.maximum(mu, 1e-6)


class SeasonalNaiveForecaster(PersistenceForecaster):
    name = "seasonal_naive"

    def mean(self, obs, season):
        mu = super().mean(obs, season)
        mu[season:] = obs[:-season]
        return mu


class SeasonalRatioForecaster(PersistenceForecaster):
    """mu_t = d_{t-1} * (yesterday's level at t / yesterday's level at t-1), using 3-interval sums
    to damp noise; sigma_t = rel_sd * mu_t."""
    name = "seasonal_ratio"

    def mean(self, obs, season):
        mu = super().mean(obs, season)
        T = obs.shape[0]
        c = np.cumsum(np.concatenate([np.zeros((1,) + obs.shape[1:]), obs]), axis=0)  # prefix sums

        def window(end):  # sum of obs[end-2 .. end]
            return c[end + 1] - c[end - 2]

        ts = np.arange(season + 3, T)
        num = window(ts - season)        # yesterday, around the target time
        den = window(ts - season - 1)    # yesterday, one interval earlier
        ratio = np.clip(num / np.maximum(den, 1e-6), 0.7, 1.4)
        mu[ts] = obs[ts - 1] * ratio
        return mu


def forecast_scores(d: np.ndarray, mu: np.ndarray, sigma: np.ndarray, upper: np.ndarray) -> dict:
    """Accuracy, Gaussian NLL (Eq. 5 used as a score) and coverage of the planning demand."""
    s = np.maximum(sigma, 1e-6)
    return {
        "MAE_Gbps": float(np.mean(np.abs(d - mu))),
        "MAPE_%": float(100 * np.mean(np.abs(d - mu) / np.maximum(d, 1e-6))),
        "NLL": float(np.mean(np.log(s) + (d - mu) ** 2 / (2 * s ** 2))),
        "coverage": float(np.mean(d <= upper)),
        "mean_reserve_Gbps": float(np.mean(upper - mu)),
    }
