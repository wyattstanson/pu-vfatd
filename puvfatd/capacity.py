"""Stage 1: effective firewall capacity estimation (Eqs. 1-3)."""
from __future__ import annotations

import numpy as np

from .config import Params
from .topology import Topology


class CapacityEstimator:
    def __init__(self, topo: Topology, p: Params, fixed: bool = False):
        self.topo, self.p, self.fixed = topo, p, fixed
        self.C_hat: np.ndarray | None = None

    def prior(self, mix: np.ndarray) -> np.ndarray:
        """Model-based prior, Eq. (1): nominal capacity scaled by the benchmarked inspection load."""
        return self.p.C0 / (mix @ self.topo.kappa)

    def update(self, n_prev: np.ndarray, y: np.ndarray, psi: np.ndarray, mix: np.ndarray,
               stale: bool) -> np.ndarray:
        if self.fixed:                      # ablation: the base paper's fixed nominal capacity
            self.C_hat = np.full(self.topo.S, self.p.C0)
            return self.C_hat
        prior = self.prior(mix)
        if self.C_hat is None:
            self.C_hat = prior.copy()
            return self.C_hat
        if stale:                           # missing data never increases the assumed capacity
            self.C_hat = np.minimum(self.C_hat, prior)
            return self.C_hat
        ok = (n_prev > 0) & (psi >= self.p.psi_min)
        meas = np.where(ok, self.p.psi_max * y / np.maximum(n_prev * psi, 1e-9), prior)        # Eq. (2)
        self.C_hat = np.where(ok, self.p.beta * self.C_hat + (1 - self.p.beta) * meas, prior)  # Eq. (3)
        return self.C_hat
