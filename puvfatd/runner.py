"""Closed-loop execution of one evaluation day (Algorithm 2 without the policy itself).

A Bundle holds everything that does not depend on the actions (true trace, observed demand with
telemetry staleness, forecasts and planning demand). The Runner keeps the per-interval state:
configuration in service, telemetry, and the capacity estimator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capacity import CapacityEstimator
from .config import Params
from .milp import solve_vfatd
from .model import Snapshot
from .reserve import plan_series
from .simulator import Outcome, Simulator
from .topology import Topology
from .traffic import Trace


@dataclass
class Bundle:
    trace: Trace
    d_obs: np.ndarray    # (T, I, K) demand as the controller sees it (stale reports repeat old values)
    mu: np.ndarray       # (T, I, K) forecast mean, Eq. (4)
    sigma: np.ndarray    # (T, I, K) forecast standard deviation
    D_hat: np.ndarray    # (T, I, K) planning demand, Eq. (8)
    q: np.ndarray        # (T, I) reserve multiplier, Eq. (7)


def observed_demand(trace: Trace) -> np.ndarray:
    obs = trace.demand.copy()
    for t in range(1, trace.T):
        if trace.stale[t]:
            obs[t] = obs[t - 1]
    return obs


def make_bundle(trace: Trace, forecaster, q0: np.ndarray, p: Params, reserve_mode: str = "adaptive") -> Bundle:
    d_obs = observed_demand(trace)
    mu, sigma = forecaster.predict(d_obs, trace.hours, p.season)
    D_hat, q = plan_series(d_obs, mu, sigma, q0, p, reserve_mode)
    return Bundle(trace, d_obs, mu, sigma, D_hat, q)


@dataclass
class Context:
    """Information available to a controller at the start of interval t."""
    t: int
    hour: float
    d_obs: np.ndarray      # (I, K) last measured demand (interval t-1)
    mu: np.ndarray
    sigma: np.ndarray
    D_hat: np.ndarray
    C_hat: np.ndarray      # (S,) estimated per-instance capacity, Eq. (3)
    n_max: np.ndarray      # (S,) from site health telemetry
    B: np.ndarray          # (L,) from link health telemetry
    b: np.ndarray          # (L,) background load
    lat: np.ndarray        # (I, S) usable path latency
    h: np.ndarray          # (S,) processing delay
    link_util: np.ndarray  # (L,)
    n_prev: np.ndarray
    x_prev: np.ndarray
    stale: bool
    mode: str

    def snapshot(self, D: np.ndarray, C: np.ndarray, constrained: bool = True) -> Snapshot:
        realistic = self.mode == "realistic"
        return Snapshot(D=D, C=C, n_max=self.n_max, B=self.B, b=self.b, lat=self.lat, h=self.h,
                        n_prev=self.n_prev, x_prev=self.x_prev,
                        link_constraints=realistic and constrained,
                        latency_constraints=realistic and constrained, coverage=constrained)


class Runner:
    def __init__(self, topo: Topology, p: Params, bundle: Bundle, mode: str = "realistic",
                 cap_mode: str = "estimated", seed: int = 0):
        self.topo, self.p, self.bundle, self.mode = topo, p, bundle, mode
        self.trace = bundle.trace
        self.sim = Simulator(topo, p, self.trace, mode, seed)
        self.est = CapacityEstimator(topo, p, fixed=(cap_mode == "fixed"))
        self.t = self.trace.eval_start
        # Algorithm 2, line 1: start from a feasible vFATD solution for the initial demand
        t0 = self.t - 1
        snap0 = Snapshot(D=self.trace.demand[t0], C=np.full(topo.S, p.C0 / 1.25), n_max=self.sim.n_max(t0),
                         B=np.where(np.isfinite(self.sim.link_cap(t0)), self.sim.link_cap(t0), 1e9),
                         b=self.trace.background[t0], lat=topo.lat, h=topo.h0.copy(),
                         n_prev=np.zeros(topo.S, int), x_prev=np.zeros((topo.I, topo.K, topo.S)),
                         link_constraints=mode == "realistic", latency_constraints=mode == "realistic")
        r = solve_vfatd(topo, p, snap0, time_limit=10.0)
        if not r.ok:
            raise RuntimeError("no feasible initial configuration")
        self.n_prev, self.x_prev = r.n, r.x
        warm = self.sim.step(t0, r.n, r.x, r.n, r.x)   # telemetry of interval t-1
        self._telemetry(warm, stale=False)

    @property
    def done(self) -> bool:
        return self.t >= self.trace.T

    def _mix(self, out: Outcome) -> np.ndarray:
        tot = out.class_load.sum(axis=1, keepdims=True)
        share = out.demand.sum(axis=0) / max(out.demand.sum(), 1e-9)
        return np.where(tot > 1e-9, out.class_load / np.maximum(tot, 1e-9), share[None, :])

    def _telemetry(self, out: Outcome, stale: bool) -> None:
        if not stale or not hasattr(self, "y"):
            self.y, self.psi, self.h, self.link_util, self.mix = out.y, out.psi, out.h, out.link_util, self._mix(out)
        self.C_hat = self.est.update(out.n, self.y, self.psi, self.mix, stale)
        self.last_t = out.t

    def context(self) -> Context:
        t, tr, topo = self.t, self.trace, self.topo
        tm = t - 1  # failures are seen through the telemetry of the previous interval
        if self.mode == "realistic":
            B = topo.B * tr.link_factor[tm]
            lat = topo.path_latency(B > 0)
            n_max = self.sim.n_max(tm)
            b = tr.background[tm] * (B > 0)
        else:
            B, lat, n_max, b = np.full(topo.L, np.inf), topo.lat, topo.n_max.copy(), np.zeros(topo.L)
        return Context(t=t, hour=float(tr.hours[t]), d_obs=self.bundle.d_obs[tm], mu=self.bundle.mu[t],
                       sigma=self.bundle.sigma[t], D_hat=self.bundle.D_hat[t], C_hat=self.C_hat.copy(),
                       n_max=n_max, B=B, b=b, lat=lat, h=self.h.copy(), link_util=self.link_util.copy(),
                       n_prev=self.n_prev.copy(), x_prev=self.x_prev.copy(), stale=bool(tr.stale[tm]),
                       mode=self.mode)

    def execute(self, n: np.ndarray, x: np.ndarray) -> Outcome:
        out = self.sim.step(self.t, self.n_prev, self.x_prev, n, x)
        # the execution unit acknowledges what was actually applied (a failed site loses instances)
        self.n_prev, self.x_prev = out.n.copy(), out.x.copy()
        self._telemetry(out, stale=bool(self.trace.stale[self.t]))
        self.t += 1
        return out
