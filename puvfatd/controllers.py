"""Controllers compared in the evaluation (Section 11 of the methodology).

Baselines decide from the last *measured* demand and the base paper's fixed nominal capacity
(reactive). PU-vFATD decides from the planning demand and the estimated capacity, and every
proposal goes through the SafetyGate.
"""
from __future__ import annotations

import time

import numpy as np

from .config import Params
from .heuristics import assign_greedy, genetic, kcentre
from .milp import solve_vfatd
from .model import Snapshot, site_load
from .runner import Context
from .safety import SafetyGate
from .topology import Topology


class Controller:
    name = "controller"
    plans = False   # True when the controller uses the planning demand (PICP is reported)

    def decide(self, ctx: Context) -> tuple[np.ndarray, np.ndarray, dict]:
        raise NotImplementedError


def _baseline_snapshot(ctx: Context, p: Params, constrained: bool) -> Snapshot:
    """Reactive view used by the baselines: measured demand, nominal capacity, loss penalised
    instead of a coverage requirement. 'constrained' adds the link/latency constraints (adapted
    baselines for the realistic comparison)."""
    snap = ctx.snapshot(D=ctx.d_obs, C=np.full(len(ctx.C_hat), p.C0), constrained=constrained)
    return snap.with_(coverage=False)


class StaticController(Controller):
    """Busy-hour allocation computed once and never changed."""

    def __init__(self, name: str, n: np.ndarray, x: np.ndarray):
        self.name, self.n, self.x = name, n, x

    @classmethod
    def busy_hour(cls, topo: Topology, p: Params, demand_hist: np.ndarray, mode: str) -> "StaticController":
        D = np.quantile(demand_hist, 0.95, axis=0)
        realistic = mode == "realistic"
        snap = Snapshot(D=D, C=np.full(topo.S, p.C0), n_max=topo.n_max, B=topo.B if realistic else np.full(topo.L, np.inf),
                        b=0.1 * topo.B if realistic else np.zeros(topo.L), lat=topo.lat, h=topo.h0,
                        n_prev=np.zeros(topo.S, int), x_prev=np.zeros((topo.I, topo.K, topo.S)),
                        link_constraints=realistic, latency_constraints=realistic, coverage=False)
        r = solve_vfatd(topo, p, snap, time_limit=30.0)
        return cls("Static (busy hour)", r.n, r.x)

    def decide(self, ctx):
        return self.n.copy(), self.x.copy(), {"decision_s": 0.0}


class LocalReactiveController(Controller):
    """Each ingress scales its own edge site to the last measured demand; no spill-over."""
    name = "Local reactive"

    def __init__(self, topo: Topology, p: Params):
        self.topo, self.p = topo, p

    def decide(self, ctx):
        t0 = time.perf_counter()
        topo, p = self.topo, self.p
        n = np.zeros(topo.S, int)
        x = np.zeros((topo.I, topo.K, topo.S))
        for i in range(topo.I):
            n[i] = min(int(np.ceil(ctx.d_obs[i].sum() / (p.rho_fw * p.C0))), int(ctx.n_max[i]))
            x[i, :, i] = 1.0
        return n, x, {"decision_s": time.perf_counter() - t0}


class VfatdMilpController(Controller):
    def __init__(self, topo: Topology, p: Params, constrained: bool):
        self.topo, self.p, self.constrained = topo, p, constrained
        self.name = "vFATD-MILP" + (" (adapted)" if constrained else "")

    def decide(self, ctx):
        t0 = time.perf_counter()
        r = solve_vfatd(self.topo, self.p, _baseline_snapshot(ctx, self.p, self.constrained), self.p.t_dec_s)
        n, x = (r.n, r.x) if r.ok else (ctx.n_prev, ctx.x_prev)
        return n, x, {"decision_s": time.perf_counter() - t0, "milp": r.status}


class KCentreController(Controller):
    def __init__(self, topo: Topology, p: Params, constrained: bool, seed: int = 0):
        self.topo, self.p, self.constrained = topo, p, constrained
        self.rng = np.random.default_rng(seed)
        self.name = "vFATD-kcentre" + (" (adapted)" if constrained else "")

    def decide(self, ctx):
        t0 = time.perf_counter()
        n, x = kcentre(self.topo, self.p, _baseline_snapshot(ctx, self.p, self.constrained), self.rng)
        return n, x, {"decision_s": time.perf_counter() - t0}


class GAController(Controller):
    def __init__(self, topo: Topology, p: Params, constrained: bool, seed: int = 0):
        self.topo, self.p, self.constrained = topo, p, constrained
        self.rng = np.random.default_rng(seed)
        self.name = "vFATD-GA" + (" (adapted)" if constrained else "")

    def decide(self, ctx):
        t0 = time.perf_counter()
        snap = _baseline_snapshot(ctx, self.p, self.constrained)
        kn, _ = kcentre(self.topo, self.p, snap, self.rng)
        n, x = genetic(self.topo, self.p, snap, self.rng, seed_n=[ctx.n_prev, kn])
        return n, x, {"decision_s": time.perf_counter() - t0}


class PUController(Controller):
    """PU-vFATD: planning demand + estimated capacity -> proposer -> SafetyGate (Algorithm 2)."""
    plans = True

    def __init__(self, name: str, topo: Topology, p: Params, proposer, gate: SafetyGate | None):
        self.name, self.topo, self.p, self.proposer, self.gate = name, topo, p, proposer, gate

    def decide(self, ctx):
        t0 = time.perf_counter()
        snap = ctx.snapshot(D=ctx.D_hat, C=ctx.C_hat, constrained=True)
        n, x = self.proposer(ctx, snap)
        t_prop = time.perf_counter() - t0
        info = {"outcome": "ungated"}
        if self.gate is not None:
            n, x, info = self.gate(snap, n, x)
        info["proposal_s"] = t_prop
        info["decision_s"] = time.perf_counter() - t0
        return n, x, info


class MilpProposer:
    """Deterministic proposer: the extended model of Stage 4 solved on the planning demand."""

    def __init__(self, topo: Topology, p: Params):
        self.topo, self.p = topo, p

    def __call__(self, ctx, snap):
        r = solve_vfatd(self.topo, self.p, snap, self.p.t_dec_s)
        return (r.n, r.x) if r.ok else (ctx.n_prev, ctx.x_prev)


class FastProposer:
    """Millisecond proposer standing in for the learned policy slot of Stage 5: route the planning
    demand to the cheapest allowed sites, then size each site to its load. It ignores the licence
    and coverage rules on purpose; the SafetyGate checks and repairs every proposal."""

    def __init__(self, topo: Topology, p: Params):
        self.topo, self.p = topo, p

    def __call__(self, ctx, snap):
        x = assign_greedy(self.topo, self.p, snap, snap.n_max)
        load = site_load(snap.D, x)
        n = np.ceil(load / np.maximum(self.p.rho_fw * snap.C, 1e-9) - 1e-9).astype(int)
        return np.clip(n, 0, snap.n_max), x
