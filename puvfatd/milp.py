"""Mixed-integer linear programs solved with HiGHS (scipy.optimize.milp).

* solve_vfatd      - minimise Eq. (9) over the feasible set (Eqs. 17-22). With link/latency/coverage
                     switched off and nominal capacity this is the base-paper style vFATD model;
                     with them switched on it is the extended model used as the PU-vFATD fallback.
* solve_projection - Eq. (38): the feasible configuration closest to a proposed action.
* solve_emergency  - infeasible demand: maximise inspected traffic, protected classes first,
                     within the instance and licence limits (no coverage requirement).
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .config import Params
from .model import Snapshot, c_off, c_on, effective_mask
from .topology import Topology


@dataclass
class MilpResult:
    n: np.ndarray | None
    x: np.ndarray | None
    status: str
    seconds: float

    @property
    def ok(self) -> bool:
        return self.n is not None


@contextlib.contextmanager
def _quiet_stdout():
    """HiGHS occasionally prints internal debug lines from C++; keep the console clean."""
    try:
        fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    sys.stdout.flush()
    saved = os.dup(fd)
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), fd)
        try:
            yield
        finally:
            os.dup2(saved, fd)
            os.close(saved)


class _Rows:
    def __init__(self):
        self.r, self.c, self.v, self.lo, self.hi = [], [], [], [], []

    def add(self, cols, vals, lo, hi):
        k = len(self.lo)
        self.r.extend([k] * len(cols))
        self.c.extend(cols)
        self.v.extend(vals)
        self.lo.append(lo)
        self.hi.append(hi)

    def constraint(self, N):
        A = coo_matrix((self.v, (self.r, self.c)), shape=(len(self.lo), N)).tocsr()
        return LinearConstraint(A, np.array(self.lo), np.array(self.hi))


def _solve(topo: Topology, p: Params, snap: Snapshot, kind: str, time_limit: float,
           n_ref=None, x_ref=None, loss_weight=None, dep_scale: float = 1.0) -> MilpResult:
    t0 = time.perf_counter()
    I, K, S = topo.I, topo.K, topo.S
    nx = I * K * S
    iN = np.arange(S)
    iX = S + np.arange(nx).reshape(I, K, S)
    iA = S + nx + np.arange(S)        # (n - n_prev)^+  or |n - n_ref|
    iB = 2 * S + nx + np.arange(S)    # (n_prev - n)^+
    iM = 3 * S + nx + np.arange(nx).reshape(I, K, S)   # |x - x_prev| or |x - x_ref|
    N = 3 * S + 2 * nx
    D = snap.D
    mask = effective_mask(topo, snap)

    c = np.zeros(N)
    lb, ub = np.zeros(N), np.full(N, np.inf)
    ub[iN] = snap.n_max
    ub[iX] = mask.astype(float)
    integrality = np.zeros(N)
    integrality[iN] = 1
    lat_f = np.where(np.isfinite(snap.lat), snap.lat, 0.0) + snap.h[None, :]
    if kind in ("cost", "emergency"):
        w = np.ones(K) if loss_weight is None else loss_weight
        c[iN] = topo.c_dep * dep_scale
        c[iX] = D[:, :, None] * (topo.c_tr[:, None, :] - p.c_loss * w[None, :, None] + p.c_lat * lat_f[:, None, :])
        c[iA] = c_on(topo, p)
        c[iB] = c_off(topo, p)
        c[iM] = p.c_mig / 2 * np.broadcast_to(D[:, :, None], (I, K, S))
        ref_n, ref_x = snap.n_prev, snap.x_prev
    elif kind == "proj":
        c[iA] = 1.0
        ub[iB] = 0.0
        c[iM] = p.proj_weight * np.broadcast_to(D[:, :, None], (I, K, S))
        ref_n, ref_x = n_ref, x_ref
    else:
        raise ValueError(kind)

    rows = _Rows()
    for s in range(S):                                                       # Eq. (18)
        cols = list(iX[:, :, s].ravel()) + [iN[s]]
        vals = list(D.ravel()) + [-p.rho_fw * snap.C[s]]
        rows.add(cols, vals, -np.inf, 0.0)
    if snap.link_constraints:                                                # Eq. (19)
        for l in range(topo.L):
            ii, ss = np.nonzero(topo.chi[l])
            if len(ii) == 0 or not np.isfinite(snap.B[l]):
                continue
            cols, vals = [], []
            for i, s in zip(ii, ss):
                cols += list(iX[i, :, s])
                vals += list(D[i, :])
            rows.add(cols, vals, -np.inf, max(0.0, p.rho_link * snap.B[l] - snap.b[l]))
    for i in range(I):                                                       # Eq. (21)
        for k in range(K):
            cols = list(iX[i, k, :])
            exact = snap.coverage and kind != "emergency" and topo.protected[k] and mask[i, k].any()
            rows.add(cols, [1.0] * S, 1.0 if exact else -np.inf, 1.0)
    if snap.coverage and kind != "emergency":                                # Eq. (22)
        rows.add(list(iX.ravel()), list(np.repeat(D.ravel(), S)), p.gamma_cov * D.sum(), np.inf)
    rows.add(list(iN), [1.0] * S, -np.inf, p.n_lic)                          # licence
    for s in range(S):                                                       # auxiliary |.| / (.)^+
        rows.add([iN[s], iA[s]], [1.0, -1.0], -np.inf, ref_n[s])
        if kind == "proj":
            rows.add([iN[s], iA[s]], [-1.0, -1.0], -np.inf, -ref_n[s])
        else:
            rows.add([iN[s], iB[s]], [-1.0, -1.0], -np.inf, -ref_n[s])
    for i in range(I):
        for k in range(K):
            for s in range(S):
                rows.add([iX[i, k, s], iM[i, k, s]], [1.0, -1.0], -np.inf, ref_x[i, k, s])
                rows.add([iX[i, k, s], iM[i, k, s]], [-1.0, -1.0], -np.inf, -ref_x[i, k, s])

    with _quiet_stdout():
        res = milp(c, constraints=[rows.constraint(N)], integrality=integrality, bounds=Bounds(lb, ub),
                   options={"time_limit": max(0.05, time_limit), "mip_rel_gap": 1e-3, "disp": False})
    secs = time.perf_counter() - t0
    if res.x is None:
        return MilpResult(None, None, {2: "infeasible", 1: "time_limit"}.get(res.status, f"status{res.status}"), secs)
    n = np.round(res.x[iN]).astype(int)
    x = np.clip(res.x[iX], 0.0, 1.0)
    rs = x.sum(axis=2, keepdims=True)
    x = np.where(rs > 1.0, x / np.maximum(rs, 1e-12), x)
    return MilpResult(n, x, "optimal" if res.status == 0 else "time_limit", secs)


def solve_vfatd(topo: Topology, p: Params, snap: Snapshot, time_limit: float = 10.0) -> MilpResult:
    return _solve(topo, p, snap, "cost", time_limit)


def solve_projection(topo: Topology, p: Params, snap: Snapshot, n_ref: np.ndarray, x_ref: np.ndarray,
                     time_limit: float = 2.0) -> MilpResult:
    return _solve(topo, p, snap, "proj", time_limit, n_ref=n_ref, x_ref=x_ref)


def solve_emergency(topo: Topology, p: Params, snap: Snapshot, time_limit: float = 2.0) -> MilpResult:
    w = np.where(topo.protected, 10.0, 1.0)   # serve protected classes first
    return _solve(topo, p, snap, "emergency", time_limit, loss_weight=w, dep_scale=0.01)
