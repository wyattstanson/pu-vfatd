"""Stage 6: safety check, repair and fallback - Algorithm 1 (SafetyGate) of the methodology."""
from __future__ import annotations

import time

import numpy as np

from .config import Params
from .heuristics import assign_greedy, kcentre, unit_cost
from .milp import solve_emergency, solve_projection, solve_vfatd
from .model import TOL, Snapshot, effective_mask, feasible, link_flow, site_load, violations
from .topology import Topology

EPS = 1e-9


class SafetyGate:
    def __init__(self, topo: Topology, p: Params, projection: bool = True, milp_fallback: bool = True,
                 proj_time: float = 2.0, milp_time: float | None = None, seed: int = 0):
        self.topo, self.p = topo, p
        self.projection, self.milp_fallback = projection, milp_fallback
        self.proj_time = proj_time
        self.milp_time = p.t_dec_s if milp_time is None else milp_time
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ Algorithm 1
    def __call__(self, snap: Snapshot, n: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        t0 = time.perf_counter()
        topo, p = self.topo, self.p
        omega = violations(topo, p, snap, n, x)                                   # line 1, Eq. (27)
        info = {"omega": omega}
        if np.all(omega <= TOL):                                                  # line 2
            return n, x, {**info, "outcome": "accepted", "gate_s": time.perf_counter() - t0}
        n2, x2 = self.greedy_repair(snap, n.copy(), x.copy())                     # lines 3-8
        if feasible(topo, p, snap, n2, x2):                                       # line 9
            return n2, x2, {**info, "outcome": "repaired", "gate_s": time.perf_counter() - t0}
        remaining = p.t_dec_s - (time.perf_counter() - t0)
        if self.projection and remaining > 0:                                     # lines 10-11, Eq. (38)
            r = solve_projection(topo, p, snap, n, x, min(remaining, self.proj_time))
            if r.ok and feasible(topo, p, snap, r.n, r.x):
                return r.n, r.x, {**info, "outcome": "projected", "gate_s": time.perf_counter() - t0}
        if feasible(topo, p, snap, snap.n_prev, snap.x_prev):                    # line 12: revalidated
            return snap.n_prev.copy(), snap.x_prev.copy(), {**info, "outcome": "fallback_prev",
                                                            "gate_s": time.perf_counter() - t0}
        remaining = p.t_dec_s - (time.perf_counter() - t0)
        if self.milp_fallback and remaining > 0:                                  # line 13
            r = solve_vfatd(topo, p, snap, min(remaining, self.milp_time))
            cand = (r.n, r.x) if r.ok else kcentre(topo, p, snap, self.rng)
            if feasible(topo, p, snap, *cand):                                    # line 14
                return cand[0], cand[1], {**info, "outcome": "fallback_vfatd", "gate_s": time.perf_counter() - t0}
        n3, x3 = self.emergency(snap)                                             # lines 15-16
        blocked = float((snap.D * np.clip(1 - x3.sum(axis=2), 0, None)).sum())
        return n3, x3, {**info, "outcome": "infeasible", "blocked_planned": blocked,
                        "gate_s": time.perf_counter() - t0}

    # ------------------------------------------------------------------ greedy repair
    def greedy_repair(self, snap: Snapshot, n: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        topo, p = self.topo, self.p
        D = snap.D
        mask = effective_mask(topo, snap)
        uc = unit_cost(topo, p, snap)
        n = np.minimum(np.maximum(n, 0), snap.n_max).astype(int)
        while n.sum() > p.n_lic:                       # licence: drop the most expensive instance first
            s = int(np.argmax(np.where(n > 0, topo.c_dep, -np.inf)))
            n[s] -= 1
        # latency and assignment: only allowed sites, every row fully assigned
        x = np.where(mask, np.clip(x, 0, None), 0.0)
        rows = x.sum(axis=2)
        for i in range(topo.I):
            for k in range(topo.K):
                if rows[i, k] > 1:
                    x[i, k] /= rows[i, k]
                elif rows[i, k] < 1 - 1e-12 and mask[i, k].any():
                    s = int(np.argmin(np.where(mask[i, k], uc[i], np.inf)))
                    x[i, k, s] += 1 - rows[i, k]
        for _ in range(4):
            changed = self._fix_capacity(snap, n, x, mask, uc)
            if snap.link_constraints:
                changed |= self._fix_links(snap, n, x, mask, uc)
            if not changed:
                break
        return n, x

    def _licence_left(self, n):
        return self.p.n_lic - int(n.sum())

    def _room(self, snap, n, x, i, s, avoid=None):
        """Traffic that site s can still take from ingress i (scaling out if allowed)."""
        topo, p = self.topo, self.p
        if avoid is not None and topo.chi[avoid, i, s] > 0:
            return 0.0
        spare = p.rho_fw * snap.C[s] * (n[s] + min(snap.n_max[s] - n[s], max(0, self._licence_left(n))))
        spare -= site_load(snap.D, x)[s]
        if snap.link_constraints and topo.paths[i][s]:
            head = p.rho_link * snap.B - snap.b - link_flow(topo, snap.D, x)
            spare = min(spare, min(head[l] for l in topo.paths[i][s]))
        return max(0.0, spare)

    def _move(self, snap, n, x, mask, uc, i, k, s_from, amount, avoid=None) -> float:
        topo, p = self.topo, self.p
        D = snap.D
        moved = 0.0
        for s in np.argsort(uc[i]):
            if amount - moved <= EPS:
                break
            if s == s_from or not mask[i, k, s]:
                continue
            take = min(amount - moved, self._room(snap, n, x, i, s, avoid), D[i, k] * x[i, k, s_from])
            if take <= EPS:
                continue
            x[i, k, s_from] -= take / D[i, k]
            x[i, k, s] += take / D[i, k]
            need = int(np.ceil(site_load(D, x)[s] / max(p.rho_fw * snap.C[s], EPS) - 1e-9))
            n[s] = max(n[s], min(need, snap.n_max[s], n[s] + max(0, self._licence_left(n))))
            moved += take
        return moved

    def _fix_capacity(self, snap, n, x, mask, uc) -> bool:
        topo, p = self.topo, self.p
        D = snap.D
        changed = False
        load = site_load(D, x)
        for s in np.argsort(-(load - p.rho_fw * snap.C * n)):
            load = site_load(D, x)
            if load[s] <= p.rho_fw * snap.C[s] * n[s] + EPS:
                continue
            need = int(np.ceil(load[s] / max(p.rho_fw * snap.C[s], EPS) - 1e-9))
            add = max(0, min(need, snap.n_max[s]) - n[s])
            add = min(add, max(0, self._licence_left(n)))
            if add:
                n[s] += add
                changed = True
            excess = load[s] - p.rho_fw * snap.C[s] * n[s]
            if excess > EPS:
                flows = sorted(((D[i, k] * x[i, k, s], i, k) for i in range(topo.I) for k in range(topo.K)
                                if x[i, k, s] > 0), reverse=True)
                for f, i, k in flows:
                    if excess <= EPS:
                        break
                    moved = self._move(snap, n, x, mask, uc, i, k, s, min(f, excess))
                    excess -= moved
                    changed |= moved > 0
        return changed

    def _fix_links(self, snap, n, x, mask, uc) -> bool:
        topo, p = self.topo, self.p
        D = snap.D
        changed = False
        over = snap.b + link_flow(topo, D, x) - p.rho_link * snap.B
        for l in np.argsort(-over):
            excess = (snap.b + link_flow(topo, D, x) - p.rho_link * snap.B)[l]
            if excess <= EPS:
                continue
            flows = sorted(((D[i, k] * x[i, k, s], i, k, s) for i in range(topo.I) for k in range(topo.K)
                            for s in range(topo.S) if topo.chi[l, i, s] > 0 and x[i, k, s] > 0), reverse=True)
            for f, i, k, s in flows:
                if excess <= EPS:
                    break
                moved = self._move(snap, n, x, mask, uc, i, k, s, min(f, excess), avoid=l)
                excess -= moved
                changed |= moved > 0
        return changed

    # ------------------------------------------------------------------ explicit infeasibility
    def emergency(self, snap: Snapshot) -> tuple[np.ndarray, np.ndarray]:
        """No feasible configuration exists: use as many instances as the limits allow, serve
        protected classes first and block the rest. Inspection is never bypassed."""
        r = solve_emergency(self.topo, self.p, snap, time_limit=min(2.0, self.p.t_dec_s))
        if r.ok:
            return r.n, r.x
        n = snap.n_max.astype(int).copy()
        while n.sum() > self.p.n_lic:
            n[int(np.argmax(np.where(n > 0, self.topo.c_dep, -np.inf)))] -= 1
        return n, assign_greedy(self.topo, self.p, snap, n)
