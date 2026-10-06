"""Heuristic baselines of the base paper, re-implemented for this simulation.

These follow the paper's description (a greedy k-centre style placement with randomised choices and
a fixed iteration budget, and a genetic algorithm over instance counts whose traffic split is
reconstructed greedily). They are our re-implementations, not the authors' code.
"""
from __future__ import annotations

import numpy as np

from .config import Params
from .model import Snapshot, costs, effective_mask
from .topology import Topology


def unit_cost(topo: Topology, p: Params, snap: Snapshot) -> np.ndarray:
    lat = np.where(np.isfinite(snap.lat), snap.lat, 0.0) + snap.h[None, :]
    return topo.c_tr + p.c_lat * lat                                           # (I, S)


def assign_greedy(topo: Topology, p: Params, snap: Snapshot, n: np.ndarray, mask: np.ndarray | None = None,
                  order: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Fill each ingress-class demand into the cheapest allowed sites with spare capacity.
    Protected classes go first; whatever does not fit stays unassigned (blocked)."""
    mask = effective_mask(topo, snap) if mask is None else mask
    D = snap.D
    x = np.zeros((topo.I, topo.K, topo.S))
    cap_left = p.rho_fw * snap.C * n
    link_left = (p.rho_link * snap.B - snap.b) if snap.link_constraints else np.full(topo.L, np.inf)
    uc = unit_cost(topo, p, snap)
    if order is None:
        order = sorted(((i, k) for i in range(topo.I) for k in range(topo.K)),
                       key=lambda ik: (not topo.protected[ik[1]], -D[ik]))
    for i, k in order:
        rem = D[i, k]
        if rem <= 1e-12:
            continue
        for s in np.argsort(uc[i]):
            if not mask[i, k, s] or cap_left[s] <= 1e-12:
                continue
            path = topo.paths[i][s]
            head = min((link_left[l] for l in path), default=np.inf)
            take = min(rem, cap_left[s], max(head, 0.0))
            if take <= 1e-12:
                continue
            x[i, k, s] += take / D[i, k]
            cap_left[s] -= take
            for l in path:
                link_left[l] -= take
            rem -= take
            if rem <= 1e-12:
                break
    return x


def kcentre(topo: Topology, p: Params, snap: Snapshot, rng: np.random.Generator,
            budget: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Greedy nearest-centre placement with randomised choice and a fixed iteration budget."""
    mask = effective_mask(topo, snap)
    n = np.zeros(topo.S, int)
    D = snap.D
    for _ in range(budget):
        x = assign_greedy(topo, p, snap, n, mask)
        left = D * np.clip(1 - x.sum(axis=2), 0, None)                         # (I, K)
        if left.sum() <= 1e-6 or n.sum() >= p.n_lic:
            break
        i = int(np.argmax(left.sum(axis=1)))
        allowed = mask[i][left[i] > 1e-9].any(axis=0) & (n < snap.n_max)
        if not allowed.any():
            break
        cand = np.flatnonzero(allowed)
        cand = cand[np.argsort(snap.lat[i, cand])]
        s = int(cand[1] if len(cand) > 1 and rng.random() < 0.3 else cand[0])   # randomised choice
        need = int(np.ceil(left[i].sum() / max(p.rho_fw * snap.C[s], 1e-9)))
        n[s] = min(snap.n_max[s], n[s] + max(1, need), n[s] + p.n_lic - int(n.sum()))
    return n, assign_greedy(topo, p, snap, n, mask)


def genetic(topo: Topology, p: Params, snap: Snapshot, rng: np.random.Generator, pop: int = 20,
            gens: int = 15, seed_n: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """GA over instance counts; the traffic split is reconstructed greedily for each candidate."""
    mask = effective_mask(topo, snap)
    hi = snap.n_max.astype(int)

    def fitness(n):
        x = assign_greedy(topo, p, snap, n, mask)
        f = costs(topo, p, snap.D, n, x, snap.n_prev, snap.x_prev, snap.lat, snap.h)["total"]
        return f + 10.0 * max(0, int(n.sum()) - p.n_lic)

    popl = [np.minimum(hi, s.astype(int)) for s in (seed_n or [])]
    while len(popl) < pop:
        base = popl[rng.integers(len(popl))] if popl else np.zeros(topo.S, int)
        popl.append(np.clip(base + rng.integers(-2, 3, topo.S), 0, hi))
    fit = np.array([fitness(n) for n in popl])
    for _ in range(gens):
        elite = [popl[j] for j in np.argsort(fit)[:2]]
        children = []
        while len(children) < pop - 2:
            a, b = (min(rng.choice(pop, 3, replace=False), key=lambda j: fit[j]) for _ in range(2))
            child = np.where(rng.random(topo.S) < 0.5, popl[a], popl[b])
            mut = rng.random(topo.S) < 0.15
            child = np.clip(child + mut * rng.integers(-1, 2, topo.S), 0, hi)
            children.append(child)
        popl = elite + children
        fit = np.array([fitness(n) for n in popl])
    best = popl[int(np.argmin(fit))]
    return best, assign_greedy(topo, p, snap, best, mask)
