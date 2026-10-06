"""Joint optimisation model of Stage 4: cost terms (Eqs. 9-16), hard constraints (Eqs. 17-22),
normalised violations (Eq. 27) and the feasibility test (Eq. 37)."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import Params
from .topology import Topology

FAMILIES = ("cap", "link", "lat", "cov")
TOL = 1e-7


@dataclass
class Snapshot:
    """What the controller believes at the start of interval t."""
    D: np.ndarray          # (I, K) planning demand
    C: np.ndarray          # (S,) per-instance capacity used for planning
    n_max: np.ndarray      # (S,) instances that fit (0 for a failed site)
    B: np.ndarray          # (L,) link capacity (0 for a failed link)
    b: np.ndarray          # (L,) background load
    lat: np.ndarray        # (I, S) path latency (inf when unusable)
    h: np.ndarray          # (S,) processing-delay estimate
    n_prev: np.ndarray     # (S,) configuration in service
    x_prev: np.ndarray     # (I, K, S)
    link_constraints: bool = True     # False = base-paper assumption of an over-provisioned network
    latency_constraints: bool = True
    coverage: bool = True             # protected classes fully assigned + served ratio >= Gamma

    def with_(self, **kw) -> "Snapshot":
        return replace(self, **kw)


def c_on(topo: Topology, p: Params) -> np.ndarray:
    """Scale-out coefficient from measured start-up and sync times, Eq. (16)."""
    return topo.c_dep * (p.t_on_s + p.t_sync_s) / p.tau_s


def c_off(topo: Topology, p: Params) -> np.ndarray:
    return topo.c_dep * p.c_off_frac


def effective_mask(topo: Topology, snap: Snapshot) -> np.ndarray:
    """(I, K, S) sites allowed for each ingress-class pair: reachable, alive and within the latency
    budget (Eq. 20). A pair with no allowed site falls back to its lowest-latency reachable site."""
    reach = np.isfinite(snap.lat) & (snap.n_max > 0)[None, :]
    tot = np.where(reach, snap.lat + snap.h[None, :], np.inf)              # (I, S)
    if snap.latency_constraints:
        ok = reach[:, None, :] & (tot[:, None, :] <= topo.t_max[None, :, None])
    else:
        ok = np.repeat(reach[:, None, :], topo.K, axis=1)
    empty = ~ok.any(axis=2)
    if empty.any():
        best = np.argmin(tot, axis=1)                                       # (I,)
        for i, k in zip(*np.nonzero(empty)):
            if np.isfinite(tot[i, best[i]]):
                ok[i, k, best[i]] = True
    return ok


def site_load(D: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.einsum("ik,iks->s", D, x)


def link_flow(topo: Topology, D: np.ndarray, x: np.ndarray) -> np.ndarray:
    per_is = np.einsum("ik,iks->is", D, x)
    return np.tensordot(topo.chi, per_is, axes=([1, 2], [0, 1]))


def costs(topo: Topology, p: Params, D: np.ndarray, n: np.ndarray, x: np.ndarray,
          n_prev: np.ndarray, x_prev: np.ndarray, lat: np.ndarray, h: np.ndarray) -> dict:
    """Cost terms of Eqs. (10)-(15) for demand D (planning demand, or actual demand afterwards)."""
    F = D[:, :, None] * x
    lat_f = np.where(np.isfinite(lat), lat, 0.0)
    out = {
        "dep": float(topo.c_dep @ n),                                                  # Eq. (10)
        "tr": float(np.einsum("is,iks->", topo.c_tr, F)),                               # Eq. (11)
        "loss": float(p.c_loss * (D * np.clip(1 - x.sum(axis=2), 0, None)).sum()),      # Eq. (12)
        "lat": float(p.c_lat * np.einsum("is,iks->", lat_f + h[None, :], F)),          # Eq. (13)
        "scale": float(c_on(topo, p) @ np.clip(n - n_prev, 0, None)
                       + c_off(topo, p) @ np.clip(n_prev - n, 0, None)),               # Eq. (14)
        "mig": float(p.c_mig / 2 * (D[:, :, None] * np.abs(x - x_prev)).sum()),        # Eq. (15)
    }
    out["total"] = sum(out.values())                                                   # Eq. (9)
    return out


def violations(topo: Topology, p: Params, snap: Snapshot, n: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Normalised violation Omega_j of each constraint family (cap, link, lat, cov), Eq. (27)."""
    D = snap.D
    total = max(float(D.sum()), 1e-9)
    load = site_load(D, x)
    cap = np.clip(load - p.rho_fw * snap.C * n, 0, None).sum()                         # Eq. (18)
    cap += (p.rho_fw * snap.C * np.clip(n - snap.n_max, 0, None)).sum()                # Eq. (17)
    cap += p.rho_fw * p.C0 * max(0.0, float(n.sum()) - p.n_lic)                        # licence
    link = 0.0
    if snap.link_constraints:                                                          # Eq. (19)
        flow = link_flow(topo, D, x)
        link = np.clip(snap.b + flow - p.rho_link * snap.B, 0, None).sum() / topo.B.sum()
    mask = effective_mask(topo, snap)                                                  # Eq. (20)
    lat = (D[:, :, None] * x)[~mask].sum()
    rows = x.sum(axis=2)                                                               # Eq. (21)
    prot = topo.protected[None, :]
    if snap.coverage:
        cov = (D * np.abs(1 - rows))[np.broadcast_to(prot, D.shape)].sum()
        cov += (D * np.clip(rows - 1, 0, None))[np.broadcast_to(~prot, D.shape)].sum()
        cov += max(0.0, p.gamma_cov * total - float((D * np.minimum(rows, 1)).sum()))  # Eq. (22)
    else:
        cov = (D * np.clip(rows - 1, 0, None)).sum()
    return np.array([cap / total, link, lat / total, cov / total])


def feasible(topo: Topology, p: Params, snap: Snapshot, n: np.ndarray, x: np.ndarray) -> bool:
    """Eq. (37): the action is in the feasible set iff every family has zero violation."""
    return bool(np.all(violations(topo, p, snap, n, x) <= TOL) and np.all(x >= -1e-12))
