"""Interval-level simulator of the telco cloud-edge network (Stage 7 'Execution and feedback').

Each interval is split in two phases to model make-before-break execution: until new instances are
ready (T^on + T^sync) the old configuration carries the traffic, afterwards the new one does.
Traffic that cannot be inspected is dropped or blocked; it is never forwarded uninspected.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Params
from .model import c_off, c_on
from .topology import Topology
from .traffic import Trace


@dataclass
class Outcome:
    t: int
    n: np.ndarray            # applied instance counts (a failed site loses its instances)
    x: np.ndarray            # applied split
    demand: np.ndarray       # (I, K) actual offered traffic
    served: np.ndarray       # (I, K) inspected and delivered, phase-weighted
    blocked: float           # never assigned to a firewall (Gbps)
    dropped_link: float      # lost on overloaded / failed links
    dropped_cap: float       # arrived at a firewall without capacity
    y: np.ndarray            # (S,) processed throughput, end of interval
    psi: np.ndarray          # (S,) CPU utilisation, end of interval
    h: np.ndarray            # (S,) processing delay, ms
    class_load: np.ndarray   # (S, K) load arriving at each site by class
    link_util: np.ndarray    # (L,)
    lat_vals: np.ndarray     # flow latencies (both phases)
    lat_w: np.ndarray        # served traffic carried by each latency value
    lat_viol: np.ndarray     # 1 where the flow latency exceeds its class budget
    cap_inst: np.ndarray     # (S,) true per-instance capacity
    cost: dict               # realised cost terms with the actual demand
    f_old: float             # fraction of the interval served by the old configuration

    @property
    def lost(self) -> float:
        return self.blocked + self.dropped_link + self.dropped_cap


class Simulator:
    def __init__(self, topo: Topology, params: Params, trace: Trace, mode: str = "realistic", seed: int = 0):
        if mode not in ("realistic", "original"):
            raise ValueError(mode)
        self.topo, self.p, self.trace, self.mode = topo, params, trace, mode
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ environment state at t
    def n_max(self, t: int) -> np.ndarray:
        if self.mode == "original":
            return self.topo.n_max.copy()
        return np.floor(self.topo.n_max * self.trace.site_factor[t]).astype(int)

    def link_cap(self, t: int) -> np.ndarray:
        if self.mode == "original":
            return np.full(self.topo.L, np.inf)
        return self.topo.B * self.trace.link_factor[t]

    def cap_per_instance(self, t: int, mix: np.ndarray) -> np.ndarray:
        """True per-instance throughput for the class mix at each site (S, K)."""
        if self.mode == "original":
            return np.full(self.topo.S, self.p.C0)
        kappa = self.topo.kappa * self.trace.kappa_factor[t]
        return self.p.C0 / (mix @ kappa)

    # ------------------------------------------------------------------ one phase
    def _phase(self, t: int, D: np.ndarray, n: np.ndarray, x: np.ndarray) -> dict:
        topo = self.topo
        F = D[:, :, None] * x
        blocked = float((D * np.clip(1 - x.sum(axis=2), 0, None)).sum())
        B = self.link_cap(t)
        flow_l = np.tensordot(topo.chi, F.sum(axis=1), axes=([1, 2], [0, 1]))
        if self.mode == "realistic":
            b = self.trace.background[t] * (B > 0)
            avail = np.clip(B - b, 0, None)
            g = np.where(flow_l > 1e-12, np.minimum(1.0, avail / np.maximum(flow_l, 1e-12)), 1.0)
            surv = np.ones((topo.I, topo.S))
            for l in range(topo.L):
                surv = np.where(topo.chi[l] > 0, np.minimum(surv, g[l]), surv)
            link_util = (b + np.minimum(flow_l, avail)) / topo.B
        else:
            surv = np.ones((topo.I, topo.S))
            link_util = flow_l / topo.B
        F1 = F * surv[:, None, :]
        dropped_link = float((F - F1).sum())

        arriving = F1.sum(axis=(0, 1))
        class_load = np.einsum("iks->sk", F1)
        mix = np.where(arriving[:, None] > 1e-9, class_load / np.maximum(arriving[:, None], 1e-9),
                       D.sum(axis=0) / max(D.sum(), 1e-9))
        cap_inst = self.cap_per_instance(t, mix)
        cap = cap_inst * n
        u = np.where(arriving > 1e-12, np.minimum(1.0, cap / np.maximum(arriving, 1e-12)), 1.0)
        served = F1 * u[None, None, :]
        psi = np.where(cap > 0, np.minimum(1.0, arriving / np.maximum(cap, 1e-12)), 0.0)
        h = topo.h0 / (1 - np.minimum(psi, 0.95))
        lat = np.where(np.isfinite(topo.lat), topo.lat, 0.0)[:, None, :] + h[None, None, :]
        lat = np.broadcast_to(lat, F.shape)
        return dict(F=F, served=served, blocked=blocked, dropped_link=dropped_link,
                    dropped_cap=float((F1 - served).sum()), y=served.sum(axis=(0, 1)), psi=psi, h=h,
                    class_load=class_load, link_util=link_util, lat=lat, cap_inst=cap_inst)

    # ------------------------------------------------------------------ one interval
    def step(self, t: int, n_prev: np.ndarray, x_prev: np.ndarray, n_req: np.ndarray, x_req: np.ndarray) -> Outcome:
        topo, p = self.topo, self.p
        D = self.trace.demand[t]
        nmax = self.n_max(t)
        n = np.minimum(np.asarray(n_req, int), nmax)        # instances on a failed site are lost
        n_old = np.minimum(np.asarray(n_prev, int), nmax)
        x = np.asarray(x_req, float)
        if np.any(n > n_old):
            f_old = p.f_scale_out
        elif np.abs(x - x_prev).sum() > 1e-9:
            f_old = p.f_reroute
        else:
            f_old = 0.0
        new = self._phase(t, D, n, x)
        old = self._phase(t, D, n_old, x_prev) if f_old > 0 else new
        w_old, w_new = f_old, 1.0 - f_old

        def mix(key):
            return w_old * old[key] + w_new * new[key]

        served = mix("served")
        lat_vals = np.concatenate([old["lat"].ravel(), new["lat"].ravel()])
        lat_w = np.concatenate([w_old * old["served"].ravel(), w_new * new["served"].ravel()])
        budget = np.broadcast_to(topo.t_max[None, :, None], new["lat"].shape)
        lat_viol = np.concatenate([(old["lat"] > budget).ravel(), (new["lat"] > budget).ravel()]).astype(float)
        lost = mix("blocked") + mix("dropped_link") + mix("dropped_cap")
        F = mix("F")
        cost = {
            "dep": float(topo.c_dep @ n),
            "tr": float(np.einsum("is,iks->", topo.c_tr, F)),
            "loss": float(p.c_loss * lost),
            "lat": float(p.c_lat * (lat_vals * lat_w).sum()),
            "scale": float(c_on(topo, p) @ np.clip(n - n_old, 0, None) + c_off(topo, p) @ np.clip(n_old - n, 0, None)),
            "mig": float(p.c_mig / 2 * (D[:, :, None] * np.abs(x - x_prev)).sum()),
        }
        cost["total"] = sum(cost.values())
        psi_meas = np.clip(new["psi"] + self.rng.normal(0, 0.01, topo.S) * (new["psi"] > 0), 0, 1)
        return Outcome(t=t, n=n, x=x, demand=D, served=served.sum(axis=2),
                       blocked=float(mix("blocked")), dropped_link=float(mix("dropped_link")),
                       dropped_cap=float(mix("dropped_cap")), y=new["y"], psi=psi_meas, h=new["h"],
                       class_load=new["class_load"], link_util=new["link_util"], lat_vals=lat_vals,
                       lat_w=lat_w, lat_viol=lat_viol, cap_inst=new["cap_inst"], cost=cost, f_old=f_old)
