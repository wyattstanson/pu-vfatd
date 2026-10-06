"""Telco cloud-edge continuum model: ingress points, computing sites, links and pre-computed paths.

The layout is a small synthetic operator network with the same structure as the base paper's
setting (4 ingress PoPs, edge / regional / cloud computing centres). It is not the Polish operator
topology, whose details are not available to the team.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from .config import Params

LOCAL_LAT_MS = 0.1  # ingress point to its co-located edge site


@dataclass
class Topology:
    ingress: list[str]
    sites: list[str]
    tiers: list[str]
    link_names: list[str]
    B: np.ndarray          # (L,) link capacity, Gbps
    link_lat: np.ndarray   # (L,) link latency, ms
    c_dep: np.ndarray      # (S,) deployment cost per instance per interval
    n_max: np.ndarray      # (S,) instances that fit in the site's CPU/memory
    h0: np.ndarray         # (S,) idle processing delay, ms
    paths: list[list[list[int]]]  # paths[i][s] = link indices (empty for local / unreachable)
    lat: np.ndarray        # (I, S) path latency, ms; inf when the site does not serve the ingress
    chi: np.ndarray        # (L, I, S) 1 if link l is on the path i -> s, Eq. (19)
    c_tr: np.ndarray       # (I, S) transport cost per Gbps
    classes: list[str]
    t_max: np.ndarray      # (K,) latency budget, ms
    kappa: np.ndarray      # (K,) relative inspection load, Eq. (1)
    protected: np.ndarray  # (K,) bool
    share: np.ndarray      # (K,) average class share of traffic
    peak: np.ndarray       # (I,) peak demand per ingress, Gbps

    @property
    def I(self) -> int:  # noqa: E743 - matches the notation of the document
        return len(self.ingress)

    @property
    def S(self) -> int:
        return len(self.sites)

    @property
    def K(self) -> int:
        return len(self.classes)

    @property
    def L(self) -> int:
        return len(self.link_names)

    def path_latency(self, link_ok: np.ndarray) -> np.ndarray:
        """Path latency with failed links removed: a path that uses a failed link becomes unusable."""
        lat = self.lat.copy()
        failed = ~link_ok
        if failed.any():
            uses_failed = np.tensordot(failed.astype(float), self.chi, axes=(0, 0)) > 0  # (I, S)
            lat[uses_failed] = np.inf
        return lat


def default_topology(params: Params) -> Topology:
    ingress = ["pop0", "pop1", "pop2", "pop3"]
    sites = ["edge0", "edge1", "edge2", "edge3", "regA", "regB", "cloud"]
    tiers = ["edge"] * 4 + ["regional"] * 2 + ["cloud"]
    nodes = ingress + ["regA", "regB", "cloud"]
    site_node = [0, 1, 2, 3, 4, 5, 6]
    # (u, v, capacity Gbps, latency ms): primary uplinks, cross-region backups, core links
    links = [
        ("pop0", "regA", 100, 2.0), ("pop1", "regA", 100, 2.5),
        ("pop2", "regB", 100, 2.0), ("pop3", "regB", 100, 2.5),
        ("pop0", "regB", 40, 4.0), ("pop2", "regA", 40, 4.0),
        ("regA", "regB", 150, 3.0), ("regA", "cloud", 200, 7.0), ("regB", "cloud", 200, 7.0),
    ]
    idx = {name: k for k, name in enumerate(nodes)}
    n_nodes = len(nodes)
    w = np.zeros((n_nodes, n_nodes))
    link_of = {}
    for l, (u, v, _, d) in enumerate(links):
        w[idx[u], idx[v]] = w[idx[v], idx[u]] = d
        link_of[(idx[u], idx[v])] = link_of[(idx[v], idx[u])] = l
    dist, pred = shortest_path(csr_matrix(w), directed=False, return_predecessors=True)

    I, S, L = len(ingress), len(sites), len(links)
    lat = np.full((I, S), np.inf)
    chi = np.zeros((L, I, S))
    paths: list[list[list[int]]] = [[[] for _ in range(S)] for _ in range(I)]
    for i in range(I):
        for s in range(S):
            if tiers[s] == "edge":
                if s == i:  # an edge site serves only its own ingress point
                    lat[i, s] = LOCAL_LAT_MS
                continue
            target = site_node[s]
            lat[i, s] = dist[i, target]
            node, hops = target, []
            while node != i:
                prev = pred[i, node]
                hops.append(link_of[(prev, node)])
                node = prev
            paths[i][s] = hops[::-1]
            chi[hops, i, s] = 1.0

    finite = np.isfinite(lat)
    c_tr = np.zeros((I, S))
    c_tr[finite] = params.c_tr_per_ms * lat[finite]
    for i in range(I):
        c_tr[i, i] = 0.0  # co-located edge site: no transport

    return Topology(
        ingress=ingress, sites=sites, tiers=tiers,
        link_names=[f"{u}-{v}" for u, v, _, _ in links],
        B=np.array([c for _, _, c, _ in links], float),
        link_lat=np.array([d for _, _, _, d in links], float),
        c_dep=np.array([3.0] * 4 + [1.6] * 2 + [1.0]),
        n_max=np.array([3] * 4 + [8] * 2 + [24]),
        h0=np.full(S, 0.3),
        paths=paths, lat=lat, chi=chi, c_tr=c_tr,
        classes=["urllc", "interactive", "video", "bulk"],
        t_max=np.array([3.5, 7.0, 15.0, 40.0]),
        kappa=np.array([1.0, 1.15, 1.4, 1.05]),
        protected=np.array([True, True, True, False]),
        share=np.array([0.10, 0.25, 0.45, 0.20]),
        peak=np.array([140.0, 100.0, 120.0, 80.0]),
    )
