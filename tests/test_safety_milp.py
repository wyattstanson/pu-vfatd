import numpy as np

from conftest import make_snapshot
from puvfatd.heuristics import assign_greedy
from puvfatd.milp import solve_projection, solve_vfatd
from puvfatd.model import feasible, link_flow, site_load
from puvfatd.safety import SafetyGate


def test_feasible_proposal_is_accepted_unchanged(topo, p, snap):
    r = solve_vfatd(topo, p, snap, 10.0)
    n, x, info = SafetyGate(topo, p)(snap, r.n, r.x)
    assert info["outcome"] == "accepted" and np.array_equal(n, r.n) and np.allclose(x, r.x)


def test_overloaded_proposal_is_repaired_to_a_feasible_one(topo, p, snap):
    r = solve_vfatd(topo, p, snap, 10.0)
    n_bad = np.maximum(r.n - 2, 0)
    n, x, info = SafetyGate(topo, p)(snap, n_bad, r.x)
    assert info["outcome"] in ("repaired", "projected")
    assert feasible(topo, p, snap, n, x)


def test_failed_link_carries_no_traffic(topo, p):
    B = topo.B.copy()
    B[7] = 0.0                                          # regA-cloud down
    snap = make_snapshot(topo, p, B=B, lat=topo.path_latency(B > 0), b=0.1 * B)
    r = solve_vfatd(topo, p, snap, 10.0)
    assert r.ok and link_flow(topo, snap.D, r.x)[7] <= 1e-9
    assert feasible(topo, p, snap, r.n, r.x)


def test_projection_of_a_feasible_action_is_itself(topo, p, snap):
    r = solve_vfatd(topo, p, snap, 10.0)
    pr = solve_projection(topo, p, snap, r.n, r.x, 5.0)
    assert pr.ok and np.array_equal(pr.n, r.n) and np.allclose(pr.x, r.x, atol=1e-6)


def test_impossible_demand_is_reported_and_never_bypasses_inspection(topo, p):
    snap = make_snapshot(topo, p, scale=3.0)            # ~1200 Gbps, more than the network can inspect
    gate = SafetyGate(topo, p)
    n, x, info = gate(snap, np.zeros(topo.S, int), np.zeros((topo.I, topo.K, topo.S)))
    assert info["outcome"] == "infeasible" and info["blocked_planned"] > 0
    assert np.all(x.sum(axis=2) <= 1 + 1e-9)            # nothing assigned twice or invented
    assert n.sum() <= p.n_lic and np.all(n <= snap.n_max)
    prot = topo.protected
    served_share = (snap.D * x.sum(axis=2)).sum(axis=0) / snap.D.sum(axis=0)
    assert served_share[prot].min() >= served_share[~prot].max() - 1e-6   # protected classes first


def test_previous_configuration_is_revalidated_before_reuse(topo, p, snap):
    r = solve_vfatd(topo, p, snap, 10.0)
    bigger = make_snapshot(topo, p, scale=1.3, n_prev=r.n, x_prev=r.x)
    assert not feasible(topo, p, bigger, r.n, r.x)       # safe before, unsafe now
    n, x, info = SafetyGate(topo, p)(bigger, r.n, r.x)
    assert info["outcome"] != "fallback_prev"
    assert info["outcome"] == "infeasible" or feasible(topo, p, bigger, n, x)


def test_greedy_assignment_respects_capacity_and_links(topo, p, snap):
    n = np.full(topo.S, 1)
    x = assign_greedy(topo, p, snap, n)
    assert np.all(site_load(snap.D, x) <= p.rho_fw * snap.C * n + 1e-6)
    assert np.all(snap.b + link_flow(topo, snap.D, x) <= p.rho_link * snap.B + 1e-6)
