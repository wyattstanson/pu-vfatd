import numpy as np

from puvfatd.milp import solve_vfatd
from puvfatd.model import c_on, costs, effective_mask, feasible, violations


def test_cost_terms_hand_checked(topo, p):
    D = np.zeros((topo.I, topo.K))
    D[0, 2] = 10.0                                    # 10 Gbps of video at pop0
    n = np.zeros(topo.S, int)
    n[4] = 1                                          # one instance at regA
    x = np.zeros((topo.I, topo.K, topo.S))
    x[0, 2, 4] = 0.5                                  # half of it to regA, half blocked
    zeros_n = np.zeros(topo.S, int)
    c = costs(topo, p, D, n, x, zeros_n, np.zeros_like(x), topo.lat, topo.h0)
    assert np.isclose(c["dep"], 1.6)                                   # Eq. (10)
    assert np.isclose(c["tr"], p.c_tr_per_ms * 2.0 * 5.0)              # Eq. (11): 5 Gbps over 2 ms
    assert np.isclose(c["loss"], p.c_loss * 5.0)                       # Eq. (12): 5 Gbps blocked
    assert np.isclose(c["lat"], p.c_lat * 5.0 * (2.0 + 0.3))           # Eq. (13)
    assert np.isclose(c["scale"], c_on(topo, p)[4])                    # Eq. (14)
    assert np.isclose(c["mig"], p.c_mig / 2 * 10.0 * 0.5)              # Eq. (15)
    assert np.isclose(c["total"], sum(v for k, v in c.items() if k != "total"))


def test_latency_mask_excludes_far_sites_for_urllc(topo, snap):
    mask = effective_mask(topo, snap)
    urllc = topo.classes.index("urllc")
    assert mask[0, urllc].tolist() == [True, False, False, False, True, False, False]  # edge0, regA only
    assert mask[0, topo.classes.index("bulk"), 6]                                       # bulk may use cloud


def test_mask_falls_back_to_lowest_latency_site(topo, snap):
    s2 = snap.with_(h=np.full(topo.S, 50.0))          # every site far too slow for every class
    mask = effective_mask(topo, s2)
    assert mask.any(axis=2).all()
    assert mask[0, 0, 0] and mask.sum(axis=2).max() == 1


def test_milp_solution_is_feasible_and_overload_is_detected(topo, p, snap):
    r = solve_vfatd(topo, p, snap, 10.0)
    assert r.ok and feasible(topo, p, snap, r.n, r.x)
    n_bad = r.n.copy()
    n_bad[np.argmax(r.n)] -= 2                         # remove capacity -> capacity family violated
    om = violations(topo, p, snap, n_bad, r.x)
    assert om[0] > 0 and not feasible(topo, p, snap, n_bad, r.x)


def test_unassigned_protected_traffic_violates_coverage(topo, p, snap):
    r = solve_vfatd(topo, p, snap, 10.0)
    x = r.x.copy()
    x[0, 0] = 0.0                                      # drop pop0 urllc (protected)
    assert violations(topo, p, snap, r.n, x)[3] > 0
