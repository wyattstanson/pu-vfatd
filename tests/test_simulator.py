import numpy as np

from puvfatd.simulator import Simulator
from puvfatd.traffic import generate_trace


def _setup(topo, p, scenario="normal"):
    tr = generate_trace(topo, p, scenario, days=2, seed=3)
    return tr, Simulator(topo, p, tr, "realistic", seed=0)


def _split_to_regional(topo):
    x = np.zeros((topo.I, topo.K, topo.S))
    x[0, :, 4] = x[1, :, 4] = 1.0     # pop0, pop1 -> regA
    x[2, :, 5] = x[3, :, 5] = 1.0     # pop2, pop3 -> regB
    return x


def test_traffic_is_conserved(topo, p):
    tr, sim = _setup(topo, p)
    t = tr.eval_start + 200
    x = _split_to_regional(topo)
    n = np.array([0, 0, 0, 0, 4, 4, 0])
    out = sim.step(t, n, x, n + np.array([0, 0, 0, 0, 2, 2, 0]), x)
    assert np.isclose(out.served.sum() + out.lost, out.demand.sum())


def test_scale_out_waits_for_new_instances(topo, p):
    tr, sim = _setup(topo, p)
    t = tr.eval_start + 240
    x = _split_to_regional(topo)
    small = np.array([0, 0, 0, 0, 1, 1, 0])
    big = np.array([0, 0, 0, 0, 8, 8, 0])
    out = sim.step(t, small, x, big, x)
    assert np.isclose(out.f_old, (p.t_on_s + p.t_sync_s) / p.tau_s)
    assert out.dropped_cap > 0          # the old, small configuration carried the first part


def test_failed_link_drops_its_traffic(topo, p):
    tr, sim = _setup(topo, p)
    t = tr.eval_start + 100
    tr.link_factor[t, 0] = 0.0          # pop0-regA down
    x = _split_to_regional(topo)
    n = np.full(topo.S, 8)
    out = sim.step(t, n, x, n, x)
    assert out.dropped_link >= tr.demand[t, 0].sum() - 1e-6


def test_firewalls_never_serve_more_than_capacity(topo, p):
    tr, sim = _setup(topo, p)
    t = tr.eval_start + 250
    x = _split_to_regional(topo)
    n = np.array([0, 0, 0, 0, 1, 1, 0])
    out = sim.step(t, n, x, n, x)
    assert np.all(out.y <= out.cap_inst * out.n + 1e-9)
    assert out.dropped_cap > 0           # overload is dropped, not forwarded uninspected
