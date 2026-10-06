import numpy as np
import pytest

from puvfatd.config import Params
from puvfatd.model import Snapshot
from puvfatd.topology import default_topology


@pytest.fixture
def p():
    return Params()


@pytest.fixture
def topo(p):
    return default_topology(p)


def make_snapshot(topo, p, scale=1.0, **kw):
    """A realistic snapshot around the evening peak (about 400 Gbps in total)."""
    D = (topo.peak[:, None] * topo.share[None, :]) * 0.9 * scale
    base = dict(D=D, C=np.full(topo.S, 40.0), n_max=topo.n_max.copy(), B=topo.B.copy(), b=0.1 * topo.B,
                lat=topo.lat.copy(), h=topo.h0.copy(), n_prev=np.zeros(topo.S, int),
                x_prev=np.zeros((topo.I, topo.K, topo.S)))
    base.update(kw)
    return Snapshot(**base)


@pytest.fixture
def snap(topo, p):
    return make_snapshot(topo, p)
