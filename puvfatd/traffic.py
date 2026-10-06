"""Synthetic, clearly labelled traffic traces with a daily cycle and controlled stress events.

The operator trace of the base paper (18.5 M flows) is not available to the team, so every result
produced with this module is on synthetic workloads that imitate its daily pattern.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Params
from .topology import Topology

SCENARIOS = ["normal", "flash_crowd", "ddos", "link_failure", "node_failure",
             "telemetry_delay", "security_profile", "mixed"]
INTERVALS_PER_HOUR = 12


def _shape(hours: np.ndarray) -> np.ndarray:
    day = 0.5 - 0.5 * np.cos(2 * np.pi * (hours - 4.5) / 24.0)
    evening = np.exp(-0.5 * ((hours - 20.5) / 2.5) ** 2)
    return 0.25 + 0.55 * day + 0.30 * evening


_SHAPE_MAX = float(_shape(np.linspace(0, 24, 2881)).max())


def daily_profile(hours: np.ndarray) -> np.ndarray:
    """Normalised daily load shape (peak 1.0): trough around 04:30, broad day, evening peak."""
    return _shape(hours) / _SHAPE_MAX


@dataclass
class Trace:
    scenario: str
    demand: np.ndarray        # (T, I, K) true offered traffic, Gbps
    link_factor: np.ndarray   # (T, L) multiplier on link capacity (0 = failed)
    site_factor: np.ndarray   # (T, S) multiplier on site capacity (0 = failed)
    kappa_factor: np.ndarray  # (T, K) multiplier on the inspection load (security-profile change)
    stale: np.ndarray         # (T,) telemetry of interval t is not delivered (old value repeated)
    background: np.ndarray    # (T, L) non-firewall traffic on each link, Gbps
    hours: np.ndarray         # (T,) hour of day
    eval_start: int           # first interval that is evaluated (earlier days are history)
    events: list[dict] = field(default_factory=list)

    @property
    def T(self) -> int:
        return self.demand.shape[0]


def _window(day: int, start_hour: float, n_intervals: int, T: int) -> tuple[int, int]:
    a = day * 288 + int(round(start_hour * INTERVALS_PER_HOUR))
    return a, min(T, a + n_intervals)


def _apply_event(kind: str, tr: Trace, topo: Topology, day: int, rng: np.random.Generator) -> None:
    T = tr.T
    if kind == "flash_crowd":
        i = int(rng.integers(topo.I))
        a, b = _window(day, rng.uniform(13, 16), int(rng.integers(16, 25)), T)
        env = np.ones(T)
        ramp, decay = 2, 6
        for t in range(a, min(T, b + decay)):
            if t < a + ramp:
                env[t] = 1 + 0.8 * (t - a + 1) / ramp
            elif t < b:
                env[t] = 1.8
            else:
                env[t] = 1.8 - 0.8 * (t - b + 1) / decay
        mult = np.array([1.2, 1.0, 1.0, 1.2])  # interactive and video grow the most
        for k in range(topo.K):
            scale = 1 + (env - 1) * (mult[k] if k in (1, 2) else 0.25)
            tr.demand[:, i, k] *= scale
        tr.events.append(dict(kind=kind, ingress=topo.ingress[i], t_start=a, t_end=b + decay))
    elif kind == "ddos":
        i = int(rng.integers(topo.I))
        a, b = _window(day, rng.uniform(18, 20), 24, T)
        tr.demand[a:b, i, 2:4] *= 2.5  # volumetric surge on video and bulk classes
        tr.events.append(dict(kind=kind, ingress=topo.ingress[i], t_start=a, t_end=b))
    elif kind == "link_failure":
        l = int(rng.choice([6, 7, 8]))  # a core link
        a, b = _window(day, rng.uniform(17, 19), 36, T)
        tr.link_factor[a:b, l] = 0.0
        tr.events.append(dict(kind=kind, link=topo.link_names[l], t_start=a, t_end=b))
    elif kind == "node_failure":
        s = int(rng.choice([4, 5]))  # a regional site
        a, b = _window(day, rng.uniform(16, 18), 24, T)
        tr.site_factor[a:b, s] = 0.0
        tr.events.append(dict(kind=kind, site=topo.sites[s], t_start=a, t_end=b))
    elif kind == "telemetry_delay":
        a, b = _window(day, rng.uniform(16, 18), 36, T)
        tr.stale[a:b] = rng.random(b - a) < 0.7  # most reports late, some get through
        tr.events.append(dict(kind=kind, t_start=a, t_end=b))
    elif kind == "security_profile":
        a, _ = _window(day, rng.uniform(11, 13), 1, T)
        b = (day + 1) * 288
        tr.kappa_factor[a:b, 2] = 1.5  # IPS enabled on the video class
        tr.events.append(dict(kind=kind, cls="video", t_start=a, t_end=min(T, b)))
    elif kind == "mixed":
        for sub in ("flash_crowd", "link_failure", "telemetry_delay"):
            _apply_event(sub, tr, topo, day, rng)
    elif kind != "normal":
        raise ValueError(f"unknown scenario {kind}")


def generate_trace(topo: Topology, params: Params, scenario: str = "normal", days: int = 2,
                   seed: int = 0, random_event_days: bool = False) -> Trace:
    """Generate `days` days of traffic; the scenario event is placed on the last day.

    With `random_event_days`, every earlier day also receives a random event (used for training
    pools so that the policy and forecaster see abnormal behaviour).
    """
    rng = np.random.default_rng(seed)
    T, I, K, L, S = days * 288, topo.I, topo.K, topo.L, topo.S
    t = np.arange(T)
    hours = (t % 288) / INTERVALS_PER_HOUR
    day = t // 288

    offsets = np.array([0.0, 0.5, -0.5, 1.0])[:I]
    prof = daily_profile((hours[:, None] - offsets[None, :]) % 24)       # (T, I)
    first_weekday = int(rng.integers(7))
    weekend = np.array([0.85 if (first_weekday + d) % 7 >= 5 else 1.0 for d in range(days)])
    day_amp = rng.normal(1.0, 0.04, size=(days, I)) * weekend[:, None]
    total = topo.peak[None, :] * prof * day_amp[day]

    noise = np.zeros((T, I))  # AR(1) multiplicative noise
    for k in range(1, T):
        noise[k] = 0.85 * noise[k - 1] + rng.normal(0, 0.025, I)
    total *= np.exp(noise)

    phase = np.array([9.0, 13.0, 21.0, 3.0])[:K]
    shares = topo.share[None, :] * (1 + 0.15 * np.sin(2 * np.pi * (hours[:, None] - phase[None, :]) / 24))
    shares /= shares.sum(axis=1, keepdims=True)
    demand = total[:, :, None] * shares[:, None, :] * np.exp(rng.normal(0, 0.02, (T, I, K)))

    tr = Trace(scenario=scenario, demand=demand, link_factor=np.ones((T, L)), site_factor=np.ones((T, S)),
               kappa_factor=np.ones((T, K)), stale=np.zeros(T, bool),
               background=0.10 * topo.B[None, :] * prof.mean(axis=1)[:, None],
               hours=hours, eval_start=(days - 1) * 288)
    if random_event_days:
        for d in range(days - 1):
            kind = str(rng.choice(SCENARIOS, p=[0.30, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05, 0.10]))
            _apply_event(kind, tr, topo, d, rng)
    _apply_event(scenario, tr, topo, days - 1, rng)
    return tr
