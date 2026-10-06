"""Evaluation metrics of Section 11 (Eqs. 40-45) plus safety-layer and timing statistics."""
from __future__ import annotations

import numpy as np

from .config import Params
from .runner import Bundle
from .simulator import Outcome
from .topology import Topology


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    m = weights > 0
    if not m.any():
        return float("nan")
    v, w = values[m], weights[m]
    order = np.argsort(v)
    cw = np.cumsum(w[order])
    return float(v[order][np.searchsorted(cw, q * cw[-1])])


def recovery_minutes(str_t: np.ndarray, t_f: int, p: Params, hold: int = 3) -> float:
    """Eq. (45): time from the event start until the per-interval served ratio is back above Gamma
    (and stays there for `hold` intervals). 0 if it never dropped; NaN if it never recovered."""
    if t_f >= len(str_t):
        return float("nan")
    seg = str_t[t_f:]
    if np.all(seg >= p.gamma_cov - 1e-12):
        return 0.0
    first_bad = int(np.argmax(seg < p.gamma_cov - 1e-12))
    for j in range(first_bad, len(seg) - hold + 1):
        if np.all(seg[j:j + hold] >= p.gamma_cov - 1e-12):
            return j * p.tau_s / 60.0
    return float("nan")


def summarise(name: str, outs: list[Outcome], infos: list[dict], bundle: Bundle, topo: Topology, p: Params,
              plans: bool) -> dict:
    tr = bundle.trace
    ts = np.array([o.t for o in outs])
    demand = np.array([o.demand.sum() for o in outs])
    served = np.array([o.served.sum() for o in outs])
    prot = topo.protected
    d_prot = sum(o.demand[:, prot].sum() for o in outs)
    s_prot = sum(o.served[:, prot].sum() for o in outs)
    cap = np.array([(o.cap_inst * o.n).sum() for o in outs])
    lat_v = np.concatenate([o.lat_vals for o in outs])
    lat_w = np.concatenate([o.lat_w for o in outs])
    lat_x = np.concatenate([o.lat_viol for o in outs])
    n = np.array([o.n for o in outs])
    str_t = served / np.maximum(demand, 1e-9)
    res = {
        "method": name,
        "scenario": tr.scenario,
        "STR": served.sum() / demand.sum(),                                         # Eq. (40)
        "STR_protected": s_prot / d_prot,
        "OPF": cap.sum() / demand.sum(),                                            # Eq. (41)
        "lat_mean_ms": float((lat_v * lat_w).sum() / max(lat_w.sum(), 1e-9)),       # Eq. (42)
        "lat_p95_ms": weighted_quantile(lat_v, lat_w, 0.95),
        "lat_p99_ms": weighted_quantile(lat_v, lat_w, 0.99),
        "lat_violation_rate": float((lat_x * lat_w).sum() / max(lat_w.sum(), 1e-9)),
        "RCF": float(np.abs(np.diff(n, axis=0)).sum(axis=1).mean()) if len(n) > 1 else 0.0,  # Eq. (43)
        "lost_Gbps_h": float(sum(o.lost for o in outs) * p.tau_s / 3600),
        "cost_total": float(sum(o.cost["total"] for o in outs)),
        **{f"cost_{k}": float(sum(o.cost[k] for o in outs)) for k in ("dep", "tr", "loss", "lat", "scale", "mig")},
        "decision_ms_mean": 1000 * float(np.mean([i.get("decision_s", 0.0) for i in infos])),
        "decision_ms_p95": 1000 * float(np.quantile([i.get("decision_s", 0.0) for i in infos], 0.95)),
    }
    if plans:                                                                       # Eq. (44)
        d_true = tr.demand[ts]
        res["PICP"] = float(np.mean(d_true <= bundle.D_hat[ts]))
    else:
        res["PICP"] = float("nan")
    t_f = next((e["t_start"] for e in tr.events if e["t_start"] >= tr.eval_start), None)
    res["T_rec_min"] = recovery_minutes(str_t, t_f - tr.eval_start, p) if t_f is not None else float("nan")
    outcomes = [i.get("outcome") for i in infos if "outcome" in i]
    for key in ("accepted", "repaired", "projected", "fallback_prev", "fallback_vfatd", "infeasible", "ungated"):
        res[f"gate_{key}"] = outcomes.count(key) / len(infos) if outcomes else float("nan")
    return res


def timeline(outs: list[Outcome], infos: list[dict], bundle: Bundle, topo: Topology) -> dict:
    """Per-interval series for plots and the web replay."""
    ts = [o.t for o in outs]
    return {
        "t": ts,
        "hour": [float(bundle.trace.hours[t]) for t in ts],
        "demand": [float(o.demand.sum()) for o in outs],
        "demand_ingress": [o.demand.sum(axis=1).round(2).tolist() for o in outs],
        "planned": [float(bundle.D_hat[t].sum()) for t in ts],
        "served": [float(o.served.sum()) for o in outs],
        "lost": [float(o.lost) for o in outs],
        "capacity": [float((o.cap_inst * o.n).sum()) for o in outs],
        "n": [o.n.tolist() for o in outs],
        "site_load": [o.y.round(2).tolist() for o in outs],
        "psi": [o.psi.round(3).tolist() for o in outs],
        "link_util": [np.round(o.link_util, 3).tolist() for o in outs],
        "flows_is": [np.einsum("ik,iks->is", o.demand, o.x).round(2).tolist() for o in outs],
        "cost": [round(o.cost["total"], 3) for o in outs],
        "outcome": [i.get("outcome", "-") for i in infos],
        "decision_ms": [round(1000 * i.get("decision_s", 0.0), 2) for i in infos],
    }
