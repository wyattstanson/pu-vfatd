"""Run every comparison of Section 11 and write results/ (tables, plots, dashboard data).

    python -m experiments.run_all            # full run (3 seeds, all scenarios), ~10 min on 8 cores
    python -m experiments.run_all --quick    # 1 seed, 3 scenarios, ~2 min

All traffic is synthetic (see puvfatd/traffic.py); numbers are simulation results, not testbed
measurements.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from puvfatd.config import Params
from puvfatd.controllers import (FastProposer, GAController, KCentreController, LocalReactiveController,
                                 MilpProposer, PUController, StaticController, VfatdMilpController)
from puvfatd.forecasting import (PersistenceForecaster, SeasonalNaiveForecaster, SeasonalRatioForecaster,
                                 forecast_scores)
from puvfatd.metrics import summarise, timeline
from puvfatd.reserve import calibrate_q0, plan_series
from puvfatd.runner import Runner, make_bundle, observed_demand
from puvfatd.safety import SafetyGate
from puvfatd.topology import default_topology
from puvfatd.traffic import generate_trace

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REALISTIC_SCENARIOS = ["normal", "flash_crowd", "ddos", "link_failure", "node_failure", "telemetry_delay",
                       "security_profile"]
ORIGINAL_SCENARIOS = ["normal", "flash_crowd", "ddos"]
ABLATION_SCENARIOS = ["flash_crowd", "ddos", "security_profile", "telemetry_delay"]
BASELINES = ["Static (busy hour)", "Local reactive", "vFATD-MILP", "vFATD-kcentre", "vFATD-GA"]
PU = ["PU-vFATD", "PU-vFATD-fast"]
ABLATIONS = {
    "PU-vFATD (full)": {},
    "w/o forecasting": {"reserve": "none"},
    "fixed reserve (+15%)": {"reserve": "fixed"},
    "fixed capacity C0": {"cap": "fixed"},
    "fast proposer w/o SafetyGate": {"method": "fast-nogate"},
    "migration/drain cost x3": {"params": {"c_mig": 0.03, "c_off_frac": 0.3}},
}
REPLAY_METHODS = ["PU-vFATD", "PU-vFATD-fast", "vFATD-MILP (adapted)", "Local reactive", "Static (busy hour)"]


# ---------------------------------------------------------------------------------------------- setup
def prepare(p: Params, topo):
    """Forecaster spread and reserve calibration on traces that are never evaluated (no leakage)."""
    history = [generate_trace(topo, p, "normal", days=4, seed=s, random_event_days=True) for s in range(4)]
    fc = SeasonalRatioForecaster().estimate_spread([h.demand for h in history], p.season)
    cal = [generate_trace(topo, p, "normal", days=3, seed=100 + s) for s in range(3)]
    preds = [fc.predict(observed_demand(c), c.hours, p.season) for c in cal]
    sl = slice(2 * p.season, None)
    q0 = calibrate_q0(np.concatenate([c.demand[sl] for c in cal]), np.concatenate([m[0][sl] for m in preds]),
                      np.concatenate([m[1][sl] for m in preds]), p.alpha)
    busy = np.concatenate([h.demand for h in history])
    return fc, q0, busy


def build_controller(method: str, topo, p: Params, busy: np.ndarray, mode: str, seed: int):
    constrained = mode == "realistic"
    if method.startswith("Static"):
        return StaticController.busy_hour(topo, p, busy, mode)
    if method == "Local reactive":
        return LocalReactiveController(topo, p)
    if method.startswith("vFATD-MILP"):
        return VfatdMilpController(topo, p, constrained)
    if method.startswith("vFATD-kcentre"):
        return KCentreController(topo, p, constrained, seed)
    if method.startswith("vFATD-GA"):
        return GAController(topo, p, constrained, seed)
    if method == "PU-vFATD":
        return PUController("PU-vFATD", topo, p, MilpProposer(topo, p), SafetyGate(topo, p, seed=seed))
    if method == "PU-vFATD-fast":
        return PUController("PU-vFATD-fast", topo, p, FastProposer(topo, p), SafetyGate(topo, p, seed=seed))
    if method == "fast-nogate":
        return PUController("fast proposer w/o SafetyGate", topo, p, FastProposer(topo, p), None)
    raise ValueError(method)


_CACHE: dict = {}


def _prepared():
    """Each worker process prepares the (deterministic) forecaster and calibration once."""
    if "prep" not in _CACHE:
        p = Params()
        _CACHE["prep"] = prepare(p, default_topology(p))
    return _CACHE["prep"]


def run_episode(job: dict) -> dict:
    p = Params().with_(**job.get("params", {}))
    topo = default_topology(p)
    fc, q0, busy = _prepared()
    tr = generate_trace(topo, p, job["scenario"], days=2, seed=job["seed"])
    b = make_bundle(tr, fc, q0, p, job.get("reserve", "adaptive"))
    ctrl = build_controller(job["method"], topo, p, busy, job["mode"], job["seed"])
    r = Runner(topo, p, b, job["mode"], cap_mode=job.get("cap", "estimated"), seed=job["seed"])
    outs, infos = [], []
    t0 = time.perf_counter()
    while not r.done:
        ctx = r.context()
        n, x, info = ctrl.decide(ctx)
        outs.append(r.execute(n, x))
        infos.append(info)
    row = summarise(job.get("label", ctrl.name), outs, infos, b, topo, p, ctrl.plans)
    row.update(comparison=job["comparison"], seed=job["seed"], wall_s=time.perf_counter() - t0,
               events="; ".join(f"{e['kind']}@{tr.hours[e['t_start']]:.1f}h"
                                f"({e.get('ingress') or e.get('link') or e.get('site') or e.get('cls', '')})"
                                for e in tr.events))
    res = {"row": row}
    if job.get("keep_timeline"):
        res["timeline"] = timeline(outs, infos, b, topo)
        res["events"] = [{**e, "t_start": e["t_start"] - tr.eval_start, "t_end": e["t_end"] - tr.eval_start}
                         for e in tr.events]
    return res


def forecast_table(p, topo, fc, q0, seeds):
    rows = []
    hist = [generate_trace(topo, p, "normal", days=4, seed=k, random_event_days=True).demand for k in range(4)]
    baselines = [PersistenceForecaster().estimate_spread(hist, p.season),
                 SeasonalNaiveForecaster().estimate_spread(hist, p.season)]
    for scen in REALISTIC_SCENARIOS:
        for s in seeds:
            tr = generate_trace(topo, p, scen, days=2, seed=s)
            obs = observed_demand(tr)
            ev = slice(tr.eval_start, None)
            for f in baselines + [fc]:
                mu, sig = f.predict(obs, tr.hours, p.season)
                D, _ = plan_series(obs, mu, sig, q0, p, "adaptive")
                rows.append({"scenario": scen, "seed": s, "forecaster": f.name,
                             **forecast_scores(tr.demand[ev], mu[ev], sig[ev], D[ev])})
    return rows


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    seeds = [11] if args.quick else [11, 12, 13]
    real_s = ["normal", "flash_crowd", "ddos"] if args.quick else REALISTIC_SCENARIOS
    orig_s = ["normal"] if args.quick else ORIGINAL_SCENARIOS
    abl_s = ["flash_crowd"] if args.quick else ABLATION_SCENARIOS

    p = Params()
    topo = default_topology(p)
    t_start = time.time()
    fc, q0, busy = _prepared()
    print(f"reserve multipliers q0 per ingress: {np.round(q0, 3)}")
    common: dict = {}

    jobs = []
    for scen in real_s:
        for s in seeds:
            for m in BASELINES + PU:
                name = m if m in PU or m.startswith(("Static", "Local")) else m + " (adapted)"
                jobs.append({**common, "comparison": "realistic", "mode": "realistic", "scenario": scen,
                             "seed": s, "method": m, "label": name,
                             "keep_timeline": s == seeds[0] and name in REPLAY_METHODS})
    for scen in orig_s:
        for s in seeds:
            for m in BASELINES + PU:
                jobs.append({**common, "comparison": "original", "mode": "original", "scenario": scen,
                             "seed": s, "method": m, "label": m})
    for scen in abl_s:
        for s in seeds:
            for label, spec in ABLATIONS.items():
                jobs.append({**common, "comparison": "ablation", "mode": "realistic", "scenario": scen, "seed": s,
                             "method": spec.get("method", "PU-vFATD"), "label": label,
                             "reserve": spec.get("reserve", "adaptive"), "cap": spec.get("cap", "estimated"),
                             "params": spec.get("params", {})})
    print(f"running {len(jobs)} simulated days on {args.workers} workers ...")
    rows, replays = [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_episode, j): j for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            res = f.result()
            row = res["row"]
            rows.append(row)
            if "timeline" in res:
                replays.setdefault(row["scenario"], {})[row["method"]] = {"timeline": res["timeline"],
                                                                         "events": res["events"]}
            if k % 20 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} done ({time.time() - t_start:.0f}s)")

    RESULTS.mkdir(exist_ok=True)
    fc_rows = forecast_table(p, topo, fc, q0, seeds)
    with open(RESULTS / "raw_results.json", "w") as fh:
        json.dump({"rows": rows, "forecast": fc_rows, "q0": q0.tolist(), "seeds": seeds,
                   "quick": args.quick}, fh)
    with open(RESULTS / "replays.json", "w") as fh:
        json.dump(replays, fh)
    from experiments.report import build_report
    build_report(RESULTS)
    print(f"done in {time.time() - t_start:.0f}s -> {RESULTS}")


if __name__ == "__main__":
    main()
