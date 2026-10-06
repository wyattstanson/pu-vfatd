"""Turn results/raw_results.json into tables (results.md, summary.csv), figures and the dashboard."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from puvfatd.config import Params  # noqa: E402
from puvfatd.topology import default_topology  # noqa: E402

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
DONE, SCORE, NOT_BUILT = "implemented", "used as a score", "not in this build"
EQ_META = {  # equation key -> (title, status, code reference)
    "prior": ("Capacity prior", DONE, "capacity.CapacityEstimator.prior"),
    "meas": ("Measured per-instance capacity", DONE, "capacity.CapacityEstimator.update"),
    "cap": ("Smoothed capacity estimate", DONE, "capacity.CapacityEstimator.update"),
    "fc": ("Forecast mean and spread", DONE, "forecasting.SeasonalRatioForecaster.predict (statistical, not trained)"),
    "nll": ("Gaussian negative log-likelihood", SCORE, "forecasting.forecast_scores (evaluation only)"),
    "resid": ("Normalised forecast error", DONE, "reserve.normalised_errors, reserve.calibrate_q0"),
    "qupd": ("Online reserve multiplier", DONE, "reserve.plan_series"),
    "plan": ("Planning demand", DONE, "reserve.plan_series"),
    "obj": ("Total cost", DONE, "model.costs, milp._solve"),
    "jdep": ("Deployment cost", DONE, "model.costs, milp._solve"),
    "jtr": ("Transport cost", DONE, "model.costs, milp._solve"),
    "jloss": ("Blocked-traffic penalty", DONE, "model.costs, milp._solve"),
    "jlat": ("Latency cost", DONE, "model.costs, milp._solve"),
    "jrec": ("Reconfiguration cost", DONE, "model.costs, milp._solve"),
    "jmig": ("Migration cost", DONE, "model.costs, milp._solve"),
    "con": ("Scale-out coefficient from timings", DONE, "model.c_on"),
    "c_res": ("CPU and memory limit", DONE, "model.violations, milp._solve"),
    "c_cap": ("Firewall capacity", DONE, "model.violations, milp._solve"),
    "c_link": ("Link capacity", DONE, "model.violations, milp._solve"),
    "c_lat": ("Latency budget", DONE, "model.effective_mask"),
    "c_asg": ("Assignment of protected classes", DONE, "model.violations, milp._solve"),
    "c_cov": ("Coverage, licence and fractions", DONE, "model.violations, milp._solve"),
    "state": ("Controller state", NOT_BUILT, "learned controller disabled; controllers.FastProposer fills the slot"),
    "act_n": ("Scaling action", NOT_BUILT, "learned controller disabled"),
    "act_x": ("Masked softmax routing", NOT_BUILT, "learned controller disabled"),
    "rew": ("Reward", NOT_BUILT, "learned controller disabled"),
    "viol": ("Normalised violation", DONE, "model.violations"),
    "cmdp_j": ("Expected reward and violation", NOT_BUILT, "learned controller disabled"),
    "cmdp": ("Constrained training problem", NOT_BUILT, "learned controller disabled"),
    "lag": ("Lagrangian", NOT_BUILT, "learned controller disabled"),
    "shaped": ("Shaped reward", NOT_BUILT, "learned controller disabled"),
    "gae": ("Advantage estimation", NOT_BUILT, "learned controller disabled"),
    "ratio": ("PPO probability ratio", NOT_BUILT, "learned controller disabled"),
    "clip": ("PPO clipped objective", NOT_BUILT, "learned controller disabled"),
    "loss": ("Actor-critic loss", NOT_BUILT, "learned controller disabled"),
    "dual": ("Dual update", NOT_BUILT, "learned controller disabled"),
    "feas": ("Feasibility test", DONE, "model.feasible"),
    "proj": ("Projection onto the feasible set", DONE, "milp.solve_projection"),
    "exec": ("Executed-action rule", DONE, "safety.SafetyGate.__call__"),
    "str": ("Served-traffic ratio", DONE, "metrics.summarise"),
    "opf": ("Over-provisioning factor", DONE, "metrics.summarise"),
    "lat": ("Mean latency", DONE, "metrics.summarise"),
    "rcf": ("Reconfiguration frequency", DONE, "metrics.summarise"),
    "picp": ("Reserve coverage", DONE, "metrics.summarise"),
    "trec": ("Recovery time", DONE, "metrics.recovery_minutes"),
}
STAGE_OF = [(1, 3, "Stage 1 · Capacity estimation"), (4, 5, "Stage 2 · Forecasting"), (6, 8, "Stage 3 · Reserve"),
            (9, 22, "Stage 4 · Optimisation model"), (23, 36, "Stage 5 · Learned controller"),
            (37, 39, "Stage 6 · Safety gate"), (40, 45, "Evaluation metrics")]


def equations_for_web() -> list[dict]:
    from latex2mathml.converter import convert
    src = json.loads((Path(__file__).resolve().parents[1] / "web" / "equations_src.json").read_text(encoding="utf-8"))
    out = []
    for e in src:
        title, status, code = EQ_META[e["key"]]
        stage = next(name for a, b, name in STAGE_OF if a <= e["n"] <= b)
        out.append({"n": e["n"], "title": title, "status": status, "code": code, "stage": stage,
                    "mathml": convert(e["latex"], display="block").replace("<mover>", '<mover accent="true">')})
    return out
METRICS = ["STR", "STR_protected", "cost_total", "OPF", "lat_p95_ms", "lat_violation_rate", "RCF", "lost_Gbps_h",
           "T_rec_min", "decision_ms_mean", "PICP", "gate_accepted", "gate_repaired", "gate_projected",
           "gate_fallback_prev", "gate_fallback_vfatd", "gate_infeasible"]
ORDER = ["Static (busy hour)", "Local reactive", "vFATD-MILP", "vFATD-MILP (adapted)", "vFATD-kcentre",
         "vFATD-kcentre (adapted)", "vFATD-GA", "vFATD-GA (adapted)", "PU-vFATD-fast", "PU-vFATD"]
COLORS = {"PU-vFATD": "#1f6f43", "PU-vFATD-fast": "#5aa469", "vFATD-MILP (adapted)": "#3b6ea8",
          "vFATD-MILP": "#3b6ea8", "vFATD-GA (adapted)": "#7a5aa6", "vFATD-GA": "#7a5aa6",
          "vFATD-kcentre (adapted)": "#a0702a", "vFATD-kcentre": "#a0702a", "Local reactive": "#b5483b",
          "Static (busy hour)": "#7d7d7d"}


def _ci(vals):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if len(v) == 0:
        return float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), 0.0
    return float(v.mean()), float(T975.get(len(v) - 1, 1.96) * v.std(ddof=1) / np.sqrt(len(v)))


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["comparison"], r["scenario"], r["method"])].append(r)
    agg = {}
    for key, rs in groups.items():
        agg[key] = {m: _ci([r.get(m) for r in rs]) for m in METRICS}
        agg[key]["n"] = len(rs)
    return agg


def overall(agg, comparison):
    """Mean over scenarios of the per-scenario means (each scenario weighted equally)."""
    by_m = defaultdict(lambda: defaultdict(list))
    for (c, s, m), v in agg.items():
        if c == comparison:
            for k in METRICS:
                by_m[m][k].append(v[k][0])
    return {m: {k: float(np.nanmean(v)) if np.isfinite(v).any() else float("nan") for k, v in d.items()}
            for m, d in by_m.items()}


def _sorted(methods):
    return sorted(methods, key=lambda m: ORDER.index(m) if m in ORDER else 100 + len(m))


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def figures(res_dir: Path, agg, replays):
    fig_dir = res_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    # 1. flash-crowd timeline
    sc = "flash_crowd" if "flash_crowd" in replays else next(iter(replays))
    rep = replays[sc]
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    any_m = next(iter(rep.values()))["timeline"]
    hrs = np.array(any_m["hour"])
    ax[0].plot(hrs, any_m["demand"], color="black", lw=1.4, label="offered demand")
    for m in ("PU-vFATD", "vFATD-MILP (adapted)", "Local reactive"):
        if m in rep:
            tl = rep[m]["timeline"]
            ax[0].plot(hrs, tl["capacity"], lw=1.1, color=COLORS[m], label=f"capacity: {m}")
            ax[1].plot(hrs, tl["lost"], lw=1.1, color=COLORS[m], label=m)
    for e in next(iter(rep.values()))["events"]:
        for a in ax:
            a.axvspan(hrs[max(0, e["t_start"])], hrs[min(len(hrs) - 1, e["t_end"] - 1)], color="#f2c94c", alpha=0.25)
    ax[0].set_ylabel("Gbps")
    ax[0].legend(loc="upper left", frameon=False, ncol=2)
    ax[0].set_title(f"Scenario '{sc}' (seed {11}): deployed inspection capacity vs demand")
    ax[1].set_ylabel("lost Gbps")
    ax[1].set_xlabel("hour of day")
    fig.tight_layout()
    fig.savefig(fig_dir / "timeline_flash_crowd.png", dpi=150)
    plt.close(fig)

    # 2. served ratio and cost per method, realistic comparison
    ov = overall(agg, "realistic")
    ms = _sorted(ov)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].barh(ms, [100 * ov[m]["STR"] for m in ms], color=[COLORS.get(m, "#999") for m in ms])
    ax[0].set_xlim(min(100 * ov[m]["STR"] for m in ms) - 1, 100.2)
    ax[0].set_xlabel("served-traffic ratio STR (%) - higher is better")
    ax[1].barh(ms, [ov[m]["cost_total"] for m in ms], color=[COLORS.get(m, "#999") for m in ms])
    ax[1].set_xlabel("total daily cost (incl. loss penalty) - lower is better")
    ax[1].set_yticklabels([])
    fig.suptitle("Realistic constraints, mean over scenarios")
    fig.tight_layout()
    fig.savefig(fig_dir / "str_cost_realistic.png", dpi=150)
    plt.close(fig)

    # 3. STR per scenario
    scen = sorted({s for (c, s, m) in agg if c == "realistic"})
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.8 / len(ms)
    for j, m in enumerate(ms):
        vals = [100 * agg.get(("realistic", s, m), {"STR": (np.nan, 0)})["STR"][0] for s in scen]
        ax.bar(np.arange(len(scen)) + j * w, vals, w, color=COLORS.get(m, "#999"), label=m)
    ax.set_xticks(np.arange(len(scen)) + 0.4 - w / 2)
    ax.set_xticklabels(scen, rotation=15)
    ax.set_ylim(85, 100.5)
    ax.set_ylabel("STR (%)")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "str_by_scenario.png", dpi=150)
    plt.close(fig)

    # 4. decision time
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(ms, [max(ov[m]["decision_ms_mean"], 1e-3) for m in ms], color=[COLORS.get(m, "#999") for m in ms])
    ax.set_xscale("log")
    ax.set_xlabel("mean decision time per interval (ms, log scale)")
    fig.tight_layout()
    fig.savefig(fig_dir / "decision_time.png", dpi=150)
    plt.close(fig)


def build_report(res_dir: Path) -> None:
    raw = json.loads((res_dir / "raw_results.json").read_text())
    replays = json.loads((res_dir / "replays.json").read_text())
    rows = raw["rows"]
    agg = aggregate(rows)

    with open(res_dir / "summary.csv", "w", newline="") as fh:
        keys = ["comparison", "scenario", "method", "seed"] + METRICS + ["cost_dep", "cost_tr", "cost_loss",
                                                                        "cost_lat", "cost_scale", "cost_mig",
                                                                        "events", "wall_s"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["comparison"], r["scenario"], r["method"], r["seed"])):
            w.writerow(r)

    figures(res_dir, agg, replays)
    seeds = raw["seeds"]
    lines = ["# PU-vFATD simulation results", "",
             f"Seeds per cell: {len(seeds)} ({', '.join(map(str, seeds))}). Values are means; "
             "+/- is the 95% confidence half-width over seeds. **All traffic is synthetic** (daily cycle "
             "plus injected events); these are simulation results, not testbed measurements.", ""]

    # forecasting
    fc = defaultdict(lambda: defaultdict(list))
    for r in raw["forecast"]:
        for k in ("MAE_Gbps", "MAPE_%", "NLL", "coverage", "mean_reserve_Gbps"):
            fc[r["forecaster"]][k].append(r[k])
    lines += ["## Forecasting (Stage 2-3)", "",
              "Coverage = share of intervals where the planning demand of Eq. (8) was at least the actual "
              f"demand (target 1 - alpha = {1 - Params().alpha:.2f}).", "",
              md_table(["forecaster", "MAE (Gbps)", "MAPE (%)", "NLL (Eq. 5 as score)", "coverage", "mean reserve (Gbps)"],
                       [[f, _fmt(np.mean(d["MAE_Gbps"])), _fmt(np.mean(d["MAPE_%"]), 2), _fmt(np.mean(d["NLL"])),
                         _fmt(np.mean(d["coverage"])), _fmt(np.mean(d["mean_reserve_Gbps"]), 2)] for f, d in fc.items()]), ""]

    for comp, title in (("realistic", "Realistic constraints (links, latency, variable capacity, failures)"),
                        ("original", "Original base-paper assumptions (over-provisioned network, fixed 50 Gbps)")):
        ov = overall(agg, comp)
        if not ov:
            continue
        lines += [f"## Comparison: {title}", "", "Mean over scenarios:", "",
                  md_table(["method", "STR", "STR protected", "daily cost", "OPF", "p95 latency (ms)",
                            "latency violations", "RCF", "lost (Gbps h)", "decision (ms)", "PICP"],
                           [[m, _fmt(ov[m]["STR"], 4), _fmt(ov[m]["STR_protected"], 4), _fmt(ov[m]["cost_total"], 0),
                             _fmt(ov[m]["OPF"], 2), _fmt(ov[m]["lat_p95_ms"], 2), _fmt(ov[m]["lat_violation_rate"], 4),
                             _fmt(ov[m]["RCF"], 2), _fmt(ov[m]["lost_Gbps_h"], 2), _fmt(ov[m]["decision_ms_mean"], 1),
                             _fmt(ov[m]["PICP"])] for m in _sorted(ov)]), ""]
        scen = sorted({s for (c, s, m) in agg if c == comp})
        ms = _sorted(ov)
        lines += ["Served-traffic ratio STR per scenario:", "",
                  md_table(["method"] + scen,
                           [[m] + [f"{_fmt(agg[(comp, s, m)]['STR'][0], 4)} +/- {_fmt(agg[(comp, s, m)]['STR'][1], 4)}"
                                   if (comp, s, m) in agg else "-" for s in scen] for m in ms]), "",
                  "Daily cost per scenario:", "",
                  md_table(["method"] + scen,
                           [[m] + [f"{_fmt(agg[(comp, s, m)]['cost_total'][0], 0)} +/- {_fmt(agg[(comp, s, m)]['cost_total'][1], 0)}"
                                   if (comp, s, m) in agg else "-" for s in scen] for m in ms]), "",
                  "Recovery time after the event (Eq. 45, minutes; 0 = never dropped below Gamma, - = not "
                  "recovered within the day or no event):", "",
                  md_table(["method"] + scen,
                           [[m] + [_fmt(agg[(comp, s, m)]["T_rec_min"][0], 0) if (comp, s, m) in agg else "-"
                                   for s in scen] for m in ms]), ""]

    ov = overall(agg, "ablation")
    if ov:
        lines += ["## Ablation study (realistic constraints)", "",
                  "Component ablations use the deterministic proposer so that each run changes one component only.", "",
                  md_table(["variant", "STR", "STR protected", "daily cost", "OPF", "lost (Gbps h)", "RCF", "PICP"],
                           [[m, _fmt(v["STR"], 4), _fmt(v["STR_protected"], 4), _fmt(v["cost_total"], 0),
                             _fmt(v["OPF"], 2), _fmt(v["lost_Gbps_h"], 2), _fmt(v["RCF"], 2), _fmt(v["PICP"])]
                            for m, v in ov.items()]), ""]

    ov = overall(agg, "realistic")
    lines += ["## Safety layer (share of intervals per SafetyGate outcome, realistic comparison)", "",
              md_table(["method", "accepted", "repaired", "projected", "fallback: previous", "fallback: vFATD",
                        "declared infeasible"],
                       [[m] + [_fmt(ov[m][f"gate_{k}"]) for k in ("accepted", "repaired", "projected", "fallback_prev",
                                                                  "fallback_vfatd", "infeasible")]
                        for m in _sorted(ov) if m.startswith("PU")]), "",
              "## Figures", "", "![timeline](figures/timeline_flash_crowd.png)", "",
              "![str and cost](figures/str_cost_realistic.png)", "", "![per scenario](figures/str_by_scenario.png)", "",
              "![decision time](figures/decision_time.png)", ""]
    (res_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")

    # dashboard data
    p = Params()
    topo = default_topology(p)
    data = {
        "topology": {"ingress": topo.ingress, "sites": topo.sites, "tiers": topo.tiers, "links": topo.link_names,
                     "B": topo.B.tolist(), "link_lat": topo.link_lat.tolist(), "n_max": topo.n_max.tolist(),
                     "classes": topo.classes, "t_max": topo.t_max.tolist(), "c_dep": topo.c_dep.tolist(),
                     "paths": topo.paths},
        "params": {"tau_s": p.tau_s, "rho_fw": p.rho_fw, "rho_link": p.rho_link, "gamma": p.gamma_cov, "C0": p.C0,
                   "alpha": p.alpha, "n_lic": p.n_lic},
        "overall": {c: overall(agg, c) for c in ("realistic", "original", "ablation")},
        "per_scenario": {f"{c}|{s}|{m}": {k: v[k][0] for k in METRICS} for (c, s, m), v in agg.items()},
        "replays": replays, "seeds": seeds, "q0": raw["q0"],
        "forecast": {f: {k: float(np.mean(v)) for k, v in d.items()} for f, d in fc.items()},
        "equations": equations_for_web(),
        "quick": raw.get("quick", False),
    }
    (res_dir / "dashboard_data.json").write_text(json.dumps(data, separators=(",", ":")))
    tpl = Path(__file__).resolve().parents[1] / "web" / "dashboard_template.html"
    if tpl.exists():
        page = tpl.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
        # bare page for publishing as an artifact; full document for opening from disk
        (res_dir / "dashboard_page.html").write_text(page, encoding="utf-8")
        doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
               '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
               '</head>\n<body>\n' + page + '\n</body>\n</html>\n')
        (res_dir / "dashboard.html").write_text(doc, encoding="utf-8")
        site = Path(__file__).resolve().parents[1] / "docs"   # GitHub Pages serves this folder
        site.mkdir(exist_ok=True)
        (site / "index.html").write_text(doc, encoding="utf-8")
        (site / ".nojekyll").write_text("", encoding="utf-8")


if __name__ == "__main__":
    build_report(Path(__file__).resolve().parents[1] / "results")
