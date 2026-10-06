# PU-vFATD — simulation implementation

**Live dashboard: https://wyattstanson.github.io/pu-vfatd/**

Group 21 (Shourya Saraf 24BDS0391, Sreevatsa Pulipati 24BCE2315, Aryansh Sinha 24BCE2312).
Implementation of the Algorithm and Methodology document for *Proactive and Uncertainty-Aware Virtual
Firewall Scaling and Placement in the Telco Cloud-Edge Continuum*, an extension of vFATD
(Kopeć-Persiński and Bęben, 2026).

## Run it

```bash
pip install -r requirements.txt
python -m pytest                      # 23 tests
python -m experiments.run_all         # 282 simulated days, about 12 minutes on 8 cores
python -m experiments.run_all --quick # about 2 minutes
```

Outputs land in `results/`: `results.md` (all tables), `summary.csv` (one row per simulated day),
`figures/*.png`, and `dashboard.html` — the interactive workspace. Open it by double-clicking; it is
one self-contained file and needs no server or internet (fonts fall back to system fonts offline).
The same page is written to `docs/index.html`, which GitHub Pages serves as the live dashboard.

## What maps to what

| Methodology | Code |
|---|---|
| Stage 1, Eqs. (1)–(3) capacity estimation | `puvfatd/capacity.py` |
| Stage 2, Eq. (4) forecasting | `puvfatd/forecasting.py` (seasonal-ratio forecaster) |
| Stage 3, Eqs. (6)–(8) reserve | `puvfatd/reserve.py` |
| Stage 4, Eqs. (9)–(22) cost and constraints | `puvfatd/model.py`, `puvfatd/milp.py` (HiGHS via SciPy) |
| Stage 6, Algorithm 1 SafetyGate, Eqs. (27), (37)–(39) | `puvfatd/safety.py` |
| Algorithm 2 online cycle, Stage 7 execution | `puvfatd/runner.py`, `puvfatd/simulator.py`, `puvfatd/controllers.py` |
| Section 11 metrics, Eqs. (40)–(45) | `puvfatd/metrics.py` |
| Baselines (static, local reactive, vFATD MILP / k-centre / GA) | `puvfatd/controllers.py`, `puvfatd/heuristics.py` |

The Equations tab of the dashboard lists all 45 equations with their status and code location.

## Scope and honest limits

- **Traffic is synthetic.** The base paper's operator trace is not available, so `traffic.py`
  generates a daily cycle on a 4-PoP, 7-site network and injects flash crowds, DDoS-like surges,
  link and site failures, telemetry delay and an IPS throughput change.
- **No learned models in this build.** The PPO controller of Stage 5 (Eqs. 23–26, 28–36) and Eq. (5)
  training are not implemented; the forecaster is statistical. The controller slot is filled by the
  extended vFATD MILP (`PU-vFATD`) and a millisecond greedy proposer (`PU-vFATD-fast`), both checked
  by the SafetyGate. A learned policy can be plugged in as another proposer.
- **No Kubernetes/SDN prototype yet**; start-up and sync times are simulation parameters
  (`config.py`), not measurements.
- The baselines are our re-implementations of the paper's methods, not the authors' code. In the
  realistic comparison they receive the same link and latency constraints ("adapted") but keep the
  paper's fixed 50 Gbps per instance and plan on the last measured demand.
- Cost coefficients are scenario choices; vary them before drawing cost conclusions.

## Headline results (3 seeds, realistic constraints, mean over 7 scenarios)

| controller | served traffic | protected served | daily cost | lost (Gbps·h) |
|---|---|---|---|---|
| PU-vFATD | 99.84% | 99.92% | 6,309 | 10.5 |
| PU-vFATD-fast | 99.86% | 99.95% | 8,428 | 9.1 |
| Local reactive | 97.90% | 97.88% | 8,384 | 132.1 |
| Static (busy hour) | 97.06% | 96.82% | 9,456 | 184.9 |
| vFATD-MILP (adapted) | 94.64% | 94.71% | 9,663 | 323.8 |

The ablation (4 stress scenarios) shows where the gain comes from: replacing the capacity estimate
with the fixed 50 Gbps drops PU-vFATD to 96.47% served; removing the SafetyGate from the fast
proposer raises loss from 15.4 to 22.3 Gbps·h on the same scenarios; removing forecasting changes
the served ratio only slightly (99.73% vs 99.77%) but breaks the reserve's coverage target (PICP
0.49 vs 0.95). Under the base paper's original assumptions every method serves above 99.5%, and the
vFATD MILP (cost 4,011) is slightly cheaper than PU-vFATD (4,148), which pays for its reserve.
