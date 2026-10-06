"""Parameters of PU-vFATD.

The orchestration values are the starting values of Table 2 in the methodology document; the cost
coefficients are scenario choices for this simulation (they are not taken from the base paper and
should be varied in a sensitivity study). Units: traffic in Gbps, latency in ms, cost per 5-minute
interval.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Params:
    # --- orchestration (Table 2) ---
    tau_s: float = 300.0       # tau: orchestration interval (s)
    window: int = 12           # W: forecast input window (intervals)
    season: int = 288          # intervals per day, used as the seasonal lag
    alpha: float = 0.05        # tolerated under-forecast probability
    eta_q: float = 0.02        # reserve step size, Eq. (7)
    beta: float = 0.7          # capacity smoothing factor, Eq. (3)
    psi_max: float = 1.0       # Eq. (2): capacity is extrapolated to full CPU; headroom comes from rho_fw
    psi_min: float = 0.15      # Eq. (3): below this utilisation the extrapolation is not trusted
    rho_fw: float = 0.85       # firewall utilisation limit, Eq. (18)
    rho_link: float = 0.80     # link utilisation limit, Eq. (19)
    gamma_cov: float = 0.999   # minimum served-traffic ratio, Eq. (22)
    t_dec_s: float = 10.0      # decision deadline T^dec
    C0: float = 50.0           # nominal throughput of one instance (base-paper value)
    n_lic: int = 40            # licence / accelerator limit N^lic

    # --- cost coefficients (per interval) ---
    c_loss: float = 1.0        # per Gbps blocked or dropped
    c_lat: float = 0.001       # per Gbps x ms
    c_tr_per_ms: float = 0.004 # transport cost per Gbps per ms of path latency
    c_mig: float = 0.01        # per Gbps of traffic moved between firewalls, Eq. (15)
    t_on_s: float = 90.0       # measured container start-up time T^on (simulation value)
    t_sync_s: float = 15.0     # rule installation / state sync time T^sync
    c_off_frac: float = 0.1    # scale-in (drain) cost as a fraction of c_dep

    # --- safety layer ---
    proj_weight: float = 0.05  # varpi in Eq. (38)

    # --- learning (Eqs. 26-36) ---
    c_int: float = 0.05        # intervention penalty in the reward
    fixed_reserve: float = 0.15  # ablation: fixed margin instead of the adaptive reserve

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    @property
    def f_scale_out(self) -> float:
        """Fraction of an interval served by the old configuration when new instances start."""
        return min(1.0, (self.t_on_s + self.t_sync_s) / self.tau_s)

    @property
    def f_reroute(self) -> float:
        """Fraction of an interval served by the old routing when only the split changes."""
        return min(1.0, self.t_sync_s / self.tau_s)
