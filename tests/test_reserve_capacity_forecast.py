import numpy as np

from puvfatd.capacity import CapacityEstimator
from puvfatd.forecasting import SeasonalRatioForecaster
from puvfatd.reserve import calibrate_q0, plan_series


def test_q0_is_the_upper_quantile_of_normalised_errors(p):
    rng = np.random.default_rng(0)
    mu = np.full((20000, 2, 3), 100.0)
    sigma = np.full_like(mu, 5.0)
    d = mu + sigma * rng.normal(size=mu.shape)
    q0 = calibrate_q0(d, mu, sigma, alpha=0.05)
    assert np.allclose(q0, 1.645, atol=0.05)          # standard-normal 95% quantile


def test_reserve_multiplier_rises_after_misses_and_falls_otherwise(p):
    T = 50
    mu = np.full((T, 1, 1), 100.0)
    sigma = np.full_like(mu, 5.0)
    high = (200.0 + 10.0 * np.arange(T))[:, None, None]   # rising above every previous plan -> misses
    _, q_up = plan_series(high, mu, sigma, np.array([1.0]), p)
    low = np.full_like(mu, 50.0)                      # never above the plan
    _, q_down = plan_series(low, mu, sigma, np.array([1.0]), p)
    assert q_up[-1, 0] > 1.0 > q_down[-1, 0]
    assert np.isclose(q_up[2, 0] - q_up[1, 0], p.eta_q * (1 - p.alpha))     # Eq. (7) step


def test_planning_demand_never_below_last_measurement(p):
    T = 10
    mu = np.full((T, 1, 1), 80.0)
    sigma = np.full_like(mu, 1.0)
    d = np.full_like(mu, 120.0)
    D, _ = plan_series(d, mu, sigma, np.array([1.0]), p)
    assert np.all(D[1:] >= 120.0)                      # outer max of Eq. (8)


def test_capacity_estimate_converges_to_measured_throughput(topo, p):
    est = CapacityEstimator(topo, p)
    mix = np.tile(topo.share, (topo.S, 1))
    est.update(np.zeros(topo.S, int), np.zeros(topo.S), np.zeros(topo.S), mix, stale=False)
    true_c = 30.0
    n = np.full(topo.S, 2)
    psi = np.full(topo.S, 0.6)
    for _ in range(40):
        C = est.update(n, n * true_c * psi, psi, mix, stale=False)   # Eqs. (2)-(3)
    assert np.allclose(C, true_c, atol=0.05)


def test_stale_telemetry_never_raises_capacity(topo, p):
    est = CapacityEstimator(topo, p)
    mix = np.tile(topo.share, (topo.S, 1))
    est.update(np.zeros(topo.S, int), np.zeros(topo.S), np.zeros(topo.S), mix, stale=False)
    before = est.C_hat.copy()
    after = est.update(np.full(topo.S, 2), np.full(topo.S, 1e4), np.full(topo.S, 0.5), mix, stale=True)
    assert np.all(after <= before + 1e-12)


def test_forecast_uses_only_past_observations(p):
    rng = np.random.default_rng(1)
    obs = 50 + 10 * rng.random((3 * p.season, 2, 2))
    fc = SeasonalRatioForecaster().estimate_spread([obs], p.season)
    mu1, _ = fc.predict(obs, np.zeros(len(obs)), p.season)
    changed = obs.copy()
    t = 2 * p.season + 17
    changed[t:] *= 3.0                                  # change the present and the future only
    mu2, _ = fc.predict(changed, np.zeros(len(obs)), p.season)
    assert np.allclose(mu1[: t + 1], mu2[: t + 1])     # forecasts up to t are unchanged


def test_seasonal_ratio_is_exact_on_a_clean_daily_pattern(p):
    h = np.arange(3 * p.season) % p.season
    obs = (100 + 40 * np.sin(2 * np.pi * h / p.season))[:, None, None].repeat(2, 1).repeat(2, 2)
    fc = SeasonalRatioForecaster().estimate_spread([obs], p.season)
    mu, _ = fc.predict(obs, np.zeros(len(obs)), p.season)
    err = np.abs(mu[2 * p.season:] - obs[2 * p.season:]) / obs[2 * p.season:]
    assert err.max() < 0.002
