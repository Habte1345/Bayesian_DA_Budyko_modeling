"""
enkf_engine.py  –  Ensemble Kalman Filter for the Two-Store Bucket Model
=========================================================================
State vector : [S (soil moisture, mm),  G (groundwater, mm)]
Observable   : Q (total streamflow, mm/month)

All functions are basin-agnostic; the caller loops over basins.
"""

import numpy as np
from model import ModelParams, two_store_model_step


# ─────────────────────────────────────────────────────────────────────────────
# Internal wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _transition(state: np.ndarray, P_t: float, PET_t: float,
                params: ModelParams) -> np.ndarray:
    """state [S, G]  →  [S_next, G_next]"""
    S, G = float(state[0]), float(state[1])
    S_n, G_n, *_ = two_store_model_step(S, G, P_t, PET_t, params)
    return np.array([S_n, G_n])


def _obs(state: np.ndarray, P_t: float, PET_t: float,
         params: ModelParams) -> np.ndarray:
    """state [S, G]  →  [Q]  (runs a full model step)"""
    S, G = float(state[0]), float(state[1])
    _, _, _, Q, *_ = two_store_model_step(S, G, P_t, PET_t, params)
    return np.array([Q])


def _clip(ensemble: np.ndarray, Smax: float) -> np.ndarray:
    """Enforce physical bounds: S ∈ [0, 1.5·Smax],  G ≥ 0"""
    out = ensemble.copy()
    out[:, 0] = np.clip(out[:, 0], 0.0, 1.5 * Smax)
    out[:, 1] = np.clip(out[:, 1], 0.0, None)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Single time-step update
# ─────────────────────────────────────────────────────────────────────────────

def enkf_update(
    ensemble: np.ndarray,
    obs_Q: float,
    R: np.ndarray,
    P_t: float,
    PET_t: float,
    params: ModelParams,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One EnKF analysis step.

    Parameters
    ----------
    ensemble : (N, 2)   prior ensemble [S, G]
    obs_Q    : scalar   observed streamflow (mm/month); NaN → skip analysis
    R        : (1, 1)   observation error covariance
    P_t      : scalar   precipitation this month (mm/month)
    PET_t    : scalar   potential ET this month  (mm/month)
    params   : ModelParams

    Returns
    -------
    updated  : (N, 2)   posterior ensemble
    H_ens    : (N, 1)   model-predicted Q for each member (before update)
    """
    N = ensemble.shape[0]

    # ── Forecast ──────────────────────────────────────────────────────────
    forecast = np.array([_transition(ensemble[i], P_t, PET_t, params) for i in range(N)])
    H_ens    = np.array([_obs(forecast[i],        P_t, PET_t, params) for i in range(N)])

    # ── If observation is missing, return open-loop forecast ─────────────
    if not np.isfinite(obs_Q):
        updated = _clip(forecast, params.Smax)
        return updated, H_ens

    observation = np.array([obs_Q])

    # ── Empirical covariances ─────────────────────────────────────────────
    dx = forecast - forecast.mean(axis=0)   # (N, 2)
    dy = H_ens    - H_ens.mean(axis=0)      # (N, 1)

    P_xy = dx.T @ dy / (N - 1)             # (2, 1)
    P_yy = (dy.T @ dy / (N - 1)).reshape(1, 1)

    # ── Kalman gain  K = P_xy · (P_yy + R)^{-1} ─────────────────────────
    K = P_xy @ np.linalg.inv(P_yy + R)     # (2, 1)

    # ── Analysis ──────────────────────────────────────────────────────────
    updated = np.empty_like(forecast)
    for i in range(N):
        innovation = observation - H_ens[i]
        updated[i] = forecast[i] + (K @ innovation).ravel()

    updated = _clip(updated, params.Smax)
    return updated, H_ens


# ─────────────────────────────────────────────────────────────────────────────
# Full time-series loop  (one basin)
# ─────────────────────────────────────────────────────────────────────────────

def enkf_loop(
    initial_ensemble: np.ndarray,
    obs_Q_series: np.ndarray,
    P_series: np.ndarray,
    PET_series: np.ndarray,
    params: ModelParams,
    obs_noise_std: float,
    inflation_noise_std: float,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Run EnKF over all T time steps for one basin.

    Parameters
    ----------
    initial_ensemble    : (N, 2)
    obs_Q_series        : (T,)   observed Q, NaN = missing
    P_series            : (T,)   precipitation  (mm/month)
    PET_series          : (T,)   potential ET   (mm/month)
    params              : ModelParams
    obs_noise_std       : scalar  σ of observation error (mm/month)
    inflation_noise_std : scalar  σ of additive covariance inflation

    Returns
    -------
    dict with keys:
        ensemble_history  (T, N, 2)   full posterior ensemble
        post_S_mean       (T,)
        post_S_ci         (T, 2)      [2.5%, 97.5%]
        post_G_mean       (T,)
        post_G_ci         (T, 2)
        post_Q_mean       (T,)
        post_Q_ci         (T, 2)
        openloop_Q        (T,)        open-loop (no assimilation) streamflow
        openloop_S        (T,)
        openloop_G        (T,)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    T = len(obs_Q_series)
    N = initial_ensemble.shape[0]
    R = np.array([[obs_noise_std ** 2]])

    ensemble = initial_ensemble.copy()
    ensemble = _clip(ensemble, params.Smax)

    # storage
    ens_hist = np.empty((T, N, 2))
    post_Q   = np.empty((T, N))

    # open-loop run (same initial state, no assimilation)
    ol_S = float(initial_ensemble[:, 0].mean())
    ol_G = float(initial_ensemble[:, 1].mean())
    ol_S_arr, ol_G_arr, ol_Q_arr = np.empty(T), np.empty(T), np.empty(T)

    for t in range(T):
        P_t   = float(P_series[t])
        PET_t = float(PET_series[t])

        # ── EnKF update ───────────────────────────────────────────────────
        ensemble, H_ens = enkf_update(
            ensemble = ensemble,
            obs_Q    = float(obs_Q_series[t]),
            R        = R,
            P_t      = P_t,
            PET_t    = PET_t,
            params   = params,
        )

        # Covariance inflation
        noise = rng.normal(0, inflation_noise_std, size=ensemble.shape)
        ensemble = _clip(ensemble + noise, params.Smax)

        ens_hist[t] = ensemble
        post_Q[t]   = H_ens.ravel()    # model-simulated Q before analysis

        # ── Open-loop step ────────────────────────────────────────────────
        ol_S, ol_G, _, ol_q, *_ = two_store_model_step(ol_S, ol_G, P_t, PET_t, params)
        ol_S_arr[t] = ol_S
        ol_G_arr[t] = ol_G
        ol_Q_arr[t] = ol_q

    # ── Summarise ─────────────────────────────────────────────────────────
    def mean_ci(arr2d):
        return arr2d.mean(axis=1), np.percentile(arr2d, [2.5, 97.5], axis=1).T

    S_ens = ens_hist[:, :, 0]
    G_ens = ens_hist[:, :, 1]

    post_S_mean, post_S_ci = mean_ci(S_ens)
    post_G_mean, post_G_ci = mean_ci(G_ens)
    post_Q_mean, post_Q_ci = mean_ci(post_Q)

    return dict(
        ensemble_history = ens_hist,
        post_S_mean      = post_S_mean,
        post_S_ci        = post_S_ci,
        post_G_mean      = post_G_mean,
        post_G_ci        = post_G_ci,
        post_Q_mean      = post_Q_mean,
        post_Q_ci        = post_Q_ci,
        openloop_Q       = ol_Q_arr,
        openloop_S       = ol_S_arr,
        openloop_G       = ol_G_arr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
    """
    Compute NSE, KGE, RMSE, and bias for paired obs/sim arrays.
    NaN timesteps are excluded.
    """
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    if len(o) == 0:
        return dict(NSE=np.nan, KGE=np.nan, RMSE=np.nan, Bias=np.nan, n=0)

    # NSE
    nse = 1.0 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2)

    # KGE
    r    = np.corrcoef(o, s)[0, 1]
    alpha = s.std() / o.std() if o.std() > 0 else np.nan
    beta  = s.mean() / o.mean() if o.mean() > 0 else np.nan
    kge   = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    rmse = float(np.sqrt(np.mean((o - s) ** 2)))
    bias = float((s.mean() - o.mean()) / o.mean() * 100) if o.mean() != 0 else np.nan

    return dict(NSE=round(float(nse), 4),
                KGE=round(float(kge), 4),
                RMSE=round(rmse, 4),
                Bias_pct=round(bias, 2),
                n=int(mask.sum()))
