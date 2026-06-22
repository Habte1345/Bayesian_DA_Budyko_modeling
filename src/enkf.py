"""
enkf.py — State-only Ensemble Kalman Filter (EnKF / EnSRF) for a two-store
hydrological model.

Design decisions
----------------
* **State vector**: X = [S, G]  (soil store, groundwater store).
  Model parameters are calibrated externally and held fixed.
* **Update scheme**: deterministic square-root EnKF (EnSRF, Whitaker &
  Hamill 2002).  No perturbed observations are required, so the update is
  free of sampling noise from the observation perturbations that afflict
  the classic stochastic EnKF (Burgers et al. 1998).
* **Sequential scalar updates**: when both ET and Q observations are
  available at the same time step they are assimilated one at a time
  (ET first, then Q).  Sequential scalar updates preserve the
  deterministic square-root property while avoiding the matrix inversions
  that a joint update would require.
* **Multiplicative inflation**: applied to ensemble *anomalies* after
  every analysis step to counteract variance underestimation caused by
  finite ensemble size and model error.
* **Relative observation error**: final R is
      R_std_eff = max(R_abs_std, R_frac * |y_obs|)
  allowing the error floor to scale with the magnitude of the observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .model import ModelParams, two_store_model_step


# ============================================================
# CONFIG
# ============================================================

@dataclass
class EnKFConfig:
    """
    State-only EnKF / EnSRF configuration.

    Attributes
    ----------
    nens : int
        Ensemble size.  Larger ensembles reduce sampling error but cost
        more compute.  Typical range: 50–500.
    inflation : float
        Multiplicative covariance inflation factor (≥ 1.0).
        ``inflation = 1.0`` means no inflation.  Values around 1.01–1.10
        are common for hydrological applications.
    R_ET_std, R_Q_std : float
        Absolute observation-error standard deviations for ET and Q.
    R_ET_frac, R_Q_frac : float
        Fractional (relative) observation-error coefficients.  The
        effective std is ``max(R_abs_std, R_frac * |y_obs|)``.
    proc_S_std, proc_G_std : float
        Standard deviation of additive Gaussian process noise injected
        into S and G before each forecast step.  Keeps the ensemble from
        collapsing in the state dimensions.
    P_std_frac, PET_std_frac : float
        Multiplicative forcing-perturbation fractions.  Each ensemble
        member receives an independently perturbed precipitation and PET.
    """

    nens: int = 300

    # Multiplicative inflation applied to anomalies after each analysis.
    inflation: float = 1.0

    # Absolute observation-error standard deviations.
    R_ET_std: float = 20.0
    R_Q_std: float = 100.0

    # Relative (fractional) observation-error coefficients.
    R_ET_frac: float = 0.0
    R_Q_frac: float = 0.0

    # Process (state) perturbation.
    proc_S_std: float = 0.3
    proc_G_std: float = 0.7

    # Forcing perturbation (multiplicative, applied per member).
    P_std_frac: float = 0.10
    PET_std_frac: float = 0.03


# ============================================================
# DIAGNOSTIC OUTPUT
# ============================================================

class EnKFDiagnostics(NamedTuple):
    """Per-time-step filter diagnostics.

    Attributes
    ----------
    innov_ET : float or None
        Innovation (y_obs − H*x_f mean) for ET; None if no observation.
    innov_Q : float or None
        Innovation for Q; None if no observation.
    spread_S : float
        Prior ensemble standard deviation of S.
    spread_G : float
        Prior ensemble standard deviation of G.
    spread_Q : float
        Prior ensemble standard deviation of predicted Q.
    """

    innov_ET: float | None
    innov_Q: float | None
    spread_S: float
    spread_G: float
    spread_Q: float


# ============================================================
# HELPERS
# ============================================================

def _validate_state_ensemble(X: np.ndarray) -> tuple[int, int]:
    """Validate shape and finiteness of a (2, nens) state ensemble.

    Returns
    -------
    nstate, nens : int
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected X shape (2, nens), got {X.shape}")
    nstate, nens = X.shape
    if nstate != 2:
        raise ValueError("Expected X shape (2, nens) where states are [S, G].")
    if nens < 2:
        raise ValueError("EnKF requires at least two ensemble members.")
    if not np.all(np.isfinite(X)):
        raise ValueError("State ensemble contains non-finite values.")
    return nstate, nens


def _clip_state_ensemble(
    X: np.ndarray,
    Smax: float,
    Gmax: float,
) -> np.ndarray:
    """Clip all ensemble members to physically valid bounds [0, Smax] × [0, Gmax]."""
    X = np.asarray(X, dtype=float).copy()
    X[0, :] = np.clip(X[0, :], 0.0, float(Smax))
    X[1, :] = np.clip(X[1, :], 0.0, float(Gmax))
    return X


def _effective_R_std(R_abs_std: float, R_frac: float, y_obs: float) -> float:
    """Compute the effective observation-error standard deviation.

    Uses the larger of the absolute and relative floors:
        R_eff = max(R_abs_std, R_frac * |y_obs|)

    This prevents the Kalman gain from becoming unreasonably large for
    very small observations while still allowing error to scale with
    observation magnitude when appropriate.
    """
    return max(float(R_abs_std), float(R_frac) * abs(float(y_obs)))


def _apply_multiplicative_inflation(
    X: np.ndarray,
    inflation: float,
) -> np.ndarray:
    """Inflate ensemble anomalies by a multiplicative factor.

    Multiplicative inflation scales the deviation of each member from the
    ensemble mean, leaving the mean unchanged:

        X_i ← x̄ + λ (X_i − x̄)

    where λ ≥ 1 is the inflation factor.  This counteracts variance
    underestimation caused by finite ensemble size and unrepresented
    model errors (Anderson & Anderson 1999).

    Parameters
    ----------
    X : ndarray, shape (2, nens)
        Ensemble matrix (modified in-place on a copy).
    inflation : float
        Multiplicative factor λ.  Must be ≥ 1.0; values below 1.0 are
        silently clamped to 1.0 (deflation is never intentional here).

    Returns
    -------
    ndarray, shape (2, nens)
    """
    lam = max(float(inflation), 1.0)
    if lam == 1.0:
        return X.copy()
    X = np.asarray(X, dtype=float).copy()
    x_mean = np.mean(X, axis=1, keepdims=True)
    X = x_mean + lam * (X - x_mean)
    return X


# ============================================================
# FORECAST — single member
# ============================================================

def propagate_one_member(
    S_curr: float,
    G_curr: float,
    P_t: float,
    PET_t: float,
    params: ModelParams,
    Smax: float,
    Gmax: float,
    rng: np.random.Generator,
    proc_S_std: float = 0.0,
    proc_G_std: float = 0.0,
    P_std_frac: float = 0.0,
    PET_std_frac: float = 0.0,
    ET_override: float | None = None,
) -> tuple[float, float, float, float]:
    """Propagate a single ensemble member one time step forward.

    Process noise is added to the *states before the model step* so that
    the perturbed states (not the post-step outputs) are clipped first.
    This ensures Q and ET are consistent with the clipped, perturbed
    initial states rather than being computed from unclipped states.

    Parameters
    ----------
    S_curr, G_curr : float
        Current soil and groundwater stores.
    P_t, PET_t : float
        Precipitation and potential evapotranspiration at time t.
    params : ModelParams
        Calibrated model parameters (held fixed).
    Smax, Gmax : float
        Physical upper bounds for S and G.
    rng : np.random.Generator
        Per-member random number generator (pass a seeded sub-generator
        from the ensemble-level RNG for reproducibility).
    proc_S_std, proc_G_std : float
        Standard deviations of additive process noise for S and G.
    P_std_frac, PET_std_frac : float
        Multiplicative forcing-perturbation fractions.
    ET_override : float or None
        If provided, overrides the ET observation used inside the model
        step (used only by special model variants).

    Returns
    -------
    S_next, G_next, ET_t, Q_t : float
        Post-step states and fluxes, all non-negative.
    """
    S_curr = float(S_curr)
    G_curr = float(G_curr)
    P_t = max(float(P_t), 0.0)
    PET_t = max(float(PET_t), 0.0)

    # --- Process noise (applied *before* clipping and model step) ---
    if proc_S_std > 0.0:
        S_curr += rng.normal(0.0, proc_S_std)
    if proc_G_std > 0.0:
        G_curr += rng.normal(0.0, proc_G_std)

    # Clip perturbed states to physical bounds *before* the model step
    # so that Q / ET are consistent with valid initial conditions.
    S_curr = float(np.clip(S_curr, 0.0, Smax))
    G_curr = float(np.clip(G_curr, 0.0, Gmax))

    # --- Forcing perturbation ---
    if P_std_frac > 0.0:
        P_i = P_t * (1.0 + rng.normal(0.0, P_std_frac))
    else:
        P_i = P_t
    if PET_std_frac > 0.0:
        PET_i = PET_t * (1.0 + rng.normal(0.0, PET_std_frac))
    else:
        PET_i = PET_t

    P_i = max(float(P_i), 0.0)
    PET_i = max(float(PET_i), 0.0)

    # --- Model step ---
    S_next, G_next, ET_t, Q_t, *_ = two_store_model_step(
        S_curr,
        G_curr,
        P_i,
        PET_i,
        params,
        ET_override=ET_override,
    )

    S_next = float(np.clip(S_next, 0.0, Smax))
    G_next = float(np.clip(G_next, 0.0, Gmax))
    ET_t = max(float(ET_t), 0.0)
    Q_t = max(float(Q_t), 0.0)

    return S_next, G_next, ET_t, Q_t


# ============================================================
# FORECAST — full ensemble
# ============================================================

def enkf_forecast_step_states(
    X: np.ndarray,
    P_t: float,
    PET_t: float,
    params_cal: ModelParams,
    Smax: float,
    Gmax: float,
    rng: np.random.Generator,
    proc_S_std: float = 0.0,
    proc_G_std: float = 0.0,
    P_std_frac: float = 0.0,
    PET_std_frac: float = 0.0,
    ET_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate the full state ensemble one time step forward.

    Each member is propagated independently through
    :func:`propagate_one_member` with its own sub-stream from `rng`.
    Using ``rng.spawn(nens)`` gives each member a statistically
    independent stream without manual seed bookkeeping.

    Parameters
    ----------
    X : ndarray, shape (2, nens)
        Current posterior state ensemble.
    P_t, PET_t : float
        Precipitation and PET forcing at time t.
    params_cal : ModelParams
        Fixed calibrated parameters.
    Smax, Gmax : float
        Physical bounds.
    rng : np.random.Generator
        Ensemble-level RNG; spawns one child per member.
    proc_S_std, proc_G_std, P_std_frac, PET_std_frac : float
        Perturbation parameters forwarded to :func:`propagate_one_member`.
    ET_override : float or None
        Forwarded to the model step if set.

    Returns
    -------
    X_f : ndarray, shape (2, nens)
        Prior (forecast) state ensemble.
    ET_ens : ndarray, shape (nens,)
        Predicted ET for each member.
    Q_ens : ndarray, shape (nens,)
        Predicted Q for each member.
    """
    _, nens = _validate_state_ensemble(X)

    X_f = np.zeros_like(X, dtype=float)
    ET_ens = np.zeros(nens, dtype=float)
    Q_ens = np.zeros(nens, dtype=float)

    # Spawn independent sub-generators so each member's noise is
    # statistically independent and the sequence is reproducible.
    member_rngs = rng.spawn(nens)

    for i in range(nens):
        S_next, G_next, ET_i, Q_i = propagate_one_member(
            S_curr=X[0, i],
            G_curr=X[1, i],
            P_t=P_t,
            PET_t=PET_t,
            params=params_cal,
            Smax=Smax,
            Gmax=Gmax,
            rng=member_rngs[i],
            proc_S_std=proc_S_std,
            proc_G_std=proc_G_std,
            P_std_frac=P_std_frac,
            PET_std_frac=PET_std_frac,
            ET_override=ET_override,
        )
        X_f[0, i] = S_next
        X_f[1, i] = G_next
        ET_ens[i] = ET_i
        Q_ens[i] = Q_i

    return X_f, ET_ens, Q_ens


# ============================================================
# ANALYSIS — deterministic scalar EnSRF update
# ============================================================

def enkf_update_deterministic_scalar(
    X_f: np.ndarray,
    y_obs: float,
    HX: np.ndarray,
    R_std: float,
    Smax: float,
    Gmax: float,
) -> np.ndarray:
    """Deterministic scalar EnSRF analysis update (Whitaker & Hamill 2002).

    Updates only states X = [S, G]; parameters are not touched.

    The deterministic square-root update avoids the perturbed-observation
    sampling noise present in the classic stochastic EnKF by analytically
    computing the posterior anomaly reduction factor β:

        β = 1 / (1 + √(R / (P_f^HH + R)))

    where P_f^HH = Var(HX) is the observation-space prior variance.  This
    means every ensemble member receives the *same* analysis increment for
    the anomaly component, whereas their mean shifts by the usual Kalman
    gain times the scalar innovation.

    Parameters
    ----------
    X_f : ndarray, shape (2, nens)
        Prior state ensemble (not modified in place).
    y_obs : float
        Scalar observation.  If non-finite the ensemble is returned
        unchanged (missing-data handling).
    HX : ndarray, shape (nens,)
        Observation-space equivalents H(X_f^i) for each member.
    R_std : float
        Effective observation-error standard deviation (already accounts
        for any relative scaling).
    Smax, Gmax : float
        Physical upper bounds used for post-analysis clipping.

    Returns
    -------
    X_a : ndarray, shape (2, nens)
        Posterior (analysis) state ensemble, clipped to valid bounds.

    Notes
    -----
    The update degenerates gracefully when the prior variance in
    observation space is negligible (ensemble collapsed):
    if ``P_f^HH + R ≤ 1e-12`` the prior is returned unchanged.
    """
    X_f = np.asarray(X_f, dtype=float).copy()
    HX = np.asarray(HX, dtype=float).reshape(-1)

    _, nens = _validate_state_ensemble(X_f)

    if HX.size != nens:
        raise ValueError(
            f"HX size ({HX.size}) must match ensemble size ({nens})."
        )

    # Missing observation — return prior unchanged.
    if not np.isfinite(y_obs):
        return _clip_state_ensemble(X_f, Smax, Gmax)

    R_var = float(R_std) ** 2

    # Ensemble means and anomalies.
    x_mean = np.mean(X_f, axis=1)          # shape (2,)
    hx_mean = np.mean(HX)                  # scalar

    X_anom = X_f - x_mean[:, None]         # shape (2, nens)
    HX_anom = HX - hx_mean                 # shape (nens,)

    # Observation-space prior variance and total innovation variance.
    var_hx = np.dot(HX_anom, HX_anom) / (nens - 1)   # P_f^HH
    s = var_hx + R_var                                 # innovation variance

    # Degenerate case: ensemble has collapsed in observation space.
    if s <= 1e-12 or not np.isfinite(s):
        return _clip_state_ensemble(X_f, Smax, Gmax)

    # Cross-covariance Cov(X, HX) — shape (2,).
    cov_x_hx = (X_anom @ HX_anom) / (nens - 1)

    # Kalman gain — shape (2,).
    K = cov_x_hx / s

    # Mean analysis update.
    innovation = float(y_obs) - hx_mean
    x_mean_a = x_mean + K * innovation

    # Deterministic anomaly update (square-root reduction factor β).
    # β ∈ (0, 1] shrinks the anomaly to account for reduced uncertainty.
    beta = 1.0 / (1.0 + np.sqrt(max(R_var / s, 0.0)))
    X_anom_a = X_anom - beta * np.outer(K, HX_anom)

    X_a = x_mean_a[:, None] + X_anom_a
    return _clip_state_ensemble(X_a, Smax, Gmax)


# ============================================================
# FULL ANALYSIS STEP (forecast + sequential updates + inflation)
# ============================================================

def enkf_analysis_step(
    X: np.ndarray,
    P_t: float,
    PET_t: float,
    params_cal: ModelParams,
    Smax: float,
    Gmax: float,
    rng: np.random.Generator,
    cfg: EnKFConfig,
    y_ET: float | None = None,
    y_Q: float | None = None,
    ET_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, EnKFDiagnostics]:
    """Run one complete EnKF cycle: forecast → [update ET] → [update Q] → inflate.

    Observations are assimilated sequentially (ET first, then Q).
    Sequential scalar updates maintain the deterministic square-root
    property with no matrix inversions.

    Inflation is applied *after* all scalar updates so it acts on the
    final posterior spread, not an intermediate one.

    Parameters
    ----------
    X : ndarray, shape (2, nens)
        Posterior ensemble from the previous time step.
    P_t, PET_t : float
        Precipitation and PET forcing for this time step.
    params_cal : ModelParams
        Fixed calibrated model parameters.
    Smax, Gmax : float
        Physical storage bounds.
    rng : np.random.Generator
        Ensemble-level RNG (mutated in place — caller retains reference).
    cfg : EnKFConfig
        Filter configuration.
    y_ET : float or None
        ET observation.  ``None`` or ``np.nan`` → skipped.
    y_Q : float or None
        Streamflow observation.  ``None`` or ``np.nan`` → skipped.
    ET_override : float or None
        Passed through to the model step (see :func:`propagate_one_member`).

    Returns
    -------
    X_a : ndarray, shape (2, nens)
        Posterior state ensemble.
    ET_ens : ndarray, shape (nens,)
        Prior predicted ET for each member.
    Q_ens : ndarray, shape (nens,)
        Prior predicted Q for each member.
    diag : EnKFDiagnostics
        Diagnostic statistics for this step.
    """
    _validate_state_ensemble(X)

    # ---- Forecast ------------------------------------------------
    X_f, ET_ens, Q_ens = enkf_forecast_step_states(
        X=X,
        P_t=P_t,
        PET_t=PET_t,
        params_cal=params_cal,
        Smax=Smax,
        Gmax=Gmax,
        rng=rng,
        proc_S_std=cfg.proc_S_std,
        proc_G_std=cfg.proc_G_std,
        P_std_frac=cfg.P_std_frac,
        PET_std_frac=cfg.PET_std_frac,
        ET_override=ET_override,
    )

    # ---- Diagnostics (prior) -------------------------------------
    spread_S = float(np.std(X_f[0, :], ddof=1))
    spread_G = float(np.std(X_f[1, :], ddof=1))
    spread_Q = float(np.std(Q_ens, ddof=1))

    innov_ET: float | None = None
    innov_Q: float | None = None

    X_a = X_f.copy()

    # ---- Sequential scalar updates --------------------------------

    # Update with ET observation first (if available).
    et_obs = float(y_ET) if y_ET is not None else float("nan")
    if np.isfinite(et_obs):
        R_std_et = _effective_R_std(cfg.R_ET_std, cfg.R_ET_frac, et_obs)
        innov_ET = et_obs - float(np.mean(ET_ens))
        X_a = enkf_update_deterministic_scalar(
            X_f=X_a,
            y_obs=et_obs,
            HX=ET_ens,
            R_std=R_std_et,
            Smax=Smax,
            Gmax=Gmax,
        )

    # Update with Q observation second (if available).
    q_obs = float(y_Q) if y_Q is not None else float("nan")
    if np.isfinite(q_obs):
        R_std_q = _effective_R_std(cfg.R_Q_std, cfg.R_Q_frac, q_obs)
        innov_Q = q_obs - float(np.mean(Q_ens))
        X_a = enkf_update_deterministic_scalar(
            X_f=X_a,
            y_obs=q_obs,
            HX=Q_ens,
            R_std=R_std_q,
            Smax=Smax,
            Gmax=Gmax,
        )

    # ---- Multiplicative inflation (post-analysis) -----------------
    X_a = _apply_multiplicative_inflation(X_a, cfg.inflation)
    X_a = _clip_state_ensemble(X_a, Smax, Gmax)

    diag = EnKFDiagnostics(
        innov_ET=innov_ET,
        innov_Q=innov_Q,
        spread_S=spread_S,
        spread_G=spread_G,
        spread_Q=spread_Q,
    )

    return X_a, ET_ens, Q_ens, diag



