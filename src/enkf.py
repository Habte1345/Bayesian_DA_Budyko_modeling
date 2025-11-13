# src/enkf.py
from dataclasses import dataclass, field
import numpy as np

from .model import ModelParams, two_store_model_step


# ---------------------------------------------------------
# EnKF Configuration
# ---------------------------------------------------------
@dataclass
class EnKFConfig:
    nens: int = 5
    # State vector structure: [S, G, Kperc, Kb, Ke, Cqq]
    state_dim: int = 6

    # Parameter random-walk SD (kept for compatibility)
    param_rw_sd: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )

    R_Q: float = 100   # Observation error covariance for Q (if used)
    R_ET: float = 20   # Observation error covariance for ET
    inflation: float = 1.05


# ---------------------------------------------------------
# EnKF Update Step
# ---------------------------------------------------------
def enkf_update(
    X: np.ndarray,
    y_obs: float,
    HX: np.ndarray,
    R: float,
    inflation: float,
    Smax: float,
    Gmax: float,
) -> np.ndarray:
    """
    EnKF analysis step.
    Smax, Gmax, and inflation are explicitly provided from run_simulation.py.
    """
    nens = X.shape[1]

    # 1. Inflation
    X_mean = np.mean(X, axis=1, keepdims=True)
    X = X_mean + inflation * (X - X_mean)

    # 2. Anomalies
    X_ano = X - np.mean(X, axis=1, keepdims=True)
    Y_ano = (HX - np.mean(HX)).reshape(1, -1)

    # Covariances
    Pxy = (X_ano @ Y_ano.T) / (nens - 1)
    Pyy = (Y_ano @ Y_ano.T) / (nens - 1) + R

    # Kalman Gain
    K = Pxy / (Pyy[0, 0] + 1e-8)

    # 3. Update
    innovation = y_obs - HX
    X_updated = X + K @ innovation.reshape(1, -1)

    # 4. Clamping using passed Smax, Gmax
    X_updated[0, :] = np.clip(X_updated[0, :], 0.0, Smax)
    X_updated[1, :] = np.clip(X_updated[1, :], 0.0, Gmax)

    return X_updated


# ---------------------------------------------------------
# EnKF Forecast Step
# ---------------------------------------------------------
def enkf_forecast_step(
    X_ens: np.ndarray,
    P_t: float,
    PET_t: float,
    params_cal: ModelParams,
    ET_B_t=None,
) -> tuple:
    """
    Forecast step using calibrated (fixed) parameters.

    Parameters in the ensemble are overwritten by the calibrated values,
    keeping the state as the only evolving component.
    """
    state_dim, nens = X_ens.shape
    X_next = np.copy(X_ens)

    Q_ens = np.zeros(nens)
    ET_ens = np.zeros(nens)

    Smax_cal = params_cal.Smax
    Gmax_cal = Smax_cal * 3

    for i in range(nens):

        # Extract current state (only S, G are used)
        S_curr, G_curr, _, _, _, _ = X_ens[:, i]

        params = params_cal
        ET_override = ET_B_t

        try:
            S_next, G_next, ET_t, Q_t_raw, _, _, _ = two_store_model_step(
                S_curr, G_curr, P_t, PET_t, params, ET_override=ET_override
            )
        except Exception:
            S_next, G_next, ET_t, Q_t_raw = S_curr, G_curr, 0.0, 0.0

        Q_t = max(Q_t_raw, 0.0)

        # Clamp states
        X_next[0, i] = np.clip(S_next, 0.0, Smax_cal)
        X_next[1, i] = np.clip(G_next, 0.0, Gmax_cal)

        # Reset parameters to calibrated values
        X_next[2, i] = params_cal.Kperc
        X_next[3, i] = params_cal.Kb
        X_next[4, i] = params_cal.Ke
        X_next[5, i] = params_cal.Cqq

        Q_ens[i] = Q_t
        ET_ens[i] = ET_t

    return X_next, Q_ens, ET_ens
