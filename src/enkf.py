# src/enkf.py (No Hardcoded Constraints)
from dataclasses import dataclass, field
import numpy as np

from .model import ModelParams, two_store_model_step

# =====================================================================
# 4. ENSEMBLE KALMAN FILTER (EnKF) - CORE FUNCTIONS
# =====================================================================


@dataclass
class EnKFConfig:
    nens: int = 300
    # [S, G, Kperc, Kb, Ke, Cqq] - keep 6D for compatibility with initial ensemble setup
    state_dim: int = 6
    # Random-walk SD (not used when parameters are fixed, but kept for compatibility)
    param_rw_sd: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    R_Q: float = 100.0   # Observation error covariance for Q (if used)
    R_ET: float = 50.0   # Observation error covariance for ET
    inflation: float = 1.05


def enkf_update(
    X: np.ndarray,
    y_obs: float,
    HX: np.ndarray,
    R: float,
    inflation: float,  # REQUIRED: No default value assigned here
    Smax: float,       # REQUIRED: No default value assigned here
    Gmax: float,       # REQUIRED: No default value assigned here
) -> np.ndarray:
    """
    Performs the EnKF analysis (update) step.
    Smax, Gmax, and inflation are explicitly passed from run_simulation.py.
    """
    nens = X.shape[1]

    # 1. Inflation (applied to the forecast ensemble X)
    X_mean = np.mean(X, axis=1, keepdims=True)
    X = X_mean + inflation * (X - X_mean)

    # 2. Anomalies and Kalman Gain calculation
    X_ano = X - np.mean(X, axis=1, keepdims=True)
    Y_ano = (HX - np.mean(HX)).reshape(1, -1)
    Pxy = (X_ano @ Y_ano.T) / (nens - 1)
    Pyy = (Y_ano @ Y_ano.T) / (nens - 1) + R
    K = Pxy / (Pyy[0, 0] + 1e-8)  # Kalman Gain

    # 3. Update
    innovation = y_obs - HX
    X_updated = X + K @ innovation.reshape(1, -1)

    # 4. Clamping (uses the passed Smax and Gmax)
    X_updated[0, :] = np.clip(X_updated[0, :], 0.0, Smax)  # S (Soil Moisture)
    X_updated[1, :] = np.clip(X_updated[1, :], 0.0, Gmax)  # G (Groundwater)

    return X_updated


def enkf_forecast_step(
    X_ens: np.ndarray,
    P_t: float,
    PET_t: float,
    params_cal: ModelParams,
    ET_B_t=None,
) -> tuple:
    """
    Forecast step using FIXED calibrated parameters (params_cal).
    Parameter entries in the ensemble are reset at the end to keep them constant.
    """
    state_dim, nens = X_ens.shape
    X_next = np.copy(X_ens)
    Q_ens = np.zeros(nens)
    ET_ens = np.zeros(nens)

    Smax_cal = params_cal.Smax
    # NOTE: Gmax_cal factor (3.0) is a structural constant of the two-store model, 
    # so it remains here or should be passed as a factor if variable.
    Gmax_cal = Smax_cal * 3.0 

    for i in range(nens):
        # 1) Extract only the states
        S_curr, G_curr, _, _, _, _ = X_ens[:, i]

        # 2) Use fixed calibrated parameters
        params = params_cal
        ET_override = ET_B_t

        try:
            # 3) Two-store model step with fixed params
            S_next, G_next, ET_t, Q_t_no_bias, _, _, _ = two_store_model_step(
                S_curr, G_curr, P_t, PET_t, params, ET_override=ET_override
            )
        except Exception:
            # Fallback to current state / zero flux on failure
            S_next, G_next, ET_t, Q_t_no_bias = S_curr, G_curr, 0.0, 0.0

        Q_t = max(Q_t_no_bias, 0.0)

        # 4) Constrain states
        X_next[0, i] = np.clip(S_next, 0.0, Smax_cal)
        X_next[1, i] = np.clip(G_next, 0.0, Gmax_cal)

        # 5) Reset parameter components to FIXED calibrated values
        X_next[2, i] = params_cal.Kperc
        X_next[3, i] = params_cal.Kb
        X_next[4, i] = params_cal.Ke
        X_next[5, i] = params_cal.Cqq

        Q_ens[i] = Q_t
        ET_ens[i] = ET_t

    return X_next, Q_ens, ET_ens