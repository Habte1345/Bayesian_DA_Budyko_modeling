# src/enkf.py
from dataclasses import dataclass
import numpy as np
from .model import ModelParams, two_store_model_step

@dataclass
class EnKFConfig:
    nens: int = 300
    inflation: float = 1.08
    R_ET_std: float = 20.0
    R_Q_std: float = 100.0
    proc_S_std: float = 0.3
    proc_G_std: float = 0.7
    P_std_frac: float = 0.10
    PET_std_frac: float = 0.03


def enkf_update_stochastic_scalar(
    X: np.ndarray,          # (2, nens)
    y_obs: float,
    HX: np.ndarray,         # (nens,)
    R_var: float,
    inflation: float,
    Smax: float,
    Gmax: float,
    rng: np.random.Generator,
) -> np.ndarray:

    state_dim, nens = X.shape
    if state_dim != 2:
        raise ValueError("Expected X shape (2, nens) for [S, G].")

    # inflation
    X_mean = X.mean(axis=1, keepdims=True)
    X = X_mean + inflation * (X - X_mean)

    # anomalies
    X_ano = X - X.mean(axis=1, keepdims=True)
    HX = HX.reshape(1, -1)
    Y_ano = HX - HX.mean(axis=1, keepdims=True)

    Pxy = (X_ano @ Y_ano.T) / (nens - 1)
    Pyy = (Y_ano @ Y_ano.T) / (nens - 1) + R_var

    K = Pxy / (Pyy[0, 0] + 1e-12)

    # perturbed obs
    eps = rng.normal(0.0, np.sqrt(R_var), size=nens)
    y_pert = y_obs + eps

    innovation = (y_pert - HX.ravel()).reshape(1, -1)
    X_updated = X + K @ innovation

    X_updated[0, :] = np.clip(X_updated[0, :], 0.0, Smax)
    X_updated[1, :] = np.clip(X_updated[1, :], 0.0, Gmax)

    return X_updated


def enkf_forecast_step_states(
    X: np.ndarray,            # (2, nens)
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
):
    """
    Forecast:
      - perturb forcing per member (stochastic)
      - propagate through model
      - return state ensemble + predicted obs ensembles (ET/Q)
    """
    _, nens = X.shape
    X_next = np.zeros_like(X)
    ET_ens = np.zeros(nens)
    Q_ens = np.zeros(nens)

    P_t = max(float(P_t), 0.0)
    PET_t = max(float(PET_t), 0.0)

    for i in range(nens):
        S_curr = float(X[0, i])
        G_curr = float(X[1, i])

        # --- forcing perturbation (KEY)
        P_i = P_t * (1.0 + rng.normal(0.0, P_std_frac))
        PET_i = PET_t * (1.0 + rng.normal(0.0, PET_std_frac))
        P_i = max(P_i, 0.0)
        PET_i = max(PET_i, 0.0)

        try:
            S_next, G_next, ET_t, Q_t_raw, *_ = two_store_model_step(
                S_curr, G_curr, P_i, PET_i, params_cal,
                ET_override=ET_override,
            )
        except Exception:
            S_next, G_next, ET_t, Q_t_raw = S_curr, G_curr, 0.0, 0.0

        # --- state process noise (extra anti-collapse)
        if proc_S_std > 0.0:
            S_next += rng.normal(0.0, proc_S_std)
        if proc_G_std > 0.0:
            G_next += rng.normal(0.0, proc_G_std)

        S_next = np.clip(S_next, 0.0, Smax)
        G_next = np.clip(G_next, 0.0, Gmax)

        X_next[0, i] = S_next
        X_next[1, i] = G_next

        ET_ens[i] = ET_t
        Q_ens[i] = max(Q_t_raw, 0.0)

    return X_next, ET_ens, Q_ens








# # src/enkf.py
# from dataclasses import dataclass, field
# import numpy as np

# from .model import ModelParams, two_store_model_step


# # ---------------------------------------------------------
# # EnKF Configuration
# # ---------------------------------------------------------
# @dataclass
# class EnKFConfig:
#     nens: int = 5
#     # State vector structure: [S, G, Kperc, Kb, Ke, Cqq]
#     state_dim: int = 6

#     # Parameter random-walk SD (kept for compatibility)
#     param_rw_sd: np.ndarray = field(
#         default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#     )

#     R_Q: float = 100   # Observation error covariance for Q (if used)
#     R_ET: float = 20   # Observation error covariance for ET
#     inflation: float = 1.05


# # ---------------------------------------------------------
# # EnKF Update Step
# # ---------------------------------------------------------
# def enkf_update(
#     X: np.ndarray,
#     y_obs: float,
#     HX: np.ndarray,
#     R: float,
#     inflation: float,
#     Smax: float,
#     Gmax: float,
# ) -> np.ndarray:
#     """
#     EnKF analysis step.
#     Smax, Gmax, and inflation are explicitly provided from run_simulation.py.
#     """
#     nens = X.shape[1]

#     # 1. Inflation
#     X_mean = np.mean(X, axis=1, keepdims=True)
#     X = X_mean + inflation * (X - X_mean)

#     # 2. Anomalies
#     X_ano = X - np.mean(X, axis=1, keepdims=True)
#     Y_ano = (HX - np.mean(HX)).reshape(1, -1)

#     # Covariances
#     Pxy = (X_ano @ Y_ano.T) / (nens - 1)
#     Pyy = (Y_ano @ Y_ano.T) / (nens - 1) + R

#     # Kalman Gain
#     K = Pxy / (Pyy[0, 0] + 1e-8)

#     # 3. Update
#     innovation = y_obs - HX
#     X_updated = X + K @ innovation.reshape(1, -1)

#     # 4. Clamping using passed Smax, Gmax
#     X_updated[0, :] = np.clip(X_updated[0, :], 0.0, Smax)
#     X_updated[1, :] = np.clip(X_updated[1, :], 0.0, Gmax)

#     return X_updated


# # ---------------------------------------------------------
# # EnKF Forecast Step
# # ---------------------------------------------------------
# def enkf_forecast_step(
#     X_ens: np.ndarray,
#     P_t: float,
#     PET_t: float,
#     params_cal: ModelParams,
#     ET_B_t=None,
# ) -> tuple:
#     """
#     Forecast step using calibrated (fixed) parameters.

#     Parameters in the ensemble are overwritten by the calibrated values,
#     keeping the state as the only evolving component.
#     """
#     state_dim, nens = X_ens.shape
#     X_next = np.copy(X_ens)

#     Q_ens = np.zeros(nens)
#     ET_ens = np.zeros(nens)

#     Smax_cal = params_cal.Smax
#     Gmax_cal = Smax_cal * 3

#     for i in range(nens):

#         # Extract current state (only S, G are used)
#         S_curr, G_curr, _, _, _, _ = X_ens[:, i]

#         params = params_cal
#         ET_override = ET_B_t

#         try:
#             S_next, G_next, ET_t, Q_t_raw, _, _, _ = two_store_model_step(
#                 S_curr, G_curr, P_t, PET_t, params, ET_override=ET_override
#             )
#         except Exception:
#             S_next, G_next, ET_t, Q_t_raw = S_curr, G_curr, 0.0, 0.0

#         Q_t = max(Q_t_raw, 0.0)

#         # Clamp states
#         X_next[0, i] = np.clip(S_next, 0.0, Smax_cal)
#         X_next[1, i] = np.clip(G_next, 0.0, Gmax_cal)

#         # Reset parameters to calibrated values
#         X_next[2, i] = params_cal.Kperc
#         X_next[3, i] = params_cal.Kb
#         X_next[4, i] = params_cal.Ke
#         X_next[5, i] = params_cal.Cqq

#         Q_ens[i] = Q_t
#         ET_ens[i] = ET_t

#     return X_next, Q_ens, ET_ens
