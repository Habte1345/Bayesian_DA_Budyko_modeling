# src/enkf.py 

import numpy as np
from dataclasses import dataclass, field
from .model import ModelParams, two_store_model_step

# =====================================================================
# 4. ENSEMBLE KALMAN FILTER (EnKF) - CORE FUNCTIONS
# =====================================================================

@dataclass
class EnKFConfig:
    nens: int = 300 
    state_dim: int = 7 # [S, G, Kperc, Kb, Ke, Cqq, bias]
    param_rw_sd: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Random walk SD
    )
    R_Q: float = 100.0   # Observation error covariance for Q (if used)
    R_ET: float = 50.0   # Observation error covariance for ET
    inflation: float = 1.05 

def enkf_update(X: np.ndarray, y_obs: float, HX: np.ndarray, 
                R: float, inflation: float = 1.05) -> np.ndarray:
    """Performs the EnKF analysis (update) step."""
    
    nens = X.shape[1]
    
    # 1. Inflation (applied to the forecast ensemble X)
    X_mean = np.mean(X, axis=1, keepdims=True)
    X = X_mean + inflation * (X - X_mean)
    
    # 2. Anomalies and Kalman Gain calculation
    X_ano = X - np.mean(X, axis=1, keepdims=True)
    Y_ano = (HX - np.mean(HX)).reshape(1, -1)
    Pxy = (X_ano @ Y_ano.T) / (nens - 1)
    Pyy = (Y_ano @ Y_ano.T) / (nens - 1) + R
    K = Pxy / (Pyy[0, 0] + 1e-8) # Kalman Gain
    
    # 3. Update
    innovation = y_obs - HX
    X_updated = X + K @ innovation.reshape(1, -1)

    X_updated[0, :] = np.clip(X_updated[0, :], 0.0, 1500.0) # S (Soil Moisture)
    X_updated[1, :] = np.clip(X_updated[1, :], 0.0, 2000.0) # G (Groundwater)
    X_updated[2, :] = np.clip(X_updated[2, :], 0.005, 0.99) # 🔥 Kperc: Looser bounds (Calibrated Kperc was 0.008)
    X_updated[3, :] = np.clip(X_updated[3, :], 0.01, 0.5)   # Kb
    X_updated[4, :] = np.clip(X_updated[4, :], 0.2, 0.9)    # Ke
    X_updated[5, :] = np.clip(X_updated[5, :], 0.1, 20.0)   # 🔥 Cqq: Loosened lower bound (Calibrated Cqq was as low as 0.161)
    X_updated[6, :] = np.clip(X_updated[6, :], -0.5, 0.5)   # bias
    
    return X_updated

# src/enkf.py - Corrected enkf_forecast_step

def enkf_forecast_step(X_ens, P_t, PET_t, Smax_cal: float, ET_B_t=None) -> tuple:
    """Performs the forecast step for the EnKF, running the two-store model."""
    
    state_dim, nens = X_ens.shape
    X_next = np.copy(X_ens) # Start X_next as a copy of X_ens to carry over parameters
    Q_ens = np.zeros(nens)
    ET_ens = np.zeros(nens)

    # Define constraints for parameters to ensure stability post-random walk
    KPERC_BOUNDS = (0.005, 0.99)
    KB_BOUNDS = (0.01, 0.5)
    KE_BOUNDS = (0.2, 0.9)
    CQQ_BOUNDS = (0.1, 20.0)
    BIAS_BOUNDS = (-0.5, 0.5)

    for i in range(nens):
        # 1. Extract states and parameters from the ensemble member
        S_curr, G_curr, Kperc, Kb, Ke, Cqq, bias = X_ens[:, i]
        
        # 2. Re-apply parameter constraints (Important if random walk occurred just before this call)
        Kperc = np.clip(Kperc, *KPERC_BOUNDS)
        Kb = np.clip(Kb, *KB_BOUNDS)
        Ke = np.clip(Ke, *KE_BOUNDS)
        Cqq = np.clip(Cqq, *CQQ_BOUNDS)
        bias = np.clip(bias, *BIAS_BOUNDS)
        
        params = ModelParams(Smax=Smax_cal, Kperc=Kperc, Kb=Kb, Ke=Ke, Cqq=Cqq) 
        ET_override = ET_B_t 
        
        try:
            # 3. Run the two-store model step (Relies on the numerically stable model in src/model.py)
            S_next, G_next, ET_t, Q_t_no_bias, _, _, _ = \
                two_store_model_step(S_curr, G_curr, P_t, PET_t, params, ET_override=ET_override)
        except Exception:
            # Fallback to current state/zero flux in case of fatal crash
            S_next, G_next, ET_t, Q_t_no_bias = S_curr, G_curr, 0.0, 0.0
            
        # 4. Total Streamflow (including the state-augmented 'bias' parameter)
        Q_t = max(Q_t_no_bias + bias, 0.0)

        # 5. Update X_next with new states and constrained parameters
        X_next[0, i] = np.clip(S_next, 0.0, Smax_cal)
        X_next[1, i] = np.clip(G_next, 0.0, Smax_cal * 3.0) 
        
        # Parameters (Indices 2-6): Use the constrained values from step 2
        X_next[2, i] = Kperc # Kperc
        X_next[3, i] = Kb    # Kb
        X_next[4, i] = Ke    # Ke
        X_next[5, i] = Cqq   # Cqq
        X_next[6, i] = bias  # bias
        
        Q_ens[i] = Q_t
        ET_ens[i] = ET_t

    return X_next, Q_ens, ET_ens