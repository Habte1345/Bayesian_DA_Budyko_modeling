# src/enkf.py

import numpy as np
from dataclasses import dataclass, field
from src.model import ModelParams, two_store_model_step

# =====================================================================
# 4. ENSEMBLE KALMAN FILTER (EnKF) - STABILITY TUNED
# =====================================================================

@dataclass
class EnKFConfig:
    """Configuration settings for the Ensemble Kalman Filter."""
    nens: int = 400 # Increased ensemble size for robustness
    state_dim: int = 7 # [S, G, Kperc, Kb, Ke, Cqq, bias]
    R_Q: float = 20.0 # Observation error variance (tuned for stability)
    R_ET: float = 5.0 
    inflation: float = 1.03 # Reduced inflation for stability
    # Minimal parameter random walk standard deviations
    param_rw_sd: np.ndarray = field(
        default_factory=lambda: np.array([5.0, 5.0, 0.0005, 0.0005, 0.001, 0.002, 0.05]) 
        # Index:                     [S, G, Kperc, Kb, Ke, Cqq, bias]
    )


def enkf_update(X: np.ndarray, y_obs: float, HX: np.ndarray, 
                R: float, inflation: float = 1.03) -> np.ndarray:
    """Performs the EnKF analysis (update) step."""
    nens = X.shape[1]
    
    # 1. Inflation
    X_mean = np.mean(X, axis=1, keepdims=True)
    X = X_mean + inflation * (X - X_mean)
    
    # 2. Anomalies and Kalman Gain 
    X_ano = X - np.mean(X, axis=1, keepdims=True)
    Y_ano = (HX - np.mean(HX)).reshape(1, -1)
    Pxy = (X_ano @ Y_ano.T) / (nens - 1)
    Pyy = (Y_ano @ Y_ano.T) / (nens - 1) + R
    K = Pxy / (Pyy[0, 0] + 1e-8)
    
    # 3. Update 
    innovation = y_obs - HX
    X_updated = X + K @ innovation.reshape(1, -1)
    
    # 4. Constraints (State and Parameter bounds)
    X_updated[0, :] = np.clip(X_updated[0, :], 0.0, 1500.0) # S 
    X_updated[1, :] = np.clip(X_updated[1, :] , 0.0, 2000.0) # G 
    X_updated[2, :] = np.clip(X_updated[2, :], 0.001, 0.9) # Kperc
    X_updated[3, :] = np.clip(X_updated[3, :], 0.001, 0.5) # Kb
    X_updated[4, :] = np.clip(X_updated[4, :], 0.1, 1.0) # Ke
    X_updated[5, :] = np.clip(X_updated[5, :], 0.1, 1.0) # Cqq
    X_updated[6, :] = np.clip(X_updated[6, :], -10.0, 10.0) # bias
    
    return X_updated


def enkf_forecast_step(X_ens, P_t, PET_t, ET_B_t=None) -> tuple:
    """Performs the EnKF forecast step (model integration) for all ensembles."""
    state_dim, nens = X_ens.shape
    X_next = np.copy(X_ens)
    Q_ens = np.zeros(nens)
    ET_ens = np.zeros(nens)

    for i in range(nens):
        S_curr = X_ens[0, i]
        G_curr = X_ens[1, i]
        Kperc = X_ens[2, i]
        Kb = X_ens[3, i]
        Ke = X_ens[4, i]
        Cqq = X_ens[5, i]
        bias = X_ens[6, i]
        
        params = ModelParams(Smax=500.0, Kperc=Kperc, Kb=Kb, Ke=Ke, Cqq=Cqq)
        
        ET_override = ET_B_t if ET_B_t is not None else None
        
        S_next, G_next, ET_t, Q_t_no_bias, _, _, _ = \
            two_store_model_step(S_curr, G_curr, P_t, PET_t, params, ET_override=ET_override)

        # Q must be >= 0.0
        Q_t = max(Q_t_no_bias + bias, 0.0)

        # Update ensemble state
        X_next[0, i] = S_next
        X_next[1, i] = G_next
        
        Q_ens[i] = Q_t
        ET_ens[i] = ET_t

    return X_next, Q_ens, ET_ens


def run_enkf_scenario(P_monthly, PET_monthly, Q_obs_nldas, ET_B_monthly, scenario: str, cfg: EnKFConfig):
    """Runs the EnKF assimilation cycle for the entire time series."""
    nmonths = len(P_monthly)
    
    # Initialize X array (CRITICAL FIX)
    X = np.zeros((cfg.state_dim, cfg.nens))
    
    # 1. Initialize ensemble X = [S, G, Kperc, Kb, Ke, Cqq, bias] - CONSERVATIVE SPREAD
    X[0, :] = np.clip(100.0 * (1.0 + 0.1 * np.random.randn(cfg.nens)), 0.0, 1500.0) # S (Tighter spread: 0.1)
    X[1, :] = np.clip(50.0 * (1.0 + 0.1 * np.random.randn(cfg.nens)), 0.0, 2000.0) # G (Tighter spread: 0.1)
    X[2, :] = np.clip(0.05 * (1.0 + 0.1 * np.random.randn(cfg.nens)), 0.001, 0.9) # Kperc (Tighter spread: 0.1)
    X[3, :] = np.clip(0.06 * (1.0 + 0.1 * np.random.randn(cfg.nens)), 0.001, 0.5) # Kb (Tighter spread: 0.1)
    X[4, :] = np.clip(0.7 * (1.0 + 0.1 * np.random.randn(cfg.nens)), 0.1, 1.0) # Ke (Tighter spread: 0.1)
    X[5, :] = np.clip(0.8 * (1.0 + 0.1 * np.random.randn(cfg.nens)), 0.1, 1.0) # Cqq (Tighter spread: 0.1)
    X[6, :] = 0.0 + 0.1 * np.random.randn(cfg.nens) # bias (Tighter spread: 0.1)
    
    Q_mean = np.zeros(nmonths)
    ET_mean = np.zeros(nmonths)
    S_mean = np.zeros(nmonths)
    G_mean = np.zeros(nmonths)
    
    for t in range(nmonths):
        # 2. Parameter Random Walk (Covariance Inflation for model uncertainty)
        X[2:, :] += cfg.param_rw_sd[2:, None] * np.random.randn(cfg.state_dim - 2, cfg.nens)
        # Re-apply parameter constraints after random walk
        X[2, :] = np.clip(X[2, :], 0.001, 0.9)
        X[3, :] = np.clip(X[3, :], 0.001, 0.5)
        X[4, :] = np.clip(X[4, :], 0.1, 1.0)
        X[5, :] = np.clip(X[5, :], 0.1, 1.0)
        X[6, :] = np.clip(X[6, :], -10.0, 10.0)
        
        # 3. Model Integration (Forecast)
        ET_B_t = ET_B_monthly[t] if scenario == 'Budyko' and not np.isnan(ET_B_monthly[t]) else None
        X_next, Q_ens, ET_ens = enkf_forecast_step(X, P_monthly[t], PET_monthly[t], ET_B_t)
        
        # 4. Analysis Step (Assimilate Q_NLDAS)
        if not np.isnan(Q_obs_nldas[t]) and Q_obs_nldas[t] > 0:
            X_updated = enkf_update(X_next, Q_obs_nldas[t], Q_ens, cfg.R_Q, cfg.inflation)
            X = X_updated
        else:
            X = X_next 
        
        # 5. Store ensemble means
        Q_mean[t] = np.mean(Q_ens)
        ET_mean[t] = np.mean(ET_ens)
        S_mean[t] = np.mean(X[0, :])
        G_mean[t] = np.mean(X[1, :])
    
    return {
        'Q_mean': Q_mean, 'ET_mean': ET_mean, 'S_mean': S_mean, 
        'G_mean': G_mean, 'X_final': X
    }