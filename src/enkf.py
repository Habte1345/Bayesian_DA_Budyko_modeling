import numpy as np
from dataclasses import dataclass, field
from typing import Dict
from .model import ModelParams, two_store_model_step
from src.param_manager import get_calibrated_params_for_basin

# =====================================================================
# 4. ENSEMBLE KALMAN FILTER (EnKF) - STABILITY TUNED
# =====================================================================

@dataclass
class EnKFConfig:
    # 1. NON-DEFAULT ARGUMENTS (Fields using default_factory must come first)
    nens: int = 400
    state_dim: int = 7
    
    # Parameter random walk standard deviations are set to ZERO (Fixed Parameters)
    param_rw_sd: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
        # Index:                        [S, G, Kperc, Kb, Ke, Cqq, bias]
    )
    
    # 2. DEFAULT ARGUMENTS (These come last)
    R_Q: float = 150.0  # Observation error variance for streamflow
    R_ET: float = 5.0   # Observation error variance for ET (if assimilated)
    inflation: float = 1.05


def enkf_update(X: np.ndarray, y_obs: float, HX: np.ndarray, 
                R: float, inflation: float = 1.05) -> np.ndarray:
    
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
    X_updated[1, :] = np.clip(X_updated[1, :], 0.0, 2000.0) # G 
    X_updated[2, :] = np.clip(X_updated[2, :], 0.001, 0.999) # Kperc 
    X_updated[3, :] = np.clip(X_updated[3, :], 0.01, 0.999) # Kb 
    X_updated[4, :] = np.clip(X_updated[4, :], 0.01, 1.0) # Ke 
    X_updated[5, :] = np.clip(X_updated[5, :], 0.01, 0.999) # Cqq 
    X_updated[6, :] = np.clip(X_updated[6, :], -15.0, 15.0) # bias 
    
    return X_updated


def enkf_forecast_step(X_ens, P_t, PET_t, Smax_cal: float, ET_B_t=None) -> tuple:
    """
    Performs the forecast step for the EnKF, running the two-store model 
    for each ensemble member. Corrected to use the calibrated Smax.
    """
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
        
        # CORRECTED: Use the dynamic Smax_cal value passed from the run_enkf_scenario function.
        params = ModelParams(Smax=Smax_cal, Kperc=Kperc, Kb=Kb, Ke=Ke, Cqq=Cqq) 
        
        ET_override = ET_B_t if ET_B_t is not None else None
        
        S_next, G_next, ET_t, Q_t_no_bias, _, _, _ = \
            two_store_model_step(S_curr, G_curr, P_t, PET_t, params, ET_override=ET_override)

        Q_t = max(Q_t_no_bias + bias, 0.0)

        X_next[0, i] = S_next
        X_next[1, i] = G_next
        
        Q_ens[i] = Q_t
        ET_ens[i] = ET_t

    return X_next, Q_ens, ET_ens


def run_enkf_scenario(P_monthly, PET_monthly, Q_obs_nldas, ET_B_monthly, 
                      scenario: str, cfg: EnKFConfig, target_basin: str,
                      S_init_nldas: float, G_init_nldas: float):
    
    nmonths = len(P_monthly)
    X = np.zeros((cfg.state_dim, cfg.nens))
    
    # ======================================================================
    # 1. Dynamically Load Calibrated Parameters (INCLUDING SMAX_Used)
    # ======================================================================
    print(f"Loading calibrated parameters for basin: {target_basin}")
    
    cal_params: Dict[str, float] = get_calibrated_params_for_basin(target_basin)
    
    if not cal_params:
         raise ValueError(f"FATAL: Calibrated parameters for {target_basin} were empty after loading. Check param_manager.py logic or file existence.")

    # Assign loaded values using bracket notation for guaranteed use.
    CALIBRATED_KPERC = cal_params['Kperc'] 
    CALIBRATED_KB = cal_params['Kb']
    CALIBRATED_KE = cal_params['Ke']
    CALIBRATED_CQQ  = cal_params['Cqq']
    CALIBRATED_BIAS = cal_params['Bias']
    CALIBRATED_SMAX  = cal_params['Smax_Used'] # <-- CRITICAL: Load the calibrated Smax
    # ======================================================================
    
    # 2. Initialize ensemble X = [S, G, Kperc, Kb, Ke, Cqq, bias] 
    
    # State variables (S and G) initialized with a spread around the NLDAS-based initial values.
    S_spread = S_init_nldas * 0.12 if S_init_nldas > 0 else 0.53
    G_spread = G_init_nldas * 0.02 if G_init_nldas > 0 else 0.23
    
    # FIXED: Use CALIBRATED_SMAX as the upper bound for S initialization
    X[0, :] = np.clip(S_init_nldas + S_spread * np.random.randn(cfg.nens), 1.0, CALIBRATED_SMAX) # S
    X[1, :] = np.clip(G_init_nldas + G_spread * np.random.randn(cfg.nens), 1.0, 2000.0) # G
    
    # Parameters initialized to the calibrated value with ZERO SPREAD (Fixed)
    X[2, :] = np.full(cfg.nens, CALIBRATED_KPERC) # Kperc
    X[3, :] = np.full(cfg.nens, CALIBRATED_KB) # Kb
    X[4, :] = np.full(cfg.nens, CALIBRATED_KE) # Ke
    X[5, :] = np.full(cfg.nens, CALIBRATED_CQQ) # Cqq
    X[6, :] = np.full(cfg.nens, CALIBRATED_BIAS) # bias
    
    Q_mean = np.zeros(nmonths)
    ET_mean = np.zeros(nmonths)
    S_mean = np.zeros(nmonths)
    G_mean = np.zeros(nmonths)
    
    for t in range(nmonths):
        # 3. Parameter Random Walk (Perturbing parameters before forecast)
        # This section is currently inactive because param_rw_sd is [0.0, 0.0, ...]
        X[2:, :] += cfg.param_rw_sd[2:, None] * np.random.randn(cfg.state_dim - 2, cfg.nens)
        
        # Re-apply parameter constraints after random walk (still necessary for safety)
        X[2, :] = np.clip(X[2, :], 0.001, 0.999)
        X[3, :] = np.clip(X[3, :], 0.01, 1.0)    # Consistent Kb constraint
        X[4, :] = np.clip(X[4, :], 0.01, 1.0)    # Consistent Ke constraint
        X[5, :] = np.clip(X[5, :], 0.01, 10.0)   # Consistent Cqq constraint
        X[6, :] = np.clip(X[6, :], -15.0, 15.0)
        
        # 4. Model Integration (Forecast)
        ET_B_t = ET_B_monthly[t] if scenario == 'Budyko' and not np.isnan(ET_B_monthly[t]) else None
        
        # CRITICAL FIX: Pass CALIBRATED_SMAX to the forecast step
        X_next, Q_ens, ET_ens = enkf_forecast_step(X, P_monthly[t], PET_monthly[t], CALIBRATED_SMAX, ET_B_t)
        
        # 5. Analysis Step (Assimilate Q_NLDAS)
        if not np.isnan(Q_obs_nldas[t]) and Q_obs_nldas[t] > 0:
            X_updated = enkf_update(X_next, Q_obs_nldas[t], Q_ens, cfg.R_Q, cfg.inflation)
            X = X_updated
        else:
            X = X_next 
        
        # 6. Store ensemble means
        Q_mean[t] = np.mean(Q_ens)
        ET_mean[t] = np.mean(ET_ens)
        S_mean[t] = np.mean(X[0, :])
        G_mean[t] = np.mean(X[1, :])
    
    return {
        'Q_mean': Q_mean, 'ET_mean': ET_mean, 'S_mean': S_mean, 
        'G_mean': G_mean, 'X_final': X
    }