# import numpy as np
# from dataclasses import dataclass, field
# from .model import ModelParams, two_store_model_step
# from src.param_manager import get_calibrated_params_for_basin

# # =====================================================================
# # 4. ENSEMBLE KALMAN FILTER (EnKF) - STABILITY TUNED
# # =====================================================================

# @dataclass
# class EnKFConfig:
#     # 1. NON-DEFAULT ARGUMENTS (Fields using default_factory must come first)
#     nens: int = 400
#     state_dim: int = 7
    
#     # MOVED HERE: This field uses default_factory, making it non-default in order
#     param_rw_sd: np.ndarray = field(
#         default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
#         # Index:                        [S, G, Kperc, Kb, Ke, Cqq, bias]
#     )
    
#     # 2. DEFAULT ARGUMENTS (These come last)
#     R_Q: float = 150.0 
#     R_ET: float = 5.0 
#     inflation: float = 1.05

#     # Parameter random walk standard deviations are set to ZERO (Fixed Parameters)
#     param_rw_sd: np.ndarray = field(
#         default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
#         # Index:                        [S, G, Kperc, Kb, Ke, Cqq, bias]
#     )


# def enkf_update(X: np.ndarray, y_obs: float, HX: np.ndarray, 
#                 R: float, inflation: float = 1.05) -> np.ndarray:
    
#     nens = X.shape[1]
    
#     # 1. Inflation
#     X_mean = np.mean(X, axis=1, keepdims=True)
#     X = X_mean + inflation * (X - X_mean)
    
#     # 2. Anomalies and Kalman Gain 
#     X_ano = X - np.mean(X, axis=1, keepdims=True)
#     Y_ano = (HX - np.mean(HX)).reshape(1, -1)
#     Pxy = (X_ano @ Y_ano.T) / (nens - 1)
#     Pyy = (Y_ano @ Y_ano.T) / (nens - 1) + R
#     K = Pxy / (Pyy[0, 0] + 1e-8)
    
#     # 3. Update 
#     innovation = y_obs - HX
#     X_updated = X + K @ innovation.reshape(1, -1)
    
#     # 4. Constraints (State and Parameter bounds)
#     X_updated[0, :] = np.clip(X_updated[0, :], 0.0, 1500.0) # S 
#     X_updated[1, :] = np.clip(X_updated[1, :] , 0.0, 2000.0) # G 
#     X_updated[2, :] = np.clip(X_updated[2, :], 0.001, 0.999) # Kperc (Expanded upper bound)
#     X_updated[3, :] = np.clip(X_updated[3, :], 0.001, 0.999) # Kb (Expanded upper bound)
#     X_updated[4, :] = np.clip(X_updated[4, :], 0.01, 1.0) # Ke (Lower bound expanded)
#     X_updated[5, :] = np.clip(X_updated[5, :], 0.01, 10.0) # Cqq (Upper bound expanded significantly)
#     X_updated[6, :] = np.clip(X_updated[6, :], -15.0, 15.0) # bias (Expanded range)
    
#     return X_updated


# def enkf_forecast_step(X_ens, P_t, PET_t, ET_B_t=None) -> tuple:
    
#     state_dim, nens = X_ens.shape
#     X_next = np.copy(X_ens)
#     Q_ens = np.zeros(nens)
#     ET_ens = np.zeros(nens)

#     for i in range(nens):
#         S_curr = X_ens[0, i]
#         G_curr = X_ens[1, i]
#         Kperc = X_ens[2, i]
#         Kb = X_ens[3, i]
#         Ke = X_ens[4, i]
#         Cqq = X_ens[5, i]
#         bias = X_ens[6, i]
        
#         params = ModelParams(Smax=500.0, Kperc=Kperc, Kb=Kb, Ke=Ke, Cqq=Cqq)
        
#         ET_override = ET_B_t if ET_B_t is not None else None
        
#         S_next, G_next, ET_t, Q_t_no_bias, _, _, _ = \
#             two_store_model_step(S_curr, G_curr, P_t, PET_t, params, ET_override=ET_override)

#         Q_t = max(Q_t_no_bias + bias, 0.0)

#         X_next[0, i] = S_next
#         X_next[1, i] = G_next
        
#         Q_ens[i] = Q_t
#         ET_ens[i] = ET_t

#     return X_next, Q_ens, ET_ens

# def run_enkf_scenario(P_monthly, PET_monthly, Q_obs_nldas, ET_B_monthly, 
#                       scenario: str, cfg: EnKFConfig, target_basin: str,
#                       S_init_nldas: float, G_init_nldas: float):
    
#     nmonths = len(P_monthly)
    
#     X = np.zeros((cfg.state_dim, cfg.nens))
    
#     # ======================================================================
#     # 1. Dynamically Load Calibrated Parameters
#     # ======================================================================
#     print(f"Loading calibrated parameters for basin: {target_basin}")
#     try:
#         cal_params = get_calibrated_params_for_basin(target_basin)
#     except FileNotFoundError as e:
#         print(f"FATAL ERROR: {e}. Using hardcoded fallback parameters.")
#         cal_params = {'Kperc': 0.25, 'Kb': 0.15, 'Ke': 0.80, 'Cqq': 1.50, 'Bias': 1.0}
#     except Exception as e:
#         print(f"FATAL ERROR during param load: {e}. Using hardcoded fallback parameters.")
#         cal_params = {'Kperc': 0.25, 'Kb': 0.15, 'Ke': 0.80, 'Cqq': 1.50, 'Bias': 1.0}

#     # Assign loaded (or fallback) values
#     CALIBRATED_KPERC = cal_params.get('Kperc', 0.25)
#     CALIBRATED_KB    = cal_params.get('Kb', 0.15)
#     CALIBRATED_KE    = cal_params.get('Ke', 0.80)
#     CALIBRATED_CQQ   = cal_params.get('Cqq', 1.50)
#     CALIBRATED_BIAS  = cal_params.get('Bias', 1.0)
#     # ======================================================================
    
#     # 2. Initialize ensemble X = [S, G, Kperc, Kb, Ke, Cqq, bias] 
    
#     # # State variables (S and G) initialized with spread around a guess
#     # X[0, :] = np.clip(100.0 + 30.0 * np.random.randn(cfg.nens), 1.0, 1500.0) # S (Initial State)
#     # X[1, :] = np.clip(50.0 + 15.0 * np.random.randn(cfg.nens), 1.0, 2000.0)  # G (Initial State)

#     X[0, :] = np.clip(0.54 + 10.0 * np.random.randn(cfg.nens), 1.0, 5.0) # S (Reduced mean and spread)
#     X[1, :] = np.clip(0.38 + 5.0 * np.random.randn(cfg.nens), 1.0, 3.0)  # G (Reduced mean and spread)
    
#     # Parameters initialized around the calibrated values with spread
#     # The spread (e.g., * (1.0 + 0.2 * np.random.randn(...))) defines the initial ensemble uncertainty.
#     # X[2, :] = np.clip(CALIBRATED_KPERC * (1.0 + 0.2 * np.random.randn(cfg.nens)), 0.001, 0.999) # Kperc
#     # X[3, :] = np.clip(CALIBRATED_KB * (1.0 + 0.3 * np.random.randn(cfg.nens)), 0.001, 0.999)    # Kb
#     # X[4, :] = np.clip(CALIBRATED_KE * (1.0 + 0.1 * np.random.randn(cfg.nens)), 0.01, 1.0)      # Ke
#     # X[5, :] = np.clip(CALIBRATED_CQQ * (1.0 + 0.3 * np.random.randn(cfg.nens)), 0.01, 10.0)    # Cqq
#     # X[6, :] = CALIBRATED_BIAS + 0.5 * np.random.randn(cfg.nens)                                # bias 

#     # Parameters initialized to the calibrated value with ZERO SPREAD (Fixed)
#     X[2, :] = np.full(cfg.nens, CALIBRATED_KPERC) # Kperc
#     X[3, :] = np.full(cfg.nens, CALIBRATED_KB)    # Kb
#     X[4, :] = np.full(cfg.nens, CALIBRATED_KE)    # Ke
#     X[5, :] = np.full(cfg.nens, CALIBRATED_CQQ)   # Cqq
#     X[6, :] = np.full(cfg.nens, CALIBRATED_BIAS)  # bias
    
#     Q_mean = np.zeros(nmonths)
#     ET_mean = np.zeros(nmonths)
#     S_mean = np.zeros(nmonths)
#     G_mean = np.zeros(nmonths)
    
#     for t in range(nmonths):
#         # 3. Parameter Random Walk (Perturbing parameters before forecast)
#         X[2:, :] += cfg.param_rw_sd[2:, None] * np.random.randn(cfg.state_dim - 2, cfg.nens)
#         # Re-apply parameter constraints after random walk
#         X[2, :] = np.clip(X[2, :], 0.001, 0.999)
#         X[3, :] = np.clip(X[3, :], 0.001, 0.999)
#         X[4, :] = np.clip(X[4, :], 0.01, 1.0)
#         X[5, :] = np.clip(X[5, :], 0.01, 10.0)
#         X[6, :] = np.clip(X[6, :], -15.0, 15.0)
        
#         # 4. Model Integration (Forecast)
#         ET_B_t = ET_B_monthly[t] if scenario == 'Budyko' and not np.isnan(ET_B_monthly[t]) else None
#         X_next, Q_ens, ET_ens = enkf_forecast_step(X, P_monthly[t], PET_monthly[t], ET_B_t)
        
#         # 5. Analysis Step (Assimilate Q_NLDAS)
#         if not np.isnan(Q_obs_nldas[t]) and Q_obs_nldas[t] > 0:
#             # Note: The enkf_update function handles the state constraints internally
#             X_updated = enkf_update(X_next, Q_obs_nldas[t], Q_ens, cfg.R_Q, cfg.inflation)
#             X = X_updated
#         else:
#             X = X_next 
        
#         # 6. Store ensemble means
#         Q_mean[t] = np.mean(Q_ens)
#         ET_mean[t] = np.mean(ET_ens)
#         S_mean[t] = np.mean(X[0, :])
#         G_mean[t] = np.mean(X[1, :])
    
#     return {
#         'Q_mean': Q_mean, 'ET_mean': ET_mean, 'S_mean': S_mean, 
#         'G_mean': G_mean, 'X_final': X
#     }


# import numpy as np
# from dataclasses import dataclass, field
# from typing import Dict # Added for better type hinting
# from .model import ModelParams, two_store_model_step
# from src.param_manager import get_calibrated_params_for_basin

# # =====================================================================
# # 4. ENSEMBLE KALMAN FILTER (EnKF) - STABILITY TUNED
# # =====================================================================

# @dataclass
# class EnKFConfig:
#     # 1. NON-DEFAULT ARGUMENTS (Fields using default_factory must come first)
#     nens: int = 400
#     state_dim: int = 7
    
#     # MOVED and CORRECTED: This field uses default_factory. Set to ZERO for FIXED PARAMETERS.
#     param_rw_sd: np.ndarray = field(
#         default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
#         # Index:                        [S, G, Kperc, Kb, Ke, Cqq, bias]
#     )
    
#     # 2. DEFAULT ARGUMENTS (These come last)
#     R_Q: float = 150.0 
#     R_ET: float = 5.0 
#     inflation: float = 1.05


# def enkf_update(X: np.ndarray, y_obs: float, HX: np.ndarray, 
#                 R: float, inflation: float = 1.05) -> np.ndarray:
    
#     nens = X.shape[1]
    
#     # 1. Inflation
#     X_mean = np.mean(X, axis=1, keepdims=True)
#     X = X_mean + inflation * (X - X_mean)
    
#     # 2. Anomalies and Kalman Gain 
#     X_ano = X - np.mean(X, axis=1, keepdims=True)
#     Y_ano = (HX - np.mean(HX)).reshape(1, -1)
#     Pxy = (X_ano @ Y_ano.T) / (nens - 1)
#     Pyy = (Y_ano @ Y_ano.T) / (nens - 1) + R
#     K = Pxy / (Pyy[0, 0] + 1e-8)
    
#     # 3. Update 
#     innovation = y_obs - HX
#     X_updated = X + K @ innovation.reshape(1, -1)
    
#     # 4. Constraints (State and Parameter bounds)
#     # NOTE: Since parameters are fixed via initialization and zero random walk,
#     # these clips are primarily for ensuring state (S, G) is positive and bounded.
#     X_updated[0, :] = np.clip(X_updated[0, :], 0.0, 1500.0) # S 
#     X_updated[1, :] = np.clip(X_updated[1, :], 0.0, 2000.0) # G 
#     X_updated[2, :] = np.clip(X_updated[2, :], 0.001, 0.999) # Kperc 
#     X_updated[3, :] = np.clip(X_updated[3, :], 0.001, 0.999) # Kb 
#     X_updated[4, :] = np.clip(X_updated[4, :], 0.01, 1.0) # Ke 
#     X_updated[5, :] = np.clip(X_updated[5, :], 0.01, 10.0) # Cqq 
#     X_updated[6, :] = np.clip(X_updated[6, :], -15.0, 15.0) # bias 
    
#     return X_updated


# def enkf_forecast_step(X_ens, P_t, PET_t, ET_B_t=None) -> tuple:
    
#     state_dim, nens = X_ens.shape
#     X_next = np.copy(X_ens)
#     Q_ens = np.zeros(nens)
#     ET_ens = np.zeros(nens)

#     for i in range(nens):
#         S_curr = X_ens[0, i]
#         G_curr = X_ens[1, i]
#         Kperc = X_ens[2, i]
#         Kb = X_ens[3, i]
#         Ke = X_ens[4, i]
#         Cqq = X_ens[5, i]
#         bias = X_ens[6, i]
        
#         params = ModelParams(Smax=500.0, Kperc=Kperc, Kb=Kb, Ke=Ke, Cqq=Cqq)
        
#         ET_override = ET_B_t if ET_B_t is not None else None
        
#         S_next, G_next, ET_t, Q_t_no_bias, _, _, _ = \
#             two_store_model_step(S_curr, G_curr, P_t, PET_t, params, ET_override=ET_override)

#         Q_t = max(Q_t_no_bias + bias, 0.0)

#         X_next[0, i] = S_next
#         X_next[1, i] = G_next
        
#         Q_ens[i] = Q_t
#         ET_ens[i] = ET_t

#     return X_next, Q_ens, ET_ens


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
    nens: int = 200
    state_dim: int = 7
    
    # MOVED and CORRECTED: This field uses default_factory. Set to ZERO for FIXED PARAMETERS.
    param_rw_sd: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
        # Index:                        [S, G, Kperc, Kb, Ke, Cqq, bias]
    )
    
    # 2. DEFAULT ARGUMENTS (These come last)
    R_Q: float = 0.5
    R_ET: float = 0.1 
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
    X_updated[0, :] = np.clip(X_updated[0, :], 0.0, 2000.0) # S 
    X_updated[1, :] = np.clip(X_updated[1, :], 0.0, 25000.0) # G 
    X_updated[2, :] = np.clip(X_updated[2, :], 0.001, 0.999) # Kperc 
    X_updated[3, :] = np.clip(X_updated[3, :], 0.001, 0.999) # Kb 
    X_updated[4, :] = np.clip(X_updated[4, :], 0.01, 1.0) # Ke 
    X_updated[5, :] = np.clip(X_updated[5, :], 0.01, 10.0) # Cqq 
    X_updated[6, :] = np.clip(X_updated[6, :], -15.0, 15.0) # bias 
    
    return X_updated


def enkf_forecast_step(X_ens, P_t, PET_t, ET_B_t=None) -> tuple:
    
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
        
        params = ModelParams(Smax=150.0, Kperc=Kperc, Kb=Kb, Ke=Ke, Cqq=Cqq)
        
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
    # 1. Dynamically Load Calibrated Parameters
    # ======================================================================
    print(f"Loading calibrated parameters for basin: {target_basin}")
    try:
        cal_params: Dict[str, float] = get_calibrated_params_for_basin(target_basin)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}. Using hardcoded fallback parameters.")
        cal_params = {'Kperc': 0.25, 'Kb': 0.15, 'Ke': 0.80, 'Cqq': 1.50, 'Bias': 1.0}
    except Exception as e:
        print(f"FATAL ERROR during param load: {e}. Using hardcoded fallback parameters.")
        cal_params = {'Kperc': 0.25, 'Kb': 0.15, 'Ke': 0.80, 'Cqq': 1.50, 'Bias': 1.0}

    # Assign loaded (or fallback) values
    CALIBRATED_KPERC = cal_params.get('Kperc', 0.8)
    CALIBRATED_KB    = cal_params.get('Kb', 0.7)
    CALIBRATED_KE    = cal_params.get('Ke', 0.80)
    CALIBRATED_CQQ   = cal_params.get('Cqq', 1.50)
    CALIBRATED_BIAS  = cal_params.get('Bias', 1.0)
    # ======================================================================
    
    # 2. Initialize ensemble X = [S, G, Kperc, Kb, Ke, Cqq, bias] 

    X[0, :] = np.clip(0.56 + 5.0 * np.random.randn(cfg.nens), 1.0, 100.0) 
    
    # Groundwater Storage (G) initialized around G_init_nldas (SoilM_0_200cm)
    X[1, :] = np.clip(0.35 + 2.0 * np.random.randn(cfg.nens), 1.0, 50.0) 
    
    # Parameters initialized to the calibrated value with ZERO SPREAD (Fixed)
    X[2, :] = np.full(cfg.nens, CALIBRATED_KPERC) # Kperc
    X[3, :] = np.full(cfg.nens, CALIBRATED_KB)    # Kb
    X[4, :] = np.full(cfg.nens, CALIBRATED_KE)    # Ke
    X[5, :] = np.full(cfg.nens, CALIBRATED_CQQ)   # Cqq
    X[6, :] = np.full(cfg.nens, CALIBRATED_BIAS)  # bias
    
    Q_mean = np.zeros(nmonths)
    ET_mean = np.zeros(nmonths)
    S_mean = np.zeros(nmonths)
    G_mean = np.zeros(nmonths)
    
    for t in range(nmonths):
        # The random walk and re-clipping steps for parameters are correctly 
        # removed from the time loop, ensuring parameters remain fixed.
        
        # 4. Model Integration (Forecast)
        ET_B_t = ET_B_monthly[t] if scenario == 'Budyko' and not np.isnan(ET_B_monthly[t]) else None
        X_next, Q_ens, ET_ens = enkf_forecast_step(X, P_monthly[t], PET_monthly[t], ET_B_t)
        
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