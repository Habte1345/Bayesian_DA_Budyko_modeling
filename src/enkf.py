# # import numpy as np
# # from dataclasses import dataclass, field
# # from typing import Dict
# # from .model import ModelParams, two_store_model_step
# # from src.param_manager import get_calibrated_params_for_basin

# # # =====================================================================
# # # 4. ENSEMBLE KALMAN FILTER (EnKF) - STABILITY TUNED
# # # =====================================================================

# # @dataclass
# # class EnKFConfig:
# #     nens: int = 300 
# #     state_dim: int = 7
# #     param_rw_sd: np.ndarray = field(
# #         # Keep Random Walk OFF (all zeros) until DA is stable on states alone
# #         default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# #     )
# #     # The magnitude of Q (up to 200 mm/month) means R_Q must be large!
# #     # Let's use 10% of the mean flow variance as a starting point, which is high for safety.
# #     R_Q: float = 100.0   # Significantly increased from 25.0
    
# #     # Budyko ET is a very rough constraint and needs a huge error bar.
# #     R_ET: float = 50.0   # Significantly increased from 10.0
    
# #     # Inflation must be close to 1.0 for stability.
# #     inflation: float = 1.05 # Lowered from 1.1

# # def enkf_update(X: np.ndarray, y_obs: float, HX: np.ndarray, 
# #                 R: float, inflation: float = 1.05) -> np.ndarray:
    
# #     nens = X.shape[1]
    
# #     # 1. Inflation
# #     X_mean = np.mean(X, axis=1, keepdims=True)
# #     X = X_mean + inflation * (X - X_mean)
    
# #     # 2. Anomalies and Kalman Gain 
# #     X_ano = X - np.mean(X, axis=1, keepdims=True)
# #     Y_ano = (HX - np.mean(HX)).reshape(1, -1)
# #     Pxy = (X_ano @ Y_ano.T) / (nens - 1)
# #     Pyy = (Y_ano @ Y_ano.T) / (nens - 1) + R
# #     K = Pxy / (Pyy[0, 0] + 1e-8)
    
# #     # 3. Update 
# #     innovation = y_obs - HX
# #     X_updated = X + K @ innovation.reshape(1, -1)
    
# #     # 4. Constraints (State and Parameter bounds)
# #     X_updated[0, :] = np.clip(X_updated[0, :], 0.0, 1500.0) # S 
# #     X_updated[1, :] = np.clip(X_updated[1, :], 0.0, 2000.0) # G 
# #     X_updated[2, :] = np.clip(X_updated[2, :], 0.001, 0.999) # Kperc 
# #     X_updated[3, :] = np.clip(X_updated[3, :], 0.01, 0.999) # Kb 
# #     X_updated[4, :] = np.clip(X_updated[4, :], 0.01, 1.0) # Ke 
# #     X_updated[5, :] = np.clip(X_updated[5, :], 0.01, 0.999) # Cqq 
# #     X_updated[6, :] = np.clip(X_updated[6, :], -15.0, 15.0) # bias 
    
# #     return X_updated


# # def enkf_forecast_step(X_ens, P_t, PET_t, Smax_cal: float, ET_B_t=None) -> tuple:
# #     """
# #     Performs the forecast step for the EnKF, running the two-store model 
# #     for each ensemble member. Corrected to use the calibrated Smax.
# #     """
# #     state_dim, nens = X_ens.shape
# #     X_next = np.copy(X_ens)
# #     Q_ens = np.zeros(nens)
# #     ET_ens = np.zeros(nens)

# #     for i in range(nens):
# #         S_curr = X_ens[0, i]
# #         G_curr = X_ens[1, i]
# #         Kperc = X_ens[2, i]
# #         Kb = X_ens[3, i]
# #         Ke = X_ens[4, i]
# #         Cqq = X_ens[5, i]
# #         bias = X_ens[6, i]
        
# #         # CORRECTED: Use the dynamic Smax_cal value passed from the run_enkf_scenario function.
# #         params = ModelParams(Smax=Smax_cal, Kperc=Kperc, Kb=Kb, Ke=Ke, Cqq=Cqq) 
        
# #         ET_override = ET_B_t if ET_B_t is not None else None
        
# #         S_next, G_next, ET_t, Q_t_no_bias, _, _, _ = \
# #             two_store_model_step(S_curr, G_curr, P_t, PET_t, params, ET_override=ET_override)

# #         Q_t = max(Q_t_no_bias + bias, 0.0)

# #         X_next[0, i] = S_next
# #         X_next[1, i] = G_next
        
# #         Q_ens[i] = Q_t
# #         ET_ens[i] = ET_t

# #     return X_next, Q_ens, ET_ens

# # # In src/enkf.py: (Updated run_enkf_scenario function)

# # def run_enkf_scenario(
# #     P_monthly, PET_monthly, Q_obs_nldas, ET_B_monthly,
# #     scenario: str, cfg: EnKFConfig, target_basin: str,
# #     S_init_nldas: float, G_init_nldas: float
# # ):
# #     """
# #     Run Ensemble Kalman Filter (EnKF) scenario for a given basin.
# #     """

# #     nmonths = len(P_monthly)
# #     X = np.zeros((cfg.state_dim, cfg.nens))

# #     # ======================================================================
# #     # 1. Dynamically Load Calibrated Parameters (INCLUDING SMAX_Used)
# #     # ======================================================================
# #     cal_params: Dict[str, float] = get_calibrated_params_for_basin(target_basin)
# #     if not cal_params:
# #         raise ValueError(f"FATAL: Calibrated parameters for {target_basin} were empty...")

# #     CALIBRATED_KPERC = cal_params['Kperc']
# #     CALIBRATED_KB = cal_params['Kb']
# #     CALIBRATED_KE = cal_params['Ke']
# #     CALIBRATED_CQQ = cal_params['Cqq']
# #     CALIBRATED_BIAS = cal_params['bias']
# #     CALIBRATED_SMAX = cal_params['Smax']  # <-- CRITICAL: Load calibrated Smax

# #     # ======================================================================
# #     # 2. Initialize Ensemble X = [S, G, Kperc, Kb, Ke, Cqq, bias]
# #     # ======================================================================
# #     S_spread = S_init_nldas * 0.12 if S_init_nldas > 0 else 0.53
# #     G_spread = G_init_nldas * 0.02 if G_init_nldas > 0 else 0.23

# #     X[0, :] = np.clip(S_init_nldas + S_spread * np.random.randn(cfg.nens), 1.0, CALIBRATED_SMAX)  # S
# #     X[1, :] = np.clip(G_init_nldas + G_spread * np.random.randn(cfg.nens), 1.0, 2000.0)           # G
# #     X[2, :] = np.full(cfg.nens, CALIBRATED_KPERC)                                                 # Kperc
# #     X[3, :] = np.full(cfg.nens, CALIBRATED_KB)                                                    # Kb
# #     X[4, :] = np.full(cfg.nens, CALIBRATED_KE)                                                    # Ke
# #     X[5, :] = np.full(cfg.nens, CALIBRATED_CQQ)                                                   # Cqq
# #     X[6, :] = np.full(cfg.nens, CALIBRATED_BIAS)                                                  # bias

# #     Q_mean = np.zeros(nmonths)
# #     ET_mean = np.zeros(nmonths)
# #     S_mean = np.zeros(nmonths)
# #     G_mean = np.zeros(nmonths)

# #     # ======================================================================
# #     # 3. EnKF Time Integration Loop
# #     # ======================================================================
# #     for t in range(nmonths):
# #         # Parameter random walk (perturbation)
# #         X[2:, :] += cfg.param_rw_sd[2:, None] * np.random.randn(cfg.state_dim - 2, cfg.nens)
# #         X[2, :] = np.clip(X[2, :], 0.001, 0.999)
# #         X[3, :] = np.clip(X[3, :], 0.01, 1.0)
# #         X[4, :] = np.clip(X[4, :], 0.01, 1.0)
# #         X[5, :] = np.clip(X[5, :], 0.01, 10.0)
# #         X[6, :] = np.clip(X[6, :], -15.0, 15.0)

# #         # Determine if Budyko ET is used as input constraint during forecast
# #         ET_override_for_forecast = (
# #             ET_B_monthly[t]
# #             if scenario == 'Budyko+DA' and not np.isnan(ET_B_monthly[t])
# #             else None
# #         )

# #         # 4. Forecast step
# #         X_next, Q_ens, ET_ens = enkf_forecast_step(
# #             X, P_monthly[t], PET_monthly[t], CALIBRATED_SMAX, ET_override_for_forecast
# #         )

# #         # 5. Analysis step (Data Assimilation)
# #         X_updated = X_next

# #         # Assimilate Q observation for 'Base+DA' and 'Budyko+DA'
# #         if scenario in ['Base+DA', 'Budyko+DA'] and not np.isnan(Q_obs_nldas[t]) and Q_obs_nldas[t] > 0:
# #             X_updated = enkf_update(X_next, Q_obs_nldas[t], Q_ens, cfg.R_Q, cfg.inflation)

# #         # Assimilate ET observation ONLY for 'Budyko+DA'
# #         if scenario == 'Budyko+DA' and ET_override_for_forecast is not None:
# #             X_updated = enkf_update(X_updated, ET_override_for_forecast, ET_ens, cfg.R_ET, cfg.inflation)

# #         # Ensemble update
# #         X = X_updated

# #         # 6. Store ensemble means
# #         Q_mean[t] = np.mean(Q_ens)
# #         ET_mean[t] = np.mean(ET_ens)
# #         S_mean[t] = np.mean(X[0, :])
# #         G_mean[t] = np.mean(X[1, :])

# #     # ======================================================================
# #     # 7. Return outputs
# #     # ======================================================================
# #     return {
# #         'Q_mean': Q_mean,
# #         'ET_mean': ET_mean,
# #         'S_mean': S_mean,
# #         'G_mean': G_mean,
# #         'X_final': X
# #     }

# import numpy as np
# from dataclasses import dataclass, field
# from typing import Dict
# from .model import ModelParams, two_store_model_step
# from src.param_manager import get_calibrated_params_for_basin

# # =====================================================================
# # 4. ENSEMBLE KALMAN FILTER (EnKF) - STABILITY TUNED
# # =====================================================================

# @dataclass
# class EnKFConfig:
#     nens: int = 300 
#     state_dim: int = 7
#     param_rw_sd: np.ndarray = field(
#         default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#     )
#     R_Q: float = 100.0   
#     # R_ET now refers to the NLDAS ET observation error
#     R_ET: float = 50.0   
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
#     X_updated[0, :] = np.clip(X_updated[0, :], 0.0, 1500.0) # S 
#     X_updated[1, :] = np.clip(X_updated[1, :], 0.0, 2000.0) # G 
#     X_updated[2, :] = np.clip(X_updated[2, :], 0.001, 0.999) # Kperc 
#     X_updated[3, :] = np.clip(X_updated[3, :], 0.01, 0.999) # Kb 
#     X_updated[4, :] = np.clip(X_updated[4, :], 0.01, 1.0) # Ke 
#     X_updated[5, :] = np.clip(X_updated[5, :], 0.01, 0.999) # Cqq 
#     X_updated[6, :] = np.clip(X_updated[6, :], -15.0, 15.0) # bias 
    
#     return X_updated


# def enkf_forecast_step(X_ens, P_t, PET_t, Smax_cal: float, ET_B_t=None) -> tuple:
#     """
#     Performs the forecast step for the EnKF, running the two-store model.
#     ET_B_t is used to override the model's standard ET calculation (Ke * PET)
#     in the Q_B scenario.
#     """
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
        
#         params = ModelParams(Smax=Smax_cal, Kperc=Kperc, Kb=Kb, Ke=Ke, Cqq=Cqq) 
        
#         ET_override = ET_B_t if ET_B_t is not None else None
        
#         # two_store_model_step runs the model and returns Q_t and ET_t
#         S_next, G_next, ET_t, Q_t_no_bias, _, _, _ = \
#             two_store_model_step(S_curr, G_curr, P_t, PET_t, params, ET_override=ET_override)

#         Q_t = max(Q_t_no_bias + bias, 0.0)

#         # IMPORTANT: Applying State/Gauging Constraints here for stability (retained)
#         X_next[0, i] = np.clip(S_next, 0.0, Smax_cal)
#         X_next[1, i] = np.clip(G_next, 0.0, 2000.0)
        
#         Q_ens[i] = Q_t
#         ET_ens[i] = ET_t

#     return X_next, Q_ens, ET_ens


# def run_enkf_scenario(
#     P_monthly, PET_monthly, ET_B_monthly, ET_NLDAS_monthly, # <- CRITICAL: Removed Q_obs_nldas from DA path
#     scenario: str, cfg: EnKFConfig, target_basin: str,
#     S_init_nldas: float, G_init_nldas: float
# ):
#     """
#     Run EnKF scenario for a given basin. Only Q_ET_DA scenario performs DA.
#     The DA uses ET_NLDAS_monthly as the observation.
#     """
#     nmonths = len(P_monthly)
#     X = np.zeros((cfg.state_dim, cfg.nens))

#     # ======================================================================
#     # 1. Parameter Loading and Initialization (CRITICAL: Loading CALIBRATED_SMAX)
#     # ======================================================================
#     cal_params: Dict[str, float] = get_calibrated_params_for_basin(target_basin)
#     if not cal_params:
#          # Simplified error handling - run_simulation.py handles the fallback
#          raise ValueError(f"FATAL: Calibrated parameters for {target_basin} were empty...")

#     CALIBRATED_KPERC = cal_params['Kperc']
#     CALIBRATED_KB = cal_params['Kb']
#     CALIBRATED_KE = cal_params['Ke']
#     CALIBRATED_CQQ = cal_params['Cqq']
#     CALIBRATED_BIAS = cal_params['bias']
#     CALIBRATED_SMAX = cal_params['Smax']  # <- CRITICAL: Load calibrated Smax

#     # [... Ensemble Initialization code is assumed to be correct ...]
#     # (Using CALIBRATED_SMAX for state clipping, and full parameter ensemble)
#     S_spread = S_init_nldas * 0.12 if S_init_nldas > 0 else 0.53
#     G_spread = G_init_nldas * 0.02 if G_init_nldas > 0 else 0.23

#     X[0, :] = np.clip(S_init_nldas + S_spread * np.random.randn(cfg.nens), 1.0, CALIBRATED_SMAX)
#     X[1, :] = np.clip(G_init_nldas + G_spread * np.random.randn(cfg.nens), 1.0, 2000.0)
#     X[2, :] = np.full(cfg.nens, CALIBRATED_KPERC)
#     X[3, :] = np.full(cfg.nens, CALIBRATED_KB)
#     X[4, :] = np.full(cfg.nens, CALIBRATED_KE)
#     X[5, :] = np.full(cfg.nens, CALIBRATED_CQQ)
#     X[6, :] = np.full(cfg.nens, CALIBRATED_BIAS)
    
#     Q_mean = np.zeros(nmonths)
#     ET_mean = np.zeros(nmonths)
#     S_mean = np.zeros(nmonths)
#     G_mean = np.zeros(nmonths)

#     # ======================================================================
#     # 2. EnKF Time Integration Loop (Modified for New Scenarios)
#     # ======================================================================
#     for t in range(nmonths):
#         # Parameter random walk (perturbation) - kept for structural completeness
#         X[2:, :] += cfg.param_rw_sd[2:, None] * np.random.randn(cfg.state_dim - 2, cfg.nens)
#         X[2, :] = np.clip(X[2, :], 0.001, 0.999)
#         X[3, :] = np.clip(X[3, :], 0.01, 1.0)
#         X[4, :] = np.clip(X[4, :], 0.01, 1.0)
#         X[5, :] = np.clip(X[5, :], 0.01, 10.0)
#         X[6, :] = np.clip(X[6, :], -15.0, 15.0)

#         # Determine if Budyko ET is used as input constraint for the Q_B scenario
#         ET_override_for_forecast = (
#             ET_B_monthly[t]
#             if scenario == 'Q_B' and not np.isnan(ET_B_monthly[t])
#             else None
#         )

#         # 3. Forecast step: Uses default ET_Ke (unless ET_override is set for Q_B)
#         X_next, Q_ens, ET_ens = enkf_forecast_step(
#             X, P_monthly[t], PET_monthly[t], CALIBRATED_SMAX, ET_override_for_forecast
#         )

#         # 4. Analysis step (Data Assimilation)
#         X_updated = X_next # Start with the forecast ensemble

#         # *** CRITICAL CHANGE: Only assimilate NLDAS ET for 'Q_ET_DA' scenario ***
#         if scenario == 'Q_ET_DA':
#             ET_obs = ET_NLDAS_monthly[t]
#             is_et_available = not np.isnan(ET_obs) and ET_obs > 0
            
#             if is_et_available:
#                 # Assimilate NLDAS ET (Evap_df)
#                 # Observation is ET, Model projection (H(X)) is the forecast ET_ens
#                 X_updated = enkf_update(X_next, ET_obs, ET_ens, cfg.R_ET, cfg.inflation)

#         # 5. Ensemble update
#         X = X_updated

#         # 6. Store ensemble means (Q_mean is the assimilated Q for Q_ET_DA)
#         Q_mean[t] = np.mean(Q_ens)
#         ET_mean[t] = np.mean(ET_ens)
#         S_mean[t] = np.mean(X[0, :])
#         G_mean[t] = np.mean(X[1, :])

#     # ======================================================================
#     # 7. Return outputs
#     # ======================================================================
#     return {
#         'Q_mean': Q_mean,
#         'ET_mean': ET_mean,
#         'S_mean': S_mean,
#         'G_mean': G_mean,
#         'X_final': X
#     }



# src/enkf.py (Simplified to core EnKF functions)

import numpy as np
from dataclasses import dataclass, field
from .model import ModelParams, two_store_model_step
# Removed get_calibrated_params_for_basin as it is only needed in run_simulation.py

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
    
    # 4. Constraints (State and Parameter bounds) - Must be consistent with calibration bounds
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
        
        # States (Indices 0, 1): MUST be clipped
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