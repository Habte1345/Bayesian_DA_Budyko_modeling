
# import sys
# import os
# import numpy as np
# import pandas as pd
# from typing import Dict, Optional, Tuple, Any, List
# import geopandas as gpd 
# from multiprocessing import Pool, cpu_count
# import warnings
# from tqdm import tqdm
# import logging
# from functools import partial

# # Ignore specific RuntimeWarnings for numerical stability
# warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
# np.seterr(all="ignore")

# # --- GLOBAL CONSTANTS & PATHS ---
# PROJECT_ROOT = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\DA\DA_Github_repo\Bayesian_DA_Budyko_modeling"
# DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
# INPUT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "input_data", "NDVI") # New directory for M and Slope
# sys.path.append(PROJECT_ROOT)
# G_MAX_CEILING = 5000.0

# # =====================================================================
# # CORE MODULE IMPORTS (fall back to safe defaults if import fails)
# # =====================================================================
# # (Fallback definitions for EnKFConfig, enkf_update, etc., are kept here
# # to ensure the script is runnable even without the 'src' modules.)
# try:
#     from src.model import ModelParams, two_store_model_step
#     from src.enkf import EnKFConfig, enkf_update, enkf_forecast_step
#     from src.param_manager import get_calibrated_params_for_basin, load_all_calibrated_params
#     from src.metrics import calculate_kge
#     from src.budyko import estimate_budyko_et, OmegaMLRModel
# except ImportError as e:
#     print(f"FATAL: CORE MODULE Import failed. Using Fallbacks. Error: {e}")
    
#     class EnKFConfig:
#         def __init__(self):
#             self.state_dim = 7
#             self.nens = 50
#             self.R_ET = 0.5
#             self.R_Q = 1.0
#             self.R_B = 0.5
#             self.inflation = 1.05
#             self.param_perturb_frac = 0.05
#             self.min_param_sd = 1e-4

#     class ModelParams:
#         def __init__(self, **kwargs):
#             self.__dict__.update(kwargs)

#     def enkf_update(X, y_obs, y_ens, R, inflation):
#         X = X.copy()
#         nx, nens = X.shape
#         y_ens = np.array(y_ens).reshape((nens,))
#         X_mean = X.mean(axis=1, keepdims=True)
#         X_ano = X - X_mean
#         y_mean = y_ens.mean()
#         y_ano = y_ens - y_mean
#         C_xo = (X_ano @ y_ano.reshape(1, -1).T) / (nens - 1)
#         C_oo = (y_ano @ y_ano.T) / (nens - 1)
#         # Note: The fallback implementation for K calculation might need review for edge cases, 
#         # but is kept here for completeness based on the provided snippet.
#         K = C_xo / (C_oo + R + 1e-12) 
#         y_obs_pert = np.random.normal(loc=y_obs, scale=np.sqrt(R), size=nens)
#         X = X + K @ (y_obs_pert.reshape(1, -1) - y_ens.reshape(1, -1))
#         X_mean_post = X.mean(axis=1, keepdims=True)
#         X = X_mean_post + inflation * (X - X_mean_post)
#         return X

#     def enkf_forecast_step(X, P_t, PET_t, CALIBRATED_SMAX, ET_B_t=None):
#         nens = X.shape[1]
#         S = X[0, :]
#         G = X[1, :]
#         bias = X[6, :]
#         # Simplified forecast model for fallback
#         Q_ens = np.maximum(0.0, 0.1 * (S + G) + bias + 0.01 * np.random.randn(nens))
#         Ke = X[4, :]
#         ET_ens = np.maximum(0.0, Ke * (PET_t if PET_t is not None else 0.0) + 0.01 * np.random.randn(nens))
#         if ET_B_t is not None and not np.isnan(ET_B_t):
#             ET_ens = ET_ens * 0.5 + (ET_B_t * 0.5)
#         X_next = X.copy()
#         return X_next, Q_ens, ET_ens

#     def get_calibrated_params_for_basin(basin_id):
#         return {'Kperc': 0.2, 'Kb': 0.05, 'Ke': 0.6, 'Cqq': 0.3, 'bias': 0.0, 'Smax': 20.0}

#     def load_all_calibrated_params():
#         return {}

#     def calculate_kge(obs, sim):
#         if len(obs) == 0:
#             return np.nan
#         # Simplified KGE calculation (just correlation) for fallback
#         return float(np.corrcoef(obs, sim)[0, 1]) 

#     def estimate_budyko_et(P, PET, M, Slope, omega_model, model='Fu'):
#         with np.errstate(divide='ignore', invalid='ignore'):
#             # Fallback estimation (still needs the inputs)
#             phi = PET / (P + 1e-12)
#             frac = P / (P + PET + 1e-12)
#             return frac * PET
    
#     class OmegaMLRModel:
#         def __init__(self, beta0, beta1, beta2):
#             pass

# # =====================================================================
# # LOGGING, IO, and ENKF HELPER FUNCTIONS
# # =====================================================================
# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s] %(levelname)s: %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

# def load_feather_df(fname: str, ddir: str) -> pd.DataFrame:
#     """Loads a Feather file from a specified directory."""
#     path = os.path.join(ddir, fname)
#     if not os.path.exists(path):
#         # Allow Budyko files to fail gracefully if the input_data directory is wrong, but log it
#         if ddir == INPUT_DATA_DIR:
#              logging.error(f"FATAL: Budyko data file not found: {path}. Check your path!")
#         else:
#             logging.warning(f"File not found: {path}. Returning empty DataFrame.")
#         return pd.DataFrame()
#     df = pd.read_feather(path)
#     df = df.dropna(axis=1, how='all')
#     if 'time' in df.columns:
#         df['time'] = pd.to_datetime(df['time'])
#         df.set_index('time', inplace=True)
#     return df

# def load_all_data():
#     """Loads all 9 input dataframes."""
#     # Data DataFrames (from DATA_DIR)
#     Rainf_df = load_feather_df("Rainf.feather", DATA_DIR)
#     PotEvap_df = load_feather_df("PotEvap.feather", DATA_DIR)
#     Evap_df = load_feather_df("EVap.feather", DATA_DIR)
#     Q_USGS_monthly = load_feather_df("Q_USGS.feather", DATA_DIR)
#     Qsb_monthly = load_feather_df("Qsb.feather", DATA_DIR)
#     Q_nldas_mm_monthly = load_feather_df("Q_nldas_mm_monthly.feather", DATA_DIR)
#     S_init_df = load_feather_df("RootMoist.feather", DATA_DIR)
#     G_init_df = load_feather_df("SoilM_0_200cm.feather", DATA_DIR)
    
#     # Budyko Parameter DataFrames (from INPUT_DATA_DIR)
#     M_df = load_feather_df("M.feather", INPUT_DATA_DIR)
#     Slope_df = load_feather_df("slope.feather", INPUT_DATA_DIR) # Single-row DataFrame
    
#     return Rainf_df, PotEvap_df, Evap_df, Q_USGS_monthly, Qsb_monthly, Q_nldas_mm_monthly, \
#            S_init_df, G_init_df, M_df, Slope_df

# def _extract_column_key(df: pd.DataFrame, basin_id: str) -> str:
#     cleaned_id = str(basin_id).zfill(8)
#     if cleaned_id in df.columns:
#         return cleaned_id
#     unpadded_id = str(basin_id).lstrip('0')
#     if unpadded_id in df.columns:
#         return unpadded_id
#     # Final robust check for partial matches (less ideal but safer for mixed formats)
#     for col in df.columns:
#         col_str = str(col)
#         if col_str.endswith(cleaned_id) or cleaned_id in col_str:
#             return col
#     # NOTE: Since pre-filtering is done, this should theoretically not be hit for TARGET_BASINS
#     raise KeyError(f"Column for basin {cleaned_id} not found.")

# def _ensure_cfg_defaults(cfg):
#     if not hasattr(cfg, 'R_Q') or cfg.R_Q is None:
#         cfg.R_Q = 1.0
#     if not hasattr(cfg, 'R_ET') or cfg.R_ET is None:
#         cfg.R_ET = 0.5
#     if not hasattr(cfg, 'R_B') or cfg.R_B is None:
#         cfg.R_B = 0.5
#     if not hasattr(cfg, 'inflation') or cfg.inflation is None:
#         cfg.inflation = 1.05
#     if not hasattr(cfg, 'param_perturb_frac') or cfg.param_perturb_frac is None:
#         cfg.param_perturb_frac = 0.05
#     if not hasattr(cfg, 'min_param_sd'):
#         cfg.min_param_sd = 1e-4
#     if not hasattr(cfg, 'nens') or cfg.nens is None:
#         cfg.nens = 50

# def _apply_param_spread(X, param_indices, cal_params, cfg):
#     param_names = ['Kperc', 'Kb', 'Ke', 'Cqq', 'bias']
#     for i, param_name in enumerate(param_names):
#         idx = param_indices[i]
#         base = cal_params.get(param_name, 0.0)
#         sd = max(cfg.param_perturb_frac * max(abs(base), 1.0), cfg.min_param_sd)
#         X[idx, :] = np.random.normal(loc=base, scale=sd, size=X.shape[1])
#         X[idx, :] = np.where(X[idx, :] < 0.0, 0.0, X[idx, :])

# def _adaptive_inflation_and_jitter(X, inflation, var_floor=1e-8, jitter_scale=1e-6):
#     var = np.var(X, axis=1, ddof=1)
#     # If variance is low, inflate ensemble spread around the mean
#     low_var_mask = var < var_floor
#     if np.any(low_var_mask):
#         X_mean = X.mean(axis=1, keepdims=True)
#         X = X_mean + inflation * (X - X_mean)
#     # Add minor jitter to prevent collapse
#     X += np.random.normal(loc=0.0, scale=jitter_scale, size=X.shape)
#     return X

# def run_enkf_scenario(P_monthly, PET_monthly, Q_USGS_monthly, Qsb_monthly, ET_B_monthly, ET_NLDAS_monthly,
#                       Q_nldas_mm_monthly_data, # This is the crucial parameter name
#                       scenario, cfg, target_basin, S_init_nldas, G_init_nldas, CALIBRATED_SMAX,
#                       all_calibrated_params):
    
#     _ensure_cfg_defaults(cfg)
#     nmonths = len(P_monthly)
#     cal_params = all_calibrated_params.get(target_basin)
#     if not cal_params:
#         cal_params = get_calibrated_params_for_basin(target_basin)
#     if not cal_params:
#         return None

#     TOTAL_DIM = 7
#     X = np.zeros((TOTAL_DIM, cfg.nens))
#     X[0, :] = np.clip(np.random.normal(S_init_nldas, 0.1, cfg.nens), 0.25, CALIBRATED_SMAX)
#     X[1, :] = np.clip(np.random.normal(G_init_nldas, 0.1, cfg.nens), 0.2, G_MAX_CEILING)

#     try:
#         # Indices 2, 3, 4, 5, 6 for Kperc, Kb, Ke, Cqq, bias parameters
#         _apply_param_spread(X, param_indices=[2, 3, 4, 5, 6], cal_params=cal_params, cfg=cfg)
#     except Exception:
#         # Fallback if spread fails
#         X[2, :] = cal_params.get('Kperc', 0.0)
#         X[3, :] = cal_params.get('Kb', 0.0)
#         X[4, :] = cal_params.get('Ke', 0.0)
#         X[5, :] = cal_params.get('Cqq', 0.0)
#         X[6, :] = cal_params.get('bias', 0.0)

#     Q_mean = np.zeros(nmonths)
#     ET_mean = np.zeros(nmonths)
#     S_mean = np.zeros(nmonths)
#     G_mean = np.zeros(nmonths)
#     X_t = X.copy()

#     for t in range(nmonths):
#         ET_B_t_forecast_input = ET_B_monthly[t] if scenario == 'Q_B' else None

#         X_next, Q_ens, ET_ens = enkf_forecast_step(
#             X_t, P_monthly[t], PET_monthly[t], CALIBRATED_SMAX,
#             ET_B_t_forecast_input
#         )

#         X_assimilated = X_next.copy()

#         # A. Assimilation for ET (NLDAS) - Only in Q_ET_DA scenario
#         if scenario == 'Q_ET_DA':
#             ET_obs = ET_NLDAS_monthly[t]
#             if not np.isnan(ET_obs) and ET_obs > 0:
#                 Rb = getattr(cfg, 'R_ET', 0.5)
#                 et_obs_pert = float(ET_obs) + np.random.normal(0.0, np.sqrt(max(Rb, 1e-12)))
#                 try:
#                     X_assimilated = enkf_update(X_assimilated, et_obs_pert, ET_ens, Rb, cfg.inflation)
#                 except Exception as e:
#                     logging.debug(f"Warning: ET assimilation failed at t={t}: {e}")

#         # B. Assimilation for Budyko Pseudo-Observation (ET_B) - Only in Q_B scenario
#         if scenario == 'Q_B':
#             ET_B_t_obs = ET_B_monthly[t]
#             if not np.isnan(ET_B_t_obs) and ET_B_t_obs > 0:
#                 Rb = getattr(cfg, 'R_B', 0.5)
#                 etb_pert = float(ET_B_t_obs) + np.random.normal(0.0, np.sqrt(max(Rb, 1e-12)))
#                 try:
#                     X_assimilated = enkf_update(X_assimilated, etb_pert, ET_ens, Rb, cfg.inflation)
#                 except Exception as e:
#                     logging.debug(f"Warning: Budyko pseudo-assimilation failed at t={t}: {e}")

#         # C. Assimilation for Streamflow (Q_NLDAS) - Runs for ALL scenarios (Q_Base, Q_B, Q_ET_DA)
#         Q_obs = Q_nldas_mm_monthly_data[t] # Correctly using the parameter
#         if not np.isnan(Q_obs) and Q_obs > 0:
#             Rq = getattr(cfg, 'R_Q', 1.0)
#             q_obs_pert = float(Q_obs) + np.random.normal(0.0, np.sqrt(max(Rq, 1e-12)))
#             try:
#                 X_assimilated = enkf_update(X_assimilated, q_obs_pert, Q_ens, Rq, cfg.inflation)
#             except Exception as e:
#                 logging.debug(f"Warning: Q assimilation failed at t={t}: {e}")

#         X_t = X_assimilated.copy()

#         # State constraints
#         X_t[0, :] = np.clip(X_t[0, :], 0.23, CALIBRATED_SMAX)
#         X_t[1, :] = np.clip(X_t[1, :], 0.24, G_MAX_CEILING)

#         try:
#             X_t = _adaptive_inflation_and_jitter(X_t, inflation=cfg.inflation, var_floor=1e-8, jitter_scale=1e-6)
#         except Exception as e:
#             logging.debug(f"Adaptive inflation/jitter failed at t={t}: {e}")

#         Q_mean[t] = np.mean(Q_ens)
#         ET_mean[t] = np.mean(ET_ens)
#         S_mean[t] = np.mean(X_t[0, :])
#         G_mean[t] = np.mean(X_t[1, :])

#     return {'Q_mean': Q_mean, 'ET_mean': ET_mean, 'S_mean': S_mean, 'G_mean': G_mean, 'X_final': X_t}


# # =====================================================================
# # PROCESS BASIN (Updated: Fixed local variable name for consistency)
# # =====================================================================
# def process_basin(
#     basin: str, all_calibrated_params: Dict[str, ModelParams],
#     Rainf_df: pd.DataFrame, PotEvap_df: pd.DataFrame, Evap_df: pd.DataFrame,
#     Q_USGS_monthly: pd.DataFrame, Q_nldas_mm_monthly: pd.DataFrame, Qsb_monthly: pd.DataFrame,
#     S_init_df: pd.DataFrame, G_init_df: pd.DataFrame,
#     M_df: pd.DataFrame, Slope_df: pd.DataFrame, # NEW BUDYKO DATA
#     cfg: EnKFConfig
# ) -> Tuple[str, Optional[pd.DataFrame], Optional[Dict[str, float]]]:

#     basin_id = str(basin).zfill(8)
#     basin_metrics = {
#         'Q_Base_Q_KGE': np.nan, 'Q_B_Q_KGE': np.nan, 'Q_ET_DA_Q_KGE': np.nan,
#         'Q_Base_ET_KGE': np.nan, 'Q_B_ET_KGE': np.nan, 'Q_ET_DA_ET_KGE': np.nan
#     }
#     dates = Q_USGS_monthly.index

#     try:
#         # --- 1. Data Extraction (Input Time Series) ---
#         P_key = _extract_column_key(Rainf_df, basin_id)
#         PET_key = _extract_column_key(PotEvap_df, basin_id)
#         Q_obs_key = _extract_column_key(Q_USGS_monthly, basin_id)
#         ET_key = _extract_column_key(Evap_df, basin_id)
#         Q_nldas_key = _extract_column_key(Q_nldas_mm_monthly, basin_id)
#         Qsb_key = _extract_column_key(Qsb_monthly, basin_id) # Need Qsb key too
#         S_init_key = _extract_column_key(S_init_df, basin_id)
#         G_init_key = _extract_column_key(G_init_df, basin_id)
        
#         P_monthly = np.clip(Rainf_df[P_key].values, 0.0, np.inf)
#         PET_monthly = np.clip(PotEvap_df[PET_key].values, 0.0, np.inf)

#         Q_obs_USGS_eval = np.clip(Q_USGS_monthly[Q_obs_key].values, 0.0, np.inf)
#         ET_NLDAS_monthly = np.clip(Evap_df[ET_key].values, 0.0, np.inf)
        
#         # 👇 CRITICAL FIX: Renamed local variable for consistency (Q_nldas_mm_monthly_data)
#         Q_nldas_mm_monthly_data = Q_nldas_mm_monthly[Q_nldas_key].values 
        
#         Qsb_monthly_data = Qsb_monthly[Qsb_key].values # Using Qsb_key

#         cal_params = all_calibrated_params.get(basin_id)
#         CALIBRATED_SMAX = cal_params['Smax'] if cal_params and 'Smax' in cal_params else 20.0

#         S_init_nldas = np.nanmean(S_init_df[S_init_key].values[-12:])
#         G_init_nldas = np.nanmean(G_init_df[G_init_key].values[-12:])

#         # --- 2. Budyko Parameter Extraction (Actual Data) ---
#         M_key = _extract_column_key(M_df, basin_id)
#         Slope_key = _extract_column_key(Slope_df, basin_id)
        
#         # M_basin is a time series
#         M_basin = M_df[M_key].values 
        
#         # Slope_basin is a single value from the first row of Slope_df
#         if Slope_df.empty:
#              raise ValueError("Slope data is empty.")
#         Slope_basin = float(Slope_df[Slope_key].iloc[0]) 

#         # --- 3. Budyko Calculation ---
#         omega_model = OmegaMLRModel(beta0=2.36, beta1=1.16, beta2=0.0)

#         ET_B_monthly = np.array([
#             estimate_budyko_et(P_monthly[t], PET_monthly[t], M_basin[t], Slope_basin, omega_model, model='Fu')
#             for t in range(len(P_monthly))
#         ])

#         # --- 4. Simulation Execution ---
#         if np.all(np.isnan(P_monthly)) or np.all(np.isnan(PET_monthly)) or np.all(np.isnan(Q_obs_USGS_eval)):
#             return basin_id, None, basin_metrics

#         basin_results_dfs = {}
#         spin_up = 60

#         for scenario in ['Q_Base', 'Q_B', 'Q_ET_DA']:
#             result = run_enkf_scenario(
#                 P_monthly, PET_monthly, Q_obs_USGS_eval, Qsb_monthly_data,
#                 ET_B_monthly, ET_NLDAS_monthly,
#                 Q_nldas_mm_monthly_data, # Using the CORRECT local variable name
#                 scenario, cfg, basin_id, S_init_nldas, G_init_nldas, CALIBRATED_SMAX,
#                 all_calibrated_params
#             )
            
#             if result is None:
#                 logging.warning(f"Skipping basin {basin_id} in scenario {scenario}: Missing calibrated parameters or failed setup.")
#                 continue

#             # Metrics Calculation (remains the same)
#             Q_sim, Q_obs = result['Q_mean'][spin_up:], Q_obs_USGS_eval[spin_up:]
#             ET_sim, ET_obs = result['ET_mean'][spin_up:], ET_NLDAS_monthly[spin_up:]
            
#             valid_mask_Q = ~(np.isnan(Q_sim) | np.isnan(Q_obs))
#             if np.any(valid_mask_Q) and len(Q_obs[valid_mask_Q]) >= 12:
#                 basin_metrics[f"{scenario}_Q_KGE"] = calculate_kge(Q_obs[valid_mask_Q], Q_sim[valid_mask_Q])

#             valid_mask_ET = ~(np.isnan(ET_sim) | np.isnan(ET_obs))
#             if np.any(valid_mask_ET) and len(ET_obs[valid_mask_ET]) >= 12:
#                 basin_metrics[f"{scenario}_ET_KGE"] = calculate_kge(ET_obs[valid_mask_ET], ET_sim[valid_mask_ET])

#             df_scenario = pd.DataFrame({
#                 f'{scenario}_Q_sim': result['Q_mean'],
#                 f'{scenario}_ET_sim': result['ET_mean'],
#                 f'{scenario}_S_sim': result['S_mean'],
#                 f'{scenario}_G_sim': result['G_mean'],
#             }, index=dates)
#             basin_results_dfs[scenario] = df_scenario
            
#         if not basin_results_dfs:
#             return basin_id, None, basin_metrics

#         # --- 5. Final DataFrame Assembly and Return ---
#         df_obs_inputs = pd.DataFrame({
#             'Date': dates,
#             'P_input': P_monthly,
#             'PET_input': PET_monthly,
#             'Q_USGS_obs': Q_obs_USGS_eval,
#             'Q_NLDAS_input': Q_nldas_mm_monthly_data, # Using the CORRECT local variable name
#             'ET_NLDAS_obs': ET_NLDAS_monthly,
#             'ET_Budyko': ET_B_monthly
#         }).set_index('Date')

#         df_final_basin = df_obs_inputs
#         for df_scenario in basin_results_dfs.values():
#             df_final_basin = df_final_basin.join(df_scenario, how='left')

#         return basin_id, df_final_basin, basin_metrics

#     except KeyError as e:
#         logging.error(f"CRITICAL ERROR (KeyError) processing basin {basin_id}: Data column lookup failed: {e}. Skipping basin.")
#         return basin_id, None, basin_metrics
#     except Exception as e:
#         logging.error(f"CRITICAL ERROR processing basin {basin_id}: {e}. Skipping basin.")
#         return basin_id, None, basin_metrics


# # =====================================================================
# # MAIN EXECUTION BLOCK (Cleaned to avoid unnecessary dataframe reassignments)
# # =====================================================================
# if __name__ == '__main__':
#     logging.info("----------✅ Loading all 9 input dataframes...")
#     try:
#         # Load all 9 time-series and parameter dataframes using their original names
#         Rainf_df, PotEvap_df, Evap_df, Q_USGS_monthly, Qsb_monthly, Q_nldas_mm_monthly, S_init_df, G_init_df, M_df, Slope_df = load_all_data()
        
#         # Check essential files
#         if Rainf_df.empty or Q_USGS_monthly.empty or M_df.empty or Slope_df.empty:
#             raise ValueError("Essential input data (Rainf, Q_USGS, M, or Slope) is empty. Check file paths and content.")
#         logging.info("----------✅ Data loading complete.")
#     except Exception as e:
#         logging.error(f"FATAL: Failed to load data. Error: {e}. Exiting.")
#         sys.exit(1)

#     # ---------------------------------------------------------------------
#     ## 🎯 Step 1: Identify Common Basins and Filter All Inputs (9 DataFrames)
#     # ---------------------------------------------------------------------
    
#     # 1. Collect all 9 input DataFrames
#     # Use the original names for the DataFrames loaded by load_all_data()
#     all_dfs: Dict[str, pd.DataFrame] = {
#         'Rainf': Rainf_df, 'PotEvap': PotEvap_df, 'Evap': Evap_df, 
#         'Q_USGS': Q_USGS_monthly, 'Qsb': Qsb_monthly, 'Q_nldas_mm': Q_nldas_mm_monthly, 
#         'S_init': S_init_df, 'G_init': G_init_df, 
#         'M': M_df, 'Slope': Slope_df 
#     }
    
#     # 2. Find the intersection of columns (Basin IDs) across all 9 DataFrames
#     all_column_sets = [set(df.columns) for df in all_dfs.values()]

#     if not all_column_sets:
#         logging.error("FATAL: No dataframes loaded.")
#         sys.exit(1)

#     # Calculate intersection and filter out non-basin ID columns
#     common_columns = all_column_sets[0].intersection(*all_column_sets[1:])
#     TARGET_BASINS = sorted([str(c).zfill(8) for c in common_columns if str(c).lower() not in ['time', 'date', 'index']])

#     logging.info(f"----------✅ Found **{len(TARGET_BASINS)}** common basins across all 9 input files.")
    
#     # 3. Filter all 9 DataFrames down to only common columns (Basin IDs)
#     # The dictionary elements are updated with the filtered content
#     for name, df in all_dfs.items():
#         cols_to_keep = [c for c in df.columns if str(c).zfill(8) in TARGET_BASINS or str(c).lower() in ['time', 'date']]
#         all_dfs[name] = df[cols_to_keep] # Update the dictionary value with the filtered DataFrame
    
#     # NOTE: The original variables (Rainf_df, PotEvap_df, etc.) are still bound to the initial, larger DataFrames. 
#     # The 'all_dfs' dictionary holds the correct, filtered DataFrames, which are passed to the partial function.

#     # ---------------------------------------------------------------------
#     ## 🎯 Step 2: Run Parallel Simulation
#     # ---------------------------------------------------------------------

#     cfg = EnKFConfig()
#     if cfg.state_dim < 7:
#         cfg.state_dim = 7
#     _ensure_cfg_defaults(cfg)

#     try:
#         all_calibrated_params = load_all_calibrated_params()
#     except Exception as e:
#         logging.error(f"FATAL: Could not load calibrated parameters. Error: {e}. Exiting.")
#         sys.exit(1)

#     results_timeseries = {}
#     results_metrics = {k: {} for k in ['Q_ET_DA_ET_KGE', 'Q_ET_DA_Q_KGE', 'Q_B_ET_KGE',
#                                       'Q_Base_ET_KGE', 'Q_Base_Q_KGE', 'Q_B_Q_KGE']}

#     logging.info(f"----------✅ Starting EnKF simulation for **{len(TARGET_BASINS)}** basins (parallel run)...")

#     # Pass the filtered DataFrames from the 'all_dfs' dictionary to the partial function
#     process_basin_partial = partial(
#         process_basin,
#         all_calibrated_params=all_calibrated_params,
#         Rainf_df=all_dfs['Rainf'], PotEvap_df=all_dfs['PotEvap'], Evap_df=all_dfs['Evap'],
#         Q_USGS_monthly=all_dfs['Q_USGS'], Qsb_monthly=all_dfs['Qsb'], 
#         Q_nldas_mm_monthly=all_dfs['Q_nldas_mm'],
#         S_init_df=all_dfs['S_init'], G_init_df=all_dfs['G_init'],
#         M_df=all_dfs['M'], Slope_df=all_dfs['Slope'], 
#         cfg=cfg
#     )

#     n_cpu = max(1, cpu_count() - 1)
#     with Pool(n_cpu) as pool:
#         results = list(tqdm(pool.imap(process_basin_partial, TARGET_BASINS), total=len(TARGET_BASINS)))

#     # Unpack results
#     for basin_id, df, metrics in results:
#         if df is not None:
#             results_timeseries[basin_id] = df
#         for k, v in metrics.items():
#             results_metrics[k][basin_id] = v

#     # ---------------------------------------------------------------------
#     ## 🎯 Step 3: Save Results
#     # ---------------------------------------------------------------------
#     logging.info("📝 Saving simulation results...")
#     out_dir = os.path.join(PROJECT_ROOT, 'Simulation_results')
#     os.makedirs(out_dir, exist_ok=True)
#     ts_dir = os.path.join(out_dir, 'enkf_timeseries')
#     os.makedirs(ts_dir, exist_ok=True)

#     for basin_id, df in results_timeseries.items():
#         df.reset_index().to_feather(os.path.join(ts_dir, f"{basin_id}_enkf_timeseries.feather"))

#     df_metrics = pd.DataFrame(results_metrics)

#     df_metrics.index.name = 'gauge_id'
#     df_metrics = df_metrics.reset_index()
#     df_metrics.to_csv(os.path.join(out_dir, 'enkf_performance_metrics.csv'), 
#                       float_format='%.4f', 
#                       index=False)
    
#     logging.info(f"✅ Simulation complete. Timeseries saved to {ts_dir}, metrics saved to {out_dir}")






# # =====================================================================
# # MAIN EXECUTION BLOCK (Updated for all 9 DataFrames)
# # =====================================================================
# if __name__ == '__main__':
#     logging.info("----------✅ Loading all 9 input dataframes...")
#     try:
#         # Load all 9 time-series and parameter dataframes
#         Rainf_df, PotEvap_df, Evap_df, Q_USGS_monthly, Qsb_monthly, Q_nldas_mm_monthly, S_init_df, G_init_df, M_df, Slope_df = load_all_data()
        
#         # Check essential files
#         if Rainf_df.empty or Q_USGS_monthly.empty or M_df.empty or Slope_df.empty:
#             raise ValueError("Essential input data (Rainf, Q_USGS, M, or Slope) is empty. Check file paths and content.")
#         logging.info("----------✅ Data loading complete.")
#     except Exception as e:
#         logging.error(f"FATAL: Failed to load data. Error: {e}. Exiting.")
#         sys.exit(1)

#     # ---------------------------------------------------------------------
#     ## 🎯 Step 1: Identify Common Basins and Filter All Inputs (9 DataFrames)
#     # ---------------------------------------------------------------------
    
#     # 1. Collect all 9 input DataFrames
#     all_dfs: Dict[str, pd.DataFrame] = {
#         'Rainf': Rainf_df, 'PotEvap': PotEvap_df, 'Evap': Evap_df, 
#         'Q_USGS': Q_USGS_monthly, 'Qsb': Qsb_monthly, 'Q_nldas_mm': Q_nldas_mm_monthly, 
#         'S_init': S_init_df, 'G_init': G_init_df, 
#         'M': M_df, 'Slope': Slope_df # NEW
#     }
    
#     # 2. Find the intersection of columns (Basin IDs) across all 9 DataFrames
#     all_column_sets = [set(df.columns) for df in all_dfs.values()]

#     if not all_column_sets:
#         logging.error("FATAL: No dataframes loaded.")
#         sys.exit(1)

#     # Calculate intersection and filter out non-basin ID columns
#     common_columns = all_column_sets[0].intersection(*all_column_sets[1:])
#     TARGET_BASINS = sorted([str(c).zfill(8) for c in common_columns if str(c).lower() not in ['time', 'date', 'index']])

#     logging.info(f"----------✅ Found **{len(TARGET_BASINS)}** common basins across all 9 input files.")
    
#     # 3. Filter all 9 DataFrames down to only common columns (Basin IDs)
#     filtered_dfs = {}
#     for name, df in all_dfs.items():
#         # Identify columns to keep: those whose padded name is in TARGET_BASINS OR are index columns
#         cols_to_keep = [c for c in df.columns if str(c).zfill(8) in TARGET_BASINS or str(c).lower() in ['time', 'date']]
#         filtered_dfs[name] = df[cols_to_keep]
    
#     # Re-assign filtered DataFrames for clarity
#     Rainf_df = filtered_dfs['Rainf']
#     PotEvap_df = filtered_dfs['PotEvap']
#     Evap_df = filtered_dfs['Evap']
#     Q_USGS_monthly = filtered_dfs['Q_USGS']
#     Qsb_monthly = filtered_dfs['Qsb']
#     Q_nldas_mm_monthly = filtered_dfs['Q_nldas_mm']
#     S_init_df = filtered_dfs['S_init']
#     G_init_df = filtered_dfs['G_init']
#     M_df = filtered_dfs['M']
#     Slope_df = filtered_dfs['Slope']


#     # ---------------------------------------------------------------------
#     ## 🎯 Step 2: Run Parallel Simulation
#     # ---------------------------------------------------------------------

#     cfg = EnKFConfig()
#     if cfg.state_dim < 7:
#         cfg.state_dim = 7
#     _ensure_cfg_defaults(cfg)

#     try:
#         all_calibrated_params = load_all_calibrated_params()
#     except Exception as e:
#         logging.error(f"FATAL: Could not load calibrated parameters. Error: {e}. Exiting.")
#         sys.exit(1)

#     results_timeseries = {}
#     results_metrics = {k: {} for k in ['Q_ET_DA_ET_KGE', 'Q_ET_DA_Q_KGE', 'Q_B_ET_KGE',
#                                       'Q_Base_ET_KGE', 'Q_Base_Q_KGE', 'Q_B_Q_KGE']}

#     logging.info(f"----------✅ Starting EnKF simulation for **{len(TARGET_BASINS)}** basins (parallel run)...")

#     # Update partial function with M_df and Slope_df
#     process_basin_partial = partial(
#         process_basin,
#         all_calibrated_params=all_calibrated_params,
#         Rainf_df=Rainf_df, PotEvap_df=PotEvap_df, Evap_df=Evap_df,
#         Q_USGS_monthly=Q_USGS_monthly, Qsb_monthly=Qsb_monthly, 
#         Q_nldas_mm_monthly=Q_nldas_mm_monthly,
#         S_init_df=S_init_df, G_init_df=G_init_df,
#         M_df=M_df, Slope_df=Slope_df, # NEW
#         cfg=cfg
#     )

#     n_cpu = max(1, cpu_count() - 1)
#     with Pool(n_cpu) as pool:
#         results = list(tqdm(pool.imap(process_basin_partial, TARGET_BASINS), total=len(TARGET_BASINS)))

#     # Unpack results
#     for basin_id, df, metrics in results:
#         if df is not None:
#             results_timeseries[basin_id] = df
#         for k, v in metrics.items():
#             results_metrics[k][basin_id] = v

#     # ---------------------------------------------------------------------
#     ## 🎯 Step 3: Save Results
#     # ---------------------------------------------------------------------
#     logging.info("📝 Saving simulation results...")
#     out_dir = os.path.join(PROJECT_ROOT, 'Simulation_results')
#     os.makedirs(out_dir, exist_ok=True)
#     ts_dir = os.path.join(out_dir, 'enkf_timeseries')
#     os.makedirs(ts_dir, exist_ok=True)

#     for basin_id, df in results_timeseries.items():
#         df.reset_index().to_feather(os.path.join(ts_dir, f"{basin_id}_enkf_timeseries.feather"))

#     df_metrics = pd.DataFrame(results_metrics)

#     df_metrics.index.name = 'gauge_id'
#     df_metrics = df_metrics.reset_index()
#     df_metrics.to_csv(os.path.join(out_dir, 'enkf_performance_metrics.csv'), 
#                        float_format='%.4f', 
#                        index=False)
    
#     logging.info(f"✅ Simulation complete. Timeseries saved to {ts_dir}, metrics saved to {out_dir}")






import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any, List
import geopandas as gpd 
from multiprocessing import Pool, cpu_count
import warnings
from tqdm import tqdm
import logging
from functools import partial

# Ignore specific RuntimeWarnings for numerical stability
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.seterr(all="ignore")

# --- GLOBAL CONSTANTS & PATHS ---
PROJECT_ROOT = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\DA\DA_Github_repo\Bayesian_DA_Budyko_modeling"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
INPUT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "input_data", "NDVI") # New directory for M and Slope
sys.path.append(PROJECT_ROOT)
G_MAX_CEILING = 5000.0

# =====================================================================
# CORE MODULE IMPORTS (fall back to safe defaults if import fails)
# =====================================================================
# (Fallback definitions for EnKFConfig, enkf_update, etc., are kept here
# to ensure the script is runnable even without the 'src' modules.)
try:
    from src.model import ModelParams, two_store_model_step
    from src.enkf import EnKFConfig, enkf_update, enkf_forecast_step
    from src.param_manager import get_calibrated_params_for_basin, load_all_calibrated_params
    from src.metrics import calculate_kge
    from src.budyko import estimate_budyko_et, OmegaMLRModel
except ImportError as e:
    print(f"FATAL: CORE MODULE Import failed. Using Fallbacks. Error: {e}")
    
    class EnKFConfig:
        def __init__(self):
            self.state_dim = 7
            self.nens = 50
            self.R_ET = 0.5
            self.R_Q = 1.0
            self.R_B = 0.5
            self.inflation = 1.05
            self.param_perturb_frac = 0.05
            self.min_param_sd = 1e-4

    class ModelParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def enkf_update(X, y_obs, y_ens, R, inflation):
        X = X.copy()
        nx, nens = X.shape
        y_ens = np.array(y_ens).reshape((nens,))
        X_mean = X.mean(axis=1, keepdims=True)
        X_ano = X - X_mean
        y_mean = y_ens.mean()
        y_ano = y_ens - y_mean
        C_xo = (X_ano @ y_ano.reshape(1, -1).T) / (nens - 1)
        C_oo = (y_ano @ y_ano.T) / (nens - 1)
        # Note: The fallback implementation for K calculation might need review for edge cases, 
        # but is kept here for completeness based on the provided snippet.
        K = C_xo / (C_oo + R + 1e-12) 
        y_obs_pert = np.random.normal(loc=y_obs, scale=np.sqrt(R), size=nens)
        X = X + K @ (y_obs_pert.reshape(1, -1) - y_ens.reshape(1, -1))
        X_mean_post = X.mean(axis=1, keepdims=True)
        X = X_mean_post + inflation * (X - X_mean_post)
        return X

    def enkf_forecast_step(X, P_t, PET_t, CALIBRATED_SMAX, ET_B_t=None):
        nens = X.shape[1]
        S = X[0, :]
        G = X[1, :]
        bias = X[6, :]
        # Simplified forecast model for fallback
        Q_ens = np.maximum(0.0, 0.1 * (S + G) + bias + 0.01 * np.random.randn(nens))
        Ke = X[4, :]
        ET_ens = np.maximum(0.0, Ke * (PET_t if PET_t is not None else 0.0) + 0.01 * np.random.randn(nens))
        if ET_B_t is not None and not np.isnan(ET_B_t):
            ET_ens = ET_ens * 0.5 + (ET_B_t * 0.5)
        X_next = X.copy()
        return X_next, Q_ens, ET_ens

    def get_calibrated_params_for_basin(basin_id):
        return {'Kperc': 0.2, 'Kb': 0.05, 'Ke': 0.6, 'Cqq': 0.3, 'bias': 0.0, 'Smax': 20.0}

    def load_all_calibrated_params():
        return {}

    def calculate_kge(obs, sim):
        if len(obs) == 0:
            return np.nan
        # Simplified KGE calculation (just correlation) for fallback
        return float(np.corrcoef(obs, sim)[0, 1]) 

    def estimate_budyko_et(P, PET, M, Slope, omega_model, model='Fu'):
        with np.errstate(divide='ignore', invalid='ignore'):
            # Fallback estimation (still needs the inputs)
            phi = PET / (P + 1e-12)
            frac = P / (P + PET + 1e-12)
            return frac * PET
    
    class OmegaMLRModel:
        def __init__(self, beta0, beta1, beta2):
            pass

# =====================================================================
# LOGGING, IO, and ENKF HELPER FUNCTIONS
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_feather_df(fname: str, ddir: str) -> pd.DataFrame:
    """Loads a Feather file from a specified directory."""
    path = os.path.join(ddir, fname)
    if not os.path.exists(path):
        # Allow Budyko files to fail gracefully if the input_data directory is wrong, but log it
        if ddir == INPUT_DATA_DIR:
             logging.error(f"FATAL: Budyko data file not found: {path}. Check your path!")
        else:
            logging.warning(f"File not found: {path}. Returning empty DataFrame.")
        return pd.DataFrame()
    df = pd.read_feather(path)
    df = df.dropna(axis=1, how='all')
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    return df

def load_all_data():
    """Loads all 9 input dataframes."""
    # Data DataFrames (from DATA_DIR)
    Rainf_df = load_feather_df("Rainf.feather", DATA_DIR)
    PotEvap_df = load_feather_df("PotEvap.feather", DATA_DIR)
    Evap_df = load_feather_df("EVap.feather", DATA_DIR)
    Q_USGS_monthly = load_feather_df("Q_USGS.feather", DATA_DIR)
    Qsb_monthly = load_feather_df("Qsb.feather", DATA_DIR)
    Q_nldas_mm_monthly = load_feather_df("Q_nldas_mm_monthly.feather", DATA_DIR)
    S_init_df = load_feather_df("RootMoist.feather", DATA_DIR)
    G_init_df = load_feather_df("SoilM_0_200cm.feather", DATA_DIR)
    
    # Budyko Parameter DataFrames (from INPUT_DATA_DIR)
    M_df = load_feather_df("M.feather", INPUT_DATA_DIR)
    Slope_df = load_feather_df("slope.feather", INPUT_DATA_DIR) # Single-row DataFrame
    
    return Rainf_df, PotEvap_df, Evap_df, Q_USGS_monthly, Qsb_monthly, Q_nldas_mm_monthly, \
           S_init_df, G_init_df, M_df, Slope_df

def _extract_column_key(df: pd.DataFrame, basin_id: str) -> str:
    cleaned_id = str(basin_id).zfill(8)
    if cleaned_id in df.columns:
        return cleaned_id
    unpadded_id = str(basin_id).lstrip('0')
    if unpadded_id in df.columns:
        return unpadded_id
    # Final robust check for partial matches (less ideal but safer for mixed formats)
    for col in df.columns:
        col_str = str(col)
        if col_str.endswith(cleaned_id) or cleaned_id in col_str:
            return col
    # NOTE: Since pre-filtering is done, this should theoretically not be hit for TARGET_BASINS
    raise KeyError(f"Column for basin {cleaned_id} not found.")

def _ensure_cfg_defaults(cfg):
    if not hasattr(cfg, 'R_Q') or cfg.R_Q is None:
        cfg.R_Q = 1.0
    if not hasattr(cfg, 'R_ET') or cfg.R_ET is None:
        cfg.R_ET = 0.5
    if not hasattr(cfg, 'R_B') or cfg.R_B is None:
        cfg.R_B = 0.5
    if not hasattr(cfg, 'inflation') or cfg.inflation is None:
        cfg.inflation = 1.05
    if not hasattr(cfg, 'param_perturb_frac') or cfg.param_perturb_frac is None:
        cfg.param_perturb_frac = 0.05
    if not hasattr(cfg, 'min_param_sd'):
        cfg.min_param_sd = 1e-4
    if not hasattr(cfg, 'nens') or cfg.nens is None:
        cfg.nens = 50

def _apply_param_spread(X, param_indices, cal_params, cfg):
    param_names = ['Kperc', 'Kb', 'Ke', 'Cqq', 'bias']
    for i, param_name in enumerate(param_names):
        idx = param_indices[i]
        base = cal_params.get(param_name, 0.0)
        sd = max(cfg.param_perturb_frac * max(abs(base), 1.0), cfg.min_param_sd)
        X[idx, :] = np.random.normal(loc=base, scale=sd, size=X.shape[1])
        X[idx, :] = np.where(X[idx, :] < 0.0, 0.0, X[idx, :])

def _adaptive_inflation_and_jitter(X, inflation, var_floor=1e-8, jitter_scale=1e-6):
    var = np.var(X, axis=1, ddof=1)
    # If variance is low, inflate ensemble spread around the mean
    low_var_mask = var < var_floor
    if np.any(low_var_mask):
        X_mean = X.mean(axis=1, keepdims=True)
        X = X_mean + inflation * (X - X_mean)
    # Add minor jitter to prevent collapse
    X += np.random.normal(loc=0.0, scale=jitter_scale, size=X.shape)
    return X

def run_enkf_scenario(P_monthly, PET_monthly, Q_USGS_monthly, Qsb_monthly, ET_B_monthly, ET_NLDAS_monthly,
                      Q_nldas_monthly_data_array, # Renamed to reflect it's the array, not the DataFrame
                      scenario, cfg, target_basin, S_init_nldas, G_init_nldas, CALIBRATED_SMAX,
                      all_calibrated_params):
    
    _ensure_cfg_defaults(cfg)
    nmonths = len(P_monthly)
    cal_params = all_calibrated_params.get(target_basin)
    if not cal_params:
        cal_params = get_calibrated_params_for_basin(target_basin)
    if not cal_params:
        return None

    TOTAL_DIM = 7
    X = np.zeros((TOTAL_DIM, cfg.nens))
    X[0, :] = np.clip(np.random.normal(S_init_nldas, 0.1, cfg.nens), 0.25, CALIBRATED_SMAX)
    X[1, :] = np.clip(np.random.normal(G_init_nldas, 0.1, cfg.nens), 0.2, G_MAX_CEILING)

    try:
        # Indices 2, 3, 4, 5, 6 for Kperc, Kb, Ke, Cqq, bias parameters
        _apply_param_spread(X, param_indices=[2, 3, 4, 5, 6], cal_params=cal_params, cfg=cfg)
    except Exception:
        # Fallback if spread fails
        X[2, :] = cal_params.get('Kperc', 0.0)
        X[3, :] = cal_params.get('Kb', 0.0)
        X[4, :] = cal_params.get('Ke', 0.0)
        X[5, :] = cal_params.get('Cqq', 0.0)
        X[6, :] = cal_params.get('bias', 0.0)

    Q_mean = np.zeros(nmonths)
    ET_mean = np.zeros(nmonths)
    S_mean = np.zeros(nmonths)
    G_mean = np.zeros(nmonths)
    X_t = X.copy()

    for t in range(nmonths):
        ET_B_t_forecast_input = ET_B_monthly[t] if scenario == 'Q_B' else None

        X_next, Q_ens, ET_ens = enkf_forecast_step(
            X_t, P_monthly[t], PET_monthly[t], CALIBRATED_SMAX,
            ET_B_t_forecast_input
        )

        X_assimilated = X_next.copy()

        # A. Assimilation for ET (NLDAS) - Only in Q_ET_DA scenario
        if scenario == 'Q_ET_DA':
            ET_obs = ET_NLDAS_monthly[t]
            if not np.isnan(ET_obs) and ET_obs > 0:
                Rb = getattr(cfg, 'R_ET', 0.5)
                et_obs_pert = float(ET_obs) + np.random.normal(0.0, np.sqrt(max(Rb, 1e-12)))
                try:
                    X_assimilated = enkf_update(X_assimilated, et_obs_pert, ET_ens, Rb, cfg.inflation)
                except Exception as e:
                    logging.debug(f"Warning: ET assimilation failed at t={t}: {e}")

        # B. Assimilation for Budyko Pseudo-Observation (ET_B) - Only in Q_B scenario
        if scenario == 'Q_B':
            ET_B_t_obs = ET_B_monthly[t]
            if not np.isnan(ET_B_t_obs) and ET_B_t_obs > 0:
                Rb = getattr(cfg, 'R_B', 0.5)
                etb_pert = float(ET_B_t_obs) + np.random.normal(0.0, np.sqrt(max(Rb, 1e-12)))
                try:
                    X_assimilated = enkf_update(X_assimilated, etb_pert, ET_ens, Rb, cfg.inflation)
                except Exception as e:
                    logging.debug(f"Warning: Budyko pseudo-assimilation failed at t={t}: {e}")

        # C. Assimilation for Streamflow (Q_NLDAS) - Runs for ALL scenarios (Q_Base, Q_B, Q_ET_DA)
        Q_obs = Q_nldas_monthly_data_array[t] # Using the renamed data array parameter
        if not np.isnan(Q_obs) and Q_obs > 0:
            Rq = getattr(cfg, 'R_Q', 1.0)
            q_obs_pert = float(Q_obs) + np.random.normal(0.0, np.sqrt(max(Rq, 1e-12)))
            try:
                X_assimilated = enkf_update(X_assimilated, q_obs_pert, Q_ens, Rq, cfg.inflation)
            except Exception as e:
                logging.debug(f"Warning: Q assimilation failed at t={t}: {e}")

        X_t = X_assimilated.copy()

        # State constraints
        X_t[0, :] = np.clip(X_t[0, :], 0.23, CALIBRATED_SMAX)
        X_t[1, :] = np.clip(X_t[1, :], 0.24, G_MAX_CEILING)

        try:
            X_t = _adaptive_inflation_and_jitter(X_t, inflation=cfg.inflation, var_floor=1e-8, jitter_scale=1e-6)
        except Exception as e:
            logging.debug(f"Adaptive inflation/jitter failed at t={t}: {e}")

        Q_mean[t] = np.mean(Q_ens)
        ET_mean[t] = np.mean(ET_ens)
        S_mean[t] = np.mean(X_t[0, :])
        G_mean[t] = np.mean(X_t[1, :])

    return {'Q_mean': Q_mean, 'ET_mean': ET_mean, 'S_mean': S_mean, 'G_mean': G_mean, 'X_final': X_t}


# =====================================================================
# PROCESS BASIN (Cleaned: No reassignment of Q_nldas_mm_monthly_data)
# =====================================================================
def process_basin(
    basin: str, all_calibrated_params: Dict[str, ModelParams],
    Rainf_df: pd.DataFrame, PotEvap_df: pd.DataFrame, Evap_df: pd.DataFrame,
    Q_USGS_monthly: pd.DataFrame, Q_nldas_mm_monthly: pd.DataFrame, Qsb_monthly: pd.DataFrame,
    S_init_df: pd.DataFrame, G_init_df: pd.DataFrame,
    M_df: pd.DataFrame, Slope_df: pd.DataFrame, # NEW BUDYKO DATA
    cfg: EnKFConfig
) -> Tuple[str, Optional[pd.DataFrame], Optional[Dict[str, float]]]:

    basin_id = str(basin).zfill(8)
    basin_metrics = {
        'Q_Base_Q_KGE': np.nan, 'Q_B_Q_KGE': np.nan, 'Q_ET_DA_Q_KGE': np.nan,
        'Q_Base_ET_KGE': np.nan, 'Q_B_ET_KGE': np.nan, 'Q_ET_DA_ET_KGE': np.nan
    }
    dates = Q_USGS_monthly.index

    try:
        # --- 1. Data Extraction (Input Time Series) ---
        P_key = _extract_column_key(Rainf_df, basin_id)
        PET_key = _extract_column_key(PotEvap_df, basin_id)
        Q_obs_key = _extract_column_key(Q_USGS_monthly, basin_id)
        ET_key = _extract_column_key(Evap_df, basin_id)
        Q_nldas_key = _extract_column_key(Q_nldas_mm_monthly, basin_id)
        Qsb_key = _extract_column_key(Qsb_monthly, basin_id) 
        S_init_key = _extract_column_key(S_init_df, basin_id)
        G_init_key = _extract_column_key(G_init_df, basin_id)
        
        P_monthly = np.clip(Rainf_df[P_key].values, 0.0, np.inf)
        PET_monthly = np.clip(PotEvap_df[PET_key].values, 0.0, np.inf)

        Q_obs_USGS_eval = np.clip(Q_USGS_monthly[Q_obs_key].values, 0.0, np.inf)
        ET_NLDAS_monthly = np.clip(Evap_df[ET_key].values, 0.0, np.inf)
        
        # Extract the necessary NumPy arrays directly
        Q_nldas_monthly_data_array = Q_nldas_mm_monthly[Q_nldas_key].values 
        Qsb_monthly_data = Qsb_monthly[Qsb_key].values 

        cal_params = all_calibrated_params.get(basin_id)
        CALIBRATED_SMAX = cal_params['Smax'] if cal_params and 'Smax' in cal_params else 20.0

        S_init_nldas = np.nanmean(S_init_df[S_init_key].values[-12:])
        G_init_nldas = np.nanmean(G_init_df[G_init_key].values[-12:])

        # --- 2. Budyko Parameter Extraction (Actual Data) ---
        M_key = _extract_column_key(M_df, basin_id)
        Slope_key = _extract_column_key(Slope_df, basin_id)
        
        # M_basin is a time series
        M_basin = M_df[M_key].values 
        
        # Slope_basin is a single value from the first row of Slope_df
        if Slope_df.empty:
             raise ValueError("Slope data is empty.")
        Slope_basin = float(Slope_df[Slope_key].iloc[0]) 

        # --- 3. Budyko Calculation ---
        omega_model = OmegaMLRModel(beta0=2.36, beta1=1.16, beta2=0.0)

        ET_B_monthly = np.array([
            estimate_budyko_et(P_monthly[t], PET_monthly[t], M_basin[t], Slope_basin, omega_model, model='Fu')
            for t in range(len(P_monthly))
        ])

        # --- 4. Simulation Execution ---
        if np.all(np.isnan(P_monthly)) or np.all(np.isnan(PET_monthly)) or np.all(np.isnan(Q_obs_USGS_eval)):
            return basin_id, None, basin_metrics

        basin_results_dfs = {}
        spin_up = 60

        for scenario in ['Q_Base', 'Q_B', 'Q_ET_DA']:
            result = run_enkf_scenario(
                P_monthly, PET_monthly, Q_obs_USGS_eval, Qsb_monthly_data,
                ET_B_monthly, ET_NLDAS_monthly,
                Q_nldas_monthly_data_array, # Using the direct array variable
                scenario, cfg, basin_id, S_init_nldas, G_init_nldas, CALIBRATED_SMAX,
                all_calibrated_params
            )
            
            if result is None:
                logging.warning(f"Skipping basin {basin_id} in scenario {scenario}: Missing calibrated parameters or failed setup.")
                continue

            # Metrics Calculation
            Q_sim, Q_obs = result['Q_mean'][spin_up:], Q_obs_USGS_eval[spin_up:]
            ET_sim, ET_obs = result['ET_mean'][spin_up:], ET_NLDAS_monthly[spin_up:]
            
            valid_mask_Q = ~(np.isnan(Q_sim) | np.isnan(Q_obs))
            if np.any(valid_mask_Q) and len(Q_obs[valid_mask_Q]) >= 12:
                basin_metrics[f"{scenario}_Q_KGE"] = calculate_kge(Q_obs[valid_mask_Q], Q_sim[valid_mask_Q])

            valid_mask_ET = ~(np.isnan(ET_sim) | np.isnan(ET_obs))
            if np.any(valid_mask_ET) and len(ET_obs[valid_mask_ET]) >= 12:
                basin_metrics[f"{scenario}_ET_KGE"] = calculate_kge(ET_obs[valid_mask_ET], ET_sim[valid_mask_ET])

            df_scenario = pd.DataFrame({
                f'{scenario}_Q_sim': result['Q_mean'],
                f'{scenario}_ET_sim': result['ET_mean'],
                f'{scenario}_S_sim': result['S_mean'],
                f'{scenario}_G_sim': result['G_mean'],
            }, index=dates)
            basin_results_dfs[scenario] = df_scenario
            
        if not basin_results_dfs:
            return basin_id, None, basin_metrics

        # --- 5. Final DataFrame Assembly and Return ---
        df_obs_inputs = pd.DataFrame({
            'Date': dates,
            'P_input': P_monthly,
            'PET_input': PET_monthly,
            'Q_USGS_obs': Q_obs_USGS_eval,
            'Q_NLDAS_input': Q_nldas_monthly_data_array, # Using the direct array variable for the final DF
            'ET_NLDAS_obs': ET_NLDAS_monthly,
            'ET_Budyko': ET_B_monthly
        }).set_index('Date')

        df_final_basin = df_obs_inputs
        for df_scenario in basin_results_dfs.values():
            df_final_basin = df_final_basin.join(df_scenario, how='left')

        return basin_id, df_final_basin, basin_metrics

    except KeyError as e:
        logging.error(f"CRITICAL ERROR (KeyError) processing basin {basin_id}: Data column lookup failed: {e}. Skipping basin.")
        return basin_id, None, basin_metrics
    except Exception as e:
        logging.error(f"CRITICAL ERROR processing basin {basin_id}: {e}. Skipping basin.")
        return basin_id, None, basin_metrics


# =====================================================================
# MAIN EXECUTION BLOCK 
# =====================================================================
if __name__ == '__main__':
    logging.info("----------✅ Loading all 9 input dataframes...")
    try:
        # Load all 9 time-series and parameter dataframes using their original names
        Rainf_df, PotEvap_df, Evap_df, Q_USGS_monthly, Qsb_monthly, Q_nldas_mm_monthly, S_init_df, G_init_df, M_df, Slope_df = load_all_data()
        
        # Check essential files
        if Rainf_df.empty or Q_USGS_monthly.empty or M_df.empty or Slope_df.empty:
            raise ValueError("Essential input data (Rainf, Q_USGS, M, or Slope) is empty. Check file paths and content.")
        logging.info("----------✅ Data loading complete.")
    except Exception as e:
        logging.error(f"FATAL: Failed to load data. Error: {e}. Exiting.")
        sys.exit(1)

    # ---------------------------------------------------------------------
    ## 🎯 Step 1: Identify Common Basins and Filter All Inputs (9 DataFrames)
    # ---------------------------------------------------------------------
    
    # 1. Collect all 9 input DataFrames
    all_dfs: Dict[str, pd.DataFrame] = {
        'Rainf': Rainf_df, 'PotEvap': PotEvap_df, 'Evap': Evap_df, 
        'Q_USGS': Q_USGS_monthly, 'Qsb': Qsb_monthly, 'Q_nldas_mm': Q_nldas_mm_monthly, 
        'S_init': S_init_df, 'G_init': G_init_df, 
        'M': M_df, 'Slope': Slope_df 
    }
    
    # 2. Find the intersection of columns (Basin IDs) across all 9 DataFrames
    all_column_sets = [set(df.columns) for df in all_dfs.values()]

    if not all_column_sets:
        logging.error("FATAL: No dataframes loaded.")
        sys.exit(1)

    # Calculate intersection and filter out non-basin ID columns
    common_columns = all_column_sets[0].intersection(*all_column_sets[1:])
    TARGET_BASINS = sorted([str(c).zfill(8) for c in common_columns if str(c).lower() not in ['time', 'date', 'index']])

    logging.info(f"----------✅ Found **{len(TARGET_BASINS)}** common basins across all 9 input files.")
    
    # 3. Filter all 9 DataFrames down to only common columns (Basin IDs)
    filtered_dfs = {}
    for name, df in all_dfs.items():
        cols_to_keep = [c for c in df.columns if str(c).zfill(8) in TARGET_BASINS or str(c).lower() in ['time', 'date']]
        filtered_dfs[name] = df[cols_to_keep] 
    
    # Re-assign filtered DataFrames for use in the partial function
    Rainf_df = filtered_dfs['Rainf']
    PotEvap_df = filtered_dfs['PotEvap']
    Evap_df = filtered_dfs['Evap']
    Q_USGS_monthly = filtered_dfs['Q_USGS']
    Qsb_monthly = filtered_dfs['Qsb']
    Q_nldas_mm_monthly = filtered_dfs['Q_nldas_mm']
    S_init_df = filtered_dfs['S_init']
    G_init_df = filtered_dfs['G_init']
    M_df = filtered_dfs['M']
    Slope_df = filtered_dfs['Slope']


    # ---------------------------------------------------------------------
    ## 🎯 Step 2: Run Parallel Simulation
    # ---------------------------------------------------------------------

    cfg = EnKFConfig()
    if cfg.state_dim < 7:
        cfg.state_dim = 7
    _ensure_cfg_defaults(cfg)

    try:
        all_calibrated_params = load_all_calibrated_params()
    except Exception as e:
        logging.error(f"FATAL: Could not load calibrated parameters. Error: {e}. Exiting.")
        sys.exit(1)

    results_timeseries = {}
    results_metrics = {k: {} for k in ['Q_ET_DA_ET_KGE', 'Q_ET_DA_Q_KGE', 'Q_B_ET_KGE',
                                      'Q_Base_ET_KGE', 'Q_Base_Q_KGE', 'Q_B_Q_KGE']}

    logging.info(f"----------✅ Starting EnKF simulation for **{len(TARGET_BASINS)}** basins (parallel run)...")

    # Pass the filtered DataFrames to the partial function
    process_basin_partial = partial(
        process_basin,
        all_calibrated_params=all_calibrated_params,
        Rainf_df=Rainf_df, PotEvap_df=PotEvap_df, Evap_df=Evap_df,
        Q_USGS_monthly=Q_USGS_monthly, Qsb_monthly=Qsb_monthly, 
        Q_nldas_mm_monthly=Q_nldas_mm_monthly,
        S_init_df=S_init_df, G_init_df=G_init_df,
        M_df=M_df, Slope_df=Slope_df, 
        cfg=cfg
    )

    n_cpu = max(1, cpu_count() - 1)
    with Pool(n_cpu) as pool:
        results = list(tqdm(pool.imap(process_basin_partial, TARGET_BASINS), total=len(TARGET_BASINS)))

    # Unpack results
    for basin_id, df, metrics in results:
        if df is not None:
            results_timeseries[basin_id] = df
        for k, v in metrics.items():
            results_metrics[k][basin_id] = v

    # ---------------------------------------------------------------------
    ## 🎯 Step 3: Save Results
    # ---------------------------------------------------------------------
    logging.info("📝 Saving simulation results...")
    out_dir = os.path.join(PROJECT_ROOT, 'Simulation_results')
    os.makedirs(out_dir, exist_ok=True)
    ts_dir = os.path.join(out_dir, 'enkf_timeseries')
    os.makedirs(ts_dir, exist_ok=True)

    for basin_id, df in results_timeseries.items():
        df.reset_index().to_feather(os.path.join(ts_dir, f"{basin_id}_enkf_timeseries.feather"))

    df_metrics = pd.DataFrame(results_metrics)

    df_metrics.index.name = 'gauge_id'
    df_metrics = df_metrics.reset_index()
    df_metrics.to_csv(os.path.join(out_dir, 'enkf_performance_metrics.csv'), 
                      float_format='%.4f', 
                      index=False)
    
    logging.info(f"✅ Simulation complete. Timeseries saved to {ts_dir}, metrics saved to {out_dir}")