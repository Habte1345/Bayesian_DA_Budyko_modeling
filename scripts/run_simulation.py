# import sys
# import os
# import numpy as np
# import pandas as pd
# from dataclasses import dataclass, field
# from typing import Dict, Optional, Tuple, Any, List
# import requests
# import io
# import geopandas as gpd
# from multiprocessing import Pool, cpu_count 

# # Suppress common NumPy warnings during calculation
# import warnings
# warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(PROJECT_ROOT)

# # --- GLOBAL CONSTANTS ---
# G_MAX_CEILING = 7500.0 # Standard maximum for G_store (3 * S_MAX_CEILING=2500)

# # =====================================================================
# # 🔥 CONSOLIDATED CORE IMPORTS (Cleaned)
# # =====================================================================
# try:
#     # Import core project modules
#     from src.model import ModelParams, two_store_model_step
#     from src.enkf import EnKFConfig, enkf_update, enkf_forecast_step
#     from src.param_manager import get_calibrated_params_for_basin, load_all_calibrated_params 
#     from src.metrics import calculate_kge
    
#     # Corrected: Only load_and_prepare_data is needed from data.data_processor
#     from data.data_processor import load_and_prepare_data 
    
#     from src.budyko import estimate_budyko_et
# except ImportError as e:
#     print(f"FATAL: Import failed. Please ensure all files exist. Error: {e}")
#     sys.exit(1)
    
# # =====================================================================
# # 🔥 CORE SCENARIO RUNNER (Unchanged)
# # =====================================================================

# def run_enkf_scenario(
#     P_monthly: np.ndarray, PET_monthly: np.ndarray, Q_USGS_monthly: np.ndarray, 
#     ET_B_monthly: np.ndarray, ET_NLDAS_monthly: np.ndarray, 
#     scenario: str, cfg: EnKFConfig, target_basin: str,
#     S_init_nldas: float, G_init_nldas: float, CALIBRATED_SMAX: float,
#     all_calibrated_params: Dict[str, ModelParams]
# ) -> Optional[Dict]:
#     """
#     Runs the model for three scenarios (Q_Base, Q_B, and Q_ET_DA).
#     """

#     nmonths = len(P_monthly)
    
#     # 1. Parameter Loading
#     cal_params = all_calibrated_params.get(target_basin)
#     if not cal_params:
#         cal_params = get_calibrated_params_for_basin(target_basin) 

#     if not cal_params:
#         print(f"❌ FATAL: Calibrated parameters not found for {target_basin}. Skipping basin.")
#         return None 
    
#     print(f"✅ Using calibrated parameters for {target_basin}. KGE: {cal_params.get('KGE', 'N/A'):.3f}")

#     CALIBRATED_KPERC = cal_params['Kperc']
#     CALIBRATED_KB = cal_params['Kb']
#     CALIBRATED_KE = cal_params['Ke']
#     CALIBRATED_CQQ = cal_params['Cqq']
#     CALIBRATED_BIAS = cal_params['bias']

#     # 2. Ensemble Initialization
#     X = np.zeros((cfg.state_dim, cfg.nens))
#     S_spread, G_spread = 20.0, 10.0
    
#     # Initial states
#     X[0, :] = np.clip(np.random.normal(S_init_nldas, S_spread, cfg.nens), 1.0, CALIBRATED_SMAX)
#     X[1, :] = np.clip(np.random.normal(G_init_nldas, G_spread, cfg.nens), 1.0, G_MAX_CEILING)
#     # Parameters with spread
#     X[2, :] = np.clip(np.random.normal(CALIBRATED_KPERC, 0.1, cfg.nens), 0.01, 0.999)
#     X[3, :] = np.clip(np.random.normal(CALIBRATED_KB, 0.1, cfg.nens), 0.01, 0.999)
#     X[4, :] = np.clip(np.random.normal(CALIBRATED_KE, 0.1, cfg.nens), 0.01, 0.999)
#     X[5, :] = np.clip(np.random.normal(CALIBRATED_CQQ, 0.1, cfg.nens), 0.01, 0.999) 
#     X[6, :] = np.clip(np.random.normal(CALIBRATED_BIAS, 0.15, cfg.nens), -10, 10)

#     Q_mean, ET_mean, S_mean, G_mean = np.zeros(nmonths), np.zeros(nmonths), np.zeros(nmonths), np.zeros(nmonths)
#     param_rows = slice(2, cfg.state_dim)

#     # 3. Time Loop (Forecast & Analysis)
#     for t in range(nmonths):
#         # Apply parameter random walk and mandatory clipping
#         X[param_rows, :] += cfg.param_rw_sd[param_rows][:, None] * np.random.randn(cfg.state_dim - 2, cfg.nens)
#         X[2, :] = np.clip(X[2, :], 0.01, 0.999)
#         X[3, :] = np.clip(X[3, :], 0.01, 0.999)
#         X[4, :] = np.clip(X[4, :], 0.01, 0.999)
#         X[5, :] = np.clip(X[5, :], 0.01, 0.999) 
#         X[6, :] = np.clip(X[6, :], -10, 10)

#         ET_B_t = ET_B_monthly[t] if scenario == 'Q_B' and not np.isnan(ET_B_monthly[t]) else None
        
#         # Forecast step
#         X_next, Q_ens, ET_ens = enkf_forecast_step(
#             X, P_monthly[t], PET_monthly[t], CALIBRATED_SMAX, ET_B_t
#         )

#         # Analysis step (Data Assimilation)
#         X_updated = X_next 
#         if scenario == 'Q_ET_DA':
#             ET_obs = ET_NLDAS_monthly[t]
#             is_et_available = not np.isnan(ET_obs) and ET_obs > 0
            
#             if is_et_available:
#                 X_updated = enkf_update(X_next, ET_obs, ET_ens, cfg.R_ET, cfg.inflation)

#         # Update ensemble and store means
#         X = X_updated
#         Q_mean[t] = np.mean(Q_ens)
#         ET_mean[t] = np.mean(ET_ens)
#         S_mean[t] = np.mean(X[0, :])
#         G_mean[t] = np.mean(X[1, :])

#     return {
#         'Q_mean': Q_mean,
#         'ET_mean': ET_mean,
#         'S_mean': S_mean,
#         'G_mean': G_mean,
#         'X_final': X
#     }

# # =====================================================================
# # 🛠️ HELPER: Robust Column Key Extraction (Unchanged)
# # =====================================================================

# def _extract_column_key(df: pd.DataFrame, basin_id: str) -> str:
#     """
#     Finds the correct column name in the DataFrame, handling both 
#     plain basin_id (e.g., '07071500') and prefixed columns.
#     """
#     cleaned_id = str(basin_id).zfill(8)
    
#     if cleaned_id in df.columns:
#         return cleaned_id
    
#     for col in df.columns:
#         if str(col).endswith(cleaned_id):
#             return col
            
#     raise KeyError(f"Column for basin {cleaned_id} not found in DataFrame. Columns are: {df.columns.tolist()}")

# # =====================================================================
# # 🚀 Worker Function for Parallel Execution (Modified)
# # =====================================================================

# def process_basin(
#     basin: str, all_calibrated_params: Dict[str, ModelParams], 
#     Rainf_df: pd.DataFrame, PotEvap_df: pd.DataFrame, Evap_df: pd.DataFrame, 
#     Q_USGS_monthly: pd.DataFrame, 
#     Q_nldas_mm_monthly: pd.DataFrame, # <-- ADDED
#     S_init_df: pd.DataFrame, G_init_df: pd.DataFrame, 
#     attrs: gpd.GeoDataFrame, cfg: EnKFConfig
# ) -> Tuple[str, Optional[pd.DataFrame], Optional[Dict[str, float]]]:
    
#     basin_id = str(basin).zfill(8) 
#     # 💡 MODIFICATION: Initialize all 6 KGE metrics
#     basin_metrics = { 
#         'Q_Base_Q_KGE': 0.0, 'Q_B_Q_KGE': 0.0, 'Q_ET_DA_Q_KGE': 0.0,
#         'Q_Base_ET_KGE': 0.0, 'Q_B_ET_KGE': 0.0, 'Q_ET_DA_ET_KGE': 0.0 
#     }

#     try:
#         # --- Load Data for Current Basin ---
#         P_key = _extract_column_key(Rainf_df, basin_id)
#         PET_key = _extract_column_key(PotEvap_df, basin_id)
#         Q_obs_key = _extract_column_key(Q_USGS_monthly, basin_id)
#         ET_key = _extract_column_key(Evap_df, basin_id)
#         Q_nldas_key = _extract_column_key(Q_nldas_mm_monthly, basin_id) # <-- ADDED
#         S_init_key = _extract_column_key(S_init_df, basin_id)
#         G_init_key = _extract_column_key(G_init_df, basin_id)
        
#         P_monthly = Rainf_df[P_key].values
#         PET_monthly = PotEvap_df[PET_key].values 
#         Q_obs_USGS_eval = Q_USGS_monthly[Q_obs_key].values 
#         ET_NLDAS_monthly = Evap_df[ET_key].values 
#         Q_nldas_monthly = Q_nldas_mm_monthly[Q_nldas_key].values # <-- ADDED
        
#         # Get Smax and initial states
#         cal_params = all_calibrated_params.get(basin_id)
#         CALIBRATED_SMAX = cal_params['Smax'] if cal_params and 'Smax' in cal_params else 2500.0

#         S_init_nldas = S_init_df.loc[S_init_df.index[0], S_init_key]
#         G_init_nldas = G_init_df.loc[G_init_df.index[0], G_init_key]
        
#         # Input/Observation preparation (Clipping)
#         Q_obs_USGS_eval = np.clip(Q_obs_USGS_eval, 0.0, np.inf) 
#         P_monthly = np.clip(P_monthly, 0.0, np.inf)
#         PET_monthly = np.clip(PET_monthly, 0.0, np.inf)
#         ET_NLDAS_monthly = np.clip(ET_NLDAS_monthly, 0.0, np.inf)
        
#         # Calculate Budyko ET
#         ET_B_monthly = np.array([estimate_budyko_et(P_monthly[t], PET_monthly[t], model='Fu', m=1.35) 
#                                  for t in range(len(P_monthly))])
        
#         # Critical Check
#         if np.all(np.isnan(P_monthly)) or np.all(np.isnan(PET_monthly)) or np.all(np.isnan(Q_obs_USGS_eval)):
#              return basin_id, None, basin_metrics

#         dates = Q_USGS_monthly.index 
#         basin_results_dfs = {} 
#         last_result = None
#         spin_up = 60 # 5 years

#         # Run the three scenarios
#         for scenario in ['Q_Base', 'Q_B', 'Q_ET_DA']:
#             result = run_enkf_scenario(P_monthly, PET_monthly, Q_obs_USGS_eval, 
#                                        ET_B_monthly, ET_NLDAS_monthly, 
#                                        scenario, cfg, basin_id, S_init_nldas, G_init_nldas, CALIBRATED_SMAX,
#                                        all_calibrated_params)
            
#             if result is None:
#                 return basin_id, None, basin_metrics 

#             last_result = result
            
#             # Store the time series results 
#             df_scenario = pd.DataFrame({
#                 'Date': dates,
#                 f'{scenario}_Q_sim': result['Q_mean'],
#                 f'{scenario}_ET_sim': result['ET_mean'],
#                 f'{scenario}_S_sim': result['S_mean'],
#                 f'{scenario}_G_sim': result['G_mean'],
#             }).set_index('Date')
#             basin_results_dfs[scenario] = df_scenario
            
#             # --- KGE Calculation for Q and ET ---
#             Q_sim, Q_obs = result['Q_mean'][spin_up:], Q_obs_USGS_eval[spin_up:]
#             ET_sim, ET_obs = result['ET_mean'][spin_up:], ET_NLDAS_monthly[spin_up:]
            
#             # KGE for Q
#             valid_mask_Q = ~(np.isnan(Q_sim) | np.isnan(Q_obs))
#             if np.any(valid_mask_Q) and len(Q_obs[valid_mask_Q]) >= 12:
#                 kge_Q = calculate_kge(Q_obs[valid_mask_Q], Q_sim[valid_mask_Q])
#                 basin_metrics[f"{scenario}_Q_KGE"] = kge_Q if not np.isnan(kge_Q) else 0.0
            
#             # KGE for ET
#             valid_mask_ET = ~(np.isnan(ET_sim) | np.isnan(ET_obs))
#             if np.any(valid_mask_ET) and len(ET_obs[valid_mask_ET]) >= 12:
#                 kge_ET = calculate_kge(ET_obs[valid_mask_ET], ET_sim[valid_mask_ET])
#                 basin_metrics[f"{scenario}_ET_KGE"] = kge_ET if not np.isnan(kge_ET) else 0.0
            
#         # Combine and return the final DataFrame
#         if last_result is not None:
#             df_obs_inputs = pd.DataFrame({
#                 'Date': dates,
#                 'P_input': P_monthly,
#                 'PET_input': PET_monthly,
#                 'Q_USGS_obs': Q_obs_USGS_eval,
#                 'Q_NLDAS_input': Q_nldas_monthly, # <-- ADDED TO OUTPUT DF
#                 'ET_NLDAS_obs': ET_NLDAS_monthly, 
#                 'ET_Budyko': ET_B_monthly
#             }).set_index('Date')
            
#             df_final_basin = df_obs_inputs
#             for df_scenario in basin_results_dfs.values():
#                 df_final_basin = df_final_basin.join(df_scenario, how='left')
            
#             return basin_id, df_final_basin, basin_metrics
#         else:
#             return basin_id, None, basin_metrics

#     except KeyError as e:
#         print(f"❌ CRITICAL ERROR (KeyError) processing basin {basin_id}: Missing column key: {e}")
#         return basin_id, None, basin_metrics
#     except Exception as e:
#         print(f"❌ CRITICAL ERROR processing basin {basin_id}: {e}")
#         return basin_id, None, basin_metrics

# # =====================================================================
# # 🔥 MAIN SIMULATION EXECUTION (Modified)
# # =====================================================================

# # Helper function to extract plain ID from prefixed column name (Unchanged)
# def get_basin_id_from_column(col: str) -> str:
#     """Assumes the basin ID is the last 8 characters of the column name."""
#     return str(col).split('_')[-1]

# if __name__ == '__main__':
    
#     print("Initializing data loading...")

#     # Load 11 input data frames
#     try:
#         Rainf_df, PotEvap_df, Evap_df, Qsb_df, M_df, \
#         Slope_df, Q_nldas_mm_monthly, Q_USGS_monthly, \
#         S_init_df, G_init_df, SM_df = load_and_prepare_data() 
        
#         print("✅ Data frames loaded successfully.")

#     except Exception as e:
#         print(f"FATAL: Data loading failed. Error: {e}")
#         sys.exit(1)
        
#     # --- Load CAMELS Attributes (Using direct request/feather for robustness) ---
#     print("Loading CAMELS attributes...")
#     try:
#         r = requests.get("https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/data/contents/camels_attributes_v2.0.feather")
#         attrs = gpd.read_feather(io.BytesIO(r.content)).reset_index(drop=False)
#         attrs['gauge_id'] = attrs['gauge_id'].astype(str).str.strip().str.zfill(8)
#         attrs.set_index('gauge_id', inplace=True)
#         print("✅ CAMELS attributes loaded successfully.")
#     except Exception as e:
#         print(f"❌ FATAL: Failed to load required CAMELS attributes. Error: {e}")
#         sys.exit(1) 
        
#     # 🚀 OPTIMIZATION: Pre-load all calibrated parameters
#     print("Pre-loading all calibrated parameters...")
#     try:
#         all_calibrated_params = load_all_calibrated_params() 
#         if not all_calibrated_params:
#              print("FATAL: No calibrated parameters were loaded. Check 'SCE_cal_params/final_calibrated_params.json'.")
#              sys.exit(1)
#         print(f"✅ Loaded parameters for {len(all_calibrated_params)} basins.")
#     except Exception as e:
#         print(f"❌ WARNING: Failed to pre-load all parameters. Parameter lookup will be slow. Error: {e}")
#         all_calibrated_params = {}

#     # --- Determine Target Basins ---
#     TARGET_BASIN_KEYS = list(Q_USGS_monthly.columns)
#     TARGET_BASINS = [get_basin_id_from_column(key) for key in TARGET_BASIN_KEYS] 

#     cfg = EnKFConfig()
    
#     # --- 🚀 Setup for Parallel Execution ---
    
#     # 1. Prepare Argument List
#     # <-- MODIFIED: Passing Q_nldas_mm_monthly now
#     tasks = [(basin, all_calibrated_params, Rainf_df, PotEvap_df, Evap_df, Q_USGS_monthly, 
#               Q_nldas_mm_monthly, # <-- ADDED ARGUMENT
#               S_init_df, G_init_df, attrs, cfg) for basin in TARGET_BASINS]

#     # 2. Run Parallel Processing
#     NUM_CORES = max(1, cpu_count() - 1) 
#     print(f"\nStarting parallel simulation on **{NUM_CORES} cores** for {len(TARGET_BASINS)} basins...")

#     results_timeseries = {} 
#     # <-- MODIFIED: Initialized with all 6 KGE metrics
#     results_metrics = { 
#         'Q_Base_Q_KGE': {}, 'Q_B_Q_KGE': {}, 'Q_ET_DA_Q_KGE': {},
#         'Q_Base_ET_KGE': {}, 'Q_B_ET_KGE': {}, 'Q_ET_DA_ET_KGE': {}
#     } 
    
#     try:
#         with Pool(NUM_CORES) as pool:
#             parallel_results = pool.starmap(process_basin, tasks)
#     except Exception as e:
#         print(f"❌ FATAL: Parallel pool failed. Error: {e}")
#         sys.exit(1)

#     # 3. Aggregate Results (Serial Step)
#     processed_count = 0
#     for basin_id, df_data, metrics in parallel_results:
#         if df_data is not None:
#             results_timeseries[basin_id] = df_data
#             for kge_key, value in metrics.items():
#                 if kge_key in results_metrics: 
#                     results_metrics[kge_key][basin_id] = value
#             processed_count += 1
#         else:
#             # Basin failed, set all 6 metrics to default 0.0
#             for kge_key in results_metrics.keys():
#                  if basin_id not in results_metrics[kge_key]: 
#                     results_metrics[kge_key][basin_id] = 0.0

#     print("\nParallel processing complete. Saving results...")
    
#     # ---------------------------------------------------------------------
#     # TIME SERIES SAVING SECTION (Unchanged)
#     # ---------------------------------------------------------------------
#     if results_timeseries:
#         output_dir = os.path.join(PROJECT_ROOT, 'Simulation_results', 'enkf_timeseries_by_basin') 
#         os.makedirs(output_dir, exist_ok=True)
        
#         saved_files = []
#         for basin_id, df_data in results_timeseries.items():
#             output_path = os.path.join(output_dir, f'{basin_id}_enkf_timeseries_ET_DA.feather')
#             df_data = df_data.reset_index(names=['Date']) 
#             df_data.to_feather(output_path) 
#             saved_files.append(output_path)
        
#         print(f"\n💾 Saved **{len(saved_files)} individual basin time series files** (as .feather) to the directory:")
#         print(f"**{output_dir}**")

#     # ---------------------------------------------------------------------
#     # OUTPUT METRICS SECTION (Modified)
#     # ---------------------------------------------------------------------
#     print("\n--- Summary Performance Metrics (Q_KGE vs Q_USGS, ET_KGE vs ET_NLDAS, after 5-year spin-up) ---")
#     df_metrics = pd.DataFrame(results_metrics)
    
#     # Filter out basins that failed based on the base streamflow KGE
#     df_metrics_summary = df_metrics[df_metrics['Q_Base_Q_KGE'] != 0.0]
    
#     print(df_metrics_summary.to_string())
    
#     total_basins = len(TARGET_BASINS)
    
#     print(f"\n🏆 SUCCESS: **{processed_count}/{total_basins}** basins processed with valid KGE")


import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any, List
import requests
import io
import geopandas as gpd
from multiprocessing import Pool, cpu_count

# Suppress common NumPy warnings during calculation
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# --- GLOBAL CONSTANTS ---
G_MAX_CEILING = 7500.0  # Standard maximum for G_store (3 * S_MAX_CEILING = 2500)

# =====================================================================
# 🔥 CONSOLIDATED CORE IMPORTS (Cleaned)
# =====================================================================
try:
    # Import core project modules
    from src.model import ModelParams, two_store_model_step
    from src.enkf import EnKFConfig, enkf_update, enkf_forecast_step
    from src.param_manager import get_calibrated_params_for_basin, load_all_calibrated_params
    from src.metrics import calculate_kge
    from data.data_processor import load_and_prepare_data
    from src.budyko import estimate_budyko_et
except ImportError as e:
    print(f"FATAL: Import failed. Please ensure all files exist. Error: {e}")
    sys.exit(1)

# =====================================================================
# 🔥 CORE SCENARIO RUNNER
# =====================================================================

def run_enkf_scenario(
    P_monthly: np.ndarray, PET_monthly: np.ndarray, Q_USGS_monthly: np.ndarray,
    ET_B_monthly: np.ndarray, ET_NLDAS_monthly: np.ndarray,
    scenario: str, cfg: EnKFConfig, target_basin: str,
    S_init_nldas: float, G_init_nldas: float, CALIBRATED_SMAX: float,
    all_calibrated_params: Dict[str, ModelParams]
) -> Optional[Dict]:
    """
    Runs the model for three scenarios (Q_Base, Q_B, and Q_ET_DA).
    """
    nmonths = len(P_monthly)

    # 1. Parameter Loading
    cal_params = all_calibrated_params.get(target_basin)
    if not cal_params:
        cal_params = get_calibrated_params_for_basin(target_basin)

    if not cal_params:
        print(f"❌ FATAL: Calibrated parameters not found for {target_basin}. Skipping basin.")
        return None

    print(f"✅ Using calibrated parameters for {target_basin}. KGE: {cal_params.get('KGE', 'N/A'):.3f}")

    CALIBRATED_KPERC = cal_params['Kperc']
    CALIBRATED_KB = cal_params['Kb']
    CALIBRATED_KE = cal_params['Ke']
    CALIBRATED_CQQ = cal_params['Cqq']
    CALIBRATED_BIAS = cal_params['bias']

    # 2. Ensemble Initialization
    X = np.zeros((cfg.state_dim, cfg.nens))
    S_spread, G_spread = 20.0, 10.0

    X[0, :] = np.clip(np.random.normal(S_init_nldas, S_spread, cfg.nens), 1.0, CALIBRATED_SMAX)
    X[1, :] = np.clip(np.random.normal(G_init_nldas, G_spread, cfg.nens), 1.0, G_MAX_CEILING)
    X[2, :] = np.clip(np.random.normal(CALIBRATED_KPERC, 0.1, cfg.nens), 0.01, 0.999)
    X[3, :] = np.clip(np.random.normal(CALIBRATED_KB, 0.1, cfg.nens), 0.01, 0.999)
    X[4, :] = np.clip(np.random.normal(CALIBRATED_KE, 0.1, cfg.nens), 0.01, 0.999)
    X[5, :] = np.clip(np.random.normal(CALIBRATED_CQQ, 0.1, cfg.nens), 0.01, 0.999)
    X[6, :] = np.clip(np.random.normal(CALIBRATED_BIAS, 0.15, cfg.nens), -10, 10)

    Q_mean, ET_mean, S_mean, G_mean = np.zeros(nmonths), np.zeros(nmonths), np.zeros(nmonths), np.zeros(nmonths)
    param_rows = slice(2, cfg.state_dim)

    # 3. Time Loop (Forecast & Analysis)
    for t in range(nmonths):
        X[param_rows, :] += cfg.param_rw_sd[param_rows][:, None] * np.random.randn(cfg.state_dim - 2, cfg.nens)
        X[2, :] = np.clip(X[2, :], 0.01, 0.999)
        X[3, :] = np.clip(X[3, :], 0.01, 0.999)
        X[4, :] = np.clip(X[4, :], 0.01, 0.999)
        X[5, :] = np.clip(X[5, :], 0.01, 0.999)
        X[6, :] = np.clip(X[6, :], -10, 10)

        ET_B_t = ET_B_monthly[t] if scenario == 'Q_B' and not np.isnan(ET_B_monthly[t]) else None

        X_next, Q_ens, ET_ens = enkf_forecast_step(
            X, P_monthly[t], PET_monthly[t], CALIBRATED_SMAX, ET_B_t
        )

        X_updated = X_next
        if scenario == 'Q_ET_DA':
            ET_obs = ET_NLDAS_monthly[t]
            is_et_available = not np.isnan(ET_obs) and ET_obs > 0
            if is_et_available:
                X_updated = enkf_update(X_next, ET_obs, ET_ens, cfg.R_ET, cfg.inflation)

        X = X_updated
        Q_mean[t] = np.mean(Q_ens)
        ET_mean[t] = np.mean(ET_ens)
        S_mean[t] = np.mean(X[0, :])
        G_mean[t] = np.mean(X[1, :])

    return {
        'Q_mean': Q_mean,
        'ET_mean': ET_mean,
        'S_mean': S_mean,
        'G_mean': G_mean,
        'X_final': X
    }

# =====================================================================
# 🛠️ HELPER: Robust Column Key Extraction
# =====================================================================

def _extract_column_key(df: pd.DataFrame, basin_id: str) -> str:
    cleaned_id = str(basin_id).zfill(8)

    if cleaned_id in df.columns:
        return cleaned_id

    for col in df.columns:
        if str(col).endswith(cleaned_id):
            return col

    raise KeyError(f"Column for basin {cleaned_id} not found in DataFrame. Columns are: {df.columns.tolist()}")

# =====================================================================
# 🚀 Worker Function for Parallel Execution
# =====================================================================

def process_basin(
    basin: str, all_calibrated_params: Dict[str, ModelParams],
    Rainf_df: pd.DataFrame, PotEvap_df: pd.DataFrame, Evap_df: pd.DataFrame,
    Q_USGS_monthly: pd.DataFrame, Q_nldas_mm_monthly: pd.DataFrame,
    S_init_df: pd.DataFrame, G_init_df: pd.DataFrame,
    attrs: gpd.GeoDataFrame, cfg: EnKFConfig
) -> Tuple[str, Optional[pd.DataFrame], Optional[Dict[str, float]]]:

    basin_id = str(basin).zfill(8)
    basin_metrics = {
        'Q_Base_Q_KGE': 0.0, 'Q_B_Q_KGE': 0.0, 'Q_ET_DA_Q_KGE': 0.0,
        'Q_Base_ET_KGE': 0.0, 'Q_B_ET_KGE': 0.0, 'Q_ET_DA_ET_KGE': 0.0
    }

    try:
        P_key = _extract_column_key(Rainf_df, basin_id)
        PET_key = _extract_column_key(PotEvap_df, basin_id)
        Q_obs_key = _extract_column_key(Q_USGS_monthly, basin_id)
        ET_key = _extract_column_key(Evap_df, basin_id)
        Q_nldas_key = _extract_column_key(Q_nldas_mm_monthly, basin_id)
        S_init_key = _extract_column_key(S_init_df, basin_id)
        G_init_key = _extract_column_key(G_init_df, basin_id)

        P_monthly = Rainf_df[P_key].values
        PET_monthly = PotEvap_df[PET_key].values
        Q_obs_USGS_eval = Q_USGS_monthly[Q_obs_key].values
        ET_NLDAS_monthly = Evap_df[ET_key].values
        Q_nldas_monthly = Q_nldas_mm_monthly[Q_nldas_key].values

        cal_params = all_calibrated_params.get(basin_id)
        CALIBRATED_SMAX = cal_params['Smax'] if cal_params and 'Smax' in cal_params else 2500.0

        S_init_nldas = S_init_df.loc[S_init_df.index[0], S_init_key]
        G_init_nldas = G_init_df.loc[G_init_df.index[0], G_init_key]

        Q_obs_USGS_eval = np.clip(Q_obs_USGS_eval, 0.0, np.inf)
        P_monthly = np.clip(P_monthly, 0.0, np.inf)
        PET_monthly = np.clip(PET_monthly, 0.0, np.inf)
        ET_NLDAS_monthly = np.clip(ET_NLDAS_monthly, 0.0, np.inf)

        ET_B_monthly = np.array([
            estimate_budyko_et(P_monthly[t], PET_monthly[t], model='Fu', m=1.35)
            for t in range(len(P_monthly))
        ])

        if np.all(np.isnan(P_monthly)) or np.all(np.isnan(PET_monthly)) or np.all(np.isnan(Q_obs_USGS_eval)):
            return basin_id, None, basin_metrics

        dates = Q_USGS_monthly.index
        basin_results_dfs = {}
        last_result = None
        spin_up = 60

        for scenario in ['Q_Base', 'Q_B', 'Q_ET_DA']:
            result = run_enkf_scenario(
                P_monthly, PET_monthly, Q_obs_USGS_eval,
                ET_B_monthly, ET_NLDAS_monthly,
                scenario, cfg, basin_id, S_init_nldas, G_init_nldas, CALIBRATED_SMAX,
                all_calibrated_params
            )

            if result is None:
                return basin_id, None, basin_metrics

            last_result = result

            df_scenario = pd.DataFrame({
                'Date': dates,
                f'{scenario}_Q_sim': result['Q_mean'],
                f'{scenario}_ET_sim': result['ET_mean'],
                f'{scenario}_S_sim': result['S_mean'],
                f'{scenario}_G_sim': result['G_mean'],
            }).set_index('Date')
            basin_results_dfs[scenario] = df_scenario

            Q_sim, Q_obs = result['Q_mean'][spin_up:], Q_obs_USGS_eval[spin_up:]
            ET_sim, ET_obs = result['ET_mean'][spin_up:], ET_NLDAS_monthly[spin_up:]

            valid_mask_Q = ~(np.isnan(Q_sim) | np.isnan(Q_obs))
            if np.any(valid_mask_Q) and len(Q_obs[valid_mask_Q]) >= 12:
                kge_Q = calculate_kge(Q_obs[valid_mask_Q], Q_sim[valid_mask_Q])
                basin_metrics[f"{scenario}_Q_KGE"] = kge_Q if not np.isnan(kge_Q) else 0.0

            valid_mask_ET = ~(np.isnan(ET_sim) | np.isnan(ET_obs))
            if np.any(valid_mask_ET) and len(ET_obs[valid_mask_ET]) >= 12:
                kge_ET = calculate_kge(ET_obs[valid_mask_ET], ET_sim[valid_mask_ET])
                basin_metrics[f"{scenario}_ET_KGE"] = kge_ET if not np.isnan(kge_ET) else 0.0

        if last_result is not None:
            df_obs_inputs = pd.DataFrame({
                'Date': dates,
                'P_input': P_monthly,
                'PET_input': PET_monthly,
                'Q_USGS_obs': Q_obs_USGS_eval,
                'Q_NLDAS_input': Q_nldas_monthly,
                'ET_NLDAS_obs': ET_NLDAS_monthly,
                'ET_Budyko': ET_B_monthly
            }).set_index('Date')

            df_final_basin = df_obs_inputs
            for df_scenario in basin_results_dfs.values():
                df_final_basin = df_final_basin.join(df_scenario, how='left')

            return basin_id, df_final_basin, basin_metrics
        else:
            return basin_id, None, basin_metrics

    except KeyError as e:
        print(f"❌ CRITICAL ERROR (KeyError) processing basin {basin_id}: Missing column key: {e}")
        return basin_id, None, basin_metrics
    except Exception as e:
        print(f"❌ CRITICAL ERROR processing basin {basin_id}: {e}")
        return basin_id, None, basin_metrics

# =====================================================================
# 🔥 MAIN SIMULATION EXECUTION
# =====================================================================

def get_basin_id_from_column(col: str) -> str:
    """Assumes the basin ID is the last 8 characters of the column name."""
    return str(col).split('_')[-1]

if __name__ == '__main__':
    
    # PROJECT_ROOT is assumed to be defined globally at the top of the script
    
    print("Initializing data loading...")

    # --- Data Loading ---
    try:
        # load_and_prepare_data() is assumed to return the 11 required DataFrames
        Rainf_df, PotEvap_df, Evap_df, Qsb_df, M_df, Slope_df, Q_nldas_mm_monthly, Q_USGS_monthly, \
        S_init_df, G_init_df, SM_df = load_and_prepare_data()
        print("✅ Data frames loaded successfully.")
    except Exception as e:
        print(f"FATAL: Data loading failed. Error: {e}")
        sys.exit(1)

    # --- CAMELS Attributes Loading ---
    print("Loading CAMELS attributes...")

    try:
        r = requests.get("https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/data/contents/camels_attributes_v2.0.feather")
        attrs = gpd.read_feather(io.BytesIO(r.content)).reset_index(drop=False)
        
        # --- FIX APPLIED HERE: Added missing .str before .strip() ---
        attrs['gauge_id'] = attrs['gauge_id'].astype(str).str.strip().str.zfill(8)
        
        attrs.set_index('gauge_id', inplace=True)
        print("✅ CAMELS attributes loaded successfully.")
    except Exception as e:
        print(f"❌ FATAL: Failed to load required CAMELS attributes. Error: {e}")
        sys.exit(1)
    # --- Calibrated Parameters Loading ---
    print("Pre-loading all calibrated parameters...")
    try:
        # load_all_calibrated_params() is expected to load data from the JSON output of calibration.py
        all_calibrated_params = load_all_calibrated_params()
        if not all_calibrated_params:
            print("FATAL: No calibrated parameters were loaded. Check 'SCE_cal_params/final_calibrated_params.json'.")
            sys.exit(1)
        print(f"✅ Loaded parameters for {len(all_calibrated_params)} basins.")
    except Exception as e:
        print(f"❌ WARNING: Failed to pre-load all parameters. Error: {e}")
        all_calibrated_params = {}

    # --- Setup Parallel Tasks ---
    TARGET_BASIN_KEYS = list(Q_USGS_monthly.columns)
    TARGET_BASINS = [get_basin_id_from_column(key) for key in TARGET_BASIN_KEYS]
    
    # EnKFConfig() and process_basin() are assumed to be defined above this block
    cfg = EnKFConfig() 

    tasks = [(basin, all_calibrated_params, Rainf_df, PotEvap_df, Evap_df, Q_USGS_monthly,
              Q_nldas_mm_monthly, S_init_df, G_init_df, attrs, cfg) for basin in TARGET_BASINS]

    NUM_CORES = max(1, cpu_count() - 1)
    print(f"\nStarting parallel simulation on **{NUM_CORES} cores** for {len(TARGET_BASINS)} basins...")

    results_timeseries = {}
    results_metrics = {
        'Q_Base_Q_KGE': {}, 'Q_B_Q_KGE': {}, 'Q_ET_DA_Q_KGE': {},
        'Q_Base_ET_KGE': {}, 'Q_B_ET_KGE': {}, 'Q_ET_DA_ET_KGE': {}
    }

    # --- Run Parallel Processing ---
    try:
        # process_basin is the worker function that returns basin_id, df_data, metrics
        with Pool(NUM_CORES) as pool:
            parallel_results = pool.starmap(process_basin, tasks)
    except Exception as e:
        print(f"❌ FATAL: Parallel pool failed. Error: {e}")
        sys.exit(1)

    # --- Aggregate Results (Serial Step) ---
    processed_count = 0
    for basin_id, df_data, metrics in parallel_results:
        if df_data is not None:
            results_timeseries[basin_id] = df_data
            for kge_key, value in metrics.items():
                if kge_key in results_metrics:
                    results_metrics[kge_key][basin_id] = value
            processed_count += 1
        else:
            # Basin failed, ensure metrics are recorded as 0.0 or initialize if missing
            for kge_key in results_metrics.keys():
                if basin_id not in results_metrics[kge_key]:
                    results_metrics[kge_key][basin_id] = 0.0

    print("\nParallel processing complete. Saving results...")

    # --- TIME SERIES SAVING SECTION ---
    if results_timeseries:
        output_dir = os.path.join(PROJECT_ROOT, 'Simulation_results', 'enkf_timeseries_by_basin')
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        for basin_id, df_data in results_timeseries.items():
            output_path = os.path.join(output_dir, f'{basin_id}_enkf_timeseries_ET_DA.feather')
            df_data = df_data.reset_index(names=['Date'])
            df_data.to_feather(output_path)
            saved_files.append(output_path)
        print(f"\n💾 Saved **{len(saved_files)} individual basin time series files** (as .feather) to the directory:")
        print(f"**{output_dir}**")

    # --- OUTPUT METRICS SECTION (Corrected for CSV save) ---
    print("\n--- Summary Performance Metrics (Q_KGE vs Q_USGS, ET_KGE vs ET_NLDAS, after 5-year spin-up) ---")
    df_metrics = pd.DataFrame(results_metrics)
    
    # Filter out basins that failed (KGE == 0.0 for Q_Base is the failure flag)
    df_metrics_summary = df_metrics[df_metrics['Q_Base_Q_KGE'] != 0.0]

    metrics_output_dir = os.path.join(PROJECT_ROOT, 'Simulation_results')
    os.makedirs(metrics_output_dir, exist_ok=True)
    metrics_output_path = os.path.join(metrics_output_dir, 'enkf_performance_metrics.csv')

    # Ensure index name is set and save the final, filtered summary to CSV
    df_metrics_summary.index.name = 'basin_id'
    df_metrics_summary.to_csv(metrics_output_path, float_format='%.4f')

    print(df_metrics_summary.to_string())
    print(f"\n💾 Saved summary KGE metrics to: **{metrics_output_path}**")
    
    total_basins = len(TARGET_BASINS)
    valid_processed_count = len(df_metrics_summary)
    
    print(f"\n🏆 SUCCESS: **{valid_processed_count}/{total_basins}** basins processed with valid KGE")