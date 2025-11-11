

# # run_simulation.py (Final Clean Version)
# import json
# import logging
# import os
# import sys
# from concurrent.futures import ProcessPoolExecutor, as_completed

# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# # ---------------------------------------------------------
# # Project paths
# # ---------------------------------------------------------
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
# RESULT_DIR = os.path.join(PROJECT_ROOT, "Simulation_results")
# os.makedirs(RESULT_DIR, exist_ok=True)

# sys.path.append(PROJECT_ROOT)

# # ---------------------------------------------------------
# # Imports
# # ---------------------------------------------------------
# from src.model import ModelParams, two_store_model_step
# from src.budyko import BudykoModelEstimator
# from src.enkf import EnKFConfig, enkf_forecast_step, enkf_update
# from src.metrics import calculate_kge, calculate_nse


# # ---------------------------------------------------------
# # Feather loader
# # ---------------------------------------------------------
# def load_feather_df(fname: str, ddir: str) -> pd.DataFrame:
#     path = os.path.join(ddir, fname)
#     if not os.path.exists(path):
#         logging.warning(f"File not found: {path}. Returning empty DataFrame.")
#         return pd.DataFrame()
#     df = pd.read_feather(path)
#     df = df.dropna(axis=1, how="all")
#     if "time" in df.columns:
#         df["time"] = pd.to_datetime(df["time"])
#         df.set_index("time", inplace=True)
#     return df


# # ---------------------------------------------------------
# # Load datasets
# # ---------------------------------------------------------
# PET_df = load_feather_df("PotEvap.feather", DATA_DIR)
# Rainf_df = load_feather_df("Rainf.feather", DATA_DIR)
# Evap_df = load_feather_df("EVap.feather", DATA_DIR)
# Q_usgs_df = load_feather_df("Q_USGS.feather", DATA_DIR)
# Q_nldas_df = load_feather_df("Q_nldas_mm_monthly.feather", DATA_DIR)
# Qsb_df = load_feather_df("Qsb.feather", DATA_DIR)
# M_df = load_feather_df("M.feather", DATA_DIR)
# Slope_basin = load_feather_df("slope.feather", DATA_DIR)

# # Ensure all datasets have the same basins
# common_cols = sorted(
#     set(Evap_df.columns)
#     & set(Qsb_df.columns)
#     & set(PET_df.columns)
#     & set(M_df.columns)
#     & set(Slope_basin.columns)
# )
# Evap_df = Evap_df[common_cols]
# Qsb_df = Qsb_df[common_cols]
# PET_df = PET_df[common_cols]
# M_df = M_df[common_cols]
# Slope_basin = Slope_basin[common_cols]

# # Align indices
# M_basin = M_df.copy()
# M_basin.index = pd.to_datetime(M_basin.index)
# M_basin = M_basin.loc[Evap_df.index]

# # ---------------------------------------------------------
# # Load calibrated parameters
# # ---------------------------------------------------------
# calibrated_path = os.path.join(
#     PROJECT_ROOT, "SCE_cal_params", "final_calibrated_params.json"
# )
# with open(calibrated_path, "r") as f:
#     calibrated_params = json.load(f)

# # ---------------------------------------------------------
# # Compute Budyko components once (exactly as in src/budyko.py)
# # ---------------------------------------------------------
# budyko = BudykoModelEstimator(
#     Evap_df=Evap_df,
#     Qsb_monthly=Qsb_df,
#     PotEvap_df=PET_df,
#     M_basin=M_basin,
#     Slope_basin=Slope_basin,
#     ke=0.95,
# )
# budyko.estimate_budyko_et()
# omega_true_all = budyko.omega_true
# omega_MLR_all = budyko.omega_MLR
# ET_B_all = budyko.ET_B


# # ---------------------------------------------------------
# # Simulation for a single basin
# # ---------------------------------------------------------
# def simulate_basin(basin_id):
#     if basin_id not in calibrated_params or basin_id not in Evap_df.columns:
#         return None, None

#     p = calibrated_params[basin_id]
#     idx = Evap_df.index  # Master index (length L=180)
#     L = len(idx)
    
#     # --- Extract aligned time series directly ---
#     PET         = PET_df[basin_id].values
#     P           = Rainf_df.get(basin_id, pd.Series(index=idx)).values
    
#     # CRITICAL FIX: Reindex Q_USGS to the master 'idx' (L=180) to avoid length mismatch
#     Q_obs_series = Q_usgs_df.get(basin_id, pd.Series(index=idx))
#     Q_obs       = Q_obs_series.reindex(idx).values 
    
#     ET_nldas    = Evap_df[basin_id].values
#     M_series    = M_basin[basin_id].values
#     Slope_series = Slope_basin[basin_id].values

#     # --- Budyko outputs (Correctly aligned to L=180) ---
#     ET_B       = ET_B_all[basin_id].values
    
#     # --- Diagnostic Check ---
#     if not (len(PET) == L and len(P) == L and len(Q_obs) == L and len(ET_B) == L):
#         logging.error(
#             f"Input arrays for {basin_id} have mismatched length. Skipping."
#         )
#         return None, None
#     # ----------------------------------------------------------------------
    
#     # Model parameters and initial states
#     Smax_cal = p['Smax']
#     Gmax_factor = p.get('Gmax_factor', 3.0)
#     Gmax_cal = Smax_cal * Gmax_factor 
#     S_init = p.get('S_init', Smax_cal)
#     G_init = p.get('G_init', Smax_cal)
    
#     model_params = ModelParams(
#         Smax=Smax_cal, Kperc=p['Kperc'], Kb=p['Kb'], Ke=p['Ke'], Cqq=p['Cqq']
#     )

#     # Base scenario ET calculation
#     ET_ke = PET * p['Ke']

#     # --- EnKF setup ---
#     config = EnKFConfig()
#     nens, inflation, R_ET = config.nens, config.inflation, config.R_ET
#     rng = np.random.default_rng(hash(basin_id) % (2**32 - 1))

#     # ---------------------------------------------------------
#     # 🎯 Budyko Scenario Case 2: Assimilate ET_nldas using ET_B as observation
#     # ---------------------------------------------------------
#     X_et_bud = np.tile(ET_nldas, (nens, 1)).T + rng.normal(0, 0.05, (L, nens)) 
#     ET_B_NLDAS_ass = np.empty(L) 
    
#     for t in range(L):
#         et_ens = X_et_bud[t, :]
#         ET_B_NLDAS_ass[t] = et_ens.mean() # Store forecast mean before assimilation
        
#         if np.isfinite(ET_B[t]):
#             # Prepare dummy state vector X for EnKF update (only Ke/row 4 is non-zero)
#             X_dummy = np.zeros((6, nens))
#             X_dummy[4, :] = et_ens 
#             HX = X_dummy[4, :].copy() # Measurement ensemble (Ke is used as ET proxy here)
            
#             X_dummy_updated = enkf_update(
#                 X_dummy, 
#                 y_obs=ET_B[t], 
#                 HX=HX, 
#                 R=R_ET, 
#                 inflation=inflation, 
#                 Smax=Smax_cal, 
#                 Gmax=Gmax_cal 
#             )

#             X_et_bud[t, :] = X_dummy_updated[4, :] 
#             ET_B_NLDAS_ass[t] = X_dummy_updated[4, :].mean() # Store assimilated mean

#     # --- Streamflow simulations ---
#     def run_model(ET_override=None):
#         S, G = S_init, G_init
#         Q_out = []
#         for t, (P_t, PET_t) in enumerate(zip(P, PET)):
#             S, G, _, Q, *_ = two_store_model_step(
#                 S, G, P_t, PET_t, model_params,
#                 ET_override=ET_override[t] if ET_override is not None else None
#             )
#             G = np.clip(G, 0, Gmax_cal)
#             Q_out.append(Q)
#         return np.asarray(Q_out)

#     # 1. Base Scenario: ET_ke (Q_ke)
#     Q_ke = run_model()
    
#     # 2. Budyko Scenario Case 1: ET_B (Q_b)
#     Q_b = run_model(ET_override=ET_B)
    
#     # 3. Budyko Scenario Case 2: ET_B_NLDAS_ass (Q_B_ass)
#     Q_B_ass = run_model(ET_override=ET_B_NLDAS_ass)

#     # --- Results ---
#     results = pd.DataFrame({
#         'time': idx,
#         'ET_ke': ET_ke,
#         'ET_B': ET_B,
#         'ET_nldas': ET_nldas,
#         'ET_B_NLDAS_ass': ET_B_NLDAS_ass, 
#         'Q_obs': Q_obs,
#         'Q_ke': Q_ke,
#         'Q_b': Q_b,
#         'Q_B_ass': Q_B_ass,
#         'omega_true': ET_B_all[basin_id].values, # Keep Budyko components for diagnostics
#         'omega_MLR': omega_MLR_all[basin_id].values,
#         'Q_nldas': Q_nldas_df.get(basin_id, pd.Series(index=idx)).reindex(idx).values,
#         'M': M_series,
#         'Slope': Slope_series
#     }).set_index('time')

#     # --- Metrics ---
#     metrics = {
#         'Q_ke_KGE': calculate_kge(Q_obs, Q_ke),
#         'Q_ke_NSE': calculate_nse(Q_obs, Q_ke),
#         'Q_b_KGE': calculate_kge(Q_obs, Q_b),
#         'Q_b_NSE': calculate_nse(Q_obs, Q_b),
#         'Q_B_ass_KGE': calculate_kge(Q_obs, Q_B_ass),
#         'Q_B_ass_NSE': calculate_nse(Q_obs, Q_B_ass),
#     }

#     return results, metrics

# # ---------------------------------------------------------
# # Parallel execution
# # ---------------------------------------------------------
# def run_and_save_basin(basin_id):
#     try:
#         result_df, metrics = simulate_basin(basin_id)
#         if result_df is not None:
#             result_path = os.path.join(
#                 RESULT_DIR, f"results_streamflow_{basin_id}.feather"
#             )
#             result_df.reset_index().to_feather(result_path)

#             rows = [
#                 {
#                     "gauge_id": basin_id,
#                     "scenario": sc,
#                     "KGE": metrics.get(f"{sc}_KGE", np.nan),
#                     "NSE": metrics.get(f"{sc}_NSE", np.nan),
#                 }
#                 for sc in ["Q_ke", "Q_b", "Q_B_ass"] 
#             ]
#             return rows
#     except Exception as e:
#         print(f"❌ Error processing {basin_id}: {e}", file=sys.stderr)
#     return []


# if __name__ == "__main__":
#     from multiprocessing import cpu_count

#     os.makedirs(RESULT_DIR, exist_ok=True)

#     all_basins = common_cols
#     all_metrics = []

#     with ProcessPoolExecutor(max_workers=max(1, cpu_count() - 1)) as executor:
#         futures = {executor.submit(run_and_save_basin, b): b for b in all_basins}
#         for f in tqdm(as_completed(futures), total=len(futures), desc="Running in parallel"):
#             all_metrics.extend(f.result())

#     pd.DataFrame(all_metrics).to_csv(
#         os.path.join(RESULT_DIR, "streamflow_performance_metrics.csv"), index=False
#     )
#     print("\n✅ All basin simulations completed and results saved.")







# run_simulation.py (Cleaned and Fixed Budyko Initialization)
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULT_DIR = os.path.join(PROJECT_ROOT, "Simulation_results")
os.makedirs(RESULT_DIR, exist_ok=True)

sys.path.append(PROJECT_ROOT)

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
from src.model import ModelParams, two_store_model_step
from src.budyko import BudykoModelEstimator # BudykoModelEstimator now expects calibrated_params
from src.enkf import EnKFConfig, enkf_forecast_step, enkf_update
from src.metrics import calculate_kge, calculate_nse


# ---------------------------------------------------------
# Feather loader
# ---------------------------------------------------------
def load_feather_df(fname: str, ddir: str) -> pd.DataFrame:
    path = os.path.join(ddir, fname)
    if not os.path.exists(path):
        logging.warning(f"File not found: {path}. Returning empty DataFrame.")
        return pd.DataFrame()
    df = pd.read_feather(path)
    df = df.dropna(axis=1, how="all")
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    return df


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------
PET_df = load_feather_df("PotEvap.feather", DATA_DIR)
Rainf_df = load_feather_df("Rainf.feather", DATA_DIR)
Evap_df = load_feather_df("EVap.feather", DATA_DIR)
Q_usgs_df = load_feather_df("Q_USGS.feather", DATA_DIR)
Q_nldas_df = load_feather_df("Q_nldas_mm_monthly.feather", DATA_DIR)
Qsb_df = load_feather_df("Qsb.feather", DATA_DIR)
M_df = load_feather_df("M.feather", DATA_DIR)
Slope_basin = load_feather_df("slope.feather", DATA_DIR)
RootMoist = load_feather_df("SoilM_0_200cm.feather", DATA_DIR)

# Ensure all datasets have the same basins
common_cols = sorted(
    set(Evap_df.columns)
    & set(Qsb_df.columns)
    & set(PET_df.columns)
    & set(M_df.columns)
    & set(Slope_basin.columns)
    & set(RootMoist.columns)
)
Evap_df = Evap_df[common_cols]
Qsb_df = Qsb_df[common_cols]
PET_df = PET_df[common_cols]
M_df = M_df[common_cols]
Slope_basin = Slope_basin[common_cols]
RootMoist = RootMoist[common_cols]

# Align indices
M_basin = M_df.copy()
M_basin.index = pd.to_datetime(M_basin.index)
M_basin = M_basin.loc[Evap_df.index]

# ---------------------------------------------------------
# Load calibrated parameters
# ---------------------------------------------------------
calibrated_path = os.path.join(
    PROJECT_ROOT, "SCE_cal_params", "final_calibrated_params.json"
)
with open(calibrated_path, "r") as f:
    calibrated_params = json.load(f)

# ---------------------------------------------------------
# Compute Budyko components once (using basin-specific Ke)
# ---------------------------------------------------------
budyko = BudykoModelEstimator(
    Evap_df=Evap_df,
    Qsb_monthly=Qsb_df,
    PotEvap_df=PET_df,
    M_basin=M_basin,
    Slope_basin=RootMoist,
    calibrated_params=calibrated_params # CRITICAL FIX: Pass the dictionary
)
budyko.estimate_budyko_et()
omega_true_all = budyko.omega_true
omega_MLR_all = budyko.omega_MLR
ET_B_all = budyko.ET_B


# ---------------------------------------------------------
# Simulation for a single basin
# ---------------------------------------------------------
def simulate_basin(basin_id):
    if basin_id not in calibrated_params or basin_id not in Evap_df.columns:
        return None, None

    p = calibrated_params[basin_id]
    idx = Evap_df.index  # Master index (length L=180)
    L = len(idx)
    
    # --- Extract aligned time series directly ---
    PET         = PET_df[basin_id].values
    P           = Rainf_df.get(basin_id, pd.Series(index=idx)).values
    
    # CRITICAL FIX: Reindex Q_USGS to the master 'idx' (L=180)
    Q_obs_series = Q_usgs_df.get(basin_id, pd.Series(index=idx))
    Q_obs       = Q_obs_series.reindex(idx).values 
    
    Q_nldas     = Q_nldas_df.get(basin_id, pd.Series(index=idx)).values
    ET_nldas    = Evap_df[basin_id].values
    M_series    = M_basin[basin_id].values
    Slope_series = RootMoist[basin_id].values

    # --- Budyko outputs (Correctly aligned to L=180) ---
    ET_B       = ET_B_all[basin_id].values
    
    # --- Diagnostic Check ---
    if not (len(PET) == L and len(P) == L and len(Q_obs) == L and len(ET_B) == L):
        logging.error(
            f"Input arrays for {basin_id} have mismatched length. Skipping."
        )
        return None, None
    # ----------------------------------------------------------------------
    
    # Model parameters and initial states
    Smax_cal = p['Smax']
    Gmax_factor = p.get('Gmax_factor', 3.0)
    Gmax_cal = Smax_cal * Gmax_factor 
    S_init = 0.52
    G_init = 0.35
    
    model_params = ModelParams(
        Smax=Smax_cal, Kperc=p['Kperc'], Kb=p['Kb'], Ke=p['Ke'], Cqq=p['Cqq']
    )

    # Base scenario ET calculation
    # ET_ke = PET * 0.85
    ET_ke = PET * p['Ke']
    # --- EnKF setup ---
    config = EnKFConfig()
    nens, inflation, R_ET = config.nens, config.inflation, config.R_ET
    rng = np.random.default_rng(hash(basin_id) % (2**32 - 1))

    # ---------------------------------------------------------
    # 🎯 Budyko Scenario Case 2: Assimilate ET_nldas using ET_B as observation (Q_B_ass)
    # ---------------------------------------------------------
    X_et_bud = np.tile(ET_nldas, (nens, 1)).T + rng.normal(0, 0.05, (L, nens)) 
    ET_B_NLDAS_ass = np.empty(L) 
    
    for t in range(L):
        et_ens = X_et_bud[t, :]
        ET_B_NLDAS_ass[t] = et_ens.mean() # Store forecast mean before assimilation
        
        if np.isfinite(ET_B[t]):
            X_dummy = np.zeros((6, nens))
            X_dummy[4, :] = et_ens 
            HX = X_dummy[4, :].copy() 
            
            X_dummy_updated = enkf_update(
                X_dummy, 
                y_obs=ET_B[t], 
                HX=HX, 
                R=R_ET, 
                inflation=inflation, 
                Smax=Smax_cal, 
                Gmax=Gmax_cal 
            )
            X_et_bud[t, :] = X_dummy_updated[4, :] 
            ET_B_NLDAS_ass[t] = X_dummy_updated[4, :].mean() # Store assimilated mean

    # --- Streamflow simulations ---
    def run_model(ET_override=None):
        S, G = S_init, G_init
        Q_out = []
        for t, (P_t, PET_t) in enumerate(zip(P, PET)):
            S, G, _, Q, *_ = two_store_model_step(
                S, G, P_t, PET_t, model_params,
                ET_override=ET_override[t] if ET_override is not None else None
            )
            G = np.clip(G, 0, Gmax_cal)
            Q_out.append(Q)
        return np.asarray(Q_out)

    # 1. Base Scenario: ET_ke (Q_ke)
    Q_ke = run_model()
    
    # 2. Budyko Scenario Case 1: ET_B (Q_b)
    Q_b = run_model(ET_override=ET_B)
    
    # 3. Budyko Scenario Case 2: ET_B_NLDAS_ass (Q_B_ass)
    Q_B_ass = run_model(ET_override=ET_B_NLDAS_ass)

    # --- Results ---
    results = pd.DataFrame({
        'time': idx,
        'ET_ke': ET_ke,
        'ET_B': ET_B,
        'ET_nldas': ET_nldas,
        'ET_B_NLDAS_ass': ET_B_NLDAS_ass, 
        'Q_obs': Q_obs,
        'Q_ke': Q_ke,
        'Q_b': Q_b,
        'Q_B_ass': Q_B_ass,
        'omega_true': omega_true_all[basin_id].values,
        'omega_MLR': omega_MLR_all[basin_id].values,
        'Q_nldas': Q_nldas_df.get(basin_id, pd.Series(index=idx)).reindex(idx).values,
        'M': M_series,
        'Slope': Slope_series
    }).set_index('time')

    # --- Metrics ---
    metrics = {
        'Q_ke_KGE': calculate_kge(Q_obs, Q_ke),
        'Q_ke_NSE': calculate_nse(Q_obs, Q_ke),
        'Q_b_KGE': calculate_kge(Q_obs, Q_b),
        'Q_b_NSE': calculate_nse(Q_obs, Q_b),
        'Q_B_ass_KGE': calculate_kge(Q_obs, Q_B_ass),
        'Q_B_ass_NSE': calculate_nse(Q_obs, Q_B_ass),
        'Q_nldas_KGE': calculate_kge(Q_obs, Q_nldas),
        'Q_nldas_NSE': calculate_nse(Q_obs, Q_nldas),
    }

    return results, metrics

# ---------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------

def run_and_save_basin(basin_id):
    try:
        result_df, metrics = simulate_basin(basin_id)
        if result_df is not None:
            result_path = os.path.join(
                RESULT_DIR, f"results_streamflow_{basin_id}.feather"
            )
            result_df.reset_index().to_feather(result_path)

            # CORRECTED LINE: Ensure all four scenarios are included here
            rows = [
                {
                    "gauge_id": basin_id,
                    "scenario": sc,
                    "KGE": metrics.get(f"{sc}_KGE", np.nan),
                    "NSE": metrics.get(f"{sc}_NSE", np.nan),
                }
                # List of all scenarios whose metrics should be saved:
                for sc in ["Q_ke", "Q_b", "Q_B_ass", "Q_nldas"] 
            ]
            return rows
    except Exception as e:
        print(f"❌ Error processing {basin_id}: {e}", file=sys.stderr)
    return []


if __name__ == "__main__":
    from multiprocessing import cpu_count

    os.makedirs(RESULT_DIR, exist_ok=True)

    all_basins = common_cols
    all_metrics = []

    with ProcessPoolExecutor(max_workers=max(1, cpu_count() - 1)) as executor:
        futures = {executor.submit(run_and_save_basin, b): b for b in all_basins}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Running in parallel for all 505 CAMELS basins"):
            all_metrics.extend(f.result())

    pd.DataFrame(all_metrics).to_csv(
        os.path.join(RESULT_DIR, "streamflow_performance_metrics.csv"), index=False
    )
    print("\n✅ All basin simulations completed and results saved.")