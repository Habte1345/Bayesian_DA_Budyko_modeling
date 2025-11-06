# # run_simulation.py

# """
# Main Simulation Script
# Performs multi-scenario streamflow simulation with EnKF assimilation and Budyko-based ET estimates.
# """

# import sys
# import os
# import json
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import logging

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.model import ModelParams, two_store_model_step
# from src.budyko import BudykoModelEstimator
# from src.enkf import enkf_forecast_step, enkf_update, EnKFConfig
# from src.metrics import calculate_kge, calculate_nse
# from src.param_manager import load_all_calibrated_params

# # -----------------------------
# # Setup Paths
# # -----------------------------
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
# RESULT_DIR = os.path.join(PROJECT_ROOT, 'Simulation_results')
# os.makedirs(RESULT_DIR, exist_ok=True)

# # -----------------------------
# # Load Feather Utility
# # -----------------------------
# def load_feather_df(fname: str, ddir: str) -> pd.DataFrame:
#     path = os.path.join(ddir, fname)
#     if not os.path.exists(path):
#         logging.warning(f"File not found: {path}. Returning empty DataFrame.")
#         return pd.DataFrame()
#     df = pd.read_feather(path)
#     df = df.dropna(axis=1, how='all')
#     if 'time' in df.columns:
#         df['time'] = pd.to_datetime(df['time'])
#         df.set_index('time', inplace=True)
#     return df

# # -----------------------------
# # Load Datasets
# # -----------------------------
# PET_df = load_feather_df("PotEvap.feather", DATA_DIR)
# Rainf_df = load_feather_df("Rainf.feather", DATA_DIR)
# Evap_df = load_feather_df("EVap.feather", DATA_DIR)
# Q_usgs_df = load_feather_df("Q_USGS.feather", DATA_DIR)
# Q_nldas_df = load_feather_df("Q_nldas_mm_monthly.feather", DATA_DIR)
# Qsb_df = load_feather_df("Qsb.feather", DATA_DIR)
# M_df = load_feather_df("M.feather", DATA_DIR)
# Slope_basin = load_feather_df("slope.feather", DATA_DIR)
# S_init_df = load_feather_df("RootMoist.feather", DATA_DIR)
# G_init_df = load_feather_df("SoilM_0_200cm.feather", DATA_DIR)

# M_basin = M_df[Slope_basin.columns]
# M_basin.index = pd.to_datetime(M_basin.index, format='%Y-%m')
# M_basin = M_basin.loc[Evap_df.index]

# # -----------------------------
# # Load Calibrated Parameters
# # -----------------------------
# calibrated_path = os.path.join(PROJECT_ROOT, "SCE_cal_params", "final_calibrated_params.json")
# with open(calibrated_path, 'r') as f:
#     calibrated_params = json.load(f)

# # -----------------------------
# # Simulation Function
# # -----------------------------
# def simulate_basin(basin_id):
#     if basin_id not in calibrated_params:
#         return None, None

#     # Extract Data
#     p = calibrated_params[basin_id]
#     PET = PET_df[basin_id].values
#     P = Rainf_df[basin_id].values
#     Q_obs = Q_usgs_df[basin_id].values
#     Q_nldas = Q_nldas_df[basin_id].values
#     Evap = Evap_df[[basin_id]]
#     Qsb = Qsb_df[[basin_id]]
#     M = M_basin[[basin_id]]
#     Slope = Slope_basin[[basin_id]]
#     S_init = S_init_df[basin_id].mean()
#     G_init = G_init_df[basin_id].mean()

#     model_params = ModelParams(Smax=p['Smax'], Kperc=p['Kperc'], Kb=p['Kb'], Ke=p['Ke'], Cqq=p['Cqq'])

#     # -----------------------------
#     # Budyko ET Estimation
#     # -----------------------------
#     budyko = BudykoModelEstimator(Evap_df=Evap, Qsb_monthly=Qsb,
#                                   PotEvap_df=PET_df[[basin_id]],
#                                   M_basin=M, Slope_basin=Slope, ke=p['Ke'])
#     ET_B = budyko.estimate_budyko_et()[basin_id].values
#     ET_ke = Evap_df[basin_id].values
#     ET_nldas = PET * p['Ke']

#     # -----------------------------
#     # Assimilation: ET_ke => ET_ke_NLDA
#     # -----------------------------
#     config = EnKFConfig()
#     nens = config.nens
#     inflation = config.inflation
#     R_ET = config.R_ET
#     # X_et = np.tile(ET_nldas[:2], (1, nens))  # Initial guess
#     X_et = np.zeros((6, nens))
#     X_et[0, :] = S_init         # Soil Moisture
#     X_et[1, :] = G_init         # Groundwater
#     X_et[2, :] = p['Kperc']     # Kperc
#     X_et[3, :] = p['Kb']        # Kb
#     X_et[4, :] = p['Ke']        # Ke (to be updated)
#     X_et[5, :] = p['Cqq']       # Cqq


#     ET_ke_NLDA = []

#     for t in range(len(P)):
#         # et_ens_mean = np.mean(X_et[0])
#         et_ens_mean = np.mean(X_et[4, :])
#         ET_ke_NLDA.append(et_ens_mean)
#         if np.isfinite(ET_ke[t]):
#             # HX = X_et[0]
#             HX = X_et[4, :]
#             X_et = enkf_update(X_et, y_obs=ET_ke[t], HX=HX, R=R_ET, inflation=inflation)

#     # -----------------------------
#     # Scenario 1: Q_ke (ET from Ke * PET)
#     # -----------------------------
#     S, G = S_init, G_init
#     Q_ke = []
#     for P_t, PET_t in zip(P, PET):
#         S, G, _, Q, *_ = two_store_model_step(S, G, P_t, PET_t, model_params)
#         Q_ke.append(Q)

#     # -----------------------------
#     # Scenario 2: Q_ke_NLDA (ET_ke_NLDA used as ET_override)
#     # -----------------------------
#     S, G = S_init, G_init
#     Q_ke_NLDA = []
#     for t, (P_t, PET_t) in enumerate(zip(P, PET)):
#         S, G, _, Q, *_ = two_store_model_step(S, G, P_t, PET_t, model_params, ET_override=ET_ke_NLDA[t])
#         Q_ke_NLDA.append(Q)

#     # -----------------------------
#     # Scenario 3: Q_b (Budyko ET used as ET_override)
#     # -----------------------------
#     S, G = S_init, G_init
#     Q_b = []
#     for t, (P_t, PET_t) in enumerate(zip(P, PET)):
#         S, G, _, Q, *_ = two_store_model_step(S, G, P_t, PET_t, model_params, ET_override=ET_B[t])
#         Q_b.append(Q)

#     # -----------------------------
#     # Scenario 4: Q_b_DA and Q_b_ETNL_DA
#     # -----------------------------
#     S, G = S_init, G_init
#     X = np.zeros((6, nens))
#     X[0, :] = S_init
#     X[1, :] = G_init
#     X[2, :] = p['Kperc']
#     X[3, :] = p['Kb']
#     X[4, :] = p['Ke']
#     X[5, :] = p['Cqq']

#     Q_b_DA = []
#     Q_b_ETNL_DA = []

#     for t, (P_t, PET_t) in enumerate(zip(P, PET)):
#         # Forecast with Budyko ET
#         X, Q_ens_budyko, _ = enkf_forecast_step(X, P_t, PET_t, Smax_cal=p['Smax'], ET_B_t=ET_B[t])
#         Q_b_DA.append(np.mean(Q_ens_budyko))

#         # Forecast with default ET (Ke * PET)
#         X, Q_ens_nldas, _ = enkf_forecast_step(X, P_t, PET_t, Smax_cal=p['Smax'], ET_B_t=None)
#         Q_b_ETNL_DA.append(np.mean(Q_ens_nldas))

#         # Assimilation
#         if np.isfinite(Q_nldas[t]):
#             X = enkf_update(X, y_obs=Q_nldas[t], HX=Q_ens_budyko, R=config.R_Q, inflation=inflation)

#     # -----------------------------
#     # Output Results
#     # -----------------------------
#     results = pd.DataFrame({
#         'time': Evap_df.index,
#         'Q_obs': Q_obs,
#         'Q_nldas': Q_nldas,
#         'Q_ke': Q_ke,
#         'Q_ke_NLDA': Q_ke_NLDA,
#         'Q_b': Q_b,
#         'Q_b_DA': Q_b_DA,
#         'Q_b_ETNL_DA': Q_b_ETNL_DA,
#     }).set_index('time')

#     metrics = {
#         'Q_ke_KGE': calculate_kge(Q_obs, Q_ke),
#         'Q_ke_NSE': calculate_nse(Q_obs, Q_ke),
#         'Q_ke_NLDA_KGE': calculate_kge(Q_obs, Q_ke_NLDA),
#         'Q_ke_NLDA_NSE': calculate_nse(Q_obs, Q_ke_NLDA),
#         'Q_b_KGE': calculate_kge(Q_obs, Q_b),
#         'Q_b_NSE': calculate_nse(Q_obs, Q_b),
#         'Q_b_DA_KGE': calculate_kge(Q_obs, Q_b_DA),
#         'Q_b_DA_NSE': calculate_nse(Q_obs, Q_b_DA),
#         'Q_b_ETNL_DA_KGE': calculate_kge(Q_obs, Q_b_ETNL_DA),
#         'Q_b_ETNL_DA_NSE': calculate_nse(Q_obs, Q_b_ETNL_DA),
#     }

#     return results, metrics

# # -----------------------------
# # Run All Basins
# # -----------------------------
# if __name__ == '__main__':
#     all_basins = sorted(set(PET_df.columns) & set(Rainf_df.columns) & set(Qsb_df.columns) &
#                         set(Evap_df.columns) & set(M_basin.columns) & set(Slope_basin.columns) &
#                         set(Q_usgs_df.columns) & set(Q_nldas_df.columns))

#     results_all, metrics_rows = {}, []

#     for basin_id in tqdm(all_basins, desc="Running full simulation"):
#         result_df, metrics = simulate_basin(basin_id)
#         if result_df is not None:
#             results_all[basin_id] = result_df
#             result_df.reset_index().to_feather(os.path.join(RESULT_DIR, f"results_streamflow_{basin_id}.feather"))

#             for scenario in ['Q_ke', 'Q_ke_NLDA', 'Q_b', 'Q_b_DA', 'Q_b_ETNL_DA']:
#                 metrics_rows.append({
#                     'gauge_id': basin_id,
#                     'scenario': scenario,
#                     'KGE': metrics.get(f'{scenario}_KGE', np.nan),
#                     'NSE': metrics.get(f'{scenario}_NSE', np.nan),
#                 })

#     pd.DataFrame(metrics_rows).to_csv(os.path.join(RESULT_DIR, "streamflow_performance_metrics.csv"), index=False)
#     print("\n✅ All basin simulations completed and results saved.")




# run_simulation.py

"""
Main Simulation Script
Performs multi-scenario streamflow simulation with EnKF assimilation and Budyko-based ET estimates.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import ModelParams, two_store_model_step
from src.budyko import BudykoModelEstimator
from src.enkf import enkf_forecast_step, enkf_update, EnKFConfig
from src.metrics import calculate_kge, calculate_nse
from src.param_manager import load_all_calibrated_params

# -----------------------------
# Setup Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULT_DIR = os.path.join(PROJECT_ROOT, 'Simulation_results')
# os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------------
# Load Feather Utility
# -----------------------------
def load_feather_df(fname: str, ddir: str) -> pd.DataFrame:
    path = os.path.join(ddir, fname)
    if not os.path.exists(path):
        logging.warning(f"File not found: {path}. Returning empty DataFrame.")
        return pd.DataFrame()
    df = pd.read_feather(path)
    df = df.dropna(axis=1, how='all')
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    return df

# -----------------------------
# Load Datasets
# -----------------------------
PET_df = load_feather_df("PotEvap.feather", DATA_DIR)
Rainf_df = load_feather_df("Rainf.feather", DATA_DIR)
Evap_df = load_feather_df("EVap.feather", DATA_DIR)
Q_usgs_df = load_feather_df("Q_USGS.feather", DATA_DIR)
Q_nldas_df = load_feather_df("Q_nldas_mm_monthly.feather", DATA_DIR)
Qsb_df = load_feather_df("Qsb.feather", DATA_DIR)
M_df = load_feather_df("M.feather", DATA_DIR)
Slope_basin = load_feather_df("slope.feather", DATA_DIR)
S_init_df = load_feather_df("RootMoist.feather", DATA_DIR)
G_init_df = load_feather_df("SoilM_0_200cm.feather", DATA_DIR)

M_basin = M_df[Slope_basin.columns]
M_basin.index = pd.to_datetime(M_basin.index, format='%Y-%m')
M_basin = M_basin.loc[Evap_df.index]

# -----------------------------
# Load Calibrated Parameters
# -----------------------------
calibrated_path = os.path.join(PROJECT_ROOT, "SCE_cal_params", "final_calibrated_params.json")
with open(calibrated_path, 'r') as f:
    calibrated_params = json.load(f)

# -----------------------------
# Simulation Function
# -----------------------------
def simulate_basin(basin_id):
    if basin_id not in calibrated_params:
        return None, None

    # Extract Data
    p = calibrated_params[basin_id]
    PET = PET_df[basin_id].values
    P = Rainf_df[basin_id].values
    Q_obs = Q_usgs_df[basin_id].values
    Q_nldas = Q_nldas_df[basin_id].values
    Evap = Evap_df[[basin_id]]
    Qsb = Qsb_df[[basin_id]]
    M = M_basin[[basin_id]]
    Slope = Slope_basin[[basin_id]]
    S_init = S_init_df[basin_id].mean()
    G_init = G_init_df[basin_id].mean()

    model_params = ModelParams(Smax=p['Smax'], Kperc=p['Kperc'], Kb=p['Kb'], Ke=p['Ke'], Cqq=p['Cqq'])

    # ET Definitions
    ET_ke = PET * p['Ke']
    ET_nldas = Evap_df[basin_id].values

    budyko = BudykoModelEstimator(Evap_df=Evap, Qsb_monthly=Qsb,
                                  PotEvap_df=PET_df[[basin_id]],
                                  M_basin=M, Slope_basin=Slope, ke=p['Ke'])
    ET_B = budyko.estimate_budyko_et()[basin_id].values

    # Assimilate ET_nldas with ET_ke as truth to produce ET_ke_NLDA
    config = EnKFConfig()
    nens = config.nens
    inflation = config.inflation
    R_ET = config.R_ET

    X_et = np.tile(ET_nldas, (nens, 1)).T  # Ensemble shape: (time, nens)
    ET_ke_NLDA = []

    for t in range(len(P)):
        et_ens = X_et[t, :]
        ET_ke_NLDA.append(np.mean(et_ens))
        if np.isfinite(ET_ke[t]):
            HX = et_ens.copy()
            X_dummy = np.zeros((6, nens))
            X_dummy[4, :] = et_ens 

            HX = X_dummy[4, :].copy()
            X_dummy_updated = enkf_update(X_dummy, y_obs=ET_ke[t], HX=HX, R=R_ET, inflation=inflation)
            X_et[t, :] = X_dummy_updated[4, :]


    # Scenario 1: Q_ke
    S, G = S_init, G_init
    Q_ke = []
    for P_t, PET_t in zip(P, PET):
        S, G, _, Q, *_ = two_store_model_step(S, G, P_t, PET_t, model_params)
        Q_ke.append(Q)

    # Scenario 2: Q_ke_NLDA
    S, G = S_init, G_init
    Q_ke_NLDA = []
    for t, (P_t, PET_t) in enumerate(zip(P, PET)):
        S, G, _, Q, *_ = two_store_model_step(S, G, P_t, PET_t, model_params, ET_override=ET_ke_NLDA[t])
        Q_ke_NLDA.append(Q)

    # Scenario 3: Q_b
    S, G = S_init, G_init
    Q_b = []
    for t, (P_t, PET_t) in enumerate(zip(P, PET)):
        S, G, _, Q, *_ = two_store_model_step(S, G, P_t, PET_t, model_params, ET_override=ET_B[t])
        Q_b.append(Q)

    # Scenario 4: Q_b_DA and Q_b_ETNL_DA
    S, G = S_init, G_init
    X = np.zeros((6, nens))
    X[0, :] = S_init
    X[1, :] = G_init
    X[2, :] = p['Kperc']
    X[3, :] = p['Kb']
    X[4, :] = p['Ke']
    X[5, :] = p['Cqq']

    Q_b_DA = []
    Q_b_ETNL_DA = []

    for t, (P_t, PET_t) in enumerate(zip(P, PET)):
        X, Q_ens_budyko, _ = enkf_forecast_step(X, P_t, PET_t, Smax_cal=p['Smax'], ET_B_t=ET_B[t])
        Q_b_DA.append(np.mean(Q_ens_budyko))

        X, Q_ens_nldas, _ = enkf_forecast_step(X, P_t, PET_t, Smax_cal=p['Smax'], ET_B_t=None)
        Q_b_ETNL_DA.append(np.mean(Q_ens_nldas))

        if np.isfinite(Q_nldas[t]):
            X = enkf_update(X, y_obs=Q_nldas[t], HX=Q_ens_budyko, R=config.R_Q, inflation=inflation)

    results = pd.DataFrame({
        'time': Evap_df.index,
        'Q_obs': Q_obs,
        'Q_nldas': Q_nldas,
        'Q_ke': Q_ke,
        'Q_ke_NLDA': Q_ke_NLDA,
        'Q_b': Q_b,
        'Q_b_DA': Q_b_DA,
        'Q_b_ETNL_DA': Q_b_ETNL_DA,
    }).set_index('time')

    metrics = {
        'Q_ke_KGE': calculate_kge(Q_obs, Q_ke),
        'Q_ke_NSE': calculate_nse(Q_obs, Q_ke),
        'Q_ke_NLDA_KGE': calculate_kge(Q_obs, Q_ke_NLDA),
        'Q_ke_NLDA_NSE': calculate_nse(Q_obs, Q_ke_NLDA),
        'Q_b_KGE': calculate_kge(Q_obs, Q_b),
        'Q_b_NSE': calculate_nse(Q_obs, Q_b),
        'Q_b_DA_KGE': calculate_kge(Q_obs, Q_b_DA),
        'Q_b_DA_NSE': calculate_nse(Q_obs, Q_b_DA),
        'Q_b_ETNL_DA_KGE': calculate_kge(Q_obs, Q_b_ETNL_DA),
        'Q_b_ETNL_DA_NSE': calculate_nse(Q_obs, Q_b_ETNL_DA),
    }

    return results, metrics

# -----------------------------
# Run All Basins
# -----------------------------
if __name__ == '__main__':
    all_basins = sorted(set(PET_df.columns) & set(Rainf_df.columns) & set(Qsb_df.columns) &
                        set(Evap_df.columns) & set(M_basin.columns) & set(Slope_basin.columns) &
                        set(Q_usgs_df.columns) & set(Q_nldas_df.columns))

    results_all, metrics_rows = {}, []

    for basin_id in tqdm(all_basins, desc="Running full simulation"):
        result_df, metrics = simulate_basin(basin_id)
        if result_df is not None:
            results_all[basin_id] = result_df
            result_df.reset_index().to_feather(os.path.join(RESULT_DIR, f"results_streamflow_{basin_id}.feather"))

            for scenario in ['Q_ke', 'Q_ke_NLDA', 'Q_b', 'Q_b_DA', 'Q_b_ETNL_DA']:
                metrics_rows.append({
                    'gauge_id': basin_id,
                    'scenario': scenario,
                    'KGE': metrics.get(f'{scenario}_KGE', np.nan),
                    'NSE': metrics.get(f'{scenario}_NSE', np.nan),
                })

    pd.DataFrame(metrics_rows).to_csv(os.path.join(RESULT_DIR, "streamflow_performance_metrics.csv"), index=False)
    print("\n🚀 All basin simulations completed and results saved.")