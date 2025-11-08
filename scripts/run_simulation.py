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
# # os.makedirs(RESULT_DIR, exist_ok=True)

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
#     S_init = 20
#     G_init = 15

#     model_params = ModelParams(Smax=p['Smax'], Kperc=p['Kperc'], Kb=p['Kb'], Ke=p['Ke'], Cqq=p['Cqq'])

#     # ET Definitions
#     ET_ke = PET * p['Ke']
#     ET_nldas = Evap_df[basin_id].values

#     budyko = BudykoModelEstimator(Evap_df=Evap, Qsb_monthly=Qsb,
#                                   PotEvap_df=PET_df[[basin_id]],
#                                   M_basin=M, Slope_basin=Slope, ke=p['Ke'])
#     ET_B = budyko.estimate_budyko_et()[basin_id].values

#     # Assimilate ET_nldas with ET_ke as truth to produce ET_ke_NLDA
#     config = EnKFConfig()
#     nens = config.nens
#     inflation = config.inflation
#     R_ET = config.R_ET

#     X_et = np.tile(ET_nldas, (nens, 1)).T  # Ensemble shape: (time, nens)
#     ET_ke_NLDA = []

#     for t in range(len(P)):
#         et_ens = X_et[t, :]
#         ET_ke_NLDA.append(np.mean(et_ens))
#         if np.isfinite(ET_ke[t]):
#             HX = et_ens.copy()
#             X_dummy = np.zeros((6, nens))
#             X_dummy[4, :] = et_ens 

#             HX = X_dummy[4, :].copy()
#             X_dummy_updated = enkf_update(X_dummy, y_obs=ET_ke[t], HX=HX, R=R_ET, inflation=inflation)
#             X_et[t, :] = X_dummy_updated[4, :]


#     # Scenario 1: Q_ke
#     S, G = S_init, G_init
#     Q_ke = []
#     for P_t, PET_t in zip(P, PET):
#         S, G, _, Q, *_ = two_store_model_step(S, G, P_t, PET_t, model_params)
#         Q_ke.append(Q)

#     # Scenario 2: Q_ke_NLDA
#     S, G = S_init, G_init
#     Q_ke_NLDA = []
#     for t, (P_t, PET_t) in enumerate(zip(P, PET)):
#         S, G, _, Q, *_ = two_store_model_step(S, G, P_t, PET_t, model_params, ET_override=ET_ke_NLDA[t])
#         Q_ke_NLDA.append(Q)

#     # Scenario 3: Q_b
#     S, G = S_init, G_init
#     Q_b = []
#     for t, (P_t, PET_t) in enumerate(zip(P, PET)):
#         S, G, _, Q, *_ = two_store_model_step(S, G, P_t, PET_t, model_params, ET_override=ET_B[t])
#         Q_b.append(Q)

#     # Scenario 4: Q_b_DA and Q_b_ETNL_DA
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
#         X, Q_ens_budyko, _ = enkf_forecast_step(X, P_t, PET_t, Smax_cal=p['Smax'], ET_B_t=ET_B[t])
#         Q_b_DA.append(np.mean(Q_ens_budyko))

#         X, Q_ens_nldas, _ = enkf_forecast_step(X, P_t, PET_t, Smax_cal=p['Smax'], ET_B_t=None)
#         Q_b_ETNL_DA.append(np.mean(Q_ens_nldas))

#         if np.isfinite(Q_nldas[t]):
#             X = enkf_update(X, y_obs=Q_nldas[t], HX=Q_ens_budyko, R=config.R_Q, inflation=inflation)

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


# from concurrent.futures import ProcessPoolExecutor, as_completed

# def run_and_save_basin(basin_id):
#     try:
#         result_df, metrics = simulate_basin(basin_id)
#         if result_df is not None:
#             result_path = os.path.join(RESULT_DIR, f"results_streamflow_{basin_id}.feather")
#             # result_df.reset_index().to_feather(result_path)

#             metrics_rows = []
#             for scenario in ['Q_ke', 'Q_ke_NLDA', 'Q_b', 'Q_b_DA', 'Q_b_ETNL_DA']:
#                 metrics_rows.append({
#                     'gauge_id': basin_id,
#                     'scenario': scenario,
#                     'KGE': metrics.get(f'{scenario}_KGE', np.nan),
#                     'NSE': metrics.get(f'{scenario}_NSE', np.nan),
#                 })
#             return metrics_rows
#     except Exception as e:
#         print(f"❌ Error processing {basin_id}: {e}")
#     return []


# if __name__ == '__main__':
#     from multiprocessing import cpu_count
#     os.makedirs(RESULT_DIR, exist_ok=True)

#     all_basins = sorted(set(PET_df.columns) & set(Rainf_df.columns) & set(Qsb_df.columns) &
#                         set(Evap_df.columns) & set(M_basin.columns) & set(Slope_basin.columns) &
#                         set(Q_usgs_df.columns) & set(Q_nldas_df.columns))

#     all_metrics = []
#     with ProcessPoolExecutor(max_workers=cpu_count() - 1) as executor:
#         futures = {executor.submit(run_and_save_basin, basin): basin for basin in all_basins}
#         for future in tqdm(as_completed(futures), total=len(futures), desc="Running in parallel"):
#             metrics_rows = future.result()
#             all_metrics.extend(metrics_rows)

#     pd.DataFrame(all_metrics).to_csv(os.path.join(RESULT_DIR, "streamflow_performance_metrics.csv"), index=False)
#     print("\n✅ All basin simulations completed and results saved.")




# # # run_simulation.py

# # """
# # Main Simulation Script
# # Performs multi-scenario streamflow simulation with EnKF assimilation and Budyko-based ET estimates.
# # """

# # import sys
# # import os
# # import json
# # import numpy as np
# # import pandas as pd
# # from tqdm import tqdm
# # import logging
# # from scipy.optimize import minimize

# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # from src.model import ModelParams, two_store_model_step
# # from src.budyko import BudykoModelEstimator
# # from src.enkf import enkf_forecast_step, enkf_update, EnKFConfig
# # from src.metrics import calculate_kge, calculate_nse
# # from src.param_manager import load_all_calibrated_params

# # # -----------------------------
# # # Setup Paths
# # # -----------------------------
# # PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# # DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
# # RESULT_DIR = os.path.join(PROJECT_ROOT, 'Simulation_results')

# # # -----------------------------
# # # Load Feather Utility
# # # -----------------------------
# # def load_feather_df(fname: str, ddir: str) -> pd.DataFrame:
# #     path = os.path.join(ddir, fname)
# #     if not os.path.exists(path):
# #         logging.warning(f"File not found: {path}. Returning empty DataFrame.")
# #         return pd.DataFrame()
# #     df = pd.read_feather(path)
# #     df = df.dropna(axis=1, how='all')
# #     if 'time' in df.columns:
# #         df['time'] = pd.to_datetime(df['time'])
# #         df.set_index('time', inplace=True)
# #     return df

# # # -----------------------------
# # # Load Datasets
# # # -----------------------------
# # PET_df = load_feather_df("PotEvap.feather", DATA_DIR)
# # Rainf_df = load_feather_df("Rainf.feather", DATA_DIR)
# # Evap_df = load_feather_df("EVap.feather", DATA_DIR)
# # Q_usgs_df = load_feather_df("Q_USGS.feather", DATA_DIR)
# # Q_nldas_df = load_feather_df("Q_nldas_mm_monthly.feather", DATA_DIR)
# # Qsb_df = load_feather_df("Qsb.feather", DATA_DIR)
# # M_df = load_feather_df("M.feather", DATA_DIR)
# # Slope_basin = load_feather_df("slope.feather", DATA_DIR)
# # S_init_df = load_feather_df("RootMoist.feather", DATA_DIR)
# # G_init_df = load_feather_df("SoilM_0_200cm.feather", DATA_DIR)
# # # S_init = 5
# # # G_init = 1.5

# # M_basin = M_df[Slope_basin.columns]
# # M_basin.index = pd.to_datetime(M_basin.index, format='%Y-%m')
# # M_basin = M_basin.loc[Evap_df.index]

# # # -----------------------------
# # # Load Calibrated Parameters
# # # -----------------------------
# # calibrated_path = os.path.join(PROJECT_ROOT, "SCE_cal_params", "final_calibrated_params.json")
# # with open(calibrated_path, 'r') as f:
# #     calibrated_params = json.load(f)

# # # -----------------------------
# # # Simulation Function 
# # # -----------------------------
# # def simulate_basin(basin_id, S_init=S_init, G_init=G_init):
# #     if basin_id not in calibrated_params:
# #         return None, None

# #     p = calibrated_params[basin_id]
# #     PET = PET_df[basin_id].values
# #     P = Rainf_df[basin_id].values
# #     Q_obs = Q_usgs_df[basin_id].values
# #     Q_nldas = Q_nldas_df[basin_id].values
# #     Evap = Evap_df[[basin_id]]
# #     Qsb = Qsb_df[[basin_id]]
# #     M = M_basin[[basin_id]]
# #     Slope = Slope_basin[[basin_id]]

# #     model_params = ModelParams(Smax=p['Smax'], Kperc=p['Kperc'], Kb=p['Kb'], Ke=p['Ke'], Cqq=p['Cqq'])

# #     ET_ke = PET * p['Ke']
# #     ET_nldas = Evap_df[basin_id].values

# #     budyko = BudykoModelEstimator(Evap_df=Evap, Qsb_monthly=Qsb,
# #                                   PotEvap_df=PET_df[[basin_id]],
# #                                   M_basin=M, Slope_basin=Slope, ke=p['Ke'])
# #     ET_B = budyko.estimate_budyko_et()[basin_id].values

# #     config = EnKFConfig()
# #     nens = config.nens
# #     inflation = config.inflation
# #     R_ET = config.R_ET

# #     X_et = np.tile(ET_nldas, (nens, 1)).T
# #     ET_ke_NLDA = []

# #     for t in range(len(P)):
# #         et_ens = X_et[t, :]
# #         ET_ke_NLDA.append(np.mean(et_ens))
# #         if np.isfinite(ET_ke[t]):
# #             HX = et_ens.copy()
# #             X_dummy = np.zeros((6, nens))
# #             X_dummy[4, :] = et_ens
# #             HX = X_dummy[4, :].copy()
# #             X_dummy_updated = enkf_update(X_dummy, y_obs=ET_ke[t], HX=HX, R=R_ET, inflation=inflation)
# #             X_et[t, :] = X_dummy_updated[4, :]

# #     S, G = S_init, G_init
# #     Q_ke = []
# #     for P_t, PET_t in zip(P, PET):
# #         S, G, _, Q, *_ = two_store_model_step(S, G, P_t, PET_t, model_params)
# #         Q_ke.append(Q)

# #     S, G = S_init, G_init
# #     Q_ke_NLDA = []
# #     for t, (P_t, PET_t) in enumerate(zip(P, PET)):
# #         S, G, _, Q, *_ = two_store_model_step(S, G, P_t, PET_t, model_params, ET_override=ET_ke_NLDA[t])
# #         Q_ke_NLDA.append(Q)

# #     S, G = S_init, G_init
# #     Q_b = []
# #     for t, (P_t, PET_t) in enumerate(zip(P, PET)):
# #         S, G, _, Q, *_ = two_store_model_step(S, G, P_t, PET_t, model_params, ET_override=ET_B[t])
# #         Q_b.append(Q)

# #     S, G = S_init, G_init
# #     X = np.zeros((6, nens))
# #     X[0, :] = S_init
# #     X[1, :] = G_init
# #     X[2, :] = p['Kperc']
# #     X[3, :] = p['Kb']
# #     X[4, :] = p['Ke']
# #     X[5, :] = p['Cqq']

# #     Q_b_DA = []
# #     Q_b_ETNL_DA = []

# #     for t, (P_t, PET_t) in enumerate(zip(P, PET)):
# #         X, Q_ens_budyko, _ = enkf_forecast_step(X, P_t, PET_t, Smax_cal=p['Smax'], ET_B_t=ET_B[t])
# #         Q_b_DA.append(np.mean(Q_ens_budyko))

# #         X, Q_ens_nldas, _ = enkf_forecast_step(X, P_t, PET_t, Smax_cal=p['Smax'], ET_B_t=None)
# #         Q_b_ETNL_DA.append(np.mean(Q_ens_nldas))

# #         if np.isfinite(Q_nldas[t]):
# #             X = enkf_update(X, y_obs=Q_nldas[t], HX=Q_ens_budyko, R=config.R_Q, inflation=inflation)
# # # 'Q_ke_NLDA', 'Q_b', 'Q_b_DA', 'Q_b_ETNL_DA'
# #     results = pd.DataFrame({
# #         'time': Evap_df.index,
# #         'Q_obs': Q_obs,
# #         'Q_nldas': Q_nldas,
# #         'Q_ke': Q_ke,
# #         'Q_B_ETnldas_DA': Q_ke_NLDA,
# #         # 'Q_b': Q_b,
# #         # 'Q_b_DA': Q_b_DA,
# #         'Q_B': Q_b_ETNL_DA,
# #     }).set_index('time')

# #     metrics = {
# #         'Q_ke_KGE': calculate_kge(Q_obs, Q_ke),
# #         'Q_ke_NSE': calculate_nse(Q_obs, Q_ke),
# #         'Q_ke_NLDA_KGE': calculate_kge(Q_obs, Q_ke_NLDA),
# #         'Q_ke_NLDA_NSE': calculate_nse(Q_obs, Q_ke_NLDA),
# #         'Q_b_KGE': calculate_kge(Q_obs, Q_b),
# #         'Q_b_NSE': calculate_nse(Q_obs, Q_b),
# #         'Q_b_DA_KGE': calculate_kge(Q_obs, Q_b_DA),
# #         'Q_b_DA_NSE': calculate_nse(Q_obs, Q_b_DA),
# #         'Q_b_ETNL_DA_KGE': calculate_kge(Q_obs, Q_b_ETNL_DA),
# #         'Q_b_ETNL_DA_NSE': calculate_nse(Q_obs, Q_b_ETNL_DA),
# #     }

# #     return results, metrics

# # def objective_init(params, basin_id):
# #     """
# #     Objective function for optimizing initial states S_init and G_init.
# #     Combines KGE and NSE to balance flow accuracy and variability.
# #     """
# #     S_init, G_init = params
# #     try:
# #         result_df, metrics = simulate_basin(basin_id, S_init, G_init)
# #         if metrics is None:
# #             return np.inf

# #         # Extract individual KGE and NSE scores
# #         kge_scores = [
# #             metrics.get('Q_ke_KGE', np.nan),
# #             metrics.get('Q_ke_NLDA_KGE', np.nan),
# #             metrics.get('Q_b_KGE', np.nan),
# #             metrics.get('Q_b_DA_KGE', np.nan),
# #             metrics.get('Q_b_ETNL_DA_KGE', np.nan)
# #         ]
# #         nse_scores = [
# #             metrics.get('Q_ke_NSE', np.nan),
# #             metrics.get('Q_ke_NLDA_NSE', np.nan),
# #             metrics.get('Q_b_NSE', np.nan),
# #             metrics.get('Q_b_DA_NSE', np.nan),
# #             metrics.get('Q_b_ETNL_DA_NSE', np.nan)
# #         ]

# #         # Drop NaNs before averaging
# #         kge_scores = [k for k in kge_scores if not np.isnan(k)]
# #         nse_scores = [n for n in nse_scores if not np.isnan(n)]

# #         if not kge_scores or not nse_scores:
# #             return np.inf

# #         # Weighted average of KGE and NSE (you can adjust weights here)
# #         kge_mean = np.mean(kge_scores)
# #         nse_mean = np.mean(nse_scores)
# #         combined_score = 0.6 * kge_mean + 0.4 * nse_mean

# #         # Return negative for minimization
# #         return -combined_score

# #     except Exception as e:
# #         print(f"❌ Optimization error for {basin_id}: {e}")
# #         return np.inf


# # from concurrent.futures import ProcessPoolExecutor, as_completed

# # def run_and_save_basin(basin_id):
# #     try:
# #         opt_result = minimize(objective_init, 
# #                               x0=[0.35, 0.21], 
# #                               args=(basin_id,),
# #                                 bounds=[(0.01, 1.0), (0.01, 1.0)], 
# #                                 method='L-BFGS-B', options={'disp': True, 'maxiter': 200, 'ftol': 1e-6})
# #         S_opt, G_opt = opt_result.x
# #         result_df, metrics = simulate_basin(basin_id, S_init=S_opt, G_init=G_opt)
# #         if result_df is not None:
# #             result_path = os.path.join(RESULT_DIR, f"results_streamflow_{basin_id}.feather")
# #             result_df.reset_index().to_feather(result_path)

# #             metrics_rows = []
# #             for scenario in ['Q_ke', 'Q_ke_NLDA', 'Q_b', 'Q_b_DA', 'Q_b_ETNL_DA']:
# #                 metrics_rows.append({
# #                     'gauge_id': basin_id,
# #                     'scenario': scenario,
# #                     'KGE': metrics.get(f'{scenario}_KGE', np.nan),
# #                     'NSE': metrics.get(f'{scenario}_NSE', np.nan),
# #                 })
# #             return metrics_rows
# #     except Exception as e:
# #         print(f"❌ Error processing {basin_id}: {e}")
# #     return []

# # if __name__ == '__main__':
# #     from multiprocessing import cpu_count
# #     os.makedirs(RESULT_DIR, exist_ok=True)

# #     all_basins = sorted(set(PET_df.columns) & set(Rainf_df.columns) & set(Qsb_df.columns) &
# #                         set(Evap_df.columns) & set(M_basin.columns) & set(Slope_basin.columns) &
# #                         set(Q_usgs_df.columns) & set(Q_nldas_df.columns))

# #     all_metrics = []
# #     with ProcessPoolExecutor(max_workers=cpu_count() - 1) as executor:
# #         futures = {executor.submit(run_and_save_basin, basin): basin for basin in all_basins}
# #         for future in tqdm(as_completed(futures), total=len(futures), desc="Running in parallel"):
# #             metrics_rows = future.result()
# #             all_metrics.extend(metrics_rows)

# #     pd.DataFrame(all_metrics).to_csv(os.path.join(RESULT_DIR, "streamflow_performance_metrics.csv"), index=False)
# #     print("\n✅ All basin simulations completed and results saved.")













import sys
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

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
os.makedirs(RESULT_DIR, exist_ok=True)

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
# # -----------------------------

def simulate_basin(basin_id):
    if basin_id not in calibrated_params:
        return None, None

    # -----------------------------
    # Extract basin series (1-D np arrays aligned to Evap_df.index)
    # -----------------------------
    p = calibrated_params[basin_id]
    idx = Evap_df.index

    PET = PET_df[basin_id].reindex(idx).values
    P = Rainf_df[basin_id].reindex(idx).values
    Q_obs = Q_usgs_df.get(basin_id, pd.Series(index=idx)).reindex(idx).values
    Q_nldas = Q_nldas_df[basin_id].reindex(idx).values
    ET_nldas = Evap_df[basin_id].reindex(idx).values
    Qsb = Qsb_df[[basin_id]].reindex(idx)  # DataFrame for Budyko estimator
    M = M_basin[[basin_id]].reindex(idx)
    Slope = Slope_basin[[basin_id]]  # 1-row, same columns
    Evap = Evap_df[[basin_id]].reindex(idx)  # for estimator

    # -----------------------------
    # Initial conditions / params
    # -----------------------------
    S_init = p.get('S_init', p['Smax'])
    G_init = p.get('G_init', p['Smax'])
    Gmax_factor = p.get('Gmax_factor', 3.0)

    model_params = ModelParams(
        Smax=p['Smax'], Kperc=p['Kperc'], Kb=p['Kb'], Ke=p['Ke'], Cqq=p['Cqq']
    )

    # -----------------------------
    # ET definitions (Base & Budyko)
    # -----------------------------
    ET_ke = PET * p['Ke']  # base scenario ET

    # Budyko estimator: try to obtain ET_B, omega_true, omega_MLR
    budyko = BudykoModelEstimator(
        Evap_df=Evap,
        Qsb_monthly=Qsb,
        PotEvap_df=PET_df[[basin_id]],
        M_basin=M,
        Slope_basin=Slope,
        ke=p['Ke']
    )

    # ET_B
    ET_B = budyko.estimate_budyko_et()[basin_id].reindex(idx).values

    # omega_true & omega_MLR (robust to method naming)
    def _safe_series(budyko, attr_names, basin_id, idx, fallback_value=np.nan):
        """
        Robustly fetch a basin-specific array (e.g., omega_true or omega_MLR)
        from a BudykoModelEstimator, checking both methods and attributes.
        """
        for name in attr_names:
            # case 1: it's a callable method
            if hasattr(budyko, name):
                out = getattr(budyko, name)()
                if isinstance(out, pd.DataFrame) and basin_id in out.columns:
                    return out[basin_id].reindex(idx).values
                elif isinstance(out, pd.Series):
                    return out.reindex(idx).values
            # case 2: it's a stored attribute
            if hasattr(budyko, name):
                obj = getattr(budyko, name)
                if isinstance(obj, pd.DataFrame) and basin_id in obj.columns:
                    return obj[basin_id].reindex(idx).values
                elif isinstance(obj, pd.Series):
                    return obj.reindex(idx).values

        # case 3: direct attribute names in common usage
        for name in ["omega_true", "omega_MLR"]:
            if hasattr(budyko, name):
                obj = getattr(budyko, name)
                if isinstance(obj, pd.DataFrame) and basin_id in obj.columns:
                    return obj[basin_id].reindex(idx).values

        # fallback: full NaN vector
        return np.full(len(idx), fallback_value, dtype=float)

    omega_true = _safe_series(budyko,["compute_omega_true", "omega_true"], basin_id,idx)
    omega_MLR = _safe_series(budyko,["fit_and_compute_omega_mlr", "omega_MLR"],basin_id,idx)

    # -----------------------------
    # EnKF config (for ET assimilation steps)
    # -----------------------------
    config = EnKFConfig()
    nens = config.nens
    inflation = config.inflation
    R_ET = config.R_ET

    rng = np.random.default_rng(hash(basin_id) % (2**32 - 1))

    # ---------- ET_ke_NLDAS_ass (assimilate NLDAS ET toward ET_ke)
    X_et_base = np.tile(ET_nldas, (nens, 1)).T + rng.normal(0, 0.05, (len(P), nens))
    ET_ke_NLDAS_ass = np.empty(len(P))
    for t in range(len(P)):
        et_ens = X_et_base[t, :]
        ET_ke_NLDAS_ass[t] = et_ens.mean()
        if np.isfinite(ET_ke[t]):
            X_dummy = np.zeros((6, nens))
            X_dummy[4, :] = et_ens  # put ET in a row for update
            HX = X_dummy[4, :].copy()
            X_dummy_updated = enkf_update(
                X_dummy, y_obs=ET_ke[t], HX=HX, R=R_ET, inflation=inflation
            )
            X_et_base[t, :] = X_dummy_updated[4, :]

    # ---------- ET_B_NLDAS_ass (assimilate NLDAS ET toward ET_B)
    X_et_bud = np.tile(ET_nldas, (nens, 1)).T + rng.normal(0, 0.05, (len(P), nens))
    ET_B_NLDAS_ass = np.empty(len(P))
    for t in range(len(P)):
        et_ens = X_et_bud[t, :]
        ET_B_NLDAS_ass[t] = et_ens.mean()
        if np.isfinite(ET_B[t]):
            X_dummy = np.zeros((6, nens))
            X_dummy[4, :] = et_ens
            HX = X_dummy[4, :].copy()
            X_dummy_updated = enkf_update(
                X_dummy, y_obs=ET_B[t], HX=HX, R=R_ET, inflation=inflation
            )
            X_et_bud[t, :] = X_dummy_updated[4, :]

    # -----------------------------
    # Streamflow simulations
    # -----------------------------
    # Scenario 1: Q_ke (deterministic, ET = ET_ke)
    S, G = S_init, G_init
    Q_ke = []
    for P_t, PET_t in zip(P, PET):
        S, G, _, Q, *_ = two_store_model_step(S, G, P_t, PET_t, model_params)
        G = np.clip(G, 0, Gmax_factor * p['Smax'])
        Q_ke.append(Q)
    Q_ke = np.asarray(Q_ke)

    # Scenario 1 DA: Q_ke_NLDA (deterministic, ET override = ET_ke_NLDAS_ass)
    S, G = S_init, G_init
    Q_ke_NLDA = []
    for t, (P_t, PET_t) in enumerate(zip(P, PET)):
        S, G, _, Q, *_ = two_store_model_step(
            S, G, P_t, PET_t, model_params, ET_override=ET_ke_NLDAS_ass[t]
        )
        G = np.clip(G, 0, Gmax_factor * p['Smax'])
        Q_ke_NLDA.append(Q)
    Q_ke_NLDA = np.asarray(Q_ke_NLDA)

    # Scenario 2 (Budyko): Q_b (deterministic, ET override = ET_B)
    S, G = S_init, G_init
    Q_b = []
    for t, (P_t, PET_t) in enumerate(zip(P, PET)):
        S, G, _, Q, *_ = two_store_model_step(
            S, G, P_t, PET_t, model_params, ET_override=ET_B[t]
        )
        G = np.clip(G, 0, Gmax_factor * p['Smax'])
        Q_b.append(Q)
    Q_b = np.asarray(Q_b)

    # Scenario 2 DA (Q DA path): Q_b_DA and (ET-NLDA path): Q_b_ETNL_DA (as you had)
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
        # Forecast using Budyko ET
        X, Q_ens_budyko, _ = enkf_forecast_step(X, P_t, PET_t, Smax_cal=p['Smax'], ET_B_t=ET_B[t])
        Q_b_DA.append(np.mean(Q_ens_budyko))

        # Forecast using model ET (or ET from NLDA path as per your enkf_forecast design)
        X, Q_ens_nldas, _ = enkf_forecast_step(X, P_t, PET_t, Smax_cal=p['Smax'], ET_B_t=None)
        Q_b_ETNL_DA.append(np.mean(Q_ens_nldas))

        # Assimilate streamflow (NLDAS Q)
        if np.isfinite(Q_nldas[t]):
            X = enkf_update(X, y_obs=Q_nldas[t], HX=Q_ens_budyko, R=config.R_Q, inflation=inflation)

    Q_b_DA = np.asarray(Q_b_DA)
    Q_b_ETNL_DA = np.asarray(Q_b_ETNL_DA)

    # -----------------------------
    # Assemble results DataFrame (everything you asked for)
    # -----------------------------
    results = pd.DataFrame({
        'time': idx,
        'omega_true': omega_true,
        'omega_MLR': omega_MLR,
        'ET_ke': ET_ke,
        'ET_B': ET_B,
        'ET_nldas': ET_nldas,
        'ET_ke_NLDAS_ass': ET_ke_NLDAS_ass,
        'ET_B_NLDAS_ass': ET_B_NLDAS_ass,
        'Q_obs': Q_obs,
        'Q_nldas': Q_nldas,
        'Q_ke': Q_ke,
        'Q_ke_NLDA': Q_ke_NLDA,
        'Q_b': Q_b,
        'Q_b_DA': Q_b_DA,
        'Q_b_ETNL_DA': Q_b_ETNL_DA,
    }).set_index('time')

    # -----------------------------
    # Metrics (unchanged list + feel free to extend)
    # -----------------------------
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
# Parallel Execution
# -----------------------------
def run_and_save_basin(basin_id):
    try:
        result_df, metrics = simulate_basin(basin_id)
        if result_df is not None:
            result_path = os.path.join(RESULT_DIR, f"results_streamflow_{basin_id}.feather")
            result_df.reset_index().to_feather(result_path)

            metrics_rows = []
            for scenario in ['Q_ke', 'Q_ke_NLDA', 'Q_b', 'Q_b_DA', 'Q_b_ETNL_DA']:
                metrics_rows.append({
                    'gauge_id': basin_id,
                    'scenario': scenario,
                    'KGE': metrics.get(f'{scenario}_KGE', np.nan),
                    'NSE': metrics.get(f'{scenario}_NSE', np.nan),
                })
            return metrics_rows
    except Exception as e:
        print(f"❌ Error processing {basin_id}: {e}")
    return []

if __name__ == '__main__':
    from multiprocessing import cpu_count
    os.makedirs(RESULT_DIR, exist_ok=True)

    all_basins = sorted(set(PET_df.columns) & set(Rainf_df.columns) & set(Qsb_df.columns) &
                        set(Evap_df.columns) & set(M_basin.columns) & set(Slope_basin.columns) &
                        set(Q_usgs_df.columns) & set(Q_nldas_df.columns))

    all_metrics = []
    with ProcessPoolExecutor(max_workers=cpu_count() - 1) as executor:
        futures = {executor.submit(run_and_save_basin, basin): basin for basin in all_basins}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running in parallel"):
            metrics_rows = future.result()
            all_metrics.extend(metrics_rows)

    pd.DataFrame(all_metrics).to_csv(os.path.join(RESULT_DIR, "streamflow_performance_metrics.csv"), index=False)
    print("\n✅ All basin simulations completed and results saved.")
