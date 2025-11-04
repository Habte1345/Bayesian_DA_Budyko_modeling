
# import os
# import sys
# import pandas as pd
# import numpy as np
# import json
# from typing import Dict, Any, Tuple, Optional
# from multiprocessing import cpu_count
# import warnings
# import spotpy
# import requests
# import io
# import geopandas as gpd

# # Suppress NumPy warnings (but still check for NaN/Inf explicitly)
# warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(PROJECT_ROOT)

# # try:
# #     # Ensure all your core modules are accessible
# #     from src.model import ModelParams, two_store_model_step
# #     from src.metrics import calculate_kge
# #     from data.data_processor import load_and_prepare_data
# # except ImportError as e:
# #     print(f"FATAL: Core module import failed. Error: {e}")
# #     sys.exit(1)

# # # Constants
# # S_MAX_CEILING = 2500.0
# # G_MAX_CEILING = S_MAX_CEILING * 3.0
# # # FIXED: Smax set to 700.0 for better initial state/fallback consistency with a general model.
# # DEFAULT_FALLBACK_PARAMS = {
# #     'Kperc': 0.85, 'Kb': 0.83, 'Ke': 0.77, 'Cqq': 20.0, 'bias': 0.0,
# #     'S_init': 20.0, 'G_init': 15.0, 'Smax': 700.0, 'KGE': 0.0
# # }
# # MIN_KGE_THRESHOLD = 0.6


# # def run_forward_model(P_data, PET_data, Q_obs, initial_state: tuple, params: ModelParams) -> tuple:
# #     nmonths = len(P_data)
# #     S_curr, G_curr, bias = initial_state
# #     Q_sim = np.zeros(nmonths)

# #     for t in range(nmonths):
# #         P_t, PET_t = max(P_data[t], 0.0), max(PET_data[t], 0.0)
# #         try:
# #             S_next, G_next, _, Q_t_no_bias, _, _, _ = two_store_model_step(
# #                 S_curr, G_curr, P_t, PET_t, params, ET_override=None
# #             )
# #         except Exception:
# #             # If model step fails (e.g., math error), use current states and zero flow
# #             S_next, G_next, Q_t_no_bias = S_curr, G_curr, 0.0

# #         S_curr = np.clip(S_next, 0.0, params.Smax)
# #         G_curr = np.clip(G_next, 0.0, G_MAX_CEILING)
# #         Q_sim[t] = max(Q_t_no_bias + bias, 0.0)

# #     # CRITICAL: Check for non-finite values in the simulated output
# #     if not np.all(np.isfinite(Q_sim)):
# #         return np.zeros(1), np.zeros(1) 

# #     spin_up = 60
# #     Q_sim_clean = Q_sim[spin_up:]
# #     Q_obs_clean = Q_obs[spin_up:]
# #     valid_mask = ~(np.isnan(Q_sim_clean) | np.isnan(Q_obs_clean))

# #     if valid_mask.sum() < 12:
# #         return np.zeros(1), np.zeros(1)

# #     return Q_sim_clean[valid_mask], Q_obs_clean[valid_mask]


# # def calculate_initial_states(SM_data, Q_obs, P_data, basin_area_km2):
# #     p_mean = np.nanmean(P_data) if not np.all(np.isnan(P_data)) else 50.0
# #     q_mean = np.nanmean(Q_obs) if not np.all(np.isnan(Q_obs)) else 5.0
# #     sm_max = 5.0

# #     # FIXED: Revert Smax estimate minimum to 200.0 to allow for smaller catchments
# #     smax_estimate = max(sm_max * 0.5, p_mean * 0.25, 200.0) 
# #     final_smax = np.clip(smax_estimate, 200.0, S_MAX_CEILING)
# #     S_init_heuristic = np.clip(final_smax * 0.35 + q_mean * 2.0, 10.0, final_smax * 0.9)
# #     G_init_heuristic = np.clip(final_smax * 0.25 + q_mean * 10.0, 10.0, final_smax * 2.0)

# #     return round(final_smax, 1), round(S_init_heuristic, 3), round(G_init_heuristic, 3)


# # def get_fallback_params(P_data, PET_data, Q_data, Smax, basin_area_km2):
# #     params = DEFAULT_FALLBACK_PARAMS.copy()
# #     # Calculate initial states based on data (Smax is passed/calculated)
# #     S_init, G_init = calculate_initial_states(P_data, Q_data, P_data, basin_area_km2)[1:]
# #     params.update({'S_init': S_init, 'G_init': G_init, 'Smax': Smax})
    
# #     model_params = ModelParams(Smax=Smax, Kperc=params['Kperc'], Kb=params['Kb'], Ke=params['Ke'], Cqq=params['Cqq'])
# #     initial_state = (params['S_init'], params['G_init'], params['bias'])
    
# #     Q_sim, Q_obs_clean = run_forward_model(P_data, PET_data, Q_data, initial_state, model_params)
    
# #     # Check if Q_sim is valid before calculating KGE
# #     if len(Q_sim) > 0 and np.all(np.isfinite(Q_sim)):
# #         kge = calculate_kge(Q_obs_clean, Q_sim)
# #         params['KGE'] = kge if np.isfinite(kge) else 0.0
# #     else:
# #         params['KGE'] = 0.0
        
# #     return params


# # def clean_dict_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
# #     # Ensure no NumPy types remain that could cause issues when saving JSON
# #     cleaned_data = {}
# #     for key, value in data.items():
# #         if isinstance(value, dict):
# #             cleaned_data[key] = clean_dict_for_json(value)
# #         elif isinstance(value, (np.float32, np.float64)):
# #             cleaned_data[key] = float(value)
# #         elif isinstance(value, (np.int32, np.int64)):
# #             cleaned_data[key] = int(value)
# #         else:
# #             cleaned_data[key] = value
# #     return cleaned_data


# # class TwoStoreModel_SCE:
# #     def __init__(self, P_data, PET_data, Q_obs, Smax, initial_S, initial_G, target_basin):
# #         self.P_data = P_data
# #         self.PET_data = PET_data
# #         self.Q_obs = Q_obs
# #         self.Smax = Smax
# #         self.initial_S_fallback = initial_S
# #         self.initial_G_fallback = initial_G
# #         self.target_basin = target_basin

# #     def parameters(self):
# #         # NOTE: Bounds are consistent with fallback parameters (e.g. 0.85 is in [0.1, 0.999])
# #         return spotpy.parameter.generate([
# #             spotpy.parameter.Uniform(0.1, 0.8, name='Kperc'),
# #             spotpy.parameter.Uniform(0.1, 0.7, name='Kb'),
# #             spotpy.parameter.Uniform(0.1, 0.999, name='Ke'),
# #             spotpy.parameter.Uniform(0.1, 20.0, name='Cqq'),
# #             spotpy.parameter.Uniform(-0.5, 0.5, name='Bias'),
# #         ])

# #     def simulation(self, params):
# #         model_params = ModelParams(Smax=self.Smax, Kperc=params['Kperc'], Kb=params['Kb'],
# #                                    Ke=params['Ke'], Cqq=params['Cqq'])
# #         initial_state = (self.initial_S_fallback, self.initial_G_fallback, params['Bias'])
# #         Q_sim, _ = run_forward_model(self.P_data, self.PET_data, self.Q_obs, initial_state, model_params)
        
# #         # Ensure the simulation result is a simple list of floats
# #         return Q_sim.tolist() if len(Q_sim) > 0 else [np.nanmean(self.Q_obs)]

# #     def evaluation(self):
# #         _, Q_obs_clean = run_forward_model(
# #             self.P_data, self.PET_data, self.Q_obs,
# #             (self.initial_S_fallback, self.initial_G_fallback, 0.0),
# #             ModelParams(Smax=self.Smax) # Use default/fallback params for evaluation array length
# #         )
# #         return Q_obs_clean.tolist() if len(Q_obs_clean) > 0 else [np.nanmean(self.Q_obs)]

# #     def objectivefunction(self, evaluation, simulation):
# #         evaluation, simulation = np.array(evaluation), np.array(simulation)
        
# #         # CRITICAL: Check for non-finite values BEFORE KGE
# #         if not np.all(np.isfinite(simulation)) or len(evaluation) < 12:
# #             return 9999.0 # Max penalty for invalid simulation/too short data
            
# #         epsilon = 1e-6
# #         Q_obs_trans = np.sqrt(np.clip(evaluation, epsilon, None))
# #         Q_sim_trans = np.sqrt(np.clip(simulation, epsilon, None))
        
# #         kge_sr = calculate_kge(Q_obs_trans, Q_sim_trans)
        
# #         # Penalty if KGE is not a valid number
# #         return -kge_sr if np.isfinite(kge_sr) else 9999.0


# # def worker_calibrate_basin(
# #     target_basin: str, P_df: pd.DataFrame, PET_df: pd.DataFrame, Q_usgs_df: pd.DataFrame,
# #     S_init_df: pd.DataFrame, G_init_df: pd.DataFrame, SM_df: pd.DataFrame,
# #     Q_nldas_df: pd.DataFrame, attrs: gpd.GeoDataFrame, reps: int
# # ) -> Tuple[str, Optional[Dict[str, Any]]]:

# #     # PID is less important in serial mode but can be kept for logging
# #     print(f"\n---> Calibrating Basin: {target_basin} (PID: {os.getpid()})...")

# #     try:
# #         P_data, PET_data = P_df[target_basin].values, PET_df[target_basin].values
# #         Q_nldas_cal = Q_nldas_df[target_basin].values
# #         Q_usgs_kge = Q_usgs_df[target_basin].values
# #         SM_data = SM_df[target_basin].values
# #         basin_area_km2 = attrs.loc[target_basin, 'area_gages2']
# #     except KeyError:
# #         print(f"❌ ERROR: Missing data for basin {target_basin}. Returning fallback.")
# #         return target_basin, None

# #     if np.all(np.isnan(Q_nldas_cal)) or np.all(np.isnan(P_data)):
# #         # FIXED: Use consistent default Smax (700.0) for fallback calculation
# #         Smax = DEFAULT_FALLBACK_PARAMS['Smax'] 
# #         return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)

# #     Smax, initial_S_heu, initial_G_heu = calculate_initial_states(SM_data, Q_usgs_kge, P_data, basin_area_km2)

# #     # Note: Initial states are constrained by data-driven heuristics and pre-calibrated values
# #     initial_S = min(S_init_df.loc[S_init_df.index[0], target_basin], initial_S_heu) \
# #         if target_basin in S_init_df.columns else initial_S_heu
# #     initial_G = min(G_init_df.loc[G_init_df.index[0], target_basin], initial_G_heu) \
# #         if target_basin in G_init_df.columns else initial_G_heu

# #     print(f" > {target_basin} | Smax={Smax:.0f} | Initial S/G={initial_S:.0f}/{initial_G:.0f}")

# #     # Use P_data for precipitation input, Q_nldas_cal as the Q_obs target for calibration
# #     model = TwoStoreModel_SCE(P_data, PET_data, Q_nldas_cal, Smax, initial_S, initial_G, target_basin)
# #     temp_dir = os.path.join(PROJECT_ROOT, 'SCE_cal_params')
# #     db_name = os.path.join(temp_dir, f'sceua_{target_basin}')

# #     try:
# #         sampler = spotpy.algorithms.sceua(model, dbname=db_name, dbformat='csv', save_sim=False)
# #         sampler.sample(repetitions=reps, ngs=70, kstop=60, peps=0.00005, pcento=0.00005)
# #         res = sampler.getdata()
# #     except Exception as e:
# #         print(f"❌ SCE-UA failed for {target_basin}: {e}. Returning fallback.")
# #         return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)
# #     finally:
# #         # CRITICAL: Clean up DB file immediately after use 
# #         if os.path.exists(db_name + '.csv'):
# #             os.remove(db_name + '.csv')


# #     if res is None or len(res) == 0:
# #         return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)

# #     best_idx, _ = spotpy.analyser.get_minlikeindex(res)
# #     best_params_raw = {
# #         'Kperc': float(res['parKperc'][best_idx][0]), 'Kb': float(res['parKb'][best_idx][0]),
# #         'Ke': float(res['parKe'][best_idx][0]), 'Cqq': float(res['parCqq'][best_idx][0]),
# #         'bias': float(res['parBias'][best_idx][0]), 'S_init': initial_S,
# #         'G_init': initial_G, 'Smax': Smax,
# #     }

# #     model_params_final = ModelParams(Smax=Smax, **{k: best_params_raw[k] for k in ['Kperc', 'Kb', 'Ke', 'Cqq']})
# #     initial_state_final = (initial_S, initial_G, best_params_raw['bias'])
    
# #     # Run against USGS Q_obs for final KGE evaluation
# #     Q_sim_final, Q_obs_clean = run_forward_model(P_data, PET_data, Q_usgs_kge, initial_state_final, model_params_final)
# #     best_kge_std = calculate_kge(Q_obs_clean, Q_sim_final)
# #     best_params = {**best_params_raw, 'KGE': best_kge_std}

# #     print(f" > SCE-UA Final (KGE vs USGS): {best_kge_std:.3f}")

# #     if best_kge_std <= MIN_KGE_THRESHOLD or not np.isfinite(best_kge_std):
# #         fallback_params = get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)
# #         if fallback_params['KGE'] > best_kge_std:
# #             print(f"  🔄 Fallback used (KGE: {fallback_params['KGE']:.3f})")
# #             return target_basin, fallback_params

# #     return target_basin, best_params


# # def load_camels_attributes():
# #     print("Loading CAMELS attributes...")
# #     r = requests.get(
# #         "https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/data/contents/camels_attributes_v2.0.feather"
# #     )
# #     attrs = gpd.read_feather(io.BytesIO(r.content)).reset_index(drop=False)
# #     attrs['gauge_id'] = attrs['gauge_id'].astype(str).str.strip().str.zfill(8)
# #     attrs.set_index('gauge_id', inplace=True)
# #     print("✅ CAMELS attributes loaded successfully.")
# #     return attrs


# # if __name__ == '__main__':
# #     print("Initializing data loading...")

# #     try:
# #         DFs = load_and_prepare_data()
# #         (P_df, PET_df, _, _, _, _, Q_nldas_mm_monthly, Q_usgs_df,
# #          S_init_df, G_init_df, SM_df) = DFs
# #         if Q_usgs_df is None or Q_usgs_df.empty:
# #             raise ValueError("Q_usgs_df is None/Empty.")
# #     except Exception as e:
# #         print(f"⚠️ DATA LOAD ERROR: {e}")
# #         sys.exit(1)

# #     try:
# #         attrs = load_camels_attributes()
# #     except Exception as e:
# #         print(f"❌ FATAL: Failed to load CAMELS attributes. Error: {e}")
# #         sys.exit(1)

# #     TARGET_BASINS = list(Q_usgs_df.columns)
    
# #     # Note: NUM_CORES is no longer used for parallelization, but kept for context.
# #     NUM_CORES = max(1, cpu_count() - 1) 

# #     print(f"\n🚀 Starting **SERIAL SCE-UA Calibration** for {len(TARGET_BASINS)} basins...")
# #     print(f"  (Multiprocessing disabled for stability on Windows.)")

# #     REPS = 2000 
    
# #     # Prepare all arguments for the serial loop
# #     tasks_args = [
# #         (basin, P_df, PET_df, Q_usgs_df, S_init_df, G_init_df, SM_df,
# #          Q_nldas_mm_monthly, attrs, REPS) 
# #         for basin in TARGET_BASINS
# #     ]

# #     results = {}
    
# #     # --- START OF SERIAL EXECUTION ---
# #     try:
# #         os.makedirs(os.path.join(PROJECT_ROOT, 'SCE_cal_params'), exist_ok=True)
        
# #         # Iterating through tasks_args and calling the worker function directly
# #         for args in tasks_args:
# #             basin, params = worker_calibrate_basin(*args)
# #             if params:
# #                 results[basin] = params
                
# #     except Exception as e:
# #         # This will catch errors from the main process during the serial execution
# #         print(f"❌ FATAL: Serial execution failed. Error: {e}")
# #         sys.exit(1) 
# #     # --- END OF SERIAL EXECUTION ---

# #     output_path = os.path.join(PROJECT_ROOT, 'SCE_cal_params', 'final_calibrated_params.json')
# #     cleaned_results = clean_dict_for_json(results)

# #     with open(output_path, 'w') as f:
# #         json.dump(cleaned_results, f, indent=2)

# #     results_list = [{'Basin': k, **v} for k, v in results.items() if v.get('KGE') is not None]
# #     df = pd.DataFrame(results_list).round(3).sort_values('KGE', ascending=False)

# #     print("\n" + "=" * 80)
# #     print("✅ ALL BASINS CALIBRATED!")
# #     print(df.to_markdown(index=False))
# #     print(f"\n💾 Saved: {output_path}")
# #     print(f"\n🏆 SUCCESS: {len(df[df['KGE'] > 0.5])}/{len(TARGET_BASINS)} basins KGE > 0.5")

# # # C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\DA\DA_Github_repo\Bayesian_DA_Budyko_modeling\scripts\calibration.py

# # import os
# # import sys
# # import pandas as pd
# # import numpy as np
# # import json
# # from typing import Dict, Any, Tuple, Optional
# # from multiprocessing import cpu_count
# # import warnings
# # import spotpy
# # import requests
# # import io
# # import geopandas as gpd

# # # Suppress NumPy warnings (but still check for NaN/Inf explicitly)
# # warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')

# # PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# # sys.path.append(PROJECT_ROOT)

# # try:
# #     # Ensure all your core modules are accessible
# #     from src.model import ModelParams, two_store_model_step
# #     from src.metrics import calculate_kge
# #     from data.data_processor import load_and_prepare_data
# # except ImportError as e:
# #     print(f"FATAL: Core module import failed. Error: {e}")
# #     sys.exit(1)

# # # Constants
# # S_MAX_CEILING = 2500.0
# # G_MAX_CEILING = S_MAX_CEILING * 3.0
# # # FIXED: Kperc and Kb updated to be consistent with the tight SCE-UA bounds.
# # DEFAULT_FALLBACK_PARAMS = {
# #     'Kperc': 0.75, 'Kb': 0.65, 'Ke': 0.77, 'Cqq': 20.0, 'bias': 0.0,
# #     'S_init': 20.0, 'G_init': 15.0, 'Smax': 700.0, 'KGE': 0.0
# # }
# # MIN_KGE_THRESHOLD = 0.6


# # def run_forward_model(P_data, PET_data, Q_obs, initial_state: tuple, params: ModelParams) -> tuple:
# #     nmonths = len(P_data)
# #     S_curr, G_curr, bias = initial_state
# #     Q_sim = np.zeros(nmonths)

# #     for t in range(nmonths):
# #         P_t, PET_t = max(P_data[t], 0.0), max(PET_data[t], 0.0)
# #         try:
# #             S_next, G_next, _, Q_t_no_bias, _, _, _ = two_store_model_step(
# #                 S_curr, G_curr, P_t, PET_t, params, ET_override=None
# #             )
# #         except Exception:
# #             # If model step fails (e.g., math error), use current states and zero flow
# #             S_next, G_next, Q_t_no_bias = S_curr, G_curr, 0.0

# #         S_curr = np.clip(S_next, 0.0, params.Smax)
# #         G_curr = np.clip(G_next, 0.0, G_MAX_CEILING)
# #         Q_sim[t] = max(Q_t_no_bias + bias, 0.0)

# #     # CRITICAL: Check for non-finite values in the simulated output
# #     if not np.all(np.isfinite(Q_sim)):
# #         return np.zeros(1), np.zeros(1) 

# #     spin_up = 60
# #     Q_sim_clean = Q_sim[spin_up:]
# #     Q_obs_clean = Q_obs[spin_up:]
# #     valid_mask = ~(np.isnan(Q_sim_clean) | np.isnan(Q_obs_clean))

# #     if valid_mask.sum() < 12:
# #         return np.zeros(1), np.zeros(1)

# #     return Q_sim_clean[valid_mask], Q_obs_clean[valid_mask]


# # def calculate_initial_states(SM_data, Q_obs, P_data, basin_area_km2):
# #     p_mean = np.nanmean(P_data) if not np.all(np.isnan(P_data)) else 50.0
# #     q_mean = np.nanmean(Q_obs) if not np.all(np.isnan(Q_obs)) else 5.0
# #     sm_max = 5.0

# #     # FIXED: Revert Smax estimate minimum to 200.0 to allow for smaller catchments
# #     smax_estimate = max(sm_max * 0.5, p_mean * 0.25, 200.0) 
# #     final_smax = np.clip(smax_estimate, 200.0, S_MAX_CEILING)
# #     S_init_heuristic = np.clip(final_smax * 0.35 + q_mean * 2.0, 10.0, final_smax * 0.9)
# #     G_init_heuristic = np.clip(final_smax * 0.25 + q_mean * 10.0, 10.0, final_smax * 2.0)

# #     return round(final_smax, 1), round(S_init_heuristic, 3), round(G_init_heuristic, 3)


# # def get_fallback_params(P_data, PET_data, Q_data, Smax, basin_area_km2):
# #     params = DEFAULT_FALLBACK_PARAMS.copy()
# #     # Calculate initial states based on data (Smax is passed/calculated)
# #     S_init, G_init = calculate_initial_states(P_data, Q_data, P_data, basin_area_km2)[1:]
# #     # NOTE: Smax passed from worker_calibrate_basin is used here for consistency, 
# #     # overriding the default 700.0 in DEFAULT_FALLBACK_PARAMS.
# #     params.update({'S_init': S_init, 'G_init': G_init, 'Smax': Smax})
    
# #     model_params = ModelParams(Smax=Smax, Kperc=params['Kperc'], Kb=params['Kb'], Ke=params['Ke'], Cqq=params['Cqq'])
# #     initial_state = (params['S_init'], params['G_init'], params['bias'])
    
# #     Q_sim, Q_obs_clean = run_forward_model(P_data, PET_data, Q_data, initial_state, model_params)
    
# #     # Check if Q_sim is valid before calculating KGE
# #     if len(Q_sim) > 0 and np.all(np.isfinite(Q_sim)):
# #         kge = calculate_kge(Q_obs_clean, Q_sim)
# #         params['KGE'] = kge if np.isfinite(kge) else 0.0
# #     else:
# #         params['KGE'] = 0.0
        
# #     return params


# # def clean_dict_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
# #     # Ensure no NumPy types remain that could cause issues when saving JSON
# #     cleaned_data = {}
# #     for key, value in data.items():
# #         if isinstance(value, dict):
# #             cleaned_data[key] = clean_dict_for_json(value)
# #         elif isinstance(value, (np.float32, np.float64)):
# #             cleaned_data[key] = float(value)
# #         elif isinstance(value, (np.int32, np.int64)):
# #             cleaned_data[key] = int(value)
# #         else:
# #             cleaned_data[key] = value
# #     return cleaned_data


# # class TwoStoreModel_SCE:
# #     def __init__(self, P_data, PET_data, Q_obs, Smax, initial_S, initial_G, target_basin):
# #         self.P_data = P_data
# #         self.PET_data = PET_data
# #         self.Q_obs = Q_obs
# #         self.Smax = Smax
# #         self.initial_S_fallback = initial_S
# #         self.initial_G_fallback = initial_G
# #         self.target_basin = target_basin

# #     def parameters(self):
# #         # Bounds used for calibration:
# #         return spotpy.parameter.generate([
# #             spotpy.parameter.Uniform(0.1, 0.8, name='Kperc'),
# #             spotpy.parameter.Uniform(0.1, 0.7, name='Kb'),
# #             spotpy.parameter.Uniform(0.1, 0.999, name='Ke'),
# #             spotpy.parameter.Uniform(0.1, 20.0, name='Cqq'),
# #             spotpy.parameter.Uniform(-0.5, 0.5, name='Bias'),
# #         ])

# #     def simulation(self, params):
# #         model_params = ModelParams(Smax=self.Smax, Kperc=params['Kperc'], Kb=params['Kb'],
# #                                    Ke=params['Ke'], Cqq=params['Cqq'])
# #         initial_state = (self.initial_S_fallback, self.initial_G_fallback, params['Bias'])
# #         Q_sim, _ = run_forward_model(self.P_data, self.PET_data, self.Q_obs, initial_state, model_params)
        
# #         # Ensure the simulation result is a simple list of floats
# #         return Q_sim.tolist() if len(Q_sim) > 0 else [np.nanmean(self.Q_obs)]

# #     def evaluation(self):
# #         _, Q_obs_clean = run_forward_model(
# #             self.P_data, self.PET_data, self.Q_obs,
# #             (self.initial_S_fallback, self.initial_G_fallback, 0.0),
# #             ModelParams(Smax=self.Smax) # Use default/fallback params for evaluation array length
# #         )
# #         return Q_obs_clean.tolist() if len(Q_obs_clean) > 0 else [np.nanmean(self.Q_obs)]

# #     def objectivefunction(self, evaluation, simulation):
# #         evaluation, simulation = np.array(evaluation), np.array(simulation)
        
# #         # CRITICAL: Check for non-finite values BEFORE KGE
# #         if not np.all(np.isfinite(simulation)) or len(evaluation) < 12:
# #             return 9999.0 # Max penalty for invalid simulation/too short data
            
# #         epsilon = 1e-6
# #         Q_obs_trans = np.sqrt(np.clip(evaluation, epsilon, None))
# #         Q_sim_trans = np.sqrt(np.clip(simulation, epsilon, None))
        
# #         kge_sr = calculate_kge(Q_obs_trans, Q_sim_trans)
        
# #         # Penalty if KGE is not a valid number
# #         return -kge_sr if np.isfinite(kge_sr) else 9999.0


# # def worker_calibrate_basin(
# #     target_basin: str, P_df: pd.DataFrame, PET_df: pd.DataFrame, Q_usgs_df: pd.DataFrame,
# #     S_init_df: pd.DataFrame, G_init_df: pd.DataFrame, SM_df: pd.DataFrame,
# #     Q_nldas_df: pd.DataFrame, attrs: gpd.GeoDataFrame, reps: int
# # ) -> Tuple[str, Optional[Dict[str, Any]]]:

# #     # PID is less important in serial mode but can be kept for logging
# #     print(f"\n---> Calibrating Basin: {target_basin} (PID: {os.getpid()})...")

# #     try:
# #         P_data, PET_data = P_df[target_basin].values, PET_df[target_basin].values
# #         Q_nldas_cal = Q_nldas_df[target_basin].values
# #         Q_usgs_kge = Q_usgs_df[target_basin].values
# #         SM_data = SM_df[target_basin].values
# #         basin_area_km2 = attrs.loc[target_basin, 'area_gages2']
# #     except KeyError:
# #         print(f"❌ ERROR: Missing data for basin {target_basin}. Returning fallback.")
# #         return target_basin, None

# #     if np.all(np.isnan(Q_nldas_cal)) or np.all(np.isnan(P_data)):
# #         # FIXED: Use consistent default Smax (700.0) for fallback calculation
# #         Smax = DEFAULT_FALLBACK_PARAMS['Smax'] 
# #         return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)

# #     Smax, initial_S_heu, initial_G_heu = calculate_initial_states(SM_data, Q_usgs_kge, P_data, basin_area_km2)

# #     # Note: Initial states are constrained by data-driven heuristics and pre-calibrated values
# #     initial_S = min(S_init_df.loc[S_init_df.index[0], target_basin], initial_S_heu) \
# #         if target_basin in S_init_df.columns else initial_S_heu
# #     initial_G = min(G_init_df.loc[G_init_df.index[0], target_basin], initial_G_heu) \
# #         if target_basin in G_init_df.columns else initial_G_heu

# #     print(f" > {target_basin} | Smax={Smax:.0f} | Initial S/G={initial_S:.0f}/{initial_G:.0f}")

# #     # Use P_data for precipitation input, Q_nldas_cal as the Q_obs target for calibration
# #     model = TwoStoreModel_SCE(P_data, PET_data, Q_nldas_cal, Smax, initial_S, initial_G, target_basin)
# #     temp_dir = os.path.join(PROJECT_ROOT, 'SCE_cal_params')
# #     db_name = os.path.join(temp_dir, f'sceua_{target_basin}')

# #     try:
# #         sampler = spotpy.algorithms.sceua(model, dbname=db_name, dbformat='csv', save_sim=False)
# #         sampler.sample(repetitions=reps, ngs=70, kstop=60, peps=0.00005, pcento=0.00005)
# #         res = sampler.getdata()
# #     except Exception as e:
# #         print(f"❌ SCE-UA failed for {target_basin}: {e}. Returning fallback.")
# #         return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)
# #     finally:
# #         # CRITICAL: Clean up DB file immediately after use 
# #         if os.path.exists(db_name + '.csv'):
# #             os.remove(db_name + '.csv')


# #     if res is None or len(res) == 0:
# #         return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)

# #     best_idx, _ = spotpy.analyser.get_minlikeindex(res)
# #     best_params_raw = {
# #         'Kperc': float(res['parKperc'][best_idx][0]), 'Kb': float(res['parKb'][best_idx][0]),
# #         'Ke': float(res['parKe'][best_idx][0]), 'Cqq': float(res['parCqq'][best_idx][0]),
# #         'bias': float(res['parBias'][best_idx][0]), 'S_init': initial_S,
# #         'G_init': initial_G, 'Smax': Smax,
# #     }

# #     model_params_final = ModelParams(Smax=Smax, **{k: best_params_raw[k] for k in ['Kperc', 'Kb', 'Ke', 'Cqq']})
# #     initial_state_final = (initial_S, initial_G, best_params_raw['bias'])
    
# #     # Run against USGS Q_obs for final KGE evaluation
# #     Q_sim_final, Q_obs_clean = run_forward_model(P_data, PET_data, Q_usgs_kge, initial_state_final, model_params_final)
# #     best_kge_std = calculate_kge(Q_obs_clean, Q_sim_final)
# #     best_params = {**best_params_raw, 'KGE': best_kge_std}

# #     print(f" > SCE-UA Final (KGE vs USGS): {best_kge_std:.3f}")

# #     if best_kge_std <= MIN_KGE_THRESHOLD or not np.isfinite(best_kge_std):
# #         # NOTE: Smax passed here is the calculated Smax for the basin, not the default 700.0.
# #         fallback_params = get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)
# #         if fallback_params['KGE'] > best_kge_std:
# #             print(f"  🔄 Fallback used (KGE: {fallback_params['KGE']:.3f})")
# #             return target_basin, fallback_params

# #     return target_basin, best_params


# # def load_camels_attributes():
# #     print("Loading CAMELS attributes...")
# #     r = requests.get(
# #         "https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/data/contents/camels_attributes_v2.0.feather"
# #     )
# #     attrs = gpd.read_feather(io.BytesIO(r.content)).reset_index(drop=False)
# #     attrs['gauge_id'] = attrs['gauge_id'].astype(str).str.strip().str.zfill(8)
# #     attrs.set_index('gauge_id', inplace=True)
# #     print("✅ CAMELS attributes loaded successfully.")
# #     return attrs


# # if __name__ == '__main__':
# #     print("Initializing data loading...")

# #     try:
# #         DFs = load_and_prepare_data()
# #         (P_df, PET_df, _, _, _, _, Q_nldas_mm_monthly, Q_usgs_df,
# #          S_init_df, G_init_df, SM_df) = DFs
# #         if Q_usgs_df is None or Q_usgs_df.empty:
# #             raise ValueError("Q_usgs_df is None/Empty.")
# #     except Exception as e:
# #         print(f"⚠️ DATA LOAD ERROR: {e}")
# #         sys.exit(1)

# #     try:
# #         attrs = load_camels_attributes()
# #     except Exception as e:
# #         print(f"❌ FATAL: Failed to load CAMELS attributes. Error: {e}")
# #         sys.exit(1)

# #     TARGET_BASINS = list(Q_usgs_df.columns)
    
# #     # Note: NUM_CORES is no longer used for parallelization, but kept for context.
# #     NUM_CORES = max(1, cpu_count() - 1) 

# #     print(f"\n🚀 Starting **SERIAL SCE-UA Calibration** for {len(TARGET_BASINS)} basins...")
# #     print(f"  (Multiprocessing disabled for stability on Windows.)")

# #     # FIXED: Reduced repetitions from 20000 to 5000 for a more efficient serial run.
# #     REPS = 2000 
    
# #     # Prepare all arguments for the serial loop
# #     tasks_args = [
# #         (basin, P_df, PET_df, Q_usgs_df, S_init_df, G_init_df, SM_df,
# #          Q_nldas_mm_monthly, attrs, REPS) 
# #         for basin in TARGET_BASINS
# #     ]

# #     results = {}
    
# #     # --- START OF SERIAL EXECUTION ---
# #     try:
# #         os.makedirs(os.path.join(PROJECT_ROOT, 'SCE_cal_params'), exist_ok=True)
        
# #         # Iterating through tasks_args and calling the worker function directly
# #         for args in tasks_args:
# #             basin, params = worker_calibrate_basin(*args)
# #             if params:
# #                 results[basin] = params
                
# #     except Exception as e:
# #         # This will catch errors from the main process during the serial execution
# #         print(f"❌ FATAL: Serial execution failed. Error: {e}")
# #         sys.exit(1) 
# #     # --- END OF SERIAL EXECUTION ---

# #     output_path = os.path.join(PROJECT_ROOT, 'SCE_cal_params', 'final_calibrated_params.json')
# #     cleaned_results = clean_dict_for_json(results)

# #     with open(output_path, 'w') as f:
# #         json.dump(cleaned_results, f, indent=2)

# #     results_list = [{'Basin': k, **v} for k, v in results.items() if v.get('KGE') is not None]
# #     df = pd.DataFrame(results_list).round(3).sort_values('KGE', ascending=False)

# #     print("\n" + "=" * 80)
# #     print("✅ ALL BASINS CALIBRATED!")
# #     print(df.to_markdown(index=False))
# #     print(f"\n💾 Saved: {output_path}")
# #     print(f"\n🏆 SUCCESS: {len(df[df['KGE'] > 0.5])}/{len(TARGET_BASINS)} basins KGE > 0.5")



# import pandas as pd
# import numpy as np
# import os
# import sys
# import json
# from typing import Dict, Any, Tuple, Optional
# from multiprocessing import Pool, cpu_count
# import warnings
# import spotpy
# import requests
# import io
# import geopandas as gpd

# warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(PROJECT_ROOT)

# try:
#     from src.model import ModelParams, two_store_model_step
#     from src.metrics import calculate_kge
#     from data.data_processor import load_and_prepare_data
# except ImportError as e:
#     print(f"FATAL: Core module import failed. Error: {e}")
#     sys.exit(1)

# S_MAX_CEILING = 2500.0
# G_MAX_CEILING = S_MAX_CEILING * 3.0
# DEFAULT_FALLBACK_PARAMS = {
#     'Kperc': 0.25, 'Kb': 0.15, 'Ke': 0.7, 'Cqq': 0.85, 'bias': 0.0,
#     'S_init': 2.0, 'G_init': 1.5, 'Smax': 200.0, 'KGE': 0.0
# }
# MIN_KGE_THRESHOLD = 0.6


# def run_forward_model(P_data, PET_data, Q_obs, initial_state: tuple, params: ModelParams) -> tuple:
#     nmonths = len(P_data)
#     S_curr, G_curr, bias = initial_state
#     Q_sim = np.zeros(nmonths)

#     for t in range(nmonths):
#         P_t, PET_t = max(P_data[t], 0.0), max(PET_data[t], 0.0)
#         try:
#             S_next, G_next, _, Q_t_no_bias, _, _, _ = two_store_model_step(
#                 S_curr, G_curr, P_t, PET_t, params, ET_override=None
#             )
#         except Exception:
#             S_next, G_next, Q_t_no_bias = S_curr, G_curr, 0.0

#         S_curr = np.clip(S_next, 0.0, params.Smax)
#         G_curr = np.clip(G_next, 0.0, G_MAX_CEILING)
#         Q_sim[t] = max(Q_t_no_bias + bias, 0.0)

#     spin_up = 60
#     Q_sim_clean = Q_sim[spin_up:]
#     Q_obs_clean = Q_obs[spin_up:]
#     valid_mask = ~(np.isnan(Q_sim_clean) | np.isnan(Q_obs_clean))

#     if valid_mask.sum() < 12:
#         return np.zeros(1), np.zeros(1)

#     return Q_sim_clean[valid_mask], Q_obs_clean[valid_mask]


# def calculate_initial_states(SM_data, Q_obs, P_data, basin_area_km2):
#     p_mean = np.nanmean(P_data) if not np.all(np.isnan(P_data)) else 50.0
#     q_mean = np.nanmean(Q_obs) if not np.all(np.isnan(Q_obs)) else 5.0
#     sm_max = 5.0

#     smax_estimate = max(sm_max, p_mean)
#     final_smax = np.clip(smax_estimate, 200.0, S_MAX_CEILING)
#     S_init_heuristic = np.clip(final_smax * 0.35 + q_mean * 2.0, 10.0, final_smax * 0.9)
#     G_init_heuristic = np.clip(final_smax * 0.25 + q_mean * 10.0, 10.0, final_smax * 2.0)

#     return round(final_smax, 1), round(S_init_heuristic, 3), round(G_init_heuristic, 3)


# def get_fallback_params(P_data, PET_data, Q_data, Smax, basin_area_km2):
#     params = DEFAULT_FALLBACK_PARAMS.copy()
#     S_init, G_init = calculate_initial_states(P_data, Q_data, P_data, basin_area_km2)[1:]
#     params.update({'S_init': S_init, 'G_init': G_init, 'Smax': Smax})
#     model_params = ModelParams(Smax=Smax, Kperc=params['Kperc'], Kb=params['Kb'], Ke=params['Ke'], Cqq=params['Cqq'])
#     initial_state = (params['S_init'], params['G_init'], params['bias'])
#     Q_sim, Q_obs_clean = run_forward_model(P_data, PET_data, Q_data, initial_state, model_params)
#     kge = calculate_kge(Q_obs_clean, Q_sim)
#     params['KGE'] = kge if not np.isnan(kge) else 0.0
#     return params


# def clean_dict_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
#     cleaned_data = {}
#     for key, value in data.items():
#         if isinstance(value, dict):
#             cleaned_data[key] = clean_dict_for_json(value)
#         elif isinstance(value, (np.float32, np.float64)):
#             cleaned_data[key] = float(value)
#         elif isinstance(value, (np.int32, np.int64)):
#             cleaned_data[key] = int(value)
#         else:
#             cleaned_data[key] = value
#     return cleaned_data


# class TwoStoreModel_SCE:
#     def __init__(self, P_data, PET_data, Q_obs, Smax, initial_S, initial_G, target_basin):
#         self.P_data = P_data
#         self.PET_data = PET_data
#         self.Q_obs = Q_obs
#         self.Smax = Smax
#         self.initial_S_fallback = initial_S
#         self.initial_G_fallback = initial_G
#         self.target_basin = target_basin

#     def parameters(self):
#         return spotpy.parameter.generate([
#             spotpy.parameter.Uniform(0.005, 0.8, name='Kperc'),
#             spotpy.parameter.Uniform(0.01, 0.6, name='Kb'),
#             spotpy.parameter.Uniform(0.2, 0.999, name='Ke'),
#             spotpy.parameter.Uniform(0.1, 20.0, name='Cqq'),
#             spotpy.parameter.Uniform(-0.5, 0.5, name='Bias'),
#         ])

#     def simulation(self, params):
#         model_params = ModelParams(Smax=self.Smax, Kperc=params['Kperc'], Kb=params['Kb'],
#                                    Ke=params['Ke'], Cqq=params['Cqq'])
#         initial_state = (self.initial_S_fallback, self.initial_G_fallback, params['Bias'])
#         Q_sim, _ = run_forward_model(self.P_data, self.PET_data, self.Q_obs, initial_state, model_params)
#         return Q_sim.tolist() if len(Q_sim) > 0 else [np.nanmean(self.Q_obs)]

#     def evaluation(self):
#         _, Q_obs_clean = run_forward_model(
#             self.P_data, self.PET_data, self.Q_obs,
#             (self.initial_S_fallback, self.initial_G_fallback, 0.0),
#             ModelParams(Smax=self.Smax)
#         )
#         return Q_obs_clean.tolist() if len(Q_obs_clean) > 0 else [np.nanmean(self.Q_obs)]

#     def objectivefunction(self, evaluation, simulation):
#         evaluation, simulation = np.array(evaluation), np.array(simulation)
#         if len(evaluation) < 12:
#             return 9999.0
#         epsilon = 1e-6
#         Q_obs_trans = np.sqrt(np.clip(evaluation, epsilon, None))
#         Q_sim_trans = np.sqrt(np.clip(simulation, epsilon, None))
#         kge_sr = calculate_kge(Q_obs_trans, Q_sim_trans)
#         return -kge_sr if not np.isnan(kge_sr) else 9999.0


# def worker_calibrate_basin(
#     target_basin: str, P_df: pd.DataFrame, PET_df: pd.DataFrame, Q_usgs_df: pd.DataFrame,
#     S_init_df: pd.DataFrame, G_init_df: pd.DataFrame, SM_df: pd.DataFrame,
#     Q_nldas_df: pd.DataFrame, attrs: gpd.GeoDataFrame, reps: int
# ) -> Tuple[str, Optional[Dict[str, Any]]]:

#     print(f"\n---> Calibrating Basin: {target_basin} (PID: {os.getpid()})...")

#     try:
#         P_data, PET_data = P_df[target_basin].values, PET_df[target_basin].values
#         Q_nldas_cal = Q_nldas_df[target_basin].values
#         Q_usgs_kge = Q_usgs_df[target_basin].values
#         SM_data = SM_df[target_basin].values
#         basin_area_km2 = attrs.loc[target_basin, 'area_gages2']
#     except KeyError:
#         print(f"❌ ERROR: Missing data for basin {target_basin}. Returning fallback.")
#         return target_basin, None

#     if np.all(np.isnan(Q_nldas_cal)) or np.all(np.isnan(P_data)):
#         return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, 700.0, basin_area_km2)

#     Smax, initial_S_heu, initial_G_heu = calculate_initial_states(SM_data, Q_usgs_kge, P_data, basin_area_km2)

#     initial_S = min(S_init_df.loc[S_init_df.index[0], target_basin], initial_S_heu) \
#         if target_basin in S_init_df.columns else initial_S_heu
#     initial_G = min(G_init_df.loc[G_init_df.index[0], target_basin], initial_G_heu) \
#         if target_basin in G_init_df.columns else initial_G_heu

#     print(f" > {target_basin} | Smax={Smax:.0f} | Initial S/G={initial_S:.0f}/{initial_G:.0f}")

#     model = TwoStoreModel_SCE(P_data, PET_data, Q_nldas_cal, Smax, initial_S, initial_G, target_basin)
#     temp_dir = os.path.join(PROJECT_ROOT, 'SCE_cal_params')
#     db_name = os.path.join(temp_dir, f'sceua_{target_basin}')

#     try:
#         sampler = spotpy.algorithms.sceua(model, dbname=db_name, dbformat='csv', save_sim=False)
#         sampler.sample(repetitions=reps, ngs=70, kstop=60, peps=0.00005, pcento=0.00005)
#         res = sampler.getdata()
#     except Exception as e:
#         print(f"❌ SCE-UA failed for {target_basin}: {e}. Returning fallback.")
#         return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)

#     if res is None or len(res) == 0:
#         return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)

#     best_idx, _ = spotpy.analyser.get_minlikeindex(res)
#     best_params_raw = {
#         'Kperc': float(res['parKperc'][best_idx][0]), 'Kb': float(res['parKb'][best_idx][0]),
#         'Ke': float(res['parKe'][best_idx][0]), 'Cqq': float(res['parCqq'][best_idx][0]),
#         'bias': float(res['parBias'][best_idx][0]), 'S_init': initial_S,
#         'G_init': initial_G, 'Smax': Smax,
#     }

#     model_params_final = ModelParams(Smax=Smax, **{k: best_params_raw[k] for k in ['Kperc', 'Kb', 'Ke', 'Cqq']})
#     initial_state_final = (initial_S, initial_G, best_params_raw['bias'])
#     Q_sim_final, Q_obs_clean = run_forward_model(P_data, PET_data, Q_usgs_kge, initial_state_final, model_params_final)
#     best_kge_std = calculate_kge(Q_obs_clean, Q_sim_final)
#     best_params = {**best_params_raw, 'KGE': best_kge_std}

#     print(f" > SCE-UA Final (KGE vs USGS): {best_kge_std:.3f}")

#     if best_kge_std <= MIN_KGE_THRESHOLD:
#         fallback_params = get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)
#         if fallback_params['KGE'] > best_kge_std:
#             print(f"  🔄 Fallback used (KGE: {fallback_params['KGE']:.3f})")
#             return target_basin, fallback_params

#     return target_basin, best_params


# def load_camels_attributes():
#     print("Loading CAMELS attributes...")
#     r = requests.get(
#         "https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/data/contents/camels_attributes_v2.0.feather"
#     )
#     attrs = gpd.read_feather(io.BytesIO(r.content)).reset_index(drop=False)
#     attrs['gauge_id'] = attrs['gauge_id'].astype(str).str.strip().str.zfill(8)
#     attrs.set_index('gauge_id', inplace=True)
#     print("✅ CAMELS attributes loaded successfully.")
#     return attrs


# if __name__ == '__main__':
#     print("Initializing data loading...")

#     try:
#         DFs = load_and_prepare_data()
#         (P_df, PET_df, _, _, _, _, Q_nldas_mm_monthly, Q_usgs_df,
#          S_init_df, G_init_df, SM_df) = DFs
#         if Q_usgs_df is None or Q_usgs_df.empty:
#             raise ValueError("Q_usgs_df is None/Empty.")
#     except Exception as e:
#         print(f"⚠️ DATA LOAD ERROR: {e}")
#         sys.exit(1)

#     try:
#         attrs = load_camels_attributes()
#     except Exception as e:
#         print(f"❌ FATAL: Failed to load CAMELS attributes. Error: {e}")
#         sys.exit(1)

#     TARGET_BASINS = list(Q_usgs_df.columns)
#     NUM_CORES = max(1, cpu_count() - 1)

#     print(f"\n🚀 Starting **PARALLEL SCE-UA Calibration** on {NUM_CORES} cores for {len(TARGET_BASINS)} basins...")

#     tasks = [
#         (basin, P_df, PET_df, Q_usgs_df, S_init_df, G_init_df, SM_df,
#          Q_nldas_mm_monthly, attrs, 50) # 2000=reps
#         for basin in TARGET_BASINS
#     ]

#     results = {}
#     try:
#         os.makedirs(os.path.join(PROJECT_ROOT, 'SCE_cal_params'), exist_ok=True)
#         with Pool(NUM_CORES) as pool:
#             parallel_results = pool.starmap(worker_calibrate_basin, tasks)

#         for basin, params in parallel_results:
#             if params:
#                 results[basin] = params
#     except Exception as e:
#         print(f"❌ FATAL: Parallel pool failed. Error: {e}")
#         sys.exit(1)

#     output_path = os.path.join(PROJECT_ROOT, 'SCE_cal_params', 'final_calibrated_params.json')
#     cleaned_results = clean_dict_for_json(results)

#     with open(output_path, 'w') as f:
#         json.dump(cleaned_results, f, indent=2)

#     results_list = [{'Basin': k, **v} for k, v in results.items() if v.get('KGE') is not None]
#     df = pd.DataFrame(results_list).round(3).sort_values('KGE', ascending=False)

#     print("\n" + "=" * 80)
#     print("✅ ALL BASINS CALIBRATED!")
#     print(df.to_markdown(index=False))
#     print(f"\n💾 Saved: {output_path}")
#     print(f"\n🏆 SUCCESS: {len(df[df['KGE'] > 0.5])}/{len(TARGET_BASINS)} basins KGE > 0.5")




# C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\DA\DA_Github_repo\Bayesian_DA_Budyko_modeling\scripts\calibration.py



# import os
# import sys
# import pandas as pd
# import numpy as np
# import json
# from typing import Dict, Any, Tuple, Optional
# from multiprocessing import cpu_count, get_context # NOTE: Pool imported via get_context
# import warnings
# import spotpy
# import requests
# import io
# import geopandas as gpd

# # Suppress NumPy warnings (but still check for NaN/Inf explicitly)
# warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(PROJECT_ROOT)

# try:
#     from src.model import ModelParams, two_store_model_step
#     from src.metrics import calculate_kge
#     from data.data_processor import load_and_prepare_data
# except ImportError as e:
#     print(f"FATAL: Core module import failed. Error: {e}")
#     sys.exit(1)

# # Constants
# S_MAX_CEILING = 2500.0
# G_MAX_CEILING = S_MAX_CEILING * 3.0
# DEFAULT_FALLBACK_PARAMS = {
#     'Kperc': 0.25, 'Kb': 0.15, 'Ke': 0.7, 'Cqq': 0.85, 'bias': 0.0,
#     'S_init': 0.5, 'G_init': 1.0, 'Smax': 700.0, 'KGE': 0.0 # Restored robust defaults
# }
# MIN_KGE_THRESHOLD = 0.6


# def run_forward_model(P_data, PET_data, Q_obs, initial_state: tuple, params: ModelParams) -> tuple:
#     """Runs the two-store hydrologic model forward for all time steps."""
#     nmonths = len(P_data)
#     S_curr, G_curr, bias = initial_state
#     Q_sim = np.zeros(nmonths)

#     for t in range(nmonths):
#         # Use np.nan_to_num to ensure finite inputs, protecting the model step
#         P_t, PET_t = np.nan_to_num(max(P_data[t], 0.0)), np.nan_to_num(max(PET_data[t], 0.0))
        
#         try:
#             S_next, G_next, _, Q_t_no_bias, _, _, _ = two_store_model_step(
#                 S_curr, G_curr, P_t, PET_t, params, ET_override=None
#             )
#         except Exception:
#             # If model step fails (e.g., math error), use current states and zero flow
#             S_next, G_next, Q_t_no_bias = S_curr, G_curr, 0.0

#         S_curr = np.clip(S_next, 0.0, params.Smax)
#         G_curr = np.clip(G_next, 0.0, G_MAX_CEILING)
#         Q_sim[t] = max(Q_t_no_bias + bias, 0.0)

#     # CRITICAL FIX: Check for non-finite values in the simulated output
#     if not np.all(np.isfinite(Q_sim)):
#         return np.zeros(1), np.zeros(1) # Return empty if non-finite numbers found

#     spin_up = 20
#     Q_sim_clean = Q_sim[spin_up:]
#     Q_obs_clean = Q_obs[spin_up:]
#     valid_mask = ~(np.isnan(Q_sim_clean) | np.isnan(Q_obs_clean))

#     if valid_mask.sum() < 12:
#         return np.zeros(1), np.zeros(1)

#     return Q_sim_clean[valid_mask], Q_obs_clean[valid_mask]


# # def calculate_initial_states(SM_data, Q_obs, P_data, basin_area_km2):
# #     """Heuristic calculation of Smax, initial S, and initial G."""
# #     # Robustly calculate means
# #     # p_mean = 0.25
# #     # q_mean = 0.15
# #     # sm_max = 1.2

# #     # # Reverting Smax heuristic for wider applicability, using the 200.0 minimum
# #     # smax_estimate = 0.02
# #     # final_smax = 0.01
# #     # S_init_heuristic = 0.02
# #     # G_init_heuristic = 0.01

# #     return round(final_smax, 1), round(S_init_heuristic, 3), round(G_init_heuristic, 3)


# def get_fallback_params(P_data, PET_data, Q_data, Smax, basin_area_km2):
#     """Calculates model parameters and KGE using fixed default parameters."""
#     params = DEFAULT_FALLBACK_PARAMS.copy()
    
#     # Calculate initial states based on current data
#     S_init, G_init = 0.2, 0.5 #calculate_initial_states(P_data, Q_data, P_data, basin_area_km2)[1:]
#     params.update({'S_init': S_init, 'G_init': G_init, 'Smax': Smax})
    
#     model_params = ModelParams(Smax=Smax, **{k: params[k] for k in ['Kperc', 'Kb', 'Ke', 'Cqq']})
#     initial_state = (params['S_init'], params['G_init'], params['bias'])
    
#     Q_sim, Q_obs_clean = run_forward_model(P_data, PET_data, Q_data, initial_state, model_params)
    
#     if len(Q_sim) > 0 and np.all(np.isfinite(Q_sim)):
#         kge = calculate_kge(Q_obs_clean, Q_sim)
#         params['KGE'] = kge if np.isfinite(kge) else 0.0
#     else:
#         params['KGE'] = 0.0
        
#     return params


# def clean_dict_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
#     """Recursively converts NumPy types to native Python types for JSON serialization."""
#     cleaned_data = {}
#     for key, value in data.items():
#         if isinstance(value, dict):
#             cleaned_data[key] = clean_dict_for_json(value)
#         elif isinstance(value, (np.float32, np.float64)):
#             cleaned_data[key] = float(value)
#         elif isinstance(value, (np.int32, np.int64)):
#             cleaned_data[key] = int(value)
#         else:
#             cleaned_data[key] = value
#     return cleaned_data


# class TwoStoreModel_SCE:
#     """SPOTPY interface for the two-store hydrologic model."""
#     def __init__(self, P_data, PET_data, Q_obs, Smax, initial_S, initial_G, target_basin):
#         self.P_data = P_data
#         self.PET_data = PET_data
#         self.Q_obs = Q_obs
#         self.Smax = Smax
#         self.initial_S_fallback = initial_S
#         self.initial_G_fallback = initial_G
#         self.target_basin = target_basin

#     def parameters(self):
#         # Using wider, more robust bounds for initial exploration
#         return spotpy.parameter.generate([
#             spotpy.parameter.Uniform(0.01, 1, name='Kperc'),
#             spotpy.parameter.Uniform(0.01, 1, name='Kb'),
#             spotpy.parameter.Uniform(0.01, 1, name='Ke'),
#             spotpy.parameter.Uniform(0.1, 1, name='Cqq'),
#             spotpy.parameter.Uniform(-0.5, 0.5, name='Bias'),
#         ])

#     def simulation(self, params):
#         model_params = ModelParams(Smax=self.Smax, Kperc=params['Kperc'], Kb=params['Kb'],
#                                    Ke=params['Ke'], Cqq=params['Cqq'])
#         initial_state = (self.initial_S_fallback, self.initial_G_fallback, params['Bias'])
#         Q_sim, _ = run_forward_model(self.P_data, self.PET_data, self.Q_obs, initial_state, model_params)
        
#         # Ensure the simulation result is a simple list of floats
#         return Q_sim.tolist() if len(Q_sim) > 0 else [np.nanmean(self.Q_obs)]

#     def evaluation(self):
#         # Runs the model with default params to get the correct length for Q_obs_clean
#         _, Q_obs_clean = run_forward_model(
#             self.P_data, self.PET_data, self.Q_obs,
#             (self.initial_S_fallback, self.initial_G_fallback, 0.0),
#             ModelParams(Smax=self.Smax)
#         )
#         return Q_obs_clean.tolist() if len(Q_obs_clean) > 0 else [np.nanmean(self.Q_obs)]

#     def objectivefunction(self, evaluation, simulation):
#         """Objective function using negative square-root KGE."""
#         evaluation, simulation = np.array(evaluation), np.array(simulation)
        
#         # CRITICAL FIX: Check for non-finite values BEFORE KGE to prevent worker crash
#         if not np.all(np.isfinite(simulation)) or len(evaluation) < 12:
#             return 9999.0 # Max penalty for invalid simulation/too short data
            
#         epsilon = 1e-6
#         Q_obs_trans = np.sqrt(np.clip(evaluation, epsilon, None))
#         Q_sim_trans = np.sqrt(np.clip(simulation, epsilon, None))
        
#         kge_sr = calculate_kge(Q_obs_trans, Q_sim_trans)
        
#         # Penalty if KGE is not a valid number
#         return -kge_sr if np.isfinite(kge_sr) else 9999.0


# # 💥 FUNDAMENTAL PERFORMANCE IMPROVEMENT:
# # The worker now takes only pre-extracted NumPy arrays and floats, eliminating Pandas overhead.
# def worker_calibrate_basin(
#     target_basin: str, 
#     P_data: np.ndarray, PET_data: np.ndarray, Q_usgs_kge: np.ndarray, 
#     S_init_val: float, G_init_val: float, SM_data: np.ndarray,
#     Q_nldas_cal: np.ndarray, basin_area_km2: float, Smax_heu: float, reps: int
# ) -> Tuple[str, Optional[Dict[str, Any]]]:

#     print(f"\n---> Calibrating Basin: {target_basin} (PID: {os.getpid()})...")

#     # Use the pre-calculated initial states and Smax_heu
#     initial_S = S_init_val
#     initial_G = G_init_val
#     Smax = Smax_heu

#     # Check for empty/NaN input data
#     if not np.all(np.isfinite(Q_nldas_cal)) or np.all(np.isnan(P_data)):
#         print(f" > Basin {target_basin} has non-finite/NaN NLDAS or P data. Returning fallback.")
#         return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)

#     print(f" > {target_basin} | Smax={Smax:.0f} | Initial S/G={initial_S:.0f}/{initial_G:.0f}")

#     # Q_nldas_cal is the Q_obs target for calibration
#     model = TwoStoreModel_SCE(P_data, PET_data, Q_nldas_cal, Smax, initial_S, initial_G, target_basin)
#     temp_dir = os.path.join(PROJECT_ROOT, 'SCE_cal_params')
#     db_name = os.path.join(temp_dir, f'sceua_{target_basin}')

#     try:
#         sampler = spotpy.algorithms.sceua(model, dbname=db_name, dbformat='csv', save_sim=False)
#         sampler.sample(repetitions=reps, ngs=70, kstop=60, peps=0.00005, pcento=0.00005)
#         res = sampler.getdata()
#     except Exception as e:
#         print(f"❌ SCE-UA failed for {target_basin}: {e}. Returning fallback.")
#         return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)
#     finally:
#         # CRITICAL FIX: Clean up DB file immediately after use
#         if os.path.exists(db_name + '.csv'):
#             os.remove(db_name + '.csv')


#     if res is None or len(res) == 0:
#         return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)

#     # Get best parameters
#     best_idx, _ = spotpy.analyser.get_minlikeindex(res)
#     best_params_raw = {
#         'Kperc': float(res['parKperc'][best_idx][0]), 'Kb': float(res['parKb'][best_idx][0]),
#         'Ke': float(res['parKe'][best_idx][0]), 'Cqq': float(res['parCqq'][best_idx][0]),
#         'bias': float(res['parBias'][best_idx][0]), 'S_init': initial_S,
#         'G_init': initial_G, 'Smax': Smax,
#     }

#     model_params_final = ModelParams(Smax=Smax, **{k: best_params_raw[k] for k in ['Kperc', 'Kb', 'Ke', 'Cqq']})
#     initial_state_final = (initial_S, initial_G, best_params_raw['bias'])
    
#     # Run against USGS Q_obs for final KGE evaluation
#     Q_sim_final, Q_obs_clean = run_forward_model(P_data, PET_data, Q_usgs_kge, initial_state_final, model_params_final)
#     best_kge_std = calculate_kge(Q_obs_clean, Q_sim_final)
#     best_params = {**best_params_raw, 'KGE': best_kge_std}

#     print(f" > SCE-UA Final (KGE vs USGS): {best_kge_std:.3f}")

#     # Check if KGE is poor or non-finite and use fallback if it's better
#     if best_kge_std <= MIN_KGE_THRESHOLD or not np.isfinite(best_kge_std):
#         fallback_params = get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)
#         if fallback_params['KGE'] > best_kge_std:
#             print(f"  🔄 Fallback used (KGE: {fallback_params['KGE']:.3f})")
#             return target_basin, fallback_params

#     return target_basin, clean_dict_for_json(best_params) # Ensure dict is clean


# def load_camels_attributes():
#     print("Loading CAMELS attributes...")
#     r = requests.get(
#         "https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/data/contents/camels_attributes_v2.0.feather"
#     )
#     attrs = gpd.read_feather(io.BytesIO(r.content)).reset_index(drop=False)
#     attrs['gauge_id'] = attrs['gauge_id'].astype(str).str.strip().str.zfill(8)
#     attrs.set_index('gauge_id', inplace=True)
#     print("✅ CAMELS attributes loaded successfully.")
#     return attrs


# if __name__ == '__main__':
#     print("Initializing data loading...")

#     try:
#         DFs = load_and_prepare_data()
#         (P_df, PET_df, _, _, _, _, Q_nldas_mm_monthly, Q_usgs_df,
#          S_init_df, G_init_df, SM_df) = DFs
#         if Q_usgs_df is None or Q_usgs_df.empty:
#             raise ValueError("Q_usgs_df is None/Empty.")
#     except Exception as e:
#         print(f"⚠️ DATA LOAD ERROR: {e}")
#         sys.exit(1)

#     try:
#         attrs = load_camels_attributes()
#     except Exception as e:
#         print(f"❌ FATAL: Failed to load CAMELS attributes. Error: {e}")
#         sys.exit(1)

#     TARGET_BASINS = list(Q_usgs_df.columns)
#     NUM_CORES = max(1, cpu_count() - 1)
    
#     # Use a small number of repetitions for quick performance test
#     REPS = 2000

#     print(f"\n🚀 Starting **PARALLEL SCE-UA Calibration** on {NUM_CORES} cores for {len(TARGET_BASINS)} basins...")

#     tasks = []
    
#     # 💥 FUNDAMENTAL IMPROVEMENT: Pre-extract all data to NumPy/floats
#     for basin in TARGET_BASINS:
#         try:
#             P_data_arr = P_df[basin].values
#             PET_data_arr = PET_df[basin].values
#             Q_usgs_kge_arr = Q_usgs_df[basin].values
#             Q_nldas_cal_arr = Q_nldas_mm_monthly[basin].values
#             SM_data_arr = SM_df[basin].values
#             basin_area_km2_val = attrs.loc[basin, 'area_gages2']
            
#             # Calculate initial state values
#             Smax_heu, initial_S_heu, initial_G_heu = 0, 0, 0
#             initial_S = min(S_init_df.loc[S_init_df.index[0], basin], initial_S_heu) \
#                 if basin in S_init_df.columns else initial_S_heu
#             initial_G = min(G_init_df.loc[G_init_df.index[0], basin], initial_G_heu) \
#                 if basin in G_init_df.columns else initial_G_heu
            
#             tasks.append(
#                 (basin, P_data_arr, PET_data_arr, Q_usgs_kge_arr, 
#                  initial_S, initial_G, SM_data_arr, 
#                  Q_nldas_cal_arr, basin_area_km2_val, Smax_heu, REPS) 
#             )
#         except KeyError:
#             print(f"Skipping basin {basin}: Missing one or more required data columns.")
#             continue

#     results = {}
#     try:
#         os.makedirs(os.path.join(PROJECT_ROOT, 'SCE_cal_params'), exist_ok=True)
        
#         # 💥 FUNDAMENTAL IMPROVEMENT: Use 'spawn' context Pool for stability and best Windows performance
#         with get_context("spawn").Pool(NUM_CORES) as pool:
#             parallel_results = pool.starmap(worker_calibrate_basin, tasks)

#         for basin, params in parallel_results:
#             if params:
#                 results[basin] = params
#     except Exception as e:
#         print(f"❌ FATAL: Parallel pool failed during execution. Error: {e}")
#         sys.exit(1)

#     output_path = os.path.join(PROJECT_ROOT, 'SCE_cal_params', 'final_calibrated_params.json')
#     cleaned_results = clean_dict_for_json(results)

#     with open(output_path, 'w') as f:
#         json.dump(cleaned_results, f, indent=2)

#     results_list = [{'Basin': k, **v} for k, v in results.items() if v.get('KGE') is not None]
#     df = pd.DataFrame(results_list).round(3).sort_values('KGE', ascending=False)

#     print("\n" + "=" * 80)
#     print("✅ ALL BASINS CALIBRATED!")
#     print(df.to_markdown(index=False))
#     print(f"\n💾 Saved: {output_path}")
#     print(f"\n🏆 SUCCESS: {len(df[df['KGE'] > 0.5])}/{len(TARGET_BASINS)} basins KGE > 0.5")




import os
import sys
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Tuple, Optional
from multiprocessing import cpu_count, get_context
import warnings
import spotpy
import requests
import io
import geopandas as gpd

# Suppress NumPy warnings (but still check for NaN/Inf explicitly)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

try:
    from src.model import ModelParams, two_store_model_step
    from src.metrics import calculate_kge
except ImportError as e:
    print(f"FATAL: Core module import failed. Error: {e}")
    sys.exit(1)

# Constants
S_MAX_CEILING = 2500.0
G_MAX_CEILING = S_MAX_CEILING * 3.0
DEFAULT_FALLBACK_PARAMS = {
    'Kperc': 0.25, 'Kb': 0.15, 'Ke': 0.7, 'Cqq': 0.85, 'bias': 0.0,
    'S_init': 0.5, 'G_init': 1.0, 'Smax': 700.0, 'KGE': 0.0
}
MIN_KGE_THRESHOLD = 0.6

# ------------------------- Forward model -------------------------
def run_forward_model(P_data, PET_data, Q_obs, initial_state: tuple, params: ModelParams) -> tuple:
    nmonths = len(P_data)
    S_curr, G_curr, bias = initial_state
    Q_sim = np.zeros(nmonths)

    for t in range(nmonths):
        P_t, PET_t = np.nan_to_num(max(P_data[t], 0.0)), np.nan_to_num(max(PET_data[t], 0.0))
        try:
            S_next, G_next, _, Q_t_no_bias, _, _, _ = two_store_model_step(
                S_curr, G_curr, P_t, PET_t, params, ET_override=None
            )
        except Exception:
            S_next, G_next, Q_t_no_bias = S_curr, G_curr, 0.0

        S_curr = np.clip(S_next, 0.0, params.Smax)
        G_curr = np.clip(G_next, 0.0, G_MAX_CEILING)
        Q_sim[t] = max(Q_t_no_bias + bias, 0.0)

    if not np.all(np.isfinite(Q_sim)):
        return np.zeros(1), np.zeros(1)

    spin_up = 20
    Q_sim_clean = Q_sim[spin_up:]
    Q_obs_clean = Q_obs[spin_up:]
    valid_mask = ~(np.isnan(Q_sim_clean) | np.isnan(Q_obs_clean))
    if valid_mask.sum() < 12:
        return np.zeros(1), np.zeros(1)

    return Q_sim_clean[valid_mask], Q_obs_clean[valid_mask]

# ------------------------- Fallback parameters -------------------------
def get_fallback_params(P_data, PET_data, Q_data, Smax, basin_area_km2):
    params = DEFAULT_FALLBACK_PARAMS.copy()
    S_init, G_init = 0.2, 0.5
    params.update({'S_init': S_init, 'G_init': G_init, 'Smax': Smax})
    model_params = ModelParams(Smax=Smax, **{k: params[k] for k in ['Kperc','Kb','Ke','Cqq']})
    initial_state = (params['S_init'], params['G_init'], params['bias'])
    Q_sim, Q_obs_clean = run_forward_model(P_data, PET_data, Q_data, initial_state, model_params)
    if len(Q_sim) > 0 and np.all(np.isfinite(Q_sim)):
        kge = calculate_kge(Q_obs_clean, Q_sim)
        params['KGE'] = kge if np.isfinite(kge) else 0.0
    else:
        params['KGE'] = 0.0
    return params

# ------------------------- JSON cleaner -------------------------
def clean_dict_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
    cleaned_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            cleaned_data[key] = clean_dict_for_json(value)
        elif isinstance(value, (np.float32, np.float64)):
            cleaned_data[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            cleaned_data[key] = int(value)
        else:
            cleaned_data[key] = value
    return cleaned_data

# ------------------------- SPOTPY Model Interface -------------------------
class TwoStoreModel_SCE:
    def __init__(self, P_data, PET_data, Q_obs, Smax, initial_S, initial_G, target_basin):
        self.P_data = P_data
        self.PET_data = PET_data
        self.Q_obs = Q_obs
        self.Smax = Smax
        self.initial_S_fallback = initial_S
        self.initial_G_fallback = initial_G
        self.target_basin = target_basin

    def parameters(self):
        return spotpy.parameter.generate([
            spotpy.parameter.Uniform(0.01, 1.0, name='Kperc'),
            spotpy.parameter.Uniform(0.01, 1.0, name='Kb'),
            spotpy.parameter.Uniform(0.01, 1.0, name='Ke'),
            spotpy.parameter.Uniform(0.1, 1.0, name='Cqq'),
            spotpy.parameter.Uniform(-0.5, 0.5, name='Bias'),
        ])

    def simulation(self, params):
        model_params = ModelParams(Smax=self.Smax, Kperc=params['Kperc'], Kb=params['Kb'],
                                   Ke=params['Ke'], Cqq=params['Cqq'])
        initial_state = (self.initial_S_fallback, self.initial_G_fallback, params['Bias'])
        Q_sim, _ = run_forward_model(self.P_data, self.PET_data, self.Q_obs, initial_state, model_params)
        return Q_sim.tolist() if len(Q_sim) > 0 else [np.nanmean(self.Q_obs)]

    def evaluation(self):
        _, Q_obs_clean = run_forward_model(self.P_data, self.PET_data, self.Q_obs,
                                           (self.initial_S_fallback, self.initial_G_fallback, 0.0),
                                           ModelParams(Smax=self.Smax))
        return Q_obs_clean.tolist() if len(Q_obs_clean) > 0 else [np.nanmean(self.Q_obs)]

    def objectivefunction(self, evaluation, simulation):
        evaluation, simulation = np.array(evaluation), np.array(simulation)
        if not np.all(np.isfinite(simulation)) or len(evaluation) < 12:
            return 9999.0
        epsilon = 1e-6
        Q_obs_trans = np.sqrt(np.clip(evaluation, epsilon, None))
        Q_sim_trans = np.sqrt(np.clip(simulation, epsilon, None))
        kge_sr = calculate_kge(Q_obs_trans, Q_sim_trans)
        return -kge_sr if np.isfinite(kge_sr) else 9999.0

# ------------------------- Worker Function -------------------------
def worker_calibrate_basin(target_basin: str, P_data: np.ndarray, PET_data: np.ndarray, Q_usgs_kge: np.ndarray,
                           S_init_val: float, G_init_val: float, SM_data: np.ndarray,
                           Q_nldas_cal: np.ndarray, basin_area_km2: float, Smax_heu: float, reps: int) -> Tuple[str, Optional[Dict[str, Any]]]:

    print(f"\n---> Calibrating Basin: {target_basin} (PID: {os.getpid()})...")
    initial_S, initial_G, Smax = S_init_val, G_init_val, Smax_heu

    if not np.all(np.isfinite(Q_nldas_cal)) or np.all(np.isnan(P_data)):
        print(f" > Basin {target_basin} has non-finite/NaN data. Returning fallback.")
        return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)

    model = TwoStoreModel_SCE(P_data, PET_data, Q_nldas_cal, Smax, initial_S, initial_G, target_basin)
    temp_dir = os.path.join(PROJECT_ROOT, 'SCE_cal_params')
    db_name = os.path.join(temp_dir, f'sceua_{target_basin}')

    try:
        sampler = spotpy.algorithms.sceua(model, dbname=db_name, dbformat='csv', save_sim=False)
        sampler.sample(repetitions=reps, ngs=70, kstop=60, peps=0.00005, pcento=0.00005)
        res = sampler.getdata()
    except Exception as e:
        print(f"❌ SCE-UA failed for {target_basin}: {e}. Using fallback.")
        return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)
    finally:
        if os.path.exists(db_name + '.csv'):
            os.remove(db_name + '.csv')

    if res is None or len(res) == 0:
        return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)

    best_idx, _ = spotpy.analyser.get_minlikeindex(res)
    best_params_raw = {
        'Kperc': float(res['parKperc'][best_idx][0]), 'Kb': float(res['parKb'][best_idx][0]),
        'Ke': float(res['parKe'][best_idx][0]), 'Cqq': float(res['parCqq'][best_idx][0]),
        'bias': float(res['parBias'][best_idx][0]), 'S_init': initial_S,
        'G_init': initial_G, 'Smax': Smax,
    }

    model_params_final = ModelParams(Smax=Smax, **{k: best_params_raw[k] for k in ['Kperc','Kb','Ke','Cqq']})
    initial_state_final = (initial_S, initial_G, best_params_raw['bias'])
    Q_sim_final, Q_obs_clean = run_forward_model(P_data, PET_data, Q_usgs_kge, initial_state_final, model_params_final)
    best_kge_std = calculate_kge(Q_obs_clean, Q_sim_final)
    best_params = {**best_params_raw, 'KGE': best_kge_std}

    if best_kge_std <= MIN_KGE_THRESHOLD or not np.isfinite(best_kge_std):
        fallback_params = get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)
        if fallback_params['KGE'] > best_kge_std:
            return target_basin, fallback_params

    return target_basin, clean_dict_for_json(best_params)

# ------------------------- CAMELS Attributes Loader -------------------------
def load_camels_attributes():
    r = requests.get(
        "https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/data/contents/camels_attributes_v2.0.feather"
    )
    attrs = gpd.read_feather(io.BytesIO(r.content)).reset_index(drop=False)
    attrs['gauge_id'] = attrs['gauge_id'].astype(str).str.strip().str.zfill(8)
    attrs.set_index('gauge_id', inplace=True)
    return attrs

# ------------------------- Main -------------------------
if __name__ == '__main__':
    print("Initializing data loading...")

    DATA_DIR = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\DA\DA_Github_repo\Bayesian_DA_Budyko_modeling\data\processed"
    Evap_df       = pd.read_feather(os.path.join(DATA_DIR, "Evap.feather")).set_index('time')
    PotEvap_df    = pd.read_feather(os.path.join(DATA_DIR, "PotEvap.feather")).set_index('time')
    Q_nldas_df    = pd.read_feather(os.path.join(DATA_DIR, "Q_nldas_mm_monthly.feather")).set_index('time')
    Q_usgs_df     = pd.read_feather(os.path.join(DATA_DIR, "Qsb.feather")).set_index('time')
    Rainf_df      = pd.read_feather(os.path.join(DATA_DIR, "Rainf.feather")).set_index('time')
    RootMoist_df  = pd.read_feather(os.path.join(DATA_DIR, "RootMoist.feather")).set_index('time')
    Soil_df       = pd.read_feather(os.path.join(DATA_DIR, "SoilM_0_200cm.feather")).set_index('time')

    try:
        attrs = load_camels_attributes()
    except Exception as e:
        print(f"❌ FATAL: Failed to load CAMELS attributes. Error: {e}")
        sys.exit(1)

    TARGET_BASINS = list(Q_usgs_df.columns)
    NUM_CORES = max(1, cpu_count() - 1)
    REPS = 2000

    tasks = []
    for basin in TARGET_BASINS:
        try:
            P_data_arr = Rainf_df[basin].values
            PET_data_arr = PotEvap_df[basin].values
            Q_usgs_kge_arr = Q_usgs_df[basin].values
            Q_nldas_cal_arr = Q_nldas_df[basin].values
            SM_data_arr = Soil_df[basin].values
            basin_area_km2_val = attrs.loc[basin, 'area_gages2']
            initial_S = 5
            initial_G = 3
            Smax_heu = 5.0
            tasks.append((basin, P_data_arr, PET_data_arr, Q_usgs_kge_arr,
                          initial_S, initial_G, SM_data_arr,
                          Q_nldas_cal_arr, basin_area_km2_val, Smax_heu, REPS))
        except KeyError:
            print(f"Skipping basin {basin}: Missing one or more required data columns.")
            continue

    results = {}
    os.makedirs(os.path.join(PROJECT_ROOT, 'SCE_cal_params'), exist_ok=True)
    with get_context("spawn").Pool(NUM_CORES) as pool:
        parallel_results = pool.starmap(worker_calibrate_basin, tasks)
    for basin, params in parallel_results:
        if params:
            results[basin] = params

    output_path = os.path.join(PROJECT_ROOT, 'SCE_cal_params', 'final_calibrated_params.json')
    cleaned_results = clean_dict_for_json(results)
    with open(output_path, 'w') as f:
        json.dump(cleaned_results, f, indent=2)

    results_list = [{'Basin': k, **v} for k, v in results.items() if v.get('KGE') is not None]
    df = pd.DataFrame(results_list).round(3).sort_values('KGE', ascending=False)

    print("\n" + "="*80)
    print("✅ ALL BASINS CALIBRATED!")
    print(df.to_markdown(index=False))
    print(f"\n💾 Saved: {output_path}")
    print(f"\n🏆 SUCCESS: {len(df[df['KGE'] > 0.5])}/{len(TARGET_BASINS)} basins KGE > 0.5")
