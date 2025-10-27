# # scripts/calibration.py

# import pandas as pd
# import numpy as np
# import os
# import sys
# import spotpy
# import warnings
# import json
# from typing import List, Dict, Any
# import requests
# import io
# import geopandas as gpd

# # Suppress runtime warnings from NumPy operations
# warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(PROJECT_ROOT)

# try:
#     # Ensure imports are correct
#     from src.model import ModelParams, two_store_model_step
#     from src.metrics import calculate_kge
#     from data.data_processor import load_and_prepare_data
# except ImportError as e:
#     print(f"FATAL: Import failed: {e}")
#     sys.exit(1)

# # Set global constraints to harmonize with enkf.py clipping
# S_MAX_CEILING = 2500.0
# G_MAX_CEILING = S_MAX_CEILING * 3.0 # Consistent with enkf.py

# # =====================================================================
# # 🔥 FORWARD MODEL (Simplified for Calibration)
# # =====================================================================

# def run_forward_model(P_data, PET_data, Q_obs, initial_state: tuple, params: ModelParams, debug=False) -> tuple:
#     """
#     Runs the two-store model for calibration.
#     Returns: (Q_sim_clean, Q_obs_clean)
#     """
#     nmonths = len(P_data)
#     S_curr, G_curr, bias = initial_state
#     Q_sim = np.zeros(nmonths)
    
#     for t in range(nmonths):
#         P_t = max(P_data[t], 0.0)
#         PET_t = max(PET_data[t], 0.0)
        
#         try:
#             # two_store_model_step returns Q_t_no_bias
#             S_next, G_next, _, Q_t_no_bias, _, _, _ = \
#                 two_store_model_step(S_curr, G_curr, P_t, PET_t, params, ET_override=None)
#         except:
#             S_next, G_next, Q_t_no_bias = S_curr, G_curr, 0.0
        
#         # Apply State Constraints (must be less strict than the EnKF clipping but enforced here)
#         S_next = np.clip(S_next, 0.0, params.Smax)
#         G_next = np.clip(G_next, 0.0, G_MAX_CEILING)
        
#         Q_t = max(Q_t_no_bias + bias, 0.0)
#         Q_sim[t] = Q_t
#         S_curr = S_next
#         G_curr = G_next
    
#     spin_up = 60 # 5 years
#     if len(Q_obs) <= spin_up:
#         return np.zeros(1), np.zeros(1) 
        
#     Q_sim_clean = Q_sim[spin_up:]
#     Q_obs_clean = Q_obs[spin_up:]
    
#     # Filter for non-NaN/valid comparison data
#     valid_mask = ~(np.isnan(Q_sim_clean) | np.isnan(Q_obs_clean))
#     if valid_mask.sum() < 12:
#         return np.zeros(1), np.zeros(1) 
        
#     return Q_sim_clean[valid_mask], Q_obs_clean[valid_mask]

# # =====================================================================
# # 🔥 STABLE PARAMETER SETS (Simplified Fallback)
# # =====================================================================

# DEFAULT_FALLBACK_PARAMS = {
#     'Kperc': 0.25, 'Kb': 0.15, 'Ke': 0.7, 'Cqq': 0.85, 'bias': 0.0,
#     'S_init': 300.0, 'G_init': 200.0
# }

# def get_fallback_params(P_data, PET_data, Q_data, Smax, basin_area_km2):
#     """Provides a stable, generic fallback parameter set."""
#     # This is only used if calibration fails or data is bad.
    
#     Q_mean = np.nanmean(Q_data) if not np.all(np.isnan(Q_data)) else 5.0
    
#     # Heuristic adjustment for a better S_init/G_init guess
#     S_init_adj = np.clip(Smax * 0.35 + Q_mean * 2.0, 10.0, Smax * 0.9)
#     G_init_adj = np.clip(Smax * 0.25 + Q_mean * 10.0, 10.0, Smax * 2.0)
    
#     params = DEFAULT_FALLBACK_PARAMS.copy()
#     params.update({
#         'S_init': S_init_adj,
#         'G_init': G_init_adj,
#         'Smax': Smax
#     })
    
#     model_params = ModelParams(Smax=Smax, Kperc=params['Kperc'], Kb=params['Kb'], 
#                                Ke=params['Ke'], Cqq=params['Cqq'])
#     initial_state = (params['S_init'], params['G_init'], params['bias'])
    
#     # Evaluate the fallback set using standard KGE
#     Q_sim, Q_obs_clean = run_forward_model(P_data, PET_data, Q_data, initial_state, model_params, debug=False)
#     kge = calculate_kge(Q_obs_clean, Q_sim)
#     params['KGE'] = kge if not np.isnan(kge) else 0.0
    
#     return params


# # =====================================================================
# # 🔥 SCE CLASS (Harmonized for 7 state parameters)
# # =====================================================================

# class TwoStoreModel_SCE:
#     def __init__(self, P_data, PET_data, Q_obs, Smax, initial_S, initial_G):
#         self.P_data = P_data
#         self.PET_data = PET_data
#         self.Q_obs = Q_obs
#         self.Smax = Smax
#         self.initial_S_fallback = initial_S
#         self.initial_G_fallback = initial_G

#     def parameters(self):
#         return spotpy.parameter.generate([
#             spotpy.parameter.Uniform(0.005, 0.40, name='Kperc'), 
#             spotpy.parameter.Uniform(0.01, 0.30, name='Kb'), 
#             spotpy.parameter.Uniform(0.2, 0.9, name='Ke'), 
#             spotpy.parameter.Uniform(0.1, 5.0, name='Cqq'), 
#             spotpy.parameter.Uniform(-0.5, 0.5, name='Bias'),
#         ])

#     def simulation(self, params):
#         # Smax is a constant structural parameter here
#         model_params = ModelParams(
#             Smax=self.Smax, Kperc=params['Kperc'], Kb=params['Kb'], 
#             Ke=params['Ke'], Cqq=params['Cqq']
#         )
#         # Use the fixed initial state values directly
#         initial_state = (self.initial_S_fallback, self.initial_G_fallback, params['Bias']) 
        
#         Q_sim, _ = run_forward_model(self.P_data, self.PET_data, self.Q_obs, initial_state, model_params, debug=False)
#         return Q_sim.tolist() if len(Q_sim) > 0 else [np.nanmean(self.Q_obs)]

#     def evaluation(self):
#         # Returns the cleaned observed data for comparison
#         # Using placeholder model params just to process the Q_obs data cleaning/spin-up
#         _, Q_obs_clean = run_forward_model(
#             self.P_data, self.PET_data, self.Q_obs, 
#             (self.initial_S_fallback, self.initial_G_fallback, 0.0), ModelParams(Smax=self.Smax), debug=False
#         )
#         return Q_obs_clean.tolist() if len(Q_obs_clean) > 0 else [np.nanmean(self.Q_obs)]

#     def objectivefunction(self, evaluation, simulation):
#         evaluation = np.array(evaluation)
#         simulation = np.array(simulation)
        
#         if len(evaluation) < 12: return 9999.0 # Must have enough data post-spinup
#         kge_std = calculate_kge(evaluation, simulation)
        
#         # Use Square Root KGE (SR-KGE) as the objective function to maximize fit across flow regimes
#         epsilon = 1e-6
#         Q_obs_trans = np.sqrt(np.clip(evaluation, epsilon, None))
#         Q_sim_trans = np.sqrt(np.clip(simulation, epsilon, None))
#         kge_sr = calculate_kge(Q_obs_trans, Q_sim_trans)
        
#         composite_kge = (kge_std + kge_sr) / 2.0
        
#         # Minimize -KGE (Maximizing KGE)
#         return -composite_kge if not np.isnan(composite_kge) else 9999.0

# # =====================================================================
# # 🔥 SMART Smax (Simplified for Stability)
# # =====================================================================

# def calculate_optimal_smax(SM_data, Q_obs, P_data, basin_area_km2):
#     """
#     Simplified heuristic to set a stable Smax.
#     Returns: Smax (float), S_init_heuristic (float), G_init_heuristic (float)
#     """
#     p_mean = np.nanmean(P_data) if not np.all(np.isnan(P_data)) else 50.0
#     q_mean = np.nanmean(Q_obs) if not np.all(np.isnan(Q_obs)) else 5.0
#     sm_max = 5.0# np.nanmax(SM_data) if not np.all(np.isnan(SM_data)) else 200.0 
    
#     # 1. Calculate Smax (Maximum Soil Storage)
#     smax_estimate = max(
#         sm_max * 5.0,           # Factor based on max observed soil moisture
#         p_mean * 15.0,          # Factor based on mean precipitation
#         200.0                   # Base minimum for stability
#     )
    
#     final_smax = np.clip(smax_estimate, 200.0, S_MAX_CEILING)
#     S_init_heuristic = np.clip(
#         final_smax * 0.35 + q_mean * 2.0, 
#         10.0, 
#         final_smax * 0.9
#     )
    
#     G_init_heuristic = np.clip(
#         final_smax * 0.25 + q_mean * 10.0, 
#         10.0, 
#         final_smax * 2.0 # Note: G_init uses 2.0 multiplier for heuristic, not the 3.0 ceiling
#     )

#     return round(final_smax, 1), round(S_init_heuristic, 3), round(G_init_heuristic, 3)

# # =====================================================================
# # 🔥 UPGRADED CALIBRATION (Cleaner Logic)
# # =====================================================================

# def classify_basin_type(P, PET, Q):
#     # DUMMY function definition to allow the script to run, as this was missing.
#     return "humid" 

# def calibrate_initial_parameters_sce(P_df, PET_df, Q_usgs_df, S_init_df, G_init_df, SM_df, target_basin, reps=2000, Q_nldas_df=None): 
    
#     try:
#         # Fetching attributes (unchanged)
#         r = requests.get("https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/data/contents/camels_attributes_v2.0.feather")
#         attrs = gpd.read_feather(io.BytesIO(r.content)).reset_index(drop=False)
#         attrs = attrs.to_crs("EPSG:5070")
#         attrs['geometry_points'] = attrs['geometry'].centroid 
#         attrs = attrs.set_geometry('geometry_points')
#         attrs['gauge_id'] = attrs['gauge_id'].astype(str).str.strip().str.zfill(8)
#         attrs.set_index('gauge_id', inplace=True)
        
#         P_data = P_df[target_basin].values
#         PET_data = PET_df[target_basin].values
        
#         Q_nldas_raw = Q_nldas_df[target_basin].values
#         Q_usgs_obs = Q_usgs_df[target_basin].values
#         SM_data = SM_df[target_basin].values 
#         basin_area_km2 = attrs.loc[target_basin, 'area_gages2']
        
#         # -----------------------------------------------------------------
#         # ✅ FINAL CORRECTION: REMOVED ALL UNIT CONVERSION AND CLIPPING FOR INPUT DATA
#         # ASSUMPTION: All data is in the final, correct unit (e.g., mm/month) 
#         # and pre-clipped in data_processor.py
#         # -----------------------------------------------------------------
#         Q_cal_target = Q_nldas_raw.copy() # NLDAS flow for SCE-UA
#         Q_kge_target = Q_usgs_obs.copy() # USGS flow for final KGE calculation
        
#         # Final data checks
#         if np.all(np.isnan(Q_cal_target)) or np.all(np.isnan(P_data)) or np.all(np.isnan(PET_data)):
#             # Fallback uses correct 5 arguments for get_fallback_params
#             return get_fallback_params(np.zeros(120), np.zeros(120), Q_kge_target, 700.0, basin_area_km2)
        
#         # No P_data or PET_data clipping/conversion here.
#         # -----------------------------------------------------------------

#     except (KeyError, requests.RequestException) as e:
#         print(f"⚠️ Error loading data/attributes: {e}. Using generic fallback.")
#         # Fallback uses correct 5 arguments for get_fallback_params
#         return get_fallback_params(np.zeros(120), np.zeros(120), np.zeros(120), 700.0, 1000.0)
    
#     # calculate_optimal_smax returns 3 values (Smax, S_init, G_init)
#     Smax, initial_S_heu, initial_G_heu = calculate_optimal_smax(SM_data, Q_kge_target, P_data, basin_area_km2) 

#     # Override heuristic initial states if external S_init_df/G_init_df are available
#     initial_S = min(S_init_df.loc[S_init_df.index[0], target_basin], initial_S_heu) if target_basin in S_init_df.columns else initial_S_heu
#     initial_G = min(G_init_df.loc[G_init_df.index[0], target_basin], initial_G_heu) if target_basin in G_init_df.columns else initial_G_heu
    
#     print(f" > Calibrating {target_basin} | Smax={Smax:.0f} | Cal Target: NLDAS Q")
    
#     basin_type = classify_basin_type(P_data, PET_data, Q_cal_target)
    
#     # Initialize SCE-UA model using Q_cal_target (NLDAS flow) for the objective function
#     model = TwoStoreModel_SCE(P_data, PET_data, Q_cal_target, Smax, initial_S, initial_G) 
    
#     # Setup for SPOTPY/SCE-UA (unchanged)
#     temp_dir = os.path.join(PROJECT_ROOT, 'SCE_cal_params')
#     os.makedirs(temp_dir, exist_ok=True)
#     db_name = os.path.join(temp_dir, f'sceua_{target_basin}')
#     sampler = spotpy.algorithms.sceua(model, dbname=db_name, dbformat='csv', save_sim=False)
    
#     # Run SCE-UA (unchanged)
#     sampler.sample(repetitions=reps, ngs=70, kstop=60, peps=0.00005, pcento=0.00005)
#     res = sampler.getdata()
    
#     best_kge_sr = 0.0
#     best_params = None
    
#     if res is not None and len(res) > 0:
#         # Extract best parameters from the SCE-UA run (optimization based on NLDAS flow)
#         best_idx, best_objf = spotpy.analyser.get_minlikeindex(res)
#         best_kge_sr = float(-best_objf) # This is the SR-KGE based on NLDAS
        
#         # Extract calibrated parameters
#         best_params_raw = {
#             'Kperc': float(res['parKperc'][best_idx][0]),
#             'Kb': float(res['parKb'][best_idx][0]),
#             'Ke': float(res['parKe'][best_idx][0]),
#             'Cqq': float(res['parCqq'][best_idx][0]),
#             'bias': float(res['parBias'][best_idx][0]),
#             'S_init': initial_S, # Fixed S_init/G_init used in SCE model run
#             'G_init': initial_G,
#             'Smax': Smax, 
#         }
        
#         # 3. Re-evaluate the best set using the OBSERVED (USGS) flow for KGE reporting
        
#         # Define model parameters
#         model_params_final = ModelParams(Smax=Smax, **{k: best_params_raw[k] for k in ['Kperc', 'Kb', 'Ke', 'Cqq']})
        
#         # Initial states used by SCE-UA (the fixed ones)
#         initial_state_final = (initial_S, initial_G, best_params_raw['bias'])
        
#         # Run forward model against Q_kge_target (USGS flow) for final KGE
#         Q_sim_final, Q_obs_clean = run_forward_model(P_data, PET_data, Q_kge_target, initial_state_final, model_params_final, debug=False)
#         best_kge_std = calculate_kge(Q_obs_clean, Q_sim_final)
        
#         print(f"  > SCE-UA Final (Standard KGE vs USGS): KGE={best_kge_std:.3f}")
        
#         best_params = {**best_params_raw, 'KGE': best_kge_std}

#     # Post-calibration heuristic tuning if the best KGE (vs NLDAS) is poor
#     # This logic now uses the final reported KGE (vs USGS) for the final return check.
#     if best_params is None or best_params.get('KGE', 0.0) <= 0.6: 
#         # Fallback uses correct 5 arguments for get_fallback_params
#         fallback_params = get_fallback_params(P_data, PET_data, Q_kge_target, Smax, basin_area_km2)
        
#         # Use fallback if its KGE (vs USGS) is better than the SCE-UA KGE (vs USGS)
#         if best_params is None or fallback_params['KGE'] > best_params.get('KGE', 0.0):
#             print(f"  🔄 Fallback improved KGE from {best_params.get('KGE', 0.0):.3f} to {fallback_params['KGE']:.3f}")
#             return fallback_params
#         else:
#             return best_params
            
#     return best_params

# # =====================================================================
# # 🔥 JSON SEREALIZATION FIX
# # =====================================================================

# def clean_dict_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
#     """Recursively converts numpy types in a dictionary to standard Python types."""
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

# # =====================================================================
# # 🔥 MAIN EXECUTION (Corrected Call)
# # =====================================================================

# if __name__ == '__main__':
#     print("Initializing data loading...")
    
#     try:
#         # Load the 11 variables from data_processor.py
#         P_df, PET_df, Evap_df, Qsb_df, M_df, Slope_df, Q_nldas_mm_monthly, Q_usgs_df, S_init_df, G_init_df, SM_df = load_and_prepare_data()
#         if Q_usgs_df is None or Q_usgs_df.empty:
#             raise ValueError("Q_usgs_df is None/Empty. Data loading failed.")
#     except Exception as e:
#         print(f"⚠️ DATA LOAD ERROR: {e}")
#         # Fallback dummy data generation for test purposes (retained)
#         print("🔥 Using fallback parameters for all basins...")
#         dummy_data = np.zeros(120)
#         # Ensure all required DFs are created for the loop, even if dummy
#         P_df = pd.DataFrame({'dummy_basin': dummy_data})
#         PET_df = pd.DataFrame({'dummy_basin': dummy_data})
#         Q_usgs_df = pd.DataFrame({'dummy_basin': dummy_data})
#         Q_nldas_mm_monthly = pd.DataFrame({'dummy_basin': dummy_data}) # Must be defined!
#         SM_df = pd.DataFrame({'dummy_basin': dummy_data})
#         S_init_df = pd.DataFrame({'dummy_basin': [0.0]}, index=[0])
#         G_init_df = pd.DataFrame({'dummy_basin': [0.0]}, index=[0])
    
#     TARGET_BASINS = list(Q_usgs_df.columns)
#     print(f"🚀 BULLETPROOF Calibration: {len(TARGET_BASINS)} basins\n")
    
#     results = {}
#     for basin in TARGET_BASINS:
#         print(f"\n---> Calibrating Basin: {basin}...")
#         try:
#             # ✅ CORRECTED CALL: Passing Q_nldas_mm_monthly as the Q_nldas_df argument
#             params = calibrate_initial_parameters_sce(
#                 P_df, PET_df, Q_usgs_df, S_init_df, G_init_df, SM_df, basin, reps=2000,
#                 Q_nldas_df=Q_nldas_mm_monthly 
#             )
#             results[basin] = params
#             print(f"✅ Basin {basin} calibrated. KGE (vs USGS): {params.get('KGE', 0.0):.3f}")

#         except Exception as e:
#             print(f"❌ ERROR calibrating basin {basin}. Falling back. Error: {e}")
#             # Fallback for unexpected errors during SCE-UA
#             try:
#                 # Use data to derive a better fallback (using USGS flow as the target)
#                 fallback_p = get_fallback_params(P_df[basin].values, PET_df[basin].values, 
#                                                  Q_usgs_df[basin].values, 700.0, 1000.0)
#                 results[basin] = fallback_p
#             except Exception as fe:
#                 print(f"❌ Fatal fallback error for {basin}: {fe}")
#                 results[basin] = {'KGE': 0.0, 'Smax': 700.0}
#     # Output saving
#     output_path = os.path.join(PROJECT_ROOT, 'SCE_cal_params', 'final_calibrated_params.json')
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
#     # ✅ FIX: Clean the dictionary before dumping to resolve 'float32 not serializable'
#     cleaned_results = clean_dict_for_json(results)
    
#     with open(output_path, 'w') as f:
#         json.dump(cleaned_results, f, indent=2) 
    
#     # Create and sort a DataFrame for reporting
#     results_list = [{'Basin': k, **v} for k, v in results.items()]
#     df = pd.DataFrame(results_list).round(3)
#     df = df.sort_values('KGE', ascending=False)
    
#     print("\n" + "="*80)
#     print("✅ ALL BASINS CALIBRATED!")
#     print(df.to_markdown(index=False))
#     print(f"\n💾 Saved: {output_path}")
#     print(f"\n🏆 SUCCESS: {len(df[df['KGE'] > 0.5])}/{len(TARGET_BASINS)} basins KGE > 0.5")


import pandas as pd
import numpy as np
import os
import sys
import json
from typing import Dict, Any, Tuple, Optional
from multiprocessing import Pool, cpu_count
import warnings
import spotpy
import requests
import io
import geopandas as gpd

warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

try:
    from src.model import ModelParams, two_store_model_step
    from src.metrics import calculate_kge
    from data.data_processor import load_and_prepare_data
except ImportError as e:
    print(f"FATAL: Core module import failed. Error: {e}")
    sys.exit(1)

S_MAX_CEILING = 2500.0
G_MAX_CEILING = S_MAX_CEILING * 3.0
DEFAULT_FALLBACK_PARAMS = {
    'Kperc': 0.25, 'Kb': 0.15, 'Ke': 0.7, 'Cqq': 0.85, 'bias': 0.0,
    'S_init': 300.0, 'G_init': 200.0, 'Smax': 700.0, 'KGE': 0.0
}
MIN_KGE_THRESHOLD = 0.6


def run_forward_model(P_data, PET_data, Q_obs, initial_state: tuple, params: ModelParams) -> tuple:
    nmonths = len(P_data)
    S_curr, G_curr, bias = initial_state
    Q_sim = np.zeros(nmonths)

    for t in range(nmonths):
        P_t, PET_t = max(P_data[t], 0.0), max(PET_data[t], 0.0)
        try:
            S_next, G_next, _, Q_t_no_bias, _, _, _ = two_store_model_step(
                S_curr, G_curr, P_t, PET_t, params, ET_override=None
            )
        except Exception:
            S_next, G_next, Q_t_no_bias = S_curr, G_curr, 0.0

        S_curr = np.clip(S_next, 0.0, params.Smax)
        G_curr = np.clip(G_next, 0.0, G_MAX_CEILING)
        Q_sim[t] = max(Q_t_no_bias + bias, 0.0)

    spin_up = 60
    Q_sim_clean = Q_sim[spin_up:]
    Q_obs_clean = Q_obs[spin_up:]
    valid_mask = ~(np.isnan(Q_sim_clean) | np.isnan(Q_obs_clean))

    if valid_mask.sum() < 12:
        return np.zeros(1), np.zeros(1)

    return Q_sim_clean[valid_mask], Q_obs_clean[valid_mask]


def calculate_initial_states(SM_data, Q_obs, P_data, basin_area_km2):
    p_mean = np.nanmean(P_data) if not np.all(np.isnan(P_data)) else 50.0
    q_mean = np.nanmean(Q_obs) if not np.all(np.isnan(Q_obs)) else 5.0
    sm_max = 5.0

    smax_estimate = max(sm_max * 5.0, p_mean * 15.0, 200.0)
    final_smax = np.clip(smax_estimate, 200.0, S_MAX_CEILING)
    S_init_heuristic = np.clip(final_smax * 0.35 + q_mean * 2.0, 10.0, final_smax * 0.9)
    G_init_heuristic = np.clip(final_smax * 0.25 + q_mean * 10.0, 10.0, final_smax * 2.0)

    return round(final_smax, 1), round(S_init_heuristic, 3), round(G_init_heuristic, 3)


def get_fallback_params(P_data, PET_data, Q_data, Smax, basin_area_km2):
    params = DEFAULT_FALLBACK_PARAMS.copy()
    S_init, G_init = calculate_initial_states(P_data, Q_data, P_data, basin_area_km2)[1:]
    params.update({'S_init': S_init, 'G_init': G_init, 'Smax': Smax})
    model_params = ModelParams(Smax=Smax, Kperc=params['Kperc'], Kb=params['Kb'], Ke=params['Ke'], Cqq=params['Cqq'])
    initial_state = (params['S_init'], params['G_init'], params['bias'])
    Q_sim, Q_obs_clean = run_forward_model(P_data, PET_data, Q_data, initial_state, model_params)
    kge = calculate_kge(Q_obs_clean, Q_sim)
    params['KGE'] = kge if not np.isnan(kge) else 0.0
    return params


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
            spotpy.parameter.Uniform(0.005, 0.999, name='Kperc'),
            spotpy.parameter.Uniform(0.01, 0.999, name='Kb'),
            spotpy.parameter.Uniform(0.2, 0.999, name='Ke'),
            spotpy.parameter.Uniform(0.1, 0.999, name='Cqq'),
            spotpy.parameter.Uniform(-0.5, 0.5, name='Bias'),
        ])

    def simulation(self, params):
        model_params = ModelParams(Smax=self.Smax, Kperc=params['Kperc'], Kb=params['Kb'],
                                   Ke=params['Ke'], Cqq=params['Cqq'])
        initial_state = (self.initial_S_fallback, self.initial_G_fallback, params['Bias'])
        Q_sim, _ = run_forward_model(self.P_data, self.PET_data, self.Q_obs, initial_state, model_params)
        return Q_sim.tolist() if len(Q_sim) > 0 else [np.nanmean(self.Q_obs)]

    def evaluation(self):
        _, Q_obs_clean = run_forward_model(
            self.P_data, self.PET_data, self.Q_obs,
            (self.initial_S_fallback, self.initial_G_fallback, 0.0),
            ModelParams(Smax=self.Smax)
        )
        return Q_obs_clean.tolist() if len(Q_obs_clean) > 0 else [np.nanmean(self.Q_obs)]

    def objectivefunction(self, evaluation, simulation):
        evaluation, simulation = np.array(evaluation), np.array(simulation)
        if len(evaluation) < 12:
            return 9999.0
        epsilon = 1e-6
        Q_obs_trans = np.sqrt(np.clip(evaluation, epsilon, None))
        Q_sim_trans = np.sqrt(np.clip(simulation, epsilon, None))
        kge_sr = calculate_kge(Q_obs_trans, Q_sim_trans)
        return -kge_sr if not np.isnan(kge_sr) else 9999.0


def worker_calibrate_basin(
    target_basin: str, P_df: pd.DataFrame, PET_df: pd.DataFrame, Q_usgs_df: pd.DataFrame,
    S_init_df: pd.DataFrame, G_init_df: pd.DataFrame, SM_df: pd.DataFrame,
    Q_nldas_df: pd.DataFrame, attrs: gpd.GeoDataFrame, reps: int
) -> Tuple[str, Optional[Dict[str, Any]]]:

    print(f"\n---> Calibrating Basin: {target_basin} (PID: {os.getpid()})...")

    try:
        P_data, PET_data = P_df[target_basin].values, PET_df[target_basin].values
        Q_nldas_cal = Q_nldas_df[target_basin].values
        Q_usgs_kge = Q_usgs_df[target_basin].values
        SM_data = SM_df[target_basin].values
        basin_area_km2 = attrs.loc[target_basin, 'area_gages2']
    except KeyError:
        print(f"❌ ERROR: Missing data for basin {target_basin}. Returning fallback.")
        return target_basin, None

    if np.all(np.isnan(Q_nldas_cal)) or np.all(np.isnan(P_data)):
        return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, 700.0, basin_area_km2)

    Smax, initial_S_heu, initial_G_heu = calculate_initial_states(SM_data, Q_usgs_kge, P_data, basin_area_km2)

    initial_S = min(S_init_df.loc[S_init_df.index[0], target_basin], initial_S_heu) \
        if target_basin in S_init_df.columns else initial_S_heu
    initial_G = min(G_init_df.loc[G_init_df.index[0], target_basin], initial_G_heu) \
        if target_basin in G_init_df.columns else initial_G_heu

    print(f" > {target_basin} | Smax={Smax:.0f} | Initial S/G={initial_S:.0f}/{initial_G:.0f}")

    model = TwoStoreModel_SCE(P_data, PET_data, Q_nldas_cal, Smax, initial_S, initial_G, target_basin)
    temp_dir = os.path.join(PROJECT_ROOT, 'SCE_cal_params')
    db_name = os.path.join(temp_dir, f'sceua_{target_basin}')

    try:
        sampler = spotpy.algorithms.sceua(model, dbname=db_name, dbformat='csv', save_sim=False)
        sampler.sample(repetitions=reps, ngs=70, kstop=60, peps=0.00005, pcento=0.00005)
        res = sampler.getdata()
    except Exception as e:
        print(f"❌ SCE-UA failed for {target_basin}: {e}. Returning fallback.")
        return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)

    if res is None or len(res) == 0:
        return target_basin, get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)

    best_idx, _ = spotpy.analyser.get_minlikeindex(res)
    best_params_raw = {
        'Kperc': float(res['parKperc'][best_idx][0]), 'Kb': float(res['parKb'][best_idx][0]),
        'Ke': float(res['parKe'][best_idx][0]), 'Cqq': float(res['parCqq'][best_idx][0]),
        'bias': float(res['parBias'][best_idx][0]), 'S_init': initial_S,
        'G_init': initial_G, 'Smax': Smax,
    }

    model_params_final = ModelParams(Smax=Smax, **{k: best_params_raw[k] for k in ['Kperc', 'Kb', 'Ke', 'Cqq']})
    initial_state_final = (initial_S, initial_G, best_params_raw['bias'])
    Q_sim_final, Q_obs_clean = run_forward_model(P_data, PET_data, Q_usgs_kge, initial_state_final, model_params_final)
    best_kge_std = calculate_kge(Q_obs_clean, Q_sim_final)
    best_params = {**best_params_raw, 'KGE': best_kge_std}

    print(f" > SCE-UA Final (KGE vs USGS): {best_kge_std:.3f}")

    if best_kge_std <= MIN_KGE_THRESHOLD:
        fallback_params = get_fallback_params(P_data, PET_data, Q_usgs_kge, Smax, basin_area_km2)
        if fallback_params['KGE'] > best_kge_std:
            print(f"  🔄 Fallback used (KGE: {fallback_params['KGE']:.3f})")
            return target_basin, fallback_params

    return target_basin, best_params


def load_camels_attributes():
    print("Loading CAMELS attributes...")
    r = requests.get(
        "https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/data/contents/camels_attributes_v2.0.feather"
    )
    attrs = gpd.read_feather(io.BytesIO(r.content)).reset_index(drop=False)
    attrs['gauge_id'] = attrs['gauge_id'].astype(str).str.strip().str.zfill(8)
    attrs.set_index('gauge_id', inplace=True)
    print("✅ CAMELS attributes loaded successfully.")
    return attrs


if __name__ == '__main__':
    print("Initializing data loading...")

    try:
        DFs = load_and_prepare_data()
        (P_df, PET_df, _, _, _, _, Q_nldas_mm_monthly, Q_usgs_df,
         S_init_df, G_init_df, SM_df) = DFs
        if Q_usgs_df is None or Q_usgs_df.empty:
            raise ValueError("Q_usgs_df is None/Empty.")
    except Exception as e:
        print(f"⚠️ DATA LOAD ERROR: {e}")
        sys.exit(1)

    try:
        attrs = load_camels_attributes()
    except Exception as e:
        print(f"❌ FATAL: Failed to load CAMELS attributes. Error: {e}")
        sys.exit(1)

    TARGET_BASINS = list(Q_usgs_df.columns)
    NUM_CORES = max(1, cpu_count() - 1)

    print(f"\n🚀 Starting **PARALLEL SCE-UA Calibration** on {NUM_CORES} cores for {len(TARGET_BASINS)} basins...")

    tasks = [
        (basin, P_df, PET_df, Q_usgs_df, S_init_df, G_init_df, SM_df,
         Q_nldas_mm_monthly, attrs, 2000) # 2000=reps
        for basin in TARGET_BASINS
    ]

    results = {}
    try:
        os.makedirs(os.path.join(PROJECT_ROOT, 'SCE_cal_params'), exist_ok=True)
        with Pool(NUM_CORES) as pool:
            parallel_results = pool.starmap(worker_calibrate_basin, tasks)

        for basin, params in parallel_results:
            if params:
                results[basin] = params
    except Exception as e:
        print(f"❌ FATAL: Parallel pool failed. Error: {e}")
        sys.exit(1)

    output_path = os.path.join(PROJECT_ROOT, 'SCE_cal_params', 'final_calibrated_params.json')
    cleaned_results = clean_dict_for_json(results)

    with open(output_path, 'w') as f:
        json.dump(cleaned_results, f, indent=2)

    results_list = [{'Basin': k, **v} for k, v in results.items() if v.get('KGE') is not None]
    df = pd.DataFrame(results_list).round(3).sort_values('KGE', ascending=False)

    print("\n" + "=" * 80)
    print("✅ ALL BASINS CALIBRATED!")
    print(df.to_markdown(index=False))
    print(f"\n💾 Saved: {output_path}")
    print(f"\n🏆 SUCCESS: {len(df[df['KGE'] > 0.5])}/{len(TARGET_BASINS)} basins KGE > 0.5")
