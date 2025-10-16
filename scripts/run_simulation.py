# scripts/run_simulation.py

import pandas as pd
import numpy as np
import warnings
import os
import sys

# Add directories to the system path to allow imports
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'data')) # Import the new data folder

from enkf import EnKFConfig, run_enkf_scenario 
from model import ModelParams, run_two_store_model
from budyko import solve_omega_true, fit_omega_mlr, fu_budyko
from metrics import calculate_nse, calculate_kge
from data_processor import load_and_prepare_data # Import the new function

warnings.filterwarnings('ignore')

# =====================================================================
# 3. MAIN PROCESSING/ORCHESTRATION FUNCTION
# =====================================================================

BURN_IN_MONTHS = 60 # 5 years (60 months) spin-up period

def process_all_basins(basin_names, P_df, PET_df, ET_obs_df, Qb_df, M_df, 
                       Slope_df, Q_nldas_df, Q_usgs_df, nens=400):
    """Orchestrates the simulation of all scenarios for all basins."""
    
    enkf_cfg = EnKFConfig(nens=nens, R_Q=20.0, inflation=1.03) 
    all_results = {}
    
    for basin_id in basin_names:
        
        try:
            # Extract time series data for the current basin
            P = P_df[basin_id].values
            PET = PET_df[basin_id].values
            Qb = Qb_df[basin_id].values 
            Q_nldas = Q_nldas_df[basin_id].values 
            Q_usgs = Q_usgs_df[basin_id].values 
            M = M_df[basin_id].values 
            Slope_val = Slope_df[basin_id].iloc[0] # Static slope value
            Slope_for_fit = np.full_like(M, Slope_val)

        except KeyError:
            print(f"Skipping basin {basin_id}: Missing required data.")
            continue
        
        # --- 3a. Budyko/Omega Pre-Calculation ---
        ET_ke = 0.7 * PET 
        omega_true = solve_omega_true(P, PET, ET_ke, Qb)
        
        omega_mlr_model = fit_omega_mlr(M, Slope_for_fit, omega_true)
        omega_mlr = omega_mlr_model.predict(M, Slope_for_fit)
        
        P_minus_dS = ET_ke + Qb 
        phi = np.divide(PET, P_minus_dS, out=np.full_like(PET, np.nan), where=P_minus_dS!=0)
        ET_B_ratio = fu_budyko(phi, omega_mlr)
        ET_B = np.clip(ET_B_ratio * P_minus_dS, 0.0, PET)
        
        # --- 3b. SCENARIO 1: Base Model (Q_ke) ---
        params_ke = ModelParams(Ke=0.7)
        results_ke = run_two_store_model(P, PET, params_ke, ET_input=ET_ke)
        Q_ke = results_ke['Q']
        
        # --- 3c. SCENARIO 2: Base Model + DA (Q_assim_base) ---
        results_assim_base = run_enkf_scenario(P, PET, Q_nldas, None, 'Base', enkf_cfg)
        Q_assim_base = results_assim_base['Q_mean']
        
        # --- 3d. SCENARIO 3: Budyko Model + DA (Q_assim_Budyko) ---
        results_assim_budyko = run_enkf_scenario(P, PET, Q_nldas, ET_B, 'Budyko', enkf_cfg)
        Q_assim_budyko = results_assim_budyko['Q_mean']
        
        # --- 3e. Evaluation (Applying Burn-in) ---
        Q_usgs_eval = Q_usgs[BURN_IN_MONTHS:]
        Q_ke_eval = Q_ke[BURN_IN_MONTHS:]
        Q_assim_base_eval = Q_assim_base[BURN_IN_MONTHS:]
        Q_assim_budyko_eval = Q_assim_budyko[BURN_IN_MONTHS:]
        
        metrics_ke = {'NSE': calculate_nse(Q_usgs_eval, Q_ke_eval), 'KGE': calculate_kge(Q_usgs_eval, Q_ke_eval)}
        metrics_base = {'NSE': calculate_nse(Q_usgs_eval, Q_assim_base_eval), 'KGE': calculate_kge(Q_usgs_eval, Q_assim_base_eval)}
        metrics_budyko = {'NSE': calculate_nse(Q_usgs_eval, Q_assim_budyko_eval), 'KGE': calculate_kge(Q_usgs_eval, Q_assim_budyko_eval)}
        
        all_results[basin_id] = {
            'Q_ke': Q_ke,
            'Q_assim_base': Q_assim_base,
            'Q_assim_Budyko': Q_assim_budyko,
            'Metrics_Q_ke': metrics_ke,
            'Metrics_Q_assim_base': metrics_base,
            'Metrics_Q_assim_Budyko': metrics_budyko,
        }
        
    return all_results

# =====================================================================
# 4. EXECUTION BLOCK (Main)
# =====================================================================

if __name__ == '__main__':
    print("--- Starting Hydrologic Data Assimilation Simulation ---")
    
    # 1. Load Data
    P_df, PET_df, ET_obs_df, Qb_df, M_df, Slope_df, Q_nldas_df, Q_usgs_df = load_and_prepare_data()

    # Check for successful data loading
    if P_df is None:
        print("\nSimulation aborted due to data loading errors.")
        sys.exit(1)
        
    BASIN_IDS = P_df.columns.tolist()
    
    print(f"\nSuccessfully prepared aligned data for {len(BASIN_IDS)} basins.")

    # 2. Run Simulation
    print("\nRunning three scenarios (Base, Base+DA, Budyko+DA)...")
    simulation_results = process_all_basins(
        BASIN_IDS, P_df, PET_df, ET_obs_df, Qb_df, M_df, Slope_df, 
        Q_nldas_df, Q_usgs_df, nens=400
    )
    print("\nSimulation complete.")
    
    # 3. Summarize Results
    summary_data = {
        f'{scenario}_{metric}': [
            simulation_results[bid][f'Metrics_{scenario}'][metric]  
            for bid in BASIN_IDS
        ]
        for scenario in ['Q_ke', 'Q_assim_base', 'Q_assim_Budyko']
        for metric in ['NSE', 'KGE']
    }
    
    summary = pd.DataFrame(summary_data, index=BASIN_IDS)
    
    print("\n--- Summary Performance Metrics (vs Q_USGS, after 5-year spin-up) ---")
    print(summary.round(3))