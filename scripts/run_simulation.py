# scripts/run_simulation.py

import pandas as pd
import numpy as np
import warnings
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT) 

from src.enkf import EnKFConfig, run_enkf_scenario
from src.model import ModelParams, run_two_store_model, calculate_fluxes_from_states 
from src.budyko import solve_omega_true, fit_omega_mlr, fu_budyko
from src.metrics import calculate_nse, calculate_kge

# For modules in 'data' folder:
from data.data_processor import load_and_prepare_data 

warnings.filterwarnings('ignore')

# =====================================================================
# 3. MAIN PROCESSING/ORCHESTRATION FUNCTION
# =====================================================================

BURN_IN_MONTHS = 60 # 5 years (60 months) spin-up period


def process_all_basins(basin_names, P_df, PET_df, ET_obs_df, Qb_df, M_df, 
                       Slope_df, Q_nldas_df, Q_usgs_df, 
                       S_init_df, G_init_df, 
                       nens=400):
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

            # Extract the NLDAS initial state for the basin
            S_init = S_init_df[basin_id].iloc[0]
            G_init = G_init_df[basin_id].iloc[0]
            
        except KeyError:
            print(f"Skipping basin {basin_id}: Missing required data.")
            continue
        
        # --- 3a. Budyko/Omega Pre-Calculation ---
        ET_ke = 0.7 * PET 
        omega_true = solve_omega_true(P, PET, ET_ke, Qb)
        
        omega_mlr_model = fit_omega_mlr(M, Slope_for_fit, omega_true)
        omega_mlr = omega_mlr_model.predict(M, Slope_for_fit) # This is the predicted omega
        
        P_minus_dS = ET_ke + Qb 
        phi = np.divide(PET, P_minus_dS, out=np.full_like(PET, np.nan), where=P_minus_dS!=0)
        ET_B_ratio = fu_budyko(phi, omega_mlr)
        ET_B = np.clip(ET_B_ratio * P_minus_dS, 0.0, PET)
        
        # --- 3b. SCENARIO 1: Base Model (Q_ke) ---
        params_ke = ModelParams(Ke=0.7)
        # Using the NLDAS initial states for the base model too for consistency
        results_ke = run_two_store_model(P, PET, params_ke, ET_input=ET_ke, initial_S=S_init, initial_G=G_init) 
        
        # --- 3c. SCENARIO 2: Base Model + DA (Q_assim_base) ---
        results_assim_base = run_enkf_scenario(
            P, PET, Q_nldas, None, 'Base', enkf_cfg, 
            target_basin=basin_id,
            S_init_nldas=S_init, 
            G_init_nldas=G_init 
        )
        
        # NEW: Post-process DA states to recalculate fluxes for Base + DA
        fluxes_base_da = calculate_fluxes_from_states(
            P, results_assim_base['ET_mean'], 
            results_assim_base['S_mean'], 
            results_assim_base['G_mean'], 
            params_ke
        )
        
        # --- 3d. SCENARIO 3: Budyko Model + DA (Q_assim_Budyko) ---
        results_assim_budyko = run_enkf_scenario(
            P, PET, Q_nldas, ET_B, 'Budyko', enkf_cfg,
            target_basin=basin_id,
            S_init_nldas=S_init, 
            G_init_nldas=G_init 
        )
        
        # NEW: Post-process DA states to recalculate fluxes for Budyko + DA
        params_budyko_da = ModelParams(Ke=0.7) 
        
        fluxes_budyko_da = calculate_fluxes_from_states(
            P, results_assim_budyko['ET_mean'], 
            results_assim_budyko['S_mean'], 
            results_assim_budyko['G_mean'], 
            params_budyko_da
        )

        # --- 3e. Evaluation (Applying Burn-in) ---
        Q_usgs_eval = Q_usgs[BURN_IN_MONTHS:]
        Q_ke_eval = results_ke['Q'][BURN_IN_MONTHS:]
        # Use the recalculated Q for DA scenarios for consistency
        Q_assim_base_eval = fluxes_base_da['Q'][BURN_IN_MONTHS:] 
        Q_assim_budyko_eval = fluxes_budyko_da['Q'][BURN_IN_MONTHS:] 
        
        metrics_ke = {'NSE': calculate_nse(Q_usgs_eval, Q_ke_eval), 'KGE': calculate_kge(Q_usgs_eval, Q_ke_eval)}
        metrics_base = {'NSE': calculate_nse(Q_usgs_eval, Q_assim_base_eval), 'KGE': calculate_kge(Q_usgs_eval, Q_assim_base_eval)}
        metrics_budyko = {'NSE': calculate_nse(Q_usgs_eval, Q_assim_budyko_eval), 'KGE': calculate_kge(Q_usgs_eval, Q_assim_budyko_eval)}
        
        # ======================================================================
        # EXPANDED RESULTS DICTIONARY
        # ======================================================================
        all_results[basin_id] = {
            # --- Observed Data ---
            'Q_usgs': Q_usgs,                               # <--- NEW: Added Observed Streamflow

            # --- Streamflow and Metrics (CORRECTED Qs used for DA) ---
            'Q_ke': results_ke['Q'],
            'Q_assim_base': fluxes_base_da['Q'],         # Recalculated Q
            'Q_assim_Budyko': fluxes_budyko_da['Q'],     # Recalculated Q
            'Metrics_Q_ke': metrics_ke,
            'Metrics_Q_assim_base': metrics_base,
            'Metrics_Q_assim_Budyko': metrics_budyko,

            # --- Base Model (No DA) - Full Fluxes/States ---
            'S_ke': results_ke['S'], 'G_ke': results_ke['G'],
            'ET_ke': results_ke['ET'], 'Qs_ke': results_ke['Qs'], 
            'Qb_ke': results_ke['Qb'], 'Perc_ke': results_ke['Perc'],
            
            # --- Base + DA Model - Mean States/Fluxes (Recalculated) ---
            'S_assim_base': results_assim_base['S_mean'], 
            'G_assim_base': results_assim_base['G_mean'],
            'ET_assim_base': results_assim_base['ET_mean'], 
            'Qs_assim_base': fluxes_base_da['Qs'],      
            'Qb_assim_base': fluxes_base_da['Qb'],      
            'Perc_assim_base': fluxes_base_da['Perc'],  
            
            # --- Budyko + DA Model - Mean States/Fluxes (Recalculated) ---
            'S_assim_Budyko': results_assim_budyko['S_mean'], 
            'G_assim_Budyko': results_assim_budyko['G_mean'],
            'ET_assim_Budyko': results_assim_budyko['ET_mean'],
            'Qs_assim_Budyko': fluxes_budyko_da['Qs'],      
            'Qb_assim_Budyko': fluxes_budyko_da['Qb'],      
            'Perc_assim_Budyko': fluxes_budyko_da['Perc'],  
            
            # --- Budyko/MLR Results ---
            'omega_true': omega_true,
            'omega_mlr': omega_mlr,
        }
        
    return all_results

# ... (Lines 1 to 206 of scripts/run_simulation.py remain unchanged) ...

# =====================================================================
# 4. EXECUTION BLOCK (Main)
# =====================================================================

if __name__ == '__main__':
    print("--- Starting Hydrologic Data Assimilation Simulation ---")
    
    # 1. Load Data
    P_df, PET_df, ET_obs_df, Qb_df, M_df, Slope_df, Q_nldas_df, Q_usgs_df, S_init_df, G_init_df = load_and_prepare_data()

    # Check for successful data loading
    if P_df is None:
        print("\nSimulation aborted due to data loading errors.")
        sys.exit(1)
        
    BASIN_IDS = P_df.columns.tolist()
    
    print(f"\nSuccessfully prepared aligned data for {len(BASIN_IDS)} basins.")

    # 2. Run Simulation
    print("\nRunning three scenarios (Base, Base+DA, Budyko+DA)...")
    simulation_results = process_all_basins(
        BASIN_IDS, 
        P_df, PET_df, ET_obs_df, Qb_df, M_df, 
        Slope_df, Q_nldas_df, Q_usgs_df,
        S_init_df, 
        G_init_df  
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

# ---------------------------------------------------------------------
# --- 4. SAVE RAW TIME SERIES RESULTS FOR NOTEBOOK ANALYSIS (FIXED) ---
# ---------------------------------------------------------------------
    
    # Define the results directory
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    # Create a new subdirectory for basin-specific time series
    ts_dir = os.path.join(results_dir, 'time_series_by_basin')
    os.makedirs(ts_dir, exist_ok=True) 

    # Define components in order of importance for column formatting
    # Q, ET, S, G are generally most important, followed by individual fluxes, then Budyko params
    ORDERED_COMPONENTS = ['Q', 'ET', 'S', 'G', 'Qs', 'Qb', 'Perc']
    SCENARIOS = ['ke', 'assim_base', 'assim_Budyko']
    BUDYKO_COMPONENTS = ['omega_true', 'omega_mlr']


    print(f"\nSaving individual basin time series to: {ts_dir}")
    
    # Iterate through each basin and create a dedicated CSV file
    for basin_id in BASIN_IDS:
        basin_data = simulation_results[basin_id]
        
        # 1. Build the list of columns in the desired order
        ordered_columns = []
        data_dict = {}
        
        # Add primary state/flux components
        for component in ORDERED_COMPONENTS:
            for scenario in SCENARIOS:
                key = f'{component}_{scenario}'
                col_name = f'{component}_{scenario}'
                
                # Check if the key exists (it should, based on results dictionary definition)
                if key in basin_data:
                    data_dict[col_name] = basin_data[key]
                    ordered_columns.append(col_name)

        # Add Budyko parameters (omega) - these are scalars repeated over time
        for component in BUDYKO_COMPONENTS:
            col_name = component
            data_dict[col_name] = basin_data[component]
            ordered_columns.append(col_name)

        # 2. Create the DataFrame using the correct index and ordered columns
        # We use the index from the Q_usgs_df since all time series share it
        basin_df = pd.DataFrame(data_dict, index=Q_usgs_df.index, columns=ordered_columns)

        # 3. Save the DataFrame
        file_name = f'{basin_id}_simulated_components.csv'
        file_path = os.path.join(ts_dir, file_name)
        basin_df.to_csv(file_path, index_label='Date')
        
    print(f"Successfully saved time series for {len(BASIN_IDS)} basins.")

    # Save the observed time series for easy comparison (Unchanged)
    Q_obs_file_path = os.path.join(results_dir, 'Q_observed_usgs.csv')
    Q_usgs_df.to_csv(Q_obs_file_path, index_label='Date')
    print(f"Saved observed streamflow to: {Q_obs_file_path}")