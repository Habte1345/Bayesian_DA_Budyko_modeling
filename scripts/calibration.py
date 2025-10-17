import pandas as pd
import numpy as np
import os
import sys
import spotpy 
import warnings
from typing import List, Dict, Any

# Ignore specific numpy warnings during optimization
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')

# Get the path to the project root (Bayesian_DA_Budyko_modeling)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# Import necessary project components
# NOTE: Ensure src/model.py, src/metrics.py, and data/data_processor.py exist and are correct
try:
    from src.model import ModelParams, two_store_model_step
    from src.metrics import calculate_nse, calculate_kge
    from data.data_processor import load_and_prepare_data
except ImportError as e:
    print(f"FATAL: One or more required modules failed to import. Check file paths and existence: {e}")
    sys.exit(1)


# =====================================================================
# 1. FORWARD MODEL RUNNER (Used by SCE-UA objective function)
# =====================================================================

def run_forward_model(P_data, PET_data, Q_obs, initial_state, params: ModelParams) -> tuple:
    """Runs the two-store model for a single parameter set."""
    nmonths = len(P_data)
    S_curr, G_curr, bias = initial_state
    
    Q_sim = np.zeros(nmonths)
    
    for t in range(nmonths):
        P_t = P_data[t]
        PET_t = PET_data[t]
        
        # ET_override=None for calibration
        S_next, G_next, _, Q_t_no_bias, _, _, _ = \
            two_store_model_step(S_curr, G_curr, P_t, PET_t, params, ET_override=None)

        # Apply bias and ensure non-negative flow
        Q_t = max(Q_t_no_bias + bias, 0.0)

        Q_sim[t] = Q_t
        
        S_curr = S_next
        G_curr = G_next
        
    # Use 10 months spin-up for the calibration objective
    spin_up_months = 10 
    
    # Check if Q_obs is long enough (it should be, but safety check)
    if len(Q_obs) <= spin_up_months:
        return np.array([]), np.array([])
        
    return Q_sim[spin_up_months:], Q_obs[spin_up_months:]


# =====================================================================
# 2. SPOTPY SCE-UA CALIBRATION CLASS
# =====================================================================

class TwoStoreModel_SCE:
    """SPOTPY class wrapper for the two-store hydrologic model."""
    
    def __init__(self, P_data, PET_data, Q_obs, Smax, initial_S, initial_G):
        self.P_data = P_data
        self.PET_data = PET_data
        self.Q_obs = Q_obs
        self.Smax = Smax
        self.initial_S = initial_S
        self.initial_G = initial_G

    def parameters(self):
        """Define the parameters and their bounds for the optimizer."""
        params = [
            # Name, Lower Bound, Upper Bound
            spotpy.parameter.Uniform(0.001, 0.999, name='Kperc'), # Percolation rate
            spotpy.parameter.Uniform(0.001, 0.999, name='Kb'),    # Baseflow recession constant
            spotpy.parameter.Uniform(0.01, 1.0, name='Ke'),       # ET coefficient
            spotpy.parameter.Uniform(0.01, 10.0, name='Cqq'),     # Quickflow coefficient
            spotpy.parameter.Uniform(-15.0, 15.0, name='Bias')    # Streamflow bias
        ]
        return spotpy.parameter.generate(params)

    def simulation(self, params):
        """Runs the model for a given set of parameters."""
        p = params 
        
        # Unpack parameters and set up ModelParams
        model_params = ModelParams(
            Smax=self.Smax, 
            Kperc=p['Kperc'], 
            Kb=p['Kb'], 
            Ke=p['Ke'], 
            Cqq=p['Cqq']
        )
        initial_state = (self.initial_S, self.initial_G, p['Bias'])
        
        # Run the forward model, only retaining the simulated streamflow for the objective
        Q_sim_cut, _ = run_forward_model(
            self.P_data, self.PET_data, self.Q_obs, initial_state, model_params
        )
        return Q_sim_cut

    def evaluation(self):
        """Returns the observed streamflow for comparison."""
        # Use a dummy run to get the cut observation series (for spin-up handling)
        _, Q_obs_cut = run_forward_model(
            self.P_data, self.PET_data, self.Q_obs, 
            (self.initial_S, self.initial_G, 0.0), ModelParams(Smax=self.Smax) 
        ) 
        return Q_obs_cut

    def objectivefunction(self, evaluation, simulation):
        """The function to be minimized: negative KGE."""
        # Ensure simulation is not empty (e.g., due to spin-up error)
        if simulation.size == 0 or evaluation.size == 0:
             return 9999.0
             
        kge = calculate_kge(evaluation, simulation)
        
        # Handle instability/NaNs by returning a high cost
        if np.isnan(kge) or kge < -100.0:
            return 9999.0
            
        # Minimize the negative KGE to maximize the KGE score
        return -kge


# =====================================================================
# 3. SINGLE-BASIN CALIBRATION EXECUTION
# =====================================================================

def calibrate_initial_parameters_sce(
    P_df: pd.DataFrame, PET_df: pd.DataFrame, Q_usgs_df: pd.DataFrame, 
    target_basin: str, repetitions: int = 5000 
) -> Dict[str, Any]:
    
    P_data = P_df[target_basin].values
    PET_data = PET_df[target_basin].values
    Q_obs = Q_usgs_df[target_basin].values
    
    # Use the corrected, realistic physical constraints/initial states
    Smax = 500.0   
    initial_S = 100.0
    initial_G = 50.0

    print(f"  > Starting SCE-UA for {target_basin} ({repetitions} repetitions)...")
    
    # 1. Initialize the SPOTPY Model
    model = TwoStoreModel_SCE(P_data, PET_data, Q_obs, Smax, initial_S, initial_G)
    
    # 2. Initialize the SCE-UA Sampler
    temp_dir = os.path.join(PROJECT_ROOT, 'SCE_UA_Results')
    os.makedirs(temp_dir, exist_ok=True)
    # db_name will save the run results, useful for analysis later
    db_name = os.path.join(temp_dir, f'sceua_cal_{target_basin}')
    
    sampler = spotpy.algorithms.sceua(model, dbname=db_name, dbformat='csv', save_sim=False)
    
    # 3. Run the Sampling
    sampler.sample(repetitions=repetitions)
    
    # 4. Analyze Results
    res = sampler.getdata() 
    
    if res is not None and len(res) > 0:
        # Get the index with the minimum objective function (-KGE) -> maximum KGE
        best_index, best_objf_raw = spotpy.analyser.get_minlikeindex(res)
        
        # Safely extract the scalar value of the minimal objective function
        best_objf = best_objf_raw.item() if hasattr(best_objf_raw, 'item') else best_objf_raw

        # The true KGE is the negative of the objective function (KGE = -(-KGE))
        best_kge = -best_objf 
        
        if best_kge > 0.0: 
            p_opt = res[best_index]
            
            # Extract parameters
            Kperc_opt = p_opt['parKperc'].item()
            Kb_opt = p_opt['parKb'].item()
            Ke_opt = p_opt['parKe'].item()
            Cqq_opt = p_opt['parCqq'].item()
            Bias_opt = p_opt['parBias'].item()

            # Recalculate NSE (and KGE for verification) with the optimal params
            params_final = ModelParams(Smax=Smax, Kperc=Kperc_opt, Kb=Kb_opt, Ke=Ke_opt, Cqq=Cqq_opt)
            initial_state_final = (initial_S, initial_G, Bias_opt)
            Q_sim_cut, Q_obs_cut = run_forward_model(P_data, PET_data, Q_obs, initial_state_final, params_final)
            
            final_nse = calculate_nse(Q_obs_cut, Q_sim_cut)

            return {
                'Kperc': Kperc_opt, 'Kb': Kb_opt, 'Ke': Ke_opt, 
                'Cqq': Cqq_opt, 'bias': Bias_opt,
                'NSE': final_nse, 'KGE': best_kge # Use the KGE value derived from the optimizer
            }
        else:
            print(f"  > FAILED: KGE <= 0.0. Best KGE: {best_kge:.4f}")
            return {}
    else:
        print("  > FAILED: SCE-UA sampler returned no data.")
        return {}


# =====================================================================
# 4. MULTI-BASIN CALIBRATION MANAGER
# =====================================================================

def calibrate_all_basins(P_df, PET_df, Q_usgs_df, basin_list: List[str], repetitions: int = 5000):
    """Loops through a list of basins and performs SCE-UA calibration on each."""
    all_calibrated_results = {}

    for TARGET_BASIN in basin_list:
        if TARGET_BASIN not in Q_usgs_df.columns:
            print(f"--- Warning: Basin {TARGET_BASIN} not found in streamflow data. Skipping. ---")
            continue

        print(f"\n=======================================================")
        print(f"  STARTING CALIBRATION FOR BASIN: {TARGET_BASIN}")
        print(f"=======================================================")

        calibrated_params = calibrate_initial_parameters_sce(
            P_df, PET_df, Q_usgs_df, target_basin=TARGET_BASIN, repetitions=repetitions
        )

        if calibrated_params:
            all_calibrated_results[TARGET_BASIN] = calibrated_params
            print(f"--- SUCCESS: {TARGET_BASIN} | KGE={calibrated_params['KGE']:.4f} ---")
        else:
            print(f"--- FAILED: {TARGET_BASIN} CALIBRATION FAILED ---")

    return all_calibrated_results


# =====================================================================
# 5. EXECUTION BLOCK (Main)
# =====================================================================

if __name__ == '__main__':
    try:
        # Load the data - assuming this function extracts the 10 target basins
        P_df, PET_df, _, _, _, _, _, Q_usgs_df = load_and_prepare_data()
    except Exception as e:
        print(f"FATAL: Data loading failed: {e}")
        sys.exit(1)

    if P_df is None:
        print("FATAL: Data loading returned None. Check data paths.")
        sys.exit(1)

    # Define the full list of basins you want to calibrate
    FULL_BASIN_LIST = ['06452000', '13340000', '06447000', '06360500', 
                       '06354000', '05057000', '07301500', '06191500', 
                       '02315500', '06784000'] 
    
    # Filter the list to only include basins present in the loaded streamflow data
    TARGET_BASINS = [b for b in FULL_BASIN_LIST if b in Q_usgs_df.columns]

    if not TARGET_BASINS:
        print("FATAL: None of the specified basins were found in the loaded streamflow data.")
        sys.exit(1)
        
    print(f"\nFound {len(TARGET_BASINS)} basins for calibration.")

    # Run the calibration loop
    final_results = calibrate_all_basins(
        P_df, PET_df, Q_usgs_df, TARGET_BASINS, repetitions=5000
    )
    
    print("\n\n" + "="*80)
    print("                     SUMMARY OF ALL CALIBRATION RESULTS")
    print("="*80)
    
    results_list = []
    
    for basin, params in final_results.items():
        results_list.append({
            'Basin': basin,
            'Kperc': params['Kperc'],
            'Kb': params['Kb'],
            'Ke': params['Ke'],
            'Cqq': params['Cqq'],
            'Bias': params['bias'],
            'NSE': params['NSE'],
            'KGE': params['KGE']
        })

    if results_list:
        # Display results as a DataFrame (cleaner output)
        results_df = pd.DataFrame(results_list)
        # Sort by KGE to see the best-performing models first
        results_df = results_df.sort_values(by='KGE', ascending=False).reset_index(drop=True)
        
        # Print the final summary table
        print(results_df.to_markdown(index=True, floatfmt=".4f"))

        print("\n\n" + "="*80)
        print("✅ NEXT STEPS:")
        print("1. Copy the Kperc, Kb, Ke, Cqq, and Bias values for the basin you wish to use.")
        print("2. Update the corresponding parameters in your EnKF script (src/enkf.py) to ensure stability.")
        print("="*80)

    else:
        print("No basins were successfully calibrated (KGE > 0.0). Review model setup.")