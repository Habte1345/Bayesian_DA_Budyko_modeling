import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, Any

# Get the path to the project root (Bayesian_DA_Budyko_modeling)
# Assuming this file is at PROJECT_ROOT/src/param_manager.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'SCE_cal_params')


def get_calibrated_params_for_basin(basin_id: str) -> Dict[str, float]:
    """
    Reads the SCE-UA calibration CSV file for a given basin and extracts 
    the parameter set that corresponds to the minimal objective function ('like1').
    
    Args:
        basin_id: The 8-digit USGS basin ID (e.g., '06452000').
        
    Returns:
        A dictionary of the best parameters (Kperc, Kb, Ke, Cqq, Bias) as floats.
    """
    file_name = f"sceua_cal_{basin_id}.csv"
    file_path = os.path.join(RESULTS_DIR, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Calibration file not found for basin {basin_id}. Expected at: {file_path}"
        )

    # 1. Load the calibration results
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

    # 2. Find the row with the minimal objective function ('like1' column)
    # The objective function is -KGE, so minimizing it maximizes KGE.
    min_like_index = df['like1'].idxmin()
    best_run = df.loc[min_like_index]

    # 3. Extract the optimal parameters
    # The 'par' prefix is consistent with SPOTPY output.
    best_params = {
        'Kperc': best_run['parKperc'],
        'Kb': best_run['parKb'],
        'Ke': best_run['parKe'],
        'Cqq': best_run['parCqq'],
        'Bias': best_run['parBias'],
        'KGE': -best_run['like1'] # For reference, KGE is the negative of the objective value
    }
    
    # Optional: Filter out parameters that resulted in NaN for robustness
    if np.any(pd.isnull(list(best_params.values())[:5])):
        print(f"Warning: Optimal parameters for {basin_id} contain NaN. Returning an empty set.")
        return {}

    return best_params

if __name__ == '__main__':
    # Example usage for testing
    TEST_BASIN = '13340000' 
    print(f"Testing parameter manager for basin: {TEST_BASIN}")
    try:
        params = get_calibrated_params_for_basin(TEST_BASIN)
        if params:
            print("\nOptimal Parameters:")
            for k, v in params.items():
                print(f"- {k}: {v:.4f}")
        else:
            print("Failed to retrieve parameters.")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")