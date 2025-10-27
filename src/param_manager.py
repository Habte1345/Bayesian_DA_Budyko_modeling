# # In src/param_manager.py:

# import pandas as pd
# import numpy as np
# import sys
# import os
# import json 
# from typing import Dict, Any

# # Get the path to the project root (Bayesian_DA_Budyko_modeling)
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# RESULTS_DIR = os.path.join(PROJECT_ROOT, 'SCE_cal_params')

# # Define the file name for the compiled results
# JSON_FILE_NAME = 'final_calibrated_params.json'

# def get_calibrated_params_for_basin(basin_id: str) -> Dict[str, float]:
#     """
#     Reads the *final compiled JSON* file created by calibration.py and extracts 
#     the parameter set for a given basin, including Smax.
    
#     Args:
#         basin_id: The USGS basin ID (can be str or int, will be standardized to '0xxxxxxx').
        
#     Returns:
#         A dictionary of the best parameters.
#     """
#     file_path = os.path.join(RESULTS_DIR, JSON_FILE_NAME)

#     if not os.path.exists(file_path):
#         raise FileNotFoundError(
#             f"Compiled calibration file not found. Please run 'calibration.py' first. Expected at: {file_path}"
#         )

#     # 1. Load the entire JSON file
#     try:
#         with open(file_path, 'r') as f:
#             all_params = json.load(f)
#     except Exception as e:
#         print(f"Error reading {file_path}: {e}")
#         return {}

#     # 🚨 CRITICAL FIX: Standardize the input basin_id for dictionary lookup
#     cleaned_basin_id = str(basin_id).strip().zfill(8)
    
#     # 2. Extract the optimal parameters dictionary for the target basin
#     best_params = all_params.get(cleaned_basin_id, {}) # Use the cleaned ID for lookup
    
#     if not best_params:
#         # This will now print the cleaned ID that failed the lookup
#         print(f"Warning: Basin {cleaned_basin_id} not found in the compiled calibration results.")
#         return {}
        
#     # 3. Ensure essential parameters are present (specifically Smax)
#     if 'Smax' not in best_params:
#         print(f"FATAL: The compiled parameters for {cleaned_basin_id} are missing the 'Smax' key.")
#         return {}

#     return best_params

# if __name__ == '__main__':
#     # Example usage for testing
#     TEST_BASIN = '13340000' 
#     print(f"Testing parameter manager for basin: {TEST_BASIN}")
#     try:
#         params = get_calibrated_params_for_basin(TEST_BASIN) 
#         if params:
#             print("\nOptimal Parameters (from JSON):")
#             for k in ['Kperc', 'Smax', 'KGE']: 
#                 if k in params:
#                     print(f"- {k}: {params[k]:.4f}")
#             print(f"All keys successfully loaded: {list(params.keys())}")
#         else:
#             print("Failed to retrieve parameters.")
#     except FileNotFoundError as e:
#         print(e)
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

# In src/param_manager.py:

import pandas as pd
import numpy as np
import sys
import os
import json 
from typing import Dict, Any, Optional

# Get the path to the project root (Bayesian_DA_Budyko_modeling)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'SCE_cal_params')

# Define the file name for the compiled results
JSON_FILE_NAME = 'final_calibrated_params.json'
COMPILED_FILE_PATH = os.path.join(RESULTS_DIR, JSON_FILE_NAME)

# --- 🚀 NEW FUNCTION FOR PARALLEL OPTIMIZATION ---

def load_all_calibrated_params() -> Dict[str, Dict]:
    """
    Loads ALL calibrated parameters from the single compiled JSON file.
    
    This is called once by the main process before parallel simulation starts.
    """
    if not os.path.exists(COMPILED_FILE_PATH):
        print(f"FATAL: Compiled calibration file not found. Expected at: {COMPILED_FILE_PATH}")
        return {}

    try:
        with open(COMPILED_FILE_PATH, 'r') as f:
            all_params = json.load(f)
        
        # Ensure all keys in the loaded JSON are standardized to 8-digit strings
        # This prevents lookup errors later.
        standardized_params = {}
        for basin_key, params in all_params.items():
            cleaned_basin_id = str(basin_key).strip().zfill(8)
            standardized_params[cleaned_basin_id] = params
            
        return standardized_params
    except Exception as e:
        print(f"FATAL: Error reading {COMPILED_FILE_PATH} for full load: {e}")
        return {}

# --- MODIFIED EXISTING FUNCTION (Simplified for internal consistency) ---

# We only keep this function for backward compatibility and as a fallback.
# In the parallel code, load_all_calibrated_params() will be called first.
def get_calibrated_params_for_basin(basin_id: str) -> Dict[str, float]:
    """
    Reads the *final compiled JSON* file and extracts the parameter set 
    for a given basin. (Kept for external/legacy calls).
    """
    
    # 🚨 Optimization: Call the full loader to use its caching/loading logic.
    all_params = load_all_calibrated_params()

    if not all_params:
        return {}

    cleaned_basin_id = str(basin_id).strip().zfill(8)
    
    best_params = all_params.get(cleaned_basin_id, {})
    
    if not best_params:
        print(f"Warning: Basin {cleaned_basin_id} not found in the compiled calibration results.")
        return {}
        
    if 'Smax' not in best_params:
        print(f"FATAL: The compiled parameters for {cleaned_basin_id} are missing the 'Smax' key.")
        return {}

    return best_params

if __name__ == '__main__':
    # Example usage for testing
    TEST_BASIN = '13340000' 
    print(f"Testing parameter manager for basin: {TEST_BASIN}")
    
    try:
        # Test the new loader
        all_p = load_all_calibrated_params()
        if all_p:
            print(f"Total basins loaded via load_all_calibrated_params: {len(all_p)}")
        
        # Test the basin getter
        params = get_calibrated_params_for_basin(TEST_BASIN) 
        if params:
            print("\nOptimal Parameters (from JSON):")
            for k in ['Kperc', 'Smax', 'KGE']: 
                if k in params:
                    print(f"- {k}: {params[k]:.4f}")
            print(f"All keys successfully loaded for {TEST_BASIN}: {list(params.keys())}")
        else:
            print(f"Failed to retrieve parameters for {TEST_BASIN}.")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")