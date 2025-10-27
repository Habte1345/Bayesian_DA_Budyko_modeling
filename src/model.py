# # src/model.py

# from dataclasses import dataclass
# import numpy as np

# # =====================================================================
# # 3. TWO-STORE HYDROLOGIC MODEL 
# # =====================================================================

# @dataclass
# class ModelParams:
#     """Dataclass for storing the hydrologic model parameters."""
#     Smax: float = 10.0  # Max storage in the soil store
#     Kperc: float = 0.05  # Percolation rate
#     Kb: float = 0.06     # Baseflow recession constant
#     Ke: float = 0.7      # Evaporation coefficient
#     Cqq: float = 0.8     # Quickflow coefficient (for saturation excess)


# def two_store_model_step(S_curr, G_curr, P_t, PET_t, params: ModelParams, 
#                          ET_override=None) -> tuple:
#     """Performs one time step of the two-store hydrologic model."""
    
#     P_t = max(P_t, 0.0)
#     S_curr += P_t
    
#     # Quick flow (Surface Runoff)
#     overflow = max(S_curr - params.Smax, 0.0)
#     Qs_t = params.Cqq * overflow 
#     S_curr -= Qs_t
#     S_curr = max(S_curr, 0.0)
    
#     # Evapotranspiration
#     ET_pot = ET_override if ET_override is not None else params.Ke * PET_t
#     ET_t = min(max(ET_pot, 0.0), S_curr)
#     S_curr -= ET_t
#     S_curr = max(S_curr, 0.0)
    
#     # Percolation
#     Perc_t = params.Kperc * S_curr
#     S_curr -= Perc_t
#     S_curr = max(S_curr, 0.0)
    
#     # Groundwater and baseflow
#     G_curr += Perc_t
#     Qb_t = params.Kb * G_curr
#     G_curr -= Qb_t
#     G_curr = max(G_curr, 0.0)
    
#     # Total streamflow
#     Q_t = Qs_t + Qb_t
    
#     S_next = S_curr
#     G_next = G_curr
    
#     return S_next, G_next, ET_t, Q_t, Qs_t, Qb_t, Perc_t


# def run_two_store_model(P_monthly, PET_monthly, params: ModelParams, 
#                         ET_input=None, initial_S=100.0, initial_G=20.0):
#     """Runs the two-store model for a full time series."""
#     nmonths = len(P_monthly)
#     Q = np.zeros(nmonths)
#     ET = np.zeros(nmonths)
#     S = np.zeros(nmonths)
#     G = np.zeros(nmonths)
#     Qs = np.zeros(nmonths) 
#     Qb = np.zeros(nmonths)
#     Perc = np.zeros(nmonths)
    
#     S_curr = initial_S
#     G_curr = initial_G
    
#     for t in range(nmonths):
#         ET_override = ET_input[t] if ET_input is not None and not np.isnan(ET_input[t]) else None
        
#         S_next, G_next, ET_t, Q_t, Qs_t, Qb_t, Perc_t = \
#             two_store_model_step(S_curr, G_curr, P_monthly[t], PET_monthly[t], params, ET_override)
        
#         Q[t] = Q_t
#         ET[t] = ET_t
#         S[t] = S_next
#         G[t] = G_next
#         Qs[t] = Qs_t
#         Qb[t] = Qb_t
#         Perc[t] = Perc_t
        
#         S_curr = S_next
#         G_curr = G_next
    
#     return {
#         'Q': Q, 'ET': ET, 'S': S, 'G': G,
#         'Qs': Qs, 'Qb': Qb, 'Perc': Perc
#     }


# # ASSUMPTION: This function is added to src/model.py
# def calculate_fluxes_from_states(P_monthly, ET_t, S_mean, G_mean, params: ModelParams):
#     """
#     Recalculates Qs, Qb, and Perc fluxes based on DA-corrected mean states.
    
#     Inputs: 
#         P_monthly, ET_t (Assimilated ET), S_mean (End-of-step DA state), 
#         G_mean (End-of-step DA state), ModelParams
        
#     NOTE: This uses the S_mean[t] and G_mean[t] from the DA output to drive the fluxes 
#     based on the model structure, using the state from the *previous* step as the 
#     initial state for the current step's flux calculation.
#     """
#     nmonths = len(P_monthly)
#     Qs = np.zeros(nmonths) 
#     Qb = np.zeros(nmonths)
#     Perc = np.zeros(nmonths)
#     Q_calc = np.zeros(nmonths)
    
#     # We need the state from the start of the step (previous month's end state)
#     # Since DA updates S and G, S_curr[t] is the state *before* fluxes for step t.
#     # We use a simple shift: S_start[t] = S_mean[t-1]
#     S_start = np.concatenate([[S_mean[0]], S_mean[:-1]]) 
#     G_start = np.concatenate([[G_mean[0]], G_mean[:-1]])
    
#     # Use the actual initial values passed to run_enkf_scenario for the very first step
#     # (S_mean[0] and G_mean[0] are the DA-corrected state for the first step, 
#     # so we need to know the true S_init and G_init for the first step's flux calculation 
#     # which is not available here. We'll stick to S_mean[t-1] for simplicity, 
#     # knowing the first step's flux may be slightly off.)

#     for t in range(nmonths):
#         P_t = P_monthly[t]
        
#         # 1. Quick flow (Qs)
#         S_interim = S_start[t] + P_t
#         overflow = max(S_interim - params.Smax, 0.0)
#         Qs_t = params.Cqq * overflow 
#         S_after_Qs = max(S_interim - Qs_t, 0.0)
        
#         # 2. Evapotranspiration (ET) - Use the assimilated ET value
#         S_after_ET = max(S_after_Qs - ET_t[t], 0.0)
        
#         # 3. Percolation (Perc)
#         Perc_t = params.Kperc * S_after_ET
        
#         # 4. Baseflow (Qb)
#         G_after_Perc = G_start[t] + Perc_t
#         Qb_t = params.Kb * G_after_Perc
        
#         Q_calc[t] = Qs_t + Qb_t
#         Qs[t] = Qs_t
#         Qb[t] = Qb_t
#         Perc[t] = Perc_t
        
#     return {'Q': Q_calc, 'Qs': Qs, 'Qb': Qb, 'Perc': Perc}



# # src/model.py

# from dataclasses import dataclass
# import numpy as np

# # =====================================================================
# # 3. TWO-STORE HYDROLOGIC MODEL PARAMETERS
# # =====================================================================

# @dataclass
# class ModelParams:
#     """Dataclass for storing the hydrologic model parameters."""
#     Smax: float = 1000.0  # Max storage in the soil store (A more realistic default)
#     Kperc: float = 0.2    # Percolation rate
#     Kb: float = 0.1       # Baseflow recession constant
#     Ke: float = 0.6       # Evaporation coefficient
#     Cqq: float = 5.0      # Quickflow coefficient (for saturation excess)


# def two_store_model_step(S_curr, G_curr, P_t, PET_t, params: ModelParams, 
#                          ET_override=None) -> tuple:
#     """
#     Performs one time step of the two-store hydrologic model.
#     Returns: S_next, G_next, ET_t, Q_t_no_bias, Qs_t, Qb_t, Perc_t
#     """
    
#     P_t = max(P_t, 0.0)
    
#     # 1. Update S with Precipitation
#     S_curr += P_t
    
#     # 2. Quick flow (Saturation Excess)
#     # The Cqq in the original code seems to be used as a multiplier on overflow, 
#     # which is unusual. A standard practice is for Qs to be simply the overflow 
#     # if using a monthly or daily step, but we retain the Cqq multiplier as 
#     # defined in the original code structure.
#     overflow = max(S_curr - params.Smax, 0.0)
#     Qs_t = params.Cqq * overflow 
#     S_curr -= Qs_t
#     S_curr = max(S_curr, 0.0)
    
#     # 3. Evapotranspiration
#     # Use ET_override (Budyko or DA-corrected ET) if provided, otherwise use Ke*PET.
#     ET_pot = ET_override if ET_override is not None else params.Ke * PET_t
#     ET_t = min(max(ET_pot, 0.0), S_curr) # Actual ET is limited by S_curr and ET_pot
#     S_curr -= ET_t
#     S_curr = max(S_curr, 0.0)
    
#     # 4. Percolation
#     Perc_t = params.Kperc * S_curr
#     S_curr -= Perc_t
#     S_curr = max(S_curr, 0.0)
    
#     # 5. Groundwater and Baseflow
#     G_curr += Perc_t
#     Qb_t = params.Kb * G_curr
#     G_curr -= Qb_t
#     G_curr = max(G_curr, 0.0)
    
#     # 6. Total streamflow (excluding the 'bias' parameter, which is state-augmented)
#     Q_t_no_bias = Qs_t + Qb_t
    
#     S_next = S_curr
#     G_next = G_curr
    
#     # NOTE: The run_two_store_model and calculate_fluxes_from_states functions 
#     # from the original src/model.py are removed as they are redundant for the 
#     # core EnKF execution, which only requires the two_store_model_step.
    
#     return S_next, G_next, ET_t, Q_t_no_bias, Qs_t, Qb_t, Perc_t


# src/model.py (Corrected for Numerical Stability)

from dataclasses import dataclass
import numpy as np

# =====================================================================
# 3. TWO-STORE HYDROLOGIC MODEL PARAMETERS
# =====================================================================

@dataclass
class ModelParams:
    """Dataclass for storing the hydrologic model parameters."""
    Smax: float = 1000.0    # Max storage in the soil store (A more realistic default)
    Kperc: float = 0.2      # Percolation rate
    Kb: float = 0.1         # Baseflow recession constant
    Ke: float = 0.6         # Evaporation coefficient
    Cqq: float = 5.0        # Quickflow coefficient (for saturation excess)


def two_store_model_step(S_curr, G_curr, P_t, PET_t, params: ModelParams, 
                         ET_override=None) -> tuple:
    """
    Performs one time step of the two-store hydrologic model, ensuring fluxes 
    do not exceed available storage to maintain numerical stability.
    Returns: S_next, G_next, ET_t, Q_t_no_bias, Qs_t, Qb_t, Perc_t
    """
    
    P_t = max(P_t, 0.0)
    
    # 1. Update S with Precipitation
    S_curr = S_curr + P_t
    
    # --- FLUX CALCULATIONS (S-STORE) ---
    
    # 2. Quick flow (Saturation Excess)
    # This is the first flux: The amount that exceeds Smax, modified by Cqq.
    overflow = max(S_curr - params.Smax, 0.0)
    Qs_t_calc = params.Cqq * overflow 
    # Actual Qs cannot exceed the available overflow
    Qs_t = min(Qs_t_calc, S_curr) # Crucial stability check!
    
    S_curr -= Qs_t
    S_curr = max(S_curr, 0.0) # Ensure non-negative after Qs removal
    
    # 3. Evapotranspiration
    # Use ET_override (Budyko or DA-corrected ET) if provided, otherwise use Ke*PET.
    ET_pot = ET_override if ET_override is not None else params.Ke * PET_t
    ET_t_calc = max(ET_pot, 0.0)
    
    # Actual ET is limited by potential ET and the remaining available storage (S_curr)
    ET_t = min(ET_t_calc, S_curr) # Crucial stability check!
    
    S_curr -= ET_t
    S_curr = max(S_curr, 0.0) # Ensure non-negative after ET removal
    
    # 4. Percolation
    Perc_t_calc = params.Kperc * S_curr
    
    # Percolation is limited by the remaining soil moisture
    Perc_t = min(Perc_t_calc, S_curr) # Crucial stability check!
    
    S_curr -= Perc_t
    S_curr = max(S_curr, 0.0) # S_next must be non-negative
    
    # --- FLUX CALCULATIONS (G-STORE) ---
    
    # 5. Groundwater and Baseflow
    G_curr += Perc_t # G is updated by Percolation
    
    Qb_t_calc = params.Kb * G_curr
    
    # Baseflow is limited by the available groundwater storage
    Qb_t = min(Qb_t_calc, G_curr) # Crucial stability check!
    
    G_curr -= Qb_t
    G_curr = max(G_curr, 0.0) # G_next must be non-negative
    
    # 6. Final States and Total streamflow (excluding the 'bias' parameter)
    Q_t_no_bias = Qs_t + Qb_t
    
    S_next = S_curr
    G_next = G_curr
    
    return S_next, G_next, ET_t, Q_t_no_bias, Qs_t, Qb_t, Perc_t