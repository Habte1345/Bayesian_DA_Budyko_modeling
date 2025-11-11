# # src/model.py

# from dataclasses import dataclass
# import numpy as np

# # =====================================================================
# # 3. TWO-STORE HYDROLOGIC MODEL PARAMETERS
# # =====================================================================

# @dataclass
# class ModelParams:
#     """Dataclass for storing the hydrologic model parameters."""
#     Smax: float = 1000.0    # Max storage in the soil store (a realistic default)
#     Kperc: float = 0.2      # Percolation rate
#     Kb: float = 0.1         # Baseflow recession constant
#     Ke: float = 0.6         # Evaporation coefficient
#     Cqq: float = 5.0        # Quickflow coefficient (for saturation excess)

# def two_store_model_step(S_curr, G_curr, P_t, PET_t, params: ModelParams, 
#                          ET_override=None) -> tuple:
#     """
#     Performs one time step of the two-store hydrologic model, ensuring fluxes 
#     do not exceed available storage to maintain numerical stability.
    
#     Returns:
#         S_next (float): Updated soil store after this time step.
#         G_next (float): Updated groundwater store after this time step.
#         ET_t (float): Actual evapotranspiration for this time step.
#         Q_t (float): Total streamflow for this time step (quickflow + baseflow).
#         Qs_t (float): Quickflow (saturation-excess runoff) for this time step.
#         Qb_t (float): Baseflow for this time step.
#         Perc_t (float): Percolation from soil to groundwater in this time step.
#     """
#     # Ensure non-negative precipitation input
#     P_t = max(P_t, 0.0)
    
#     # 1. Update S (soil storage) with precipitation
#     S_curr += P_t

#     # 2. Quickflow (Saturation Excess runoff)
#     overflow = max(S_curr - params.Smax, 0.0)   # Water beyond soil storage capacity
#     Qs_t_calc = params.Cqq * overflow           # Potential quickflow from overflow
#     Qs_t = min(Qs_t_calc, S_curr)               # Actual quickflow cannot exceed current soil water
#     S_curr -= Qs_t
#     S_curr = max(S_curr, 0.0)                   # Ensure non-negative soil storage
    
#     # 3. Evapotranspiration (ET)
#     # Use ET_override (e.g., Budyko or data-assimilated ET) if provided; otherwise use Ke * PET.
#     ET_pot = ET_override if ET_override is not None else params.Ke * PET_t
#     ET_t_calc = max(ET_pot, 0.0)                # Potential ET cannot be negative
#     ET_t = min(ET_t_calc, S_curr)               # Actual ET is limited by available soil water
#     S_curr -= ET_t
#     S_curr = max(S_curr, 0.0)                   # Ensure non-negative soil storage after ET
    
#     # 4. Percolation from soil to groundwater
#     Perc_t_calc = params.Kperc * S_curr
#     Perc_t = min(Perc_t_calc, S_curr)           # Percolation cannot exceed available soil water
#     S_curr -= Perc_t
#     S_curr = max(S_curr, 0.0)                   # Ensure soil storage doesn't go negative
    
#     # 5. Groundwater update and Baseflow generation
#     G_curr += Perc_t                            # Add percolated water to groundwater store
#     Qb_t_calc = params.Kb * G_curr
#     Qb_t = min(Qb_t_calc, G_curr)               # Baseflow cannot exceed groundwater storage
#     G_curr -= Qb_t
#     G_curr = max(G_curr, 0.0)                   # Ensure non-negative groundwater storage
    
#     # 6. Final states and total streamflow
#     Q_t = Qs_t + Qb_t                           # Total streamflow (quickflow + baseflow)
#     S_next = S_curr
#     G_next = G_curr
    
#     return S_next, G_next, ET_t, Q_t, Qs_t, Qb_t, Perc_t



# from dataclasses import dataclass
# import numpy as np

# @dataclass
# class ModelParams:
#     """Dataclass for storing the hydrologic model parameters."""
#     Smax: float = 1000.0    # Max storage in the soil store
#     Kperc: float = 0.2      # Percolation rate (fraction per time step)
#     Kb: float = 0.1         # Baseflow recession constant
#     Ke: float = 0.6         # Evaporation coefficient (fraction of PET)
#     Cqq: float = 5.0        # Quickflow coefficient for saturation-excess runoff

# def two_store_model_step(S_curr, G_curr, P_t, PET_t, params: ModelParams, ET_override=None):
#     """
#     Performs one time step of the two-store hydrologic model, ensuring fluxes 
#     do not exceed available storage (for numerical stability and physical realism).
#     Returns: S_next, G_next, ET_t, Q_t, Qs_t, Qb_t, Perc_t
#        S_next  -> updated soil storage after this time step
#        G_next  -> updated groundwater storage after this time step
#        ET_t    -> actual evapotranspiration for this time step
#        Q_t     -> total streamflow (quickflow + baseflow) for this time step
#        Qs_t    -> quickflow (direct runoff) for this time step
#        Qb_t    -> baseflow for this time step
#        Perc_t  -> percolation amount from soil to groundwater this time step
#     """
#     # Ensure non-negative precipitation input
#     P_t = max(P_t, 0.0)
    
#     # 1. Soil storage update with precipitation
#     S_curr += P_t

#     # 2. Quickflow from saturation excess
#     overflow = max(S_curr - params.Smax, 0.0)        # water beyond soil capacity
#     Qs_t_calc = params.Cqq * overflow                # potential quickflow from overflow
#     Qs_t = min(Qs_t_calc, S_curr)                    # quickflow cannot exceed available water
#     S_curr -= Qs_t
#     S_curr = max(S_curr, 0.0)                        # ensure non-negative soil storage

#     # 3. Evapotranspiration from soil store
#     ET_pot = ET_override if ET_override is not None else params.Ke * PET_t
#     ET_pot = max(ET_pot, 0.0)                        # no negative ET demand
#     ET_t = min(ET_pot, S_curr)                       # ET limited by available soil water
#     S_curr -= ET_t
#     S_curr = max(S_curr, 0.0)                        # ensure non-negative soil storage

#     # 4. Percolation from soil to groundwater
#     Perc_t_calc = params.Kperc * S_curr
#     Perc_t = min(Perc_t_calc, S_curr)                # percolation limited by available soil water
#     S_curr -= Perc_t
#     S_curr = max(S_curr, 0.0)                        # ensure non-negative soil storage

#     # 5. Groundwater update and baseflow
#     G_curr += Perc_t                                # add percolation to groundwater store
#     Qb_t_calc = params.Kb * G_curr
#     Qb_t = min(Qb_t_calc, G_curr)                   # baseflow cannot exceed available groundwater
#     G_curr -= Qb_t
#     G_curr = max(G_curr, 0.0)                       # ensure non-negative groundwater storage

#     # 6. Final states and total streamflow (bias removed)
#     Q_t = Qs_t + Qb_t                                # total streamflow for this step (quick + base)
#     S_next = S_curr
#     G_next = G_curr

#     return S_next, G_next, ET_t, Q_t, Qs_t, Qb_t, Perc_t


from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True) # Added frozen=True for immutability of parameters
class ModelParams:
    """Dataclass for storing the hydrologic model parameters.
    Parameters are now required, forcing the use of calibrated values.
    """
    Smax: float          # Max storage in the soil store (mm)
    Kperc: float         # Percolation rate (fraction per time step)
    Kb: float            # Baseflow recession constant
    Ke: float            # Evaporation coefficient (fraction of PET)
    Cqq: float           # Quickflow coefficient for saturation-excess runoff


def two_store_model_step(S_curr, G_curr, P_t, PET_t, params: ModelParams, ET_override=None):
    """
    Performs one time step of the two-store hydrologic model, ensuring fluxes 
    do not exceed available storage (for numerical stability and physical realism).
    Returns: S_next, G_next, ET_t, Q_t, Qs_t, Qb_t, Perc_t
        S_next  -> updated soil storage after this time step
        G_next  -> updated groundwater storage after this time step
        ET_t    -> actual evapotranspiration for this time step
        Q_t     -> total streamflow (quickflow + baseflow) for this time step
        Qs_t    -> quickflow (direct runoff) for this time step
        Qb_t    -> baseflow for this time step
        Perc_t  -> percolation amount from soil to groundwater this time step
    """
    # Ensure non-negative precipitation input
    P_t = max(P_t, 0.0)
    
    # 1. Soil storage update with precipitation
    S_curr += P_t

    # 2. Quickflow from saturation excess
    overflow = max(S_curr - params.Smax, 0.0)      # water beyond soil capacity
    Qs_t_calc = params.Cqq * overflow              # potential quickflow from overflow
    Qs_t = min(Qs_t_calc, S_curr)                  # quickflow cannot exceed available water
    S_curr -= Qs_t
    S_curr = max(S_curr, 0.0)                      # ensure non-negative soil storage

    # 3. Evapotranspiration from soil store
    ET_pot = ET_override if ET_override is not None else params.Ke * PET_t
    ET_pot = max(ET_pot, 0.0)                      # no negative ET demand
    ET_t = min(ET_pot, S_curr)                     # ET limited by available soil water
    S_curr -= ET_t
    S_curr = max(S_curr, 0.0)                      # ensure non-negative soil storage

    # 4. Percolation from soil to groundwater
    Perc_t_calc = params.Kperc * S_curr
    Perc_t = min(Perc_t_calc, S_curr)              # percolation limited by available soil water
    S_curr -= Perc_t
    S_curr = max(S_curr, 0.0)                      # ensure non-negative soil storage

    # 5. Groundwater update and baseflow
    G_curr += Perc_t                               # add percolation to groundwater store
    Qb_t_calc = params.Kb * G_curr
    Qb_t = min(Qb_t_calc, G_curr)                  # baseflow cannot exceed available groundwater
    G_curr -= Qb_t
    G_curr = max(G_curr, 0.0)                      # ensure non-negative groundwater storage

    # 6. Final states and total streamflow (bias removed)
    Q_t = Qs_t + Qb_t                              # total streamflow for this step (quick + base)
    S_next = S_curr
    G_next = G_curr

    return S_next, G_next, ET_t, Q_t, Qs_t, Qb_t, Perc_t