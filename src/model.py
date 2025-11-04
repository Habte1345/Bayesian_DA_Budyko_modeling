# src/model.py 

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

    # 2. Quick flow (Saturation Excess)
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
    Perc_t = min(Perc_t_calc, S_curr) # Crucial stability check!
    
    S_curr -= Perc_t
    S_curr = max(S_curr, 0.0) # S_next must be non-negative
    
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