# src/model.py

from dataclasses import dataclass
import numpy as np

# =====================================================================
# 3. TWO-STORE HYDROLOGIC MODEL 
# =====================================================================

@dataclass
class ModelParams:
    """Dataclass for storing the hydrologic model parameters."""
    Smax: float = 10.0  # Max storage in the soil store
    Kperc: float = 0.05  # Percolation rate
    Kb: float = 0.06     # Baseflow recession constant
    Ke: float = 0.7      # Evaporation coefficient
    Cqq: float = 0.8     # Quickflow coefficient (for saturation excess)


def two_store_model_step(S_curr, G_curr, P_t, PET_t, params: ModelParams, 
                         ET_override=None) -> tuple:
    """Performs one time step of the two-store hydrologic model."""
    
    P_t = max(P_t, 0.0)
    S_curr += P_t
    
    # Quick flow (Surface Runoff)
    overflow = max(S_curr - params.Smax, 0.0)
    Qs_t = params.Cqq * overflow 
    S_curr -= Qs_t
    S_curr = max(S_curr, 0.0)
    
    # Evapotranspiration
    ET_pot = ET_override if ET_override is not None else params.Ke * PET_t
    ET_t = min(max(ET_pot, 0.0), S_curr)
    S_curr -= ET_t
    S_curr = max(S_curr, 0.0)
    
    # Percolation
    Perc_t = params.Kperc * S_curr
    S_curr -= Perc_t
    S_curr = max(S_curr, 0.0)
    
    # Groundwater and baseflow
    G_curr += Perc_t
    Qb_t = params.Kb * G_curr
    G_curr -= Qb_t
    G_curr = max(G_curr, 0.0)
    
    # Total streamflow
    Q_t = Qs_t + Qb_t
    
    S_next = S_curr
    G_next = G_curr
    
    return S_next, G_next, ET_t, Q_t, Qs_t, Qb_t, Perc_t


def run_two_store_model(P_monthly, PET_monthly, params: ModelParams, 
                        ET_input=None, initial_S=100.0, initial_G=20.0):
    """Runs the two-store model for a full time series."""
    nmonths = len(P_monthly)
    Q = np.zeros(nmonths)
    ET = np.zeros(nmonths)
    S = np.zeros(nmonths)
    G = np.zeros(nmonths)
    Qs = np.zeros(nmonths)
    Qb = np.zeros(nmonths)
    Perc = np.zeros(nmonths)
    
    S_curr = initial_S
    G_curr = initial_G
    
    for t in range(nmonths):
        ET_override = ET_input[t] if ET_input is not None and not np.isnan(ET_input[t]) else None
        
        S_next, G_next, ET_t, Q_t, Qs_t, Qb_t, Perc_t = \
            two_store_model_step(S_curr, G_curr, P_monthly[t], PET_monthly[t], params, ET_override)
        
        Q[t] = Q_t
        ET[t] = ET_t
        S[t] = S_next
        G[t] = G_next
        Qs[t] = Qs_t
        Qb[t] = Qb_t
        Perc[t] = Perc_t
        
        S_curr = S_next
        G_curr = G_next
    
    return {
        'Q': Q, 'ET': ET, 'S': S, 'G': G,
        'Qs': Qs, 'Qb': Qb, 'Perc': Perc
    }