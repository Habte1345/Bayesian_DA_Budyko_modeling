from dataclasses import dataclass, field
import numpy as np

# ---------------------------------------------------------
# Model Parameters Dataclass
# ---------------------------------------------------------
@dataclass(frozen=True)  # immutable parameter set
class ModelParams:
    """
    Dataclass for storing hydrologic model parameters.
    Parameters are required, ensuring calibrated values are used.
    """
    Smax: float     # Max soil storage (mm)
    Kperc: float    # Percolation rate (fraction per timestep)
    Kb: float       # Baseflow recession constant
    Ke: float       # Evaporation coefficient (fraction of PET)
    Cqq: float      # Quickflow coefficient


# ---------------------------------------------------------
# Two-Store Model Hydrologic Time Step
# ---------------------------------------------------------
def two_store_model_step(
    S_curr, G_curr, P_t, PET_t, params: ModelParams, ET_override=None
):
    """
    Executes a single timestep of the two-store hydrologic model.

    Returns:
        S_next  : updated soil storage
        G_next  : updated groundwater storage
        ET_t    : actual evapotranspiration
        Q_t     : total streamflow (Q_s + Q_b)
        Qs_t    : quickflow (saturation-excess runoff)
        Qb_t    : baseflow
        Perc_t  : percolation to groundwater
    """

    # -----------------------------------------------------
    # 1. Precipitation added to soil store
    # -----------------------------------------------------
    P_t = max(P_t, 0.0)
    S_curr += P_t

    # -----------------------------------------------------
    # 2. Quickflow from saturation-excess runoff
    # -----------------------------------------------------
    overflow = max(S_curr - params.Smax, 0.0)
    Qs_t_calc = params.Cqq * overflow
    Qs_t = min(Qs_t_calc, S_curr)
    S_curr -= Qs_t
    S_curr = max(S_curr, 0.0)

    # -----------------------------------------------------
    # 3. Evapotranspiration loss
    # -----------------------------------------------------
    ET_pot = ET_override if ET_override is not None else params.Ke * PET_t
    ET_pot = max(ET_pot, 0.0)
    ET_t = min(ET_pot, S_curr)
    S_curr -= ET_t
    S_curr = max(S_curr, 0.0)

    # -----------------------------------------------------
    # 4. Percolation from soil to groundwater
    # -----------------------------------------------------
    Perc_t_calc = params.Kperc * S_curr
    Perc_t = min(Perc_t_calc, S_curr)
    S_curr -= Perc_t
    S_curr = max(S_curr, 0.0)

    # -----------------------------------------------------
    # 5. Groundwater update and baseflow recession
    # -----------------------------------------------------
    G_curr += Perc_t
    Qb_t_calc = params.Kb * G_curr
    Qb_t = min(Qb_t_calc, G_curr)
    G_curr -= Qb_t
    G_curr = max(G_curr, 0.0)

    # -----------------------------------------------------
    # 6. Total streamflow for this timestep
    # -----------------------------------------------------
    Q_t = Qs_t + Qb_t

    return S_curr, G_curr, ET_t, Q_t, Qs_t, Qb_t, Perc_t
