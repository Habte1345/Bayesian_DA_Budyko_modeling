# src/model.py
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------
# Model Parameters Dataclass
# ---------------------------------------------------------
@dataclass(frozen=True)
class ModelParams:
    Smax: float
    Kperc: float
    Kb: float
    Ke: float
    Cqq: float

    # NEW (soil-moisture stress parameters)
    Sfc_frac: float = 0.30   # field capacity fraction of Smax
    beta_et: float = 2.0     # nonlinearity for ET stress


# ---------------------------------------------------------
# Soil moisture stress function
# ---------------------------------------------------------
def soil_moisture_stress(S: float, Smax: float, Sfc_frac: float, beta: float) -> float:
    """
    Stress in [0,1]. ET increases with S.
    """
    Sfc = max(Sfc_frac * Smax, 1e-6)
    x = np.clip(S / Sfc, 0.0, 1.0)
    return float(x ** beta)


# ---------------------------------------------------------
# Two-Store Model Hydrologic Time Step
# ---------------------------------------------------------
def two_store_model_step(
    S_curr, G_curr, P_t, PET_t, params: ModelParams, ET_override=None
):
    # -----------------------------------------------------
    # 1. Precipitation added to soil store
    # -----------------------------------------------------
    P_t = max(float(P_t), 0.0)
    PET_t = max(float(PET_t), 0.0)

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
    # 3. Evapotranspiration loss (STATE-DEPENDENT)
    # -----------------------------------------------------
    if ET_override is not None:
        ET_pot = max(float(ET_override), 0.0)
        ET_t = min(ET_pot, S_curr)
    else:
        # Potential ET
        ET_pot = max(params.Ke * PET_t, 0.0)

        # NEW: moisture stress depends on S
        stress = soil_moisture_stress(
            S=S_curr,
            Smax=params.Smax,
            Sfc_frac=params.Sfc_frac,
            beta=params.beta_et,
        )

        ET_act = ET_pot * stress

        # water availability
        ET_t = min(ET_act, S_curr)

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
    # 6. Total streamflow
    # -----------------------------------------------------
    Q_t = Qs_t + Qb_t

    return S_curr, G_curr, ET_t, Q_t, Qs_t, Qb_t, Perc_t
