"""
src/model.py  –  Two-Store Bucket Hydrological Model
Reproduced verbatim from the user's original file.
"""
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

    # soil-moisture stress parameters
    Sfc_frac: float = 0.30   # field capacity fraction of Smax
    beta_et: float  = 2.0    # nonlinearity for ET stress


# ---------------------------------------------------------
# Soil moisture stress function
# ---------------------------------------------------------
def soil_moisture_stress(S: float, Smax: float, Sfc_frac: float, beta: float) -> float:
    """Stress in [0,1]. ET increases with S."""
    Smax = max(float(Smax), 1e-12)
    Sfc  = max(float(Sfc_frac) * Smax, 1e-6)
    S    = float(S)
    x    = np.clip(S / Sfc, 0.0, 1.0)
    return float(x ** float(beta))


# ---------------------------------------------------------
# Two-Store Model Hydrologic Time Step
# ---------------------------------------------------------
def two_store_model_step(
    S_curr, G_curr, P_t, PET_t, params: ModelParams, ET_override=None
):
    """
    One timestep update.

    Returns:
        S_next, G_next, ET, Q, Qs, Qb, Perc, dS
    """
    S_prev = float(S_curr)

    P_t   = max(float(P_t)   if np.isfinite(P_t)   else 0.0, 0.0)
    PET_t = max(float(PET_t) if np.isfinite(PET_t) else 0.0, 0.0)

    S_curr = float(S_curr) + P_t
    G_curr = float(G_curr)

    # 2. Quickflow from saturation-excess runoff
    overflow    = max(S_curr - float(params.Smax), 0.0)
    Cqq         = float(params.Cqq)
    if not np.isfinite(Cqq) or Cqq < 0.0:
        Cqq = 0.0
    Qs_t        = min(max(Cqq * overflow, 0.0), S_curr)
    S_curr      = max(S_curr - Qs_t, 0.0)

    # 3. Evapotranspiration loss (state-dependent)
    if ET_override is not None and np.isfinite(float(ET_override)):
        ET_t = min(max(float(ET_override), 0.0), S_curr)
    else:
        Ke   = float(params.Ke)
        if not np.isfinite(Ke) or Ke < 0.0:
            Ke = 0.0
        ET_pot = Ke * PET_t
        stress = soil_moisture_stress(S_curr, float(params.Smax),
                                      float(params.Sfc_frac), float(params.beta_et))
        ET_t   = min(max(ET_pot * stress, 0.0), S_curr)

    S_curr = max(S_curr - ET_t, 0.0)

    # 4. Percolation from soil to groundwater
    Kperc  = float(params.Kperc)
    if not np.isfinite(Kperc) or Kperc < 0.0:
        Kperc = 0.0
    Perc_t = min(max(Kperc * S_curr, 0.0), S_curr)
    S_curr = max(S_curr - Perc_t, 0.0)

    # 5. Groundwater update and baseflow recession
    G_curr += Perc_t
    Kb     = float(params.Kb)
    if not np.isfinite(Kb) or Kb < 0.0:
        Kb = 0.0
    Qb_t   = min(max(Kb * G_curr, 0.0), G_curr)
    G_curr = max(G_curr - Qb_t, 0.0)

    # 6. Total streamflow
    Q_t = Qs_t + Qb_t
    dS  = float(S_curr - S_prev)

    return S_curr, G_curr, ET_t, Q_t, Qs_t, Qb_t, Perc_t, dS
