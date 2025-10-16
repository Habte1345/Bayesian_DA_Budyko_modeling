# src/budyko.py

import numpy as np
from dataclasses import dataclass
from scipy.optimize import fsolve

# =====================================================================
# 1. BUDYKO FRAMEWORK - Fu's Equation 
# =====================================================================

def fu_budyko(phi: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Calculates the ET/P ratio based on Fu's Budyko equation."""
    phi_safe = np.maximum(phi, 1e-6)
    om_safe = np.maximum(omega, 1.0)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        term = np.power(1.0 + np.power(phi_safe, om_safe), 1.0 / om_safe)
        result = 1.0 + phi_safe - term
        result = np.maximum(result, 1e-6)
    return result


def solve_omega_true(P, PET, ET_ke, Qb):
    """Numerically solves for the Budyko parameter (omega) for a catchment."""
    valid_P = P > 1e-6
    valid_PET = PET > 1e-6
    valid_ET_ke = ET_ke > 1e-6
    valid_Qb = Qb >= 0.0
    valid_idx = valid_P & valid_PET & valid_ET_ke & valid_Qb
    omega_true = np.full_like(P, np.nan, dtype=float)

    P_valid = P[valid_idx]
    PET_valid = PET[valid_idx]
    ET_ke_valid = ET_ke[valid_idx]
    Qb_valid = Qb[valid_idx]
    
    P_minus_dS_valid = np.maximum(ET_ke_valid + Qb_valid, 1e-6)
    
    phi_valid = PET_valid / P_minus_dS_valid
    et_ratio_valid = np.clip(ET_ke_valid / P_minus_dS_valid, 1e-6, 0.999)

    for i in range(len(P_valid)):
        phi = phi_valid[i]
        et_ratio = et_ratio_valid[i]
        
        def objective_func(omega):
            if omega < 1.0: return 1e10
            try:
                f_val = 1.0 + phi - np.power(1.0 + np.power(phi, omega), 1.0/omega)
                return (f_val - et_ratio)
            except: return 1e10
        
        try:
            omega_solution, infodict, ier, msg = fsolve(objective_func, x0=2.5, full_output=True)
            if ier == 1:
                omega = omega_solution[0]
                omega_true[np.where(valid_idx)[0][i]] = np.clip(omega, 1.0, 10.0)
        except:
            pass

    return omega_true


# =====================================================================
# 2. OMEGA MLR MODEL - Dynamic Catchment Characteristics
# =====================================================================

@dataclass
class OmegaMLRModel:
    """Dataclass for storing Multiple Linear Regression coefficients for omega."""
    beta0: float
    beta1: float
    beta2: float
    
    def predict(self, M: np.ndarray, Slope: np.ndarray) -> np.ndarray:
        """Predicts omega based on vegetation cover (M) and Slope."""
        omega_pred = self.beta0 + self.beta1 * M + self.beta2 * Slope
        return np.clip(omega_pred, 1.0, 10.0)


def fit_omega_mlr(M: np.ndarray, Slope: np.ndarray, omega_true: np.ndarray) -> OmegaMLRModel:
    """Fits an MLR model (omega ~ beta0 + beta1*M + beta2*Slope) to omega_true."""
    valid_idx = ~(np.isnan(M) | np.isnan(Slope) | np.isnan(omega_true))
    
    if valid_idx.sum() < 3:
        # Fallback to default coefficients if not enough data
        return OmegaMLRModel(beta0=2.36, beta1=1.16, beta2=0.0)
    
    M_valid = M[valid_idx]
    # Assuming Slope is static, take the first valid value
    Slope_val = Slope[valid_idx][0] if len(Slope[valid_idx]) > 0 else 0.0
    Slope_valid = np.full_like(M_valid, Slope_val)
    omega_valid = omega_true[valid_idx]
    
    X = np.vstack([np.ones_like(M_valid), M_valid, Slope_valid]).T
    
    try:
        coef, _, _, _ = np.linalg.lstsq(X, omega_valid, rcond=None)
        
        if np.any(np.isnan(coef)) or len(coef) < 3:
            # Fallback to simple M-only regression
            X_M = np.vstack([np.ones_like(M_valid), M_valid]).T
            coef_M, _, _, _ = np.linalg.lstsq(X_M, omega_valid, rcond=None)
            return OmegaMLRModel(beta0=coef_M[0], beta1=coef_M[1], beta2=0.0)
            
        return OmegaMLRModel(beta0=coef[0], beta1=coef[1], beta2=coef[2])
    except np.linalg.LinAlgError:
        return OmegaMLRModel(beta0=2.36, beta1=1.16, beta2=0.0)