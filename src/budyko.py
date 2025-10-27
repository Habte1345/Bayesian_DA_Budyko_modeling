# # src/budyko.py

# import numpy as np
# from dataclasses import dataclass
# from scipy.optimize import fsolve

# # =====================================================================
# # 1. BUDYKO FRAMEWORK - Fu's Equation 
# # =====================================================================

# def fu_budyko(phi: np.ndarray, omega: np.ndarray) -> np.ndarray:
#     """Calculates the ET/P ratio based on Fu's Budyko equation."""
#     phi_safe = np.maximum(phi, 1e-6)
#     om_safe = np.maximum(omega, 1.0)
#     with np.errstate(over='ignore', under='ignore', invalid='ignore'):
#         term = np.power(1.0 + np.power(phi_safe, om_safe), 1.0 / om_safe)
#         result = 1.0 + phi_safe - term
#         result = np.maximum(result, 1e-6)
#     return result


# def solve_omega_true(P, PET, ET_ke, Qb):
#     """Numerically solves for the Budyko parameter (omega) for a catchment."""
#     valid_P = P > 1e-6
#     valid_PET = PET > 1e-6
#     valid_ET_ke = ET_ke > 1e-6
#     valid_Qb = Qb >= 0.0
#     valid_idx = valid_P & valid_PET & valid_ET_ke & valid_Qb
#     omega_true = np.full_like(P, np.nan, dtype=float)

#     P_valid = P[valid_idx]
#     PET_valid = PET[valid_idx]
#     ET_ke_valid = ET_ke[valid_idx]
#     Qb_valid = Qb[valid_idx]
    
#     P_minus_dS_valid = np.maximum(ET_ke_valid + Qb_valid, 1e-6)
    
#     phi_valid = PET_valid / P_minus_dS_valid
#     et_ratio_valid = np.clip(ET_ke_valid / P_minus_dS_valid, 1e-6, 0.999)

#     for i in range(len(P_valid)):
#         phi = phi_valid[i]
#         et_ratio = et_ratio_valid[i]
        
#         def objective_func(omega):
#             if omega < 1.0: return 1e10
#             try:
#                 f_val = 1.0 + phi - np.power(1.0 + np.power(phi, omega), 1.0/omega)
#                 return (f_val - et_ratio)
#             except: return 1e10
        
#         try:
#             omega_solution, infodict, ier, msg = fsolve(objective_func, x0=2.5, full_output=True)
#             if ier == 1:
#                 omega = omega_solution[0]
#                 omega_true[np.where(valid_idx)[0][i]] = np.clip(omega, 0.1, 1000000.0)
#         except:
#             pass

#     return omega_true


# # =====================================================================
# # 2. OMEGA MLR MODEL - Dynamic Catchment Characteristics
# # =====================================================================

# @dataclass
# class OmegaMLRModel:
#     """Dataclass for storing Multiple Linear Regression coefficients for omega."""
#     beta0: float
#     beta1: float
#     beta2: float
    
#     def predict(self, M: np.ndarray, Slope: np.ndarray) -> np.ndarray:
#         """Predicts omega based on vegetation cover (M) and Slope."""
#         omega_pred = self.beta0 + self.beta1 * M + self.beta2 * Slope
#         return np.clip(omega_pred, 1.0, 10.0)


# def fit_omega_mlr(M: np.ndarray, Slope: np.ndarray, omega_true: np.ndarray) -> OmegaMLRModel:
#     """Fits an MLR model (omega ~ beta0 + beta1*M + beta2*Slope) to omega_true."""
#     valid_idx = ~(np.isnan(M) | np.isnan(Slope) | np.isnan(omega_true))
    
#     if valid_idx.sum() < 3:
#         # Fallback to default coefficients if not enough data
#         return OmegaMLRModel(beta0=2.36, beta1=1.16, beta2=0.0)
    
#     M_valid = M[valid_idx]
#     # Assuming Slope is static, take the first valid value
#     Slope_val = Slope[valid_idx][0] if len(Slope[valid_idx]) > 0 else 0.0
#     Slope_valid = np.full_like(M_valid, Slope_val)
#     omega_valid = omega_true[valid_idx]
    
#     X = np.vstack([np.ones_like(M_valid), M_valid, Slope_valid]).T
    
#     try:
#         coef, _, _, _ = np.linalg.lstsq(X, omega_valid, rcond=None)
        
#         if np.any(np.isnan(coef)) or len(coef) < 3:
#             # Fallback to simple M-only regression
#             X_M = np.vstack([np.ones_like(M_valid), M_valid]).T
#             coef_M, _, _, _ = np.linalg.lstsq(X_M, omega_valid, rcond=None)
#             return OmegaMLRModel(beta0=coef_M[0], beta1=coef_M[1], beta2=0.0)
            
#         return OmegaMLRModel(beta0=coef[0], beta1=coef[1], beta2=coef[2])
#     except np.linalg.LinAlgError:
#         return OmegaMLRModel(beta0=2.36, beta1=1.16, beta2=0.0)


# src/budyko.py

import numpy as np
from dataclasses import dataclass
from scipy.optimize import fsolve

# =====================================================================
# 1. BUDYKO FRAMEWORK - Fu's Equation 
# =====================================================================

def fu_budyko(phi: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Calculates the ET/P ratio based on Fu's Budyko equation."""
    # Ensure numerical stability
    phi_safe = np.maximum(phi, 1e-6)
    om_safe = np.maximum(omega, 1.0)
    
    with np.errstate(all='ignore'):
        # Fu's equation: E/P = 1 + phi - (1 + phi^omega)^(1/omega)
        term = np.power(1.0 + np.power(phi_safe, om_safe), 1.0 / om_safe)
        result = 1.0 + phi_safe - term
        
    return np.clip(result, 1e-6, 1.0) # ET/P ratio must be between 0 and 1


def solve_omega_true(P, PET, ET_obs, Qb_obs):
    """
    Numerically solves for the Budyko parameter (omega) for a catchment.
    
    Inputs are typically long-term means or annual values, but applied here 
    to monthly time series, which is a common practice in DA-based omega estimation.
    """
    valid_P = P > 1e-6
    valid_ET_obs = ET_obs > 1e-6
    valid_idx = valid_P & valid_ET_obs
    omega_true = np.full_like(P, np.nan, dtype=float)

    P_valid = P[valid_idx]
    PET_valid = PET[valid_idx]
    ET_obs_valid = ET_obs[valid_idx]
    Qb_obs_valid = Qb_obs[valid_idx]
    
    # Calculate the effective P (P - dS/dt, approximated by P - dS/dt = ET + Q)
    # Since we are solving for a steady-state approximation, we use 
    # P_eff = ET + Q_total = ET_obs + (Qs_obs + Qb_obs) 
    # NOTE: The original code used P_minus_dS = ET_ke + Qb. We must ensure P_eff > 0.
    P_eff_valid = np.maximum(P_valid, 1e-6) # Use P as the primary driver for aridity index
    
    # Aridity index: phi = PET / P_eff
    phi_valid = PET_valid / P_eff_valid
    
    # Evaporative ratio: E_ratio = ET / P_eff
    et_ratio_valid = np.clip(ET_obs_valid / P_eff_valid, 1e-6, 0.999)

    for i in range(len(P_valid)):
        phi = phi_valid[i]
        et_ratio = et_ratio_valid[i]
        
        # The objective function is f(omega) = Fu(phi, omega) - et_ratio = 0
        def objective_func(omega):
            if omega < 1.0: return 1e10 # Constraint: omega >= 1.0
            try:
                # Fu's equation (simplified for scalar)
                f_val = 1.0 + phi - np.power(1.0 + np.power(phi, omega), 1.0/omega)
                return (f_val - et_ratio)
            except: 
                return 1e10
        
        try:
            # Solve for omega, starting near the typical range
            omega_solution, infodict, ier, msg = fsolve(objective_func, x0=2.5, full_output=True, xtol=1e-4)
            if ier == 1:
                omega = omega_solution[0]
                omega_true[np.where(valid_idx)[0][i]] = np.clip(omega, 1.0, 10.0) # Clip to realistic range
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
        return np.clip(omega_pred, 1.0, 10.0) # Ensure Omega is within realistic bounds


def fit_omega_mlr(M: np.ndarray, Slope: np.ndarray, omega_true: np.ndarray) -> OmegaMLRModel:
    """Fits an MLR model (omega ~ beta0 + beta1*M + beta2*Slope) to omega_true."""
    valid_idx = ~(np.isnan(M) | np.isnan(Slope) | np.isnan(omega_true))
    
    if valid_idx.sum() < 3:
        # Fallback to default coefficients if not enough data
        return OmegaMLRModel(beta0=2.36, beta1=1.16, beta2=0.0)
    
    M_valid = M[valid_idx]
    # NOTE: Assuming Slope is static, we still create a vector of the static value
    Slope_valid = Slope[valid_idx] 
    omega_valid = omega_true[valid_idx]
    
    # Construct the design matrix X for MLR: [Ones, M, Slope]
    X = np.vstack([np.ones_like(M_valid), M_valid, Slope_valid]).T
    
    try:
        coef, _, _, _ = np.linalg.lstsq(X, omega_valid, rcond=None)
        
        if np.any(np.isnan(coef)) or len(coef) < 3:
            # Fallback to simple M-only regression (omitting Slope)
            X_M = np.vstack([np.ones_like(M_valid), M_valid]).T
            coef_M, _, _, _ = np.linalg.lstsq(X_M, omega_valid, rcond=None)
            return OmegaMLRModel(beta0=coef_M[0], beta1=coef_M[1], beta2=0.0)
            
        return OmegaMLRModel(beta0=coef[0], beta1=coef[1], beta2=coef[2])
    except np.linalg.LinAlgError:
        # Fallback if the matrix is singular
        return OmegaMLRModel(beta0=2.36, beta1=1.16, beta2=0.0)
    

# =====================================================================
# 3. HELPER FUNCTION: ESTIMATE BUDYKO ET FOR SCENARIOS
# =====================================================================

def estimate_budyko_et(P, PET, model='Fu', m=1.35):
    """
    Estimate ET using Fu Budyko model with adjusted m (a helper function 
    used in run_simulation.py for scenario definition).
    """
    # Use Fu's equation with a static omega=m, and assume omega=m is applied to Aridity Index (PET/P)
    if P <= 0 or PET <= 0:
        return 0.0
    
    aridity = PET / P
    
    # Fu's equation for E/P (where m acts as omega)
    # E/P = 1 + phi - (1 + phi^omega)^(1/omega)
    try:
        E_P_ratio = 1.0 + aridity - (1.0 + aridity ** m) ** (1.0 / m)
        ET = P * E_P_ratio
    except Exception:
        # Fallback for numerical instability
        ET = 0.0 
        
    return np.clip(ET, 0.0, min(PET, P * 0.999)) # ET cannot exceed P or PET