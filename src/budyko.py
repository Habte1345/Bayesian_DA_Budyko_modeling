# # src/budyko.py
# import numpy as np
# import pandas as pd
# from dataclasses import dataclass
# from scipy.optimize import fsolve
# import warnings

# # Suppress runtime warnings that often occur with complex numerical operations
# warnings.filterwarnings("ignore", category=RuntimeWarning)

# @dataclass
# class OmegaMLRModel:
#     beta0: float
#     beta1: float
#     beta2: float

#     def predict(self, M: np.ndarray, Slope: np.ndarray) -> np.ndarray:
#         omega_MLR = self.beta0 + self.beta1 * M + self.beta2 * Slope
#         return np.clip(omega_MLR, 1.0, 10.0)

# class BudykoModelEstimator:
#     def __init__(self, Evap_df: pd.DataFrame, Qsb_monthly: pd.DataFrame, 
#                  PotEvap_df: pd.DataFrame, M_basin: pd.DataFrame, 
#                  Slope_basin: pd.DataFrame, ke: float = 0.68):
#         self.Evap_df = Evap_df
#         self.Qsb_monthly = Qsb_monthly
#         self.PotEvap_df = PotEvap_df
#         self.M_basin = M_basin
#         self.Slope_basin = Slope_basin
#         self.ke = ke

#         self.omega_true = None
#         self.omega_MLR = None
#         self.ET_B = None

#     def _solve_for_omega(self, ET, QB, PET, omega_guess=1.0):
#         if (ET + QB) == 0:
#             return np.nan
#         ratio_ET = ET / (ET + QB)
#         ratio_PET = PET / (ET + QB)
#         def f(omega):
#             return 1 + ratio_PET - (1 + ratio_PET**omega)**(1/omega) - ratio_ET
#         try:
#             sol = fsolve(f, x0=omega_guess, xtol=1e-8)
#             return sol[0] if np.isfinite(sol[0]) else np.nan
#         except:
#             return np.nan

#     def compute_omega_true(self) -> pd.DataFrame:
#         Evap_df = self.Evap_df
#         Qsb_monthly = self.Qsb_monthly
#         PotEvap_df = self.PotEvap_df
#         ke = self.ke
#         omega_true = pd.DataFrame(index=Evap_df.index, columns=Evap_df.columns, dtype=float)

#         for col in PotEvap_df.columns:
#             QB_col = Qsb_monthly[col].values
#             PET_col = PotEvap_df[col].values
#             ET_Ke = ke * PET_col
#             omega_values = np.array([
#                 self._solve_for_omega(ET, QB, PET)
#                 for ET, QB, PET in zip(ET_Ke, QB_col, PET_col)
#             ])
#             omega_true[col] = omega_values

#         self.omega_true = omega_true
#         return self.omega_true

    
#     def fit_and_compute_omega_mlr(self) -> pd.DataFrame:
#         """
#         Fit multiple linear regression (MLR) for ω as a function of M and Slope.
#         Assumes M_basin and Slope_basin have the same shape (T × B).
#         """
#         if self.omega_true is None:
#             self.compute_omega_true()

#         # Extract matrices (T = time steps, B = basins)
#         M = self.M_basin.values            # shape (T, B)
#         Slope = self.Slope_basin.values    # shape (T, B)
#         y = self.omega_true.values         # shape (T, B)

#         rows = []
#         targets = []
#         T, B = M.shape

#         # Build regression dataset from all basins/time steps
#         for i in range(B):
#             Mi = M[:, i]
#             Si = Slope[:, i]
#             yi = y[:, i]
#             mask = ~(np.isnan(Mi) | np.isnan(Si) | np.isnan(yi))
#             if mask.any():
#                 Xi = np.column_stack([np.ones(mask.sum()), Mi[mask], Si[mask]])
#                 rows.append(Xi)
#                 targets.append(yi[mask])

#         if not rows:
#             raise ValueError("No valid data to fit omega MLR.")

#         # Stack all basins together
#         X = np.vstack(rows)
#         Y = np.concatenate(targets)

#         # Solve least-squares regression
#         beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

#         # Predict ω for each basin/time step
#         Omega_hat = np.empty_like(M, dtype=float)
#         for i in range(B):
#             Omega_hat[:, i] = beta[0] + beta[1]*M[:, i] + beta[2]*Slope[:, i]

#         # Enforce ω ≥ 1
#         Omega_hat = np.where(Omega_hat < 1.0, 1.0, Omega_hat)

#         # Store and return as DataFrame
#         self.omega_MLR = pd.DataFrame(
#             Omega_hat, index=self.M_basin.index, columns=self.M_basin.columns
#         )
#         return self.omega_MLR

#     def estimate_budyko_et(self) -> pd.DataFrame:
#         if self.omega_MLR is None:
#             self.fit_and_compute_omega_mlr()

#         Qb = self.Qsb_monthly
#         PET_df = self.PotEvap_df
#         omega_MLR = self.omega_MLR
#         P_values = (self.Evap_df + Qb).values
#         PET_values = PET_df.values
#         aridity = np.divide(PET_values, P_values, out=np.zeros_like(PET_values), where=P_values > 0)

#         with np.errstate(over='ignore', invalid='ignore'):
#             omega_values = omega_MLR.values
#             E_ratio = 1.0 + aridity - np.power(1.0 + np.power(aridity, omega_values), 1.0 / omega_values)

#         E_ratio = np.nan_to_num(E_ratio, nan=0.0, posinf=0.0, neginf=0.0)
#         ET_est = P_values * E_ratio
#         ET_est = np.clip(ET_est, 0.0, np.minimum(PET_values, P_values))
#         self.ET_B = pd.DataFrame(ET_est, index=PET_df.index, columns=PET_df.columns)
#         return self.ET_B

#     def OmegaTrue_OmegaMLR_BudykoET(self) -> pd.DataFrame:
#         print("Starting Budyko Model Estimation Workflow...")
#         self.compute_omega_true()
#         self.fit_and_compute_omega_mlr()
#         self.estimate_budyko_et()
#         print("Workflow complete.")
#         return self.ET_B

# if __name__ == '__main__':
#     print("BudykoModelEstimator class defined. Run a separate script to execute the full workflow.")



# # src/budyko.py (Corrected for Basin-Specific Ke)
# import numpy as np
# import pandas as pd
# from dataclasses import dataclass
# from scipy.optimize import fsolve
# import warnings

# # Suppress runtime warnings that often occur with complex numerical operations
# warnings.filterwarnings("ignore", category=RuntimeWarning)

# @dataclass
# class OmegaMLRModel:
#     beta0: float
#     beta1: float
#     beta2: float

#     def predict(self, M: np.ndarray, Slope: np.ndarray) -> np.ndarray:
#         omega_MLR = self.beta0 + self.beta1 * M + self.beta2 * Slope
#         return np.clip(omega_MLR, 1.0, 10.0)

# class BudykoModelEstimator:
#     def __init__(self, Evap_df: pd.DataFrame, Qsb_monthly: pd.DataFrame, 
#                  PotEvap_df: pd.DataFrame, M_basin: pd.DataFrame, 
#                  Slope_basin: pd.DataFrame, calibrated_params: dict): # CHANGED: Pass calibrated_params
#         self.Evap_df = Evap_df
#         self.Qsb_monthly = Qsb_monthly
#         self.PotEvap_df = PotEvap_df
#         self.M_basin = M_basin
#         self.Slope_basin = Slope_basin
#         self.calibrated_params = calibrated_params # NEW: Store calibrated parameters

#         self.omega_true = None
#         self.omega_MLR = None
#         self.ET_B = None

#     def _solve_for_omega(self, ET, QB, PET, omega_guess=1.0):
#         if (ET + QB) == 0:
#             return np.nan
#         ratio_ET = ET / (ET + QB)
#         ratio_PET = PET / (ET + QB)
#         def f(omega):
#             return 1 + ratio_PET - (1 + ratio_PET**omega)**(1/omega) - ratio_ET
#         try:
#             sol = fsolve(f, x0=omega_guess, xtol=1e-8)
#             return sol[0] if np.isfinite(sol[0]) else np.nan
#         except:
#             return np.nan

#     def compute_omega_true(self) -> pd.DataFrame:
#         Evap_df = self.Evap_df
#         Qsb_monthly = self.Qsb_monthly
#         PotEvap_df = self.PotEvap_df
#         omega_true = pd.DataFrame(index=Evap_df.index, columns=Evap_df.columns, dtype=float)

#         for col in PotEvap_df.columns:
#             # CRITICAL FIX: Get the basin-specific Ke from the calibrated parameters
#             if col not in self.calibrated_params or 'Ke' not in self.calibrated_params[col]:
#                  print(f"Warning: Ke parameter missing for basin {col}. Skipping omega_true calculation.")
#                  continue

#             ke_basin = self.calibrated_params[col]['Ke'] # Use the calibrated Ke
            
#             QB_col = Qsb_monthly[col].values
#             PET_col = PotEvap_df[col].values
            
#             # Use the basin-specific ke_basin to calculate ET_Ke
#             ET_Ke = ke_basin * PET_col 
            
#             omega_values = np.array([
#                 self._solve_for_omega(ET, QB, PET)
#                 for ET, QB, PET in zip(ET_Ke, QB_col, PET_col)
#             ])
#             omega_true[col] = omega_values

#         self.omega_true = omega_true
#         return self.omega_true

    
#     def fit_and_compute_omega_mlr(self) -> pd.DataFrame:
#         """
#         Fit multiple linear regression (MLR) for ω as a function of M and Slope.
#         Assumes M_basin and Slope_basin have the same shape (T × B).
#         """
#         if self.omega_true is None:
#             self.compute_omega_true()

#         # Extract matrices (T = time steps, B = basins)
#         M = self.M_basin.values          # shape (T, B)
#         Slope = self.Slope_basin.values    # shape (T, B)
#         y = self.omega_true.values         # shape (T, B)

#         rows = []
#         targets = []
#         T, B = M.shape

#         # Build regression dataset from all basins/time steps
#         for i in range(B):
#             Mi = M[:, i]
#             Si = Slope[:, i]
#             yi = y[:, i]
#             mask = ~(np.isnan(Mi) | np.isnan(Si) | np.isnan(yi))
#             if mask.any():
#                 Xi = np.column_stack([np.ones(mask.sum()), Mi[mask], Si[mask]])
#                 rows.append(Xi)
#                 targets.append(yi[mask])

#         if not rows:
#             raise ValueError("No valid data to fit omega MLR.")

#         # Stack all basins together
#         X = np.vstack(rows)
#         Y = np.concatenate(targets)

#         # Solve least-squares regression
#         beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

#         # Predict ω for each basin/time step
#         Omega_hat = np.empty_like(M, dtype=float)
#         for i in range(B):
#             Omega_hat[:, i] = beta[0] + beta[1]*M[:, i] + beta[2]*Slope[:, i]

#         # Enforce ω ≥ 1
#         Omega_hat = np.where(Omega_hat < 1.0, 1.0, Omega_hat)

#         # Store and return as DataFrame
#         self.omega_MLR = pd.DataFrame(
#             Omega_hat, index=self.M_basin.index, columns=self.M_basin.columns
#         )
#         return self.omega_MLR

#     def estimate_budyko_et(self) -> pd.DataFrame:
#         if self.omega_MLR is None:
#             self.fit_and_compute_omega_mlr()

#         Qb = self.Qsb_monthly
#         PET_df = self.PotEvap_df
#         omega_MLR = self.omega_MLR
#         P_values = (self.Evap_df + Qb).values
#         PET_values = PET_df.values
#         aridity = np.divide(PET_values, P_values, out=np.zeros_like(PET_values), where=P_values > 0)

#         with np.errstate(over='ignore', invalid='ignore'):
#             omega_values = omega_MLR.values
#             E_ratio = 1.0 + aridity - np.power(1.0 + np.power(aridity, omega_values), 1.0 / omega_values)

#         E_ratio = np.nan_to_num(E_ratio, nan=0.0, posinf=0.0, neginf=0.0)
#         ET_est = P_values * E_ratio
#         ET_est = np.clip(ET_est, 0.0, np.minimum(PET_values, P_values))
#         self.ET_B = pd.DataFrame(ET_est, index=PET_df.index, columns=PET_df.columns)
#         return self.ET_B

#     def OmegaTrue_OmegaMLR_BudykoET(self) -> pd.DataFrame:
#         print("Starting Budyko Model Estimation Workflow...")
#         self.compute_omega_true()
#         self.fit_and_compute_omega_mlr()
#         self.estimate_budyko_et()
#         print("Workflow complete.")
#         return self.ET_B

# if __name__ == '__main__':
#     print("BudykoModelEstimator class defined. Run a separate script to execute the full workflow.")


# src/budyko.py (Corrected for Basin-Specific Ke with History Prints)
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import fsolve
import warnings

# Suppress runtime warnings that often occur with complex numerical operations
warnings.filterwarnings("ignore", category=RuntimeWarning)

@dataclass
class OmegaMLRModel:
    beta0: float
    beta1: float
    beta2: float

    def predict(self, M: np.ndarray, Slope: np.ndarray) -> np.ndarray:
        omega_MLR = self.beta0 + self.beta1 * M + self.beta2 * Slope
        return np.clip(omega_MLR, 1.0, 5.0)

class BudykoModelEstimator:
    def __init__(self, Evap_df: pd.DataFrame, Qsb_monthly: pd.DataFrame, 
                 PotEvap_df: pd.DataFrame, M_basin: pd.DataFrame, 
                 Slope_basin: pd.DataFrame, calibrated_params: dict): # CHANGED: Pass calibrated_params
        self.Evap_df = Evap_df
        self.Qsb_monthly = Qsb_monthly
        self.PotEvap_df = PotEvap_df
        self.M_basin = M_basin
        self.Slope_basin = Slope_basin
        self.calibrated_params = calibrated_params # NEW: Store calibrated parameters

        self.omega_true = None
        self.omega_MLR = None
        self.ET_B = None

    def _solve_for_omega(self, ET, QB, PET, omega_guess=1.0):
        if (ET + QB) == 0:
            return np.nan
        ratio_ET = ET / (ET + QB)
        ratio_PET = PET / (ET + QB)
        def f(omega):
            return 1 + ratio_PET - (1 + ratio_PET**omega)**(1/omega) - ratio_ET
        try:
            sol = fsolve(f, x0=omega_guess, xtol=1e-8)
            
            # --- CRITICAL CHANGE: Bound omega_true to [0.0, 10.0] ---
            omega_val = sol[0] if np.isfinite(sol[0]) else np.nan
            # Apply the requested bounds to omega_true
            return np.clip(omega_val, 0.0, 500.0) if np.isfinite(omega_val) else np.nan
            # --------------------------------------------------------
            
        except:
            return np.nan

    def compute_omega_true(self) -> pd.DataFrame:
        # ADDED PRINT HISTORY
        # print("--- [History] Starting calculation of basin-specific omega_true...")
        
        Evap_df = self.Evap_df
        Qsb_monthly = self.Qsb_monthly
        PotEvap_df = self.PotEvap_df
        omega_true = pd.DataFrame(index=Evap_df.index, columns=Evap_df.columns, dtype=float)

        for col in PotEvap_df.columns:
            # CRITICAL FIX: Get the basin-specific Ke from the calibrated parameters
            if col not in self.calibrated_params or 'Ke' not in self.calibrated_params[col]:
                #   print(f"Warning: Ke parameter missing for basin {col}. Skipping omega_true calculation.")
                continue

            ke_basin = self.calibrated_params[col]['Ke'] # Use the calibrated Ke
            
            QB_col = Qsb_monthly[col].values
            PET_col = PotEvap_df[col].values
            
            # Use the basin-specific ke_basin to calculate ET_Ke
            ET_Ke = ke_basin * PET_col  #ke_basin 
            
            omega_values = np.array([
                self._solve_for_omega(ET, QB, PET)
                for ET, QB, PET in zip(ET_Ke, QB_col, PET_col)
            ])
            omega_true[col] = omega_values

        self.omega_true = omega_true
        # ADDED PRINT HISTORY
        # print("--- [History] Completed calculation of omega_true.")
        return self.omega_true

    
    def fit_and_compute_omega_mlr(self) -> pd.DataFrame:
        """
        Fit multiple linear regression (MLR) for ω as a function of M and Slope.
        Assumes M_basin and Slope_basin have the same shape (T × B).
        """
        # ADDED PRINT HISTORY
        # print("--- [History] Starting fitting of omega_true with NDVI (M) and Slope...")
        
        if self.omega_true is None:
            self.compute_omega_true()

        # Extract matrices (T = time steps, B = basins)
        M = self.M_basin.values          # shape (T, B)
        Slope = self.Slope_basin.values    # shape (T, B)
        y = self.omega_true.values         # shape (T, B)

        rows = []
        targets = []
        T, B = M.shape

        # Build regression dataset from all basins/time steps
        for i in range(B):
            Mi = M[:, i]
            Si = Slope[:, i]
            yi = y[:, i]
            mask = ~(np.isnan(Mi) | np.isnan(Si) | np.isnan(yi))
            if mask.any():
                Xi = np.column_stack([np.ones(mask.sum()), Mi[mask], Si[mask]])
                rows.append(Xi)
                targets.append(yi[mask])

        if not rows:
            raise ValueError("No valid data to fit omega MLR.")

        # Stack all basins together
        X = np.vstack(rows)
        Y = np.concatenate(targets)

        # Solve least-squares regression
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

        # Predict ω for each basin/time step
        Omega_hat = np.empty_like(M, dtype=float)
        for i in range(B):
            Omega_hat[:, i] = beta[0] + beta[1]*M[:, i] + beta[2]*Slope[:, i]

        # Enforce ω ≥ 1 (This remains the original logic for omega_MLR)
        Omega_hat = np.where(Omega_hat < 1.0, 1.0, Omega_hat)

        # Store and return as DataFrame
        self.omega_MLR = pd.DataFrame(
            Omega_hat, index=self.M_basin.index, columns=self.M_basin.columns
        )
        # ADDED PRINT HISTORY
        # print("--- [History] Completed fitting of omega_MLR.")
        return self.omega_MLR

    def estimate_budyko_et(self) -> pd.DataFrame:
        # ADDED PRINT HISTORY
        # print("--- [History] Starting Budyko ET modeling using omega_MLR...")
        
        if self.omega_MLR is None:
            self.fit_and_compute_omega_mlr()

        Qb = self.Qsb_monthly
        PET_df = self.PotEvap_df
        omega_MLR = self.omega_MLR
        P_values = (self.Evap_df + Qb).values
        PET_values = PET_df.values
        aridity = np.divide(PET_values, P_values, out=np.zeros_like(PET_values), where=P_values > 0)

        with np.errstate(over='ignore', invalid='ignore'):
            omega_values = omega_MLR.values
            E_ratio = 1.0 + aridity - np.power(1.0 + np.power(aridity, omega_values), 1.0 / omega_values)

        E_ratio = np.nan_to_num(E_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        ET_est = P_values * E_ratio
        ET_est = np.clip(ET_est, 0.0, np.minimum(PET_values, P_values))
        self.ET_B = pd.DataFrame(ET_est, index=PET_df.index, columns=PET_df.columns)
        
        # ADDED PRINT HISTORY
        # print("--- [History] Completed Budyko ET modeling (ET_B).")
        return self.ET_B

    def OmegaTrue_OmegaMLR_BudykoET(self) -> pd.DataFrame:
        # print("Starting Budyko Model Estimation Workflow...")
        self.compute_omega_true()
        self.fit_and_compute_omega_mlr()
        self.estimate_budyko_et()
        print("Workflow complete.")
        return self.ET_B

if __name__ == '__main__':
    print("BudykoModelEstimator class defined. Run a separate script to execute the full workflow.")