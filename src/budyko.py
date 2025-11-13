# src/budyko.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import fsolve
import warnings

# Suppress runtime warnings that often occur with complex numerical operations
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------
# Ω via Multiple Linear Regression
# ---------------------------------------------------------
@dataclass
class OmegaMLRModel:
    beta0: float
    beta1: float
    beta2: float

    def predict(self, M: np.ndarray, Slope: np.ndarray) -> np.ndarray:
        omega_MLR = self.beta0 + self.beta1 * M + self.beta2 * Slope
        return np.clip(omega_MLR, 1.0, 5.0)


# ---------------------------------------------------------
# Budyko Model Estimator
# ---------------------------------------------------------
class BudykoModelEstimator:
    def __init__(
        self,
        Evap_df: pd.DataFrame,
        Qsb_monthly: pd.DataFrame,
        PotEvap_df: pd.DataFrame,
        M_basin: pd.DataFrame,
        Slope_basin: pd.DataFrame,
        calibrated_params: dict,
    ):
        self.Evap_df = Evap_df
        self.Qsb_monthly = Qsb_monthly
        self.PotEvap_df = PotEvap_df
        self.M_basin = M_basin
        self.Slope_basin = Slope_basin
        self.calibrated_params = calibrated_params

        self.omega_true = None
        self.omega_MLR = None
        self.ET_B = None

    # -----------------------------------------------------
    # Solve for ω_true
    # -----------------------------------------------------
    def _solve_for_omega(self, ET, QB, PET, omega_guess=1.0):
        if (ET + QB) == 0:
            return np.nan

        ratio_ET = ET / (ET + QB)
        ratio_PET = PET / (ET + QB)

        def f(omega):
            return 1 + ratio_PET - (1 + ratio_PET ** omega) ** (1 / omega) - ratio_ET

        try:
            sol = fsolve(f, x0=omega_guess, xtol=1e-8)
            omega_val = sol[0] if np.isfinite(sol[0]) else np.nan
            return np.clip(omega_val, 0.0, 10.0) if np.isfinite(omega_val) else np.nan

        except:
            return np.nan

    # -----------------------------------------------------
    # Compute ω_true using calibrated Ke values
    # -----------------------------------------------------
    def compute_omega_true(self) -> pd.DataFrame:
        Evap_df = self.Evap_df
        Qsb_monthly = self.Qsb_monthly
        PotEvap_df = self.PotEvap_df

        omega_true = pd.DataFrame(index=Evap_df.index, columns=Evap_df.columns, dtype=float)

        for col in PotEvap_df.columns:
            if col not in self.calibrated_params or "Ke" not in self.calibrated_params[col]:
                continue

            ke_basin = self.calibrated_params[col]["Ke"]

            QB_col = Qsb_monthly[col].values
            PET_col = PotEvap_df[col].values

            ET_Ke = ke_basin * PET_col

            omega_values = np.array([
                self._solve_for_omega(ET, QB, PET)
                for ET, QB, PET in zip(ET_Ke, QB_col, PET_col)
            ])

            omega_true[col] = omega_values

        self.omega_true = omega_true
        return self.omega_true

    # -----------------------------------------------------
    # Fit ω_MLR using NDVI (M) and slope
    # -----------------------------------------------------
    def fit_and_compute_omega_mlr(self) -> pd.DataFrame:
        if self.omega_true is None:
            self.compute_omega_true()

        M = self.M_basin.values
        Slope = self.Slope_basin.values
        y = self.omega_true.values

        rows = []
        targets = []
        T, B = M.shape

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

        X = np.vstack(rows)
        Y = np.concatenate(targets)

        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

        Omega_hat = np.empty_like(M, dtype=float)
        for i in range(B):
            Omega_hat[:, i] = beta[0] + beta[1] * M[:, i] + beta[2] * Slope[:, i]

        Omega_hat = np.where(Omega_hat < 1.0, 1.0, Omega_hat)

        self.omega_MLR = pd.DataFrame(
            Omega_hat,
            index=self.M_basin.index,
            columns=self.M_basin.columns,
        )

        return self.omega_MLR

    # -----------------------------------------------------
    # Compute Budyko ET using ω_MLR
    # -----------------------------------------------------
    def estimate_budyko_et(self) -> pd.DataFrame:
        if self.omega_MLR is None:
            self.fit_and_compute_omega_mlr()

        Qb = self.Qsb_monthly
        PET_df = self.PotEvap_df
        omega_MLR = self.omega_MLR

        P_values = (self.Evap_df + Qb).values
        PET_values = PET_df.values

        aridity = np.divide(
            PET_values, P_values,
            out=np.zeros_like(PET_values),
            where=P_values > 0
        )

        with np.errstate(over="ignore", invalid="ignore"):
            omega_values = omega_MLR.values
            term = np.power(aridity, omega_values)
            E_ratio = 1.0 + aridity - np.power(1.0 + term, 1.0 / omega_values)

        E_ratio = np.nan_to_num(E_ratio, nan=0.0, posinf=0.0, neginf=0.0)

        ET_est = P_values * E_ratio
        ET_est = np.clip(ET_est, 0.0, np.minimum(PET_values, P_values))

        self.ET_B = pd.DataFrame(ET_est, index=PET_df.index, columns=PET_df.columns)
        return self.ET_B

    # -----------------------------------------------------
    # Full workflow pipeline
    # -----------------------------------------------------
    def OmegaTrue_OmegaMLR_BudykoET(self) -> pd.DataFrame:
        self.compute_omega_true()
        self.fit_and_compute_omega_mlr()
        self.estimate_budyko_et()
        print("Workflow complete.")
        return self.ET_B


# ---------------------------------------------------------
# Script entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    print("BudykoModelEstimator class defined. Run a separate script to execute the full workflow.")
