# src/budyko.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import fsolve
import warnings

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
        return np.clip(omega_MLR, 2.0, 5.0)


# ---------------------------------------------------------
# Budyko Model Estimator (STATE-AWARE)
# ---------------------------------------------------------
class BudykoModelEstimator:
    """
    State-aware Budyko implementation.

    Key change:
      - Budyko ratios use P_eff = P - dS (effective water available for partitioning)
      - dS is supplied externally (e.g., from two-store model states), so this file
        does NOT try to infer storage change.

    Supports BOTH Ke sources (for omega_true inversion target ET_ke):
      1) CALIBRATED MODE:
          - calibrated_params dict provided
          - uses calibrated_params[basin]["Ke"] (constant per basin)
      2) UNCALIBRATED MODE:
          - Ke_df provided (DataFrame aligned with PET)
          - uses Ke_df[basin] (time-varying)

    Inputs expected:
      - P_df: precipitation (same index/columns as PET)
      - dS_df: storage change (S_t - S_{t-1}) same index/columns
    """

    def __init__(
        self,
        P_df: pd.DataFrame,
        dS_df: pd.DataFrame,
        PotEvap_df: pd.DataFrame,
        M_basin: pd.DataFrame,
        Slope_basin: pd.DataFrame,
        calibrated_params: dict | None = None,
        Ke_df: pd.DataFrame | None = None,
        Peff_min: float = 1e-6,
    ):
        self.P_df = P_df
        self.dS_df = dS_df
        self.PotEvap_df = PotEvap_df
        self.M_basin = M_basin
        self.Slope_basin = Slope_basin

        self.calibrated_params = calibrated_params if calibrated_params is not None else {}
        self.Ke_df = Ke_df

        # lower bound for effective precipitation to avoid division-by-zero / negative
        self.Peff_min = float(Peff_min)

        self.omega_true = None
        self.omega_MLR = None
        self.ET_B = None

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------
    def _compute_peff(self, P: np.ndarray, dS: np.ndarray) -> np.ndarray:
        """
        Effective water for Budyko partitioning at monthly scale:
          P_eff = P - dS
        Clipped to Peff_min to avoid invalid ratios.
        """
        P = P.astype(float)
        dS = dS.astype(float)
        Peff = P - dS
        Peff = np.where(np.isfinite(Peff), Peff, np.nan)
        Peff = np.where(Peff > self.Peff_min, Peff, self.Peff_min)
        return Peff

    # -----------------------------------------------------
    # Solve for ω_true from Budyko using (ET_ke / P_eff) and (PET / P_eff)
    # -----------------------------------------------------
    def _solve_for_omega_from_ratios(self, ratio_ET: float, ratio_PET: float, omega_guess: float = 2.0):
        ratio_ET = float(ratio_ET) if np.isfinite(ratio_ET) else np.nan
        ratio_PET = float(ratio_PET) if np.isfinite(ratio_PET) else np.nan

        if not np.isfinite(ratio_ET) or not np.isfinite(ratio_PET):
            return np.nan

        # physical sanity (soft): ratio_ET in [0, 1] typically; keep robust
        if ratio_PET <= 0:
            return np.nan

        def f(omega):
            return 1.0 + ratio_PET - (1.0 + ratio_PET ** omega) ** (1.0 / omega) - ratio_ET

        try:
            sol = fsolve(f, x0=omega_guess, xtol=1e-8)
            omega_val = sol[0] if np.isfinite(sol[0]) else np.nan
            return np.clip(omega_val, 1.5, 5.0) if np.isfinite(omega_val) else np.nan
        except Exception:
            return np.nan

    # -----------------------------------------------------
    # Compute ω_true (state-aware): uses P_eff = P - dS
    # omega_true is inferred such that Budyko ET ratio matches ET_ke / P_eff
    # -----------------------------------------------------
    def compute_omega_true(self) -> pd.DataFrame:
        P_df = self.P_df
        dS_df = self.dS_df
        PET_df = self.PotEvap_df

        # align safely
        idx = PET_df.index
        cols = PET_df.columns

        omega_true = pd.DataFrame(index=idx, columns=cols, dtype=float)

        # precompute arrays
        P_all = P_df.reindex(idx)[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
        dS_all = dS_df.reindex(idx)[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
        PET_all = PET_df.reindex(idx)[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)

        Peff_all = self._compute_peff(P_all, dS_all)

        for j, basin in enumerate(cols):
            # Decide Ke source
            if self.Ke_df is not None:
                if basin not in self.Ke_df.columns:
                    continue
                Ke_series = pd.to_numeric(self.Ke_df.reindex(idx)[basin], errors="coerce").values.astype(float)
            elif basin in self.calibrated_params and "Ke" in self.calibrated_params[basin]:
                Ke_val = float(self.calibrated_params[basin]["Ke"])
                Ke_series = np.full(len(idx), Ke_val, dtype=float)
            else:
                continue

            PET_col = PET_all[:, j]
            Peff_col = Peff_all[:, j]

            ET_ke = Ke_series * PET_col

            ratio_ET = np.divide(
                ET_ke, Peff_col,
                out=np.full_like(ET_ke, np.nan, dtype=float),
                where=np.isfinite(ET_ke) & np.isfinite(Peff_col) & (Peff_col > 0)
            )
            ratio_PET = np.divide(
                PET_col, Peff_col,
                out=np.full_like(PET_col, np.nan, dtype=float),
                where=np.isfinite(PET_col) & np.isfinite(Peff_col) & (Peff_col > 0)
            )

            omega_vals = np.array(
                [self._solve_for_omega_from_ratios(re, rp) for re, rp in zip(ratio_ET, ratio_PET)],
                dtype=float
            )

            omega_true[basin] = omega_vals

        self.omega_true = omega_true
        return self.omega_true

    # -----------------------------------------------------
    # Fit ω_MLR using M and slope (same logic, now target is state-aware omega_true)
    # -----------------------------------------------------
    def fit_and_compute_omega_mlr(self) -> pd.DataFrame:
        if self.omega_true is None:
            self.compute_omega_true()

        # Ensure numeric and aligned
        idx = self.omega_true.index
        cols = self.omega_true.columns

        M = self.M_basin.reindex(idx)[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
        Slope = self.Slope_basin.reindex(idx)[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
        y = self.omega_true.reindex(idx)[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)

        rows = []
        targets = []
        T, B = M.shape

        for i in range(B):
            Mi = M[:, i]
            Si = Slope[:, i]
            yi = y[:, i]

            mask = np.isfinite(Mi) & np.isfinite(Si) & np.isfinite(yi)
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
        Omega_hat = np.where(Omega_hat > 50.0, 50.0, Omega_hat)

        self.omega_MLR = pd.DataFrame(Omega_hat, index=idx, columns=cols)
        return self.omega_MLR

    # -----------------------------------------------------
    # Compute Budyko ET using ω_MLR and P_eff = P - dS (STATE-AWARE)
    # -----------------------------------------------------
    def estimate_budyko_et(self) -> pd.DataFrame:
        if self.omega_true is None:
            self.compute_omega_true()

        if self.omega_MLR is None:
            try:
                self.fit_and_compute_omega_mlr()
            except Exception:
                self.omega_MLR = pd.DataFrame(
                    index=self.omega_true.index,
                    columns=self.omega_true.columns,
                    data=np.nan,
                )

        idx = self.PotEvap_df.index
        cols = self.PotEvap_df.columns

        P = self.P_df.reindex(idx)[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
        dS = self.dS_df.reindex(idx)[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
        PET = self.PotEvap_df.reindex(idx)[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
        omega = self.omega_MLR.reindex(idx)[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)

        Peff = self._compute_peff(P, dS)

        # aridity = PET / Peff
        aridity = np.divide(
            PET, Peff,
            out=np.full_like(PET, np.nan, dtype=float),
            where=np.isfinite(PET) & np.isfinite(Peff) & (Peff > 0)
        )

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            term = np.power(aridity, omega)
            E_ratio = 1.0 + aridity - np.power(1.0 + term, 1.0 / omega)

        E_ratio = np.where(np.isfinite(E_ratio), E_ratio, np.nan)

        ET_est = Peff * E_ratio

        # physical bounds: 0 <= ET <= min(Peff, PET) (componentwise)
        ET_cap = np.nanmin(np.stack([Peff, PET]), axis=0)
        ET_est = np.where(np.isfinite(ET_est), ET_est, np.nan)
        ET_est = np.clip(ET_est, 0.0, ET_cap)

        self.ET_B = pd.DataFrame(ET_est, index=idx, columns=cols)
        return self.ET_B




























# # src/budyko.py
# import numpy as np
# import pandas as pd
# from dataclasses import dataclass
# from scipy.optimize import fsolve
# import warnings

# warnings.filterwarnings("ignore", category=RuntimeWarning)


# # ---------------------------------------------------------
# # Ω via Multiple Linear Regression
# # ---------------------------------------------------------
# @dataclass
# class OmegaMLRModel:
#     beta0: float
#     beta1: float
#     beta2: float

#     def predict(self, M: np.ndarray, Slope: np.ndarray) -> np.ndarray:
#         omega_MLR = self.beta0 + self.beta1 * M + self.beta2 * Slope
#         return np.clip(omega_MLR, 1.0, 50.0)


# # ---------------------------------------------------------
# # Budyko Model Estimator
# # ---------------------------------------------------------
# class BudykoModelEstimator:
#     """
#     Supports BOTH modes:

#     1) CALIBRATED MODE:
#         - calibrated_params dict provided
#         - uses calibrated_params[basin]["Ke"]

#     2) UNCALIBRATED MODE:
#         - Ke_df provided (DataFrame with same index/cols as PET)
#         - uses Ke_df[basin]
#     """

#     def __init__(
#         self,
#         Evap_df: pd.DataFrame,
#         Qsb_monthly: pd.DataFrame,
#         PotEvap_df: pd.DataFrame,
#         M_basin: pd.DataFrame,
#         Slope_basin: pd.DataFrame,
#         calibrated_params: dict | None = None,
#         Ke_df: pd.DataFrame | None = None,
#     ):
#         self.Evap_df = Evap_df
#         self.Qsb_monthly = Qsb_monthly
#         self.PotEvap_df = PotEvap_df
#         self.M_basin = M_basin
#         self.Slope_basin = Slope_basin

#         self.calibrated_params = calibrated_params if calibrated_params is not None else {}
#         self.Ke_df = Ke_df  # ✅ new for UNCALIBRATED mode

#         self.omega_true = None
#         self.omega_MLR = None
#         self.ET_B = None

#     # -----------------------------------------------------
#     # Solve for ω_true
#     # -----------------------------------------------------
#     def _solve_for_omega(self, ET, QB, PET, omega_guess=2.0):
#         # make sure we are working with floats
#         ET = float(ET) if np.isfinite(ET) else np.nan
#         QB = float(QB) if np.isfinite(QB) else np.nan
#         PET = float(PET) if np.isfinite(PET) else np.nan

#         if not np.isfinite(ET) or not np.isfinite(QB) or not np.isfinite(PET):
#             return np.nan

#         denom = ET + QB
#         if denom <= 0:
#             return np.nan

#         ratio_ET = ET / denom
#         ratio_PET = PET / denom

#         def f(omega):
#             return 1 + ratio_PET - (1 + ratio_PET ** omega) ** (1 / omega) - ratio_ET

#         try:
#             sol = fsolve(f, x0=omega_guess, xtol=1e-8)
#             omega_val = sol[0] if np.isfinite(sol[0]) else np.nan
#             return np.clip(omega_val, 0.1, 10.0) if np.isfinite(omega_val) else np.nan
#         except Exception:
#             return np.nan

#     # -----------------------------------------------------
#     # Compute ω_true using Ke source (calibrated OR provided Ke_df)
#     # -----------------------------------------------------
#     def compute_omega_true(self) -> pd.DataFrame:
#         Evap_df = self.Evap_df
#         Qsb_monthly = self.Qsb_monthly
#         PotEvap_df = self.PotEvap_df

#         omega_true = pd.DataFrame(index=Evap_df.index, columns=Evap_df.columns, dtype=float)

#         for basin in PotEvap_df.columns:

#             # -------------------------------------------------
#             # ✅ Decide where Ke comes from
#             # -------------------------------------------------
#             if self.Ke_df is not None:
#                 if basin not in self.Ke_df.columns:
#                     continue
#                 Ke_series = pd.to_numeric(self.Ke_df[basin], errors="coerce").values

#             elif basin in self.calibrated_params and "Ke" in self.calibrated_params[basin]:
#                 Ke_val = float(self.calibrated_params[basin]["Ke"])
#                 Ke_series = np.full(len(PotEvap_df.index), Ke_val, dtype=float)

#             else:
#                 # no Ke available -> skip basin
#                 continue

#             QB_col = pd.to_numeric(Qsb_monthly[basin], errors="coerce").values
#             PET_col = pd.to_numeric(PotEvap_df[basin], errors="coerce").values

#             ET_ke = Ke_series * PET_col

#             omega_values = np.array([
#                 self._solve_for_omega(ET, QB, PET)
#                 for ET, QB, PET in zip(ET_ke, QB_col, PET_col)
#             ])

#             omega_true[basin] = omega_values

#         self.omega_true = omega_true
#         return self.omega_true

#     # -----------------------------------------------------
#     # Fit ω_MLR using M and slope
#     # -----------------------------------------------------
#     def fit_and_compute_omega_mlr(self) -> pd.DataFrame:
#         if self.omega_true is None:
#             self.compute_omega_true()

#         # ✅ Ensure numeric
#         M = self.M_basin.apply(pd.to_numeric, errors="coerce").values.astype(float)
#         Slope = self.Slope_basin.apply(pd.to_numeric, errors="coerce").values.astype(float)
#         y = self.omega_true.apply(pd.to_numeric, errors="coerce").values.astype(float)

#         rows = []
#         targets = []
#         T, B = M.shape

#         for i in range(B):
#             Mi = M[:, i]
#             Si = Slope[:, i]
#             yi = y[:, i]

#             mask = np.isfinite(Mi) & np.isfinite(Si) & np.isfinite(yi)
#             if mask.any():
#                 Xi = np.column_stack([np.ones(mask.sum()), Mi[mask], Si[mask]])
#                 rows.append(Xi)
#                 targets.append(yi[mask])

#         if not rows:
#             raise ValueError("No valid data to fit omega MLR.")

#         X = np.vstack(rows)
#         Y = np.concatenate(targets)

#         beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

#         Omega_hat = np.empty_like(M, dtype=float)
#         for i in range(B):
#             Omega_hat[:, i] = beta[0] + beta[1] * M[:, i] + beta[2] * Slope[:, i]

#         Omega_hat = np.where(Omega_hat < 1.0, 1.0, Omega_hat)

#         self.omega_MLR = pd.DataFrame(
#             Omega_hat,
#             index=self.M_basin.index,
#             columns=self.M_basin.columns,
#         )

#         return self.omega_MLR

#     # -----------------------------------------------------
#     # Compute Budyko ET using ω_MLR
#     # -----------------------------------------------------
#     def estimate_budyko_et(self) -> pd.DataFrame:
#         if self.omega_true is None:
#             self.compute_omega_true()

#         # ---------------------------------------------
#         # ✅ fit omega_MLR, but don't crash
#         # ---------------------------------------------
#         if self.omega_MLR is None:
#             try:
#                 self.fit_and_compute_omega_mlr()
#             except Exception:
#                 # if MLR fails -> set omega_MLR NaN and still compute ET_B robustly
#                 self.omega_MLR = pd.DataFrame(
#                     index=self.omega_true.index,
#                     columns=self.omega_true.columns,
#                     data=np.nan,
#                 )

#         Qb = self.Qsb_monthly.apply(pd.to_numeric, errors="coerce")
#         PET_df = self.PotEvap_df.apply(pd.to_numeric, errors="coerce")
#         omega_MLR = self.omega_MLR.apply(pd.to_numeric, errors="coerce")

#         P_values = (self.Evap_df.apply(pd.to_numeric, errors="coerce") + Qb).values.astype(float)
#         PET_values = PET_df.values.astype(float)
#         omega_values = omega_MLR.values.astype(float)

#         aridity = np.divide(
#             PET_values, P_values,
#             out=np.full_like(PET_values, np.nan),
#             where=P_values > 0
#         )

#         with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
#             term = np.power(aridity, omega_values)
#             E_ratio = 1.0 + aridity - np.power(1.0 + term, 1.0 / omega_values)

#         E_ratio = np.where(np.isfinite(E_ratio), E_ratio, np.nan)

#         ET_est = P_values * E_ratio

#         # clip to physical bounds
#         ET_est = np.where(np.isfinite(ET_est), ET_est, np.nan)
#         ET_est = np.clip(ET_est, 0.0, np.nanmin(np.stack([PET_values, P_values]), axis=0))

#         self.ET_B = pd.DataFrame(ET_est, index=PET_df.index, columns=PET_df.columns)
#         return self.ET_B












# # src/budyko.py
# import numpy as np
# import pandas as pd
# from dataclasses import dataclass
# from scipy.optimize import fsolve
# import warnings

# # Suppress runtime warnings that often occur with complex numerical operations
# warnings.filterwarnings("ignore", category=RuntimeWarning)


# # ---------------------------------------------------------
# # Ω via Multiple Linear Regression
# # ---------------------------------------------------------
# @dataclass
# class OmegaMLRModel:
#     beta0: float
#     beta1: float
#     beta2: float

#     def predict(self, M: np.ndarray, Slope: np.ndarray) -> np.ndarray:
#         omega_MLR = self.beta0 + self.beta1 * M + self.beta2 * Slope
#         return np.clip(omega_MLR, 3.0, 20.0)


# # ---------------------------------------------------------
# # Budyko Model Estimator
# # ---------------------------------------------------------
# class BudykoModelEstimator:
#     def __init__(
#         self,
#         Evap_df: pd.DataFrame,
#         Qsb_monthly: pd.DataFrame,
#         PotEvap_df: pd.DataFrame,
#         M_basin: pd.DataFrame,
#         Slope_basin: pd.DataFrame,
#         calibrated_params: dict,
#     ):
#         self.Evap_df = Evap_df
#         self.Qsb_monthly = Qsb_monthly
#         self.PotEvap_df = PotEvap_df
#         self.M_basin = M_basin
#         self.Slope_basin = Slope_basin
#         self.calibrated_params = calibrated_params

#         self.omega_true = None
#         self.omega_MLR = None
#         self.ET_B = None

#     # -----------------------------------------------------
#     # Solve for ω_true
#     # -----------------------------------------------------
#     def _solve_for_omega(self, ET, QB, PET, omega_guess=1.0):
#         if (ET + QB) == 0:
#             return np.nan

#         ratio_ET = ET / (ET + QB)
#         ratio_PET = PET / (ET + QB)

#         def f(omega):
#             return 1 + ratio_PET - (1 + ratio_PET ** omega) ** (1 / omega) - ratio_ET

#         try:
#             sol = fsolve(f, x0=omega_guess, xtol=1e-8)
#             omega_val = sol[0] if np.isfinite(sol[0]) else np.nan
#             return np.clip(omega_val, 0.0, 10.0) if np.isfinite(omega_val) else np.nan

#         except:
#             return np.nan

#     # -----------------------------------------------------
#     # Compute ω_true using calibrated Ke values
#     # -----------------------------------------------------
#     def compute_omega_true(self) -> pd.DataFrame:
#         Evap_df = self.Evap_df
#         Qsb_monthly = self.Qsb_monthly
#         PotEvap_df = self.PotEvap_df

#         omega_true = pd.DataFrame(index=Evap_df.index, columns=Evap_df.columns, dtype=float)

#         for col in PotEvap_df.columns:
#             if col not in self.calibrated_params or "Ke" not in self.calibrated_params[col]:
#                 continue

#             ke_basin = self.calibrated_params[col]["Ke"]

#             QB_col = Qsb_monthly[col].values
#             PET_col = PotEvap_df[col].values

#             ET_Ke = ke_basin * PET_col

#             omega_values = np.array([
#                 self._solve_for_omega(ET, QB, PET)
#                 for ET, QB, PET in zip(ET_Ke, QB_col, PET_col)
#             ])

#             omega_true[col] = omega_values

#         self.omega_true = omega_true
#         return self.omega_true

#     # -----------------------------------------------------
#     # Fit ω_MLR using NDVI (M) and slope
#     # -----------------------------------------------------
#     def fit_and_compute_omega_mlr(self) -> pd.DataFrame:
#         if self.omega_true is None:
#             self.compute_omega_true()

#         M = self.M_basin.values
#         Slope = self.Slope_basin.values
#         y = self.omega_true.values

#         rows = []
#         targets = []
#         T, B = M.shape

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

#         X = np.vstack(rows)
#         Y = np.concatenate(targets)

#         beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

#         Omega_hat = np.empty_like(M, dtype=float)
#         for i in range(B):
#             Omega_hat[:, i] = beta[0] + beta[1] * M[:, i] + beta[2] * Slope[:, i]

#         Omega_hat = np.where(Omega_hat < 1.0, 1.0, Omega_hat)

#         self.omega_MLR = pd.DataFrame(
#             Omega_hat,
#             index=self.M_basin.index,
#             columns=self.M_basin.columns,
#         )

#         return self.omega_MLR

#     # -----------------------------------------------------
#     # Compute Budyko ET using ω_MLR
#     # -----------------------------------------------------
#     def estimate_budyko_et(self) -> pd.DataFrame:
#         if self.omega_MLR is None:
#             self.fit_and_compute_omega_mlr()

#         Qb = self.Qsb_monthly
#         PET_df = self.PotEvap_df
#         omega_MLR = self.omega_MLR

#         P_values = (self.Evap_df + Qb).values
#         PET_values = PET_df.values

#         aridity = np.divide(
#             PET_values, P_values,
#             out=np.zeros_like(PET_values),
#             where=P_values > 0
#         )

#         with np.errstate(over="ignore", invalid="ignore"):
#             omega_values = omega_MLR.values
#             term = np.power(aridity, omega_values)
#             E_ratio = 1.0 + aridity - np.power(1.0 + term, 1.0 / omega_values)

#         E_ratio = np.nan_to_num(E_ratio, nan=0.0, posinf=0.0, neginf=0.0)

#         ET_est = P_values * E_ratio
#         ET_est = np.clip(ET_est, 0.0, np.minimum(PET_values, P_values))

#         self.ET_B = pd.DataFrame(ET_est, index=PET_df.index, columns=PET_df.columns)
#         return self.ET_B

#     # -----------------------------------------------------
#     # Full workflow pipeline
#     # -----------------------------------------------------
#     def OmegaTrue_OmegaMLR_BudykoET(self) -> pd.DataFrame:
#         self.compute_omega_true()
#         self.fit_and_compute_omega_mlr()
#         self.estimate_budyko_et()
#         print("Workflow complete.")
#         return self.ET_B


# # ---------------------------------------------------------
# # Script entry point
# # ---------------------------------------------------------
# if __name__ == "__main__":
#     print("BudykoModelEstimator class defined. Run a separate script to execute the full workflow.")
