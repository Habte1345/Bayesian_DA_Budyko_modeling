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
#         if self.omega_true is None:
#             self.compute_omega_true()

#         M = self.M_basin.values
#         Slope = self.Slope_basin.values
#         omega_true = self.omega_true.values

#         if Slope.ndim == 2 and Slope.shape[0] == 1:
#             Slope_flat = Slope.flatten()
#         elif Slope.ndim == 1:
#             Slope_flat = Slope
#         else:
#             raise ValueError(f"Slope shape not recognized: {Slope.shape}")

#         n_time, n_basins = M.shape
#         omega_fitted = np.full_like(M, np.nan, dtype=float)

#         for i in range(n_basins):
#             M_col = M[:, i]
#             Slope_val = Slope_flat[i]
#             Slope_col = np.full(n_time, Slope_val)
#             y_true = omega_true[:, i]
#             valid_idx = ~(np.isnan(M_col) | np.isnan(Slope_col) | np.isnan(y_true))

#             if valid_idx.sum() >= 3:
#                 X = np.vstack([np.ones(valid_idx.sum()), M_col[valid_idx], Slope_col[valid_idx]]).T
#                 y = y_true[valid_idx]
#                 try:
#                     coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
#                     omega_fitted[:, i] = coef[0] + coef[1] * M_col + coef[2] * Slope_col
#                 except np.linalg.LinAlgError:
#                     omega_fitted[:, i] = np.nan
#             else:
#                 omega_fitted[:, i] = np.nan

#         omega_fitted_clipped = np.clip(omega_fitted, 1.0, 10.0)
#         self.omega_MLR = pd.DataFrame(omega_fitted_clipped,
#                                       index=self.M_basin.index,
#                                       columns=self.M_basin.columns)
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


# src/budyko.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------
# Fu (1981) Budyko helper
# -----------------------------
def fu_et_ratio(phi, omega):
    """
    Fu equation in ratio form: E/(P-ΔS) = 1 + φ - [1 + φ^ω]^(1/ω)
    where φ = PET / (P-ΔS).
    Supports broadcasting across arrays.
    """
    with np.errstate(over='ignore', invalid='ignore'):
        term = np.power(1.0 + np.power(phi, omega), 1.0 / omega)
        er = 1.0 + phi - term
    # clean-up
    er = np.where(np.isfinite(er), er, np.nan)
    return er


@dataclass
class OmegaMLRModel:
    beta0: float
    beta1: float
    beta2: float

    def predict(self, M: np.ndarray, Slope: np.ndarray) -> np.ndarray:
        omega = self.beta0 + self.beta1 * M + self.beta2 * Slope
        return np.clip(omega, 1.0, 10.0)


class BudykoModelEstimator:
    """
    Robust, alignment-safe estimator for:
      - omega_true (diagnosed from ET_ke, Qb, PET)
      - omega_MLR (fitted from M and Slope against omega_true)
      - ET_B (computed using omega_MLR via Fu)
    All inputs are hard-aligned to Evap_df.index; slope is broadcast to time.
    """

    def __init__(self,
                 Evap_df: pd.DataFrame,            # ET_nldas (used for index & P-ΔS proxy with Qb)
                 Qsb_monthly: pd.DataFrame,        # baseflow Qb
                 PotEvap_df: pd.DataFrame,         # PET
                 M_basin: pd.DataFrame,            # NDVI-derived M
                 Slope_basin: pd.DataFrame,        # 1×N slope
                 ke: float = 0.68):
        # Keep references
        self.Evap_df_raw = Evap_df.copy()
        self.Qsb_raw = Qsb_monthly.copy()
        self.PET_raw = PotEvap_df.copy()
        self.M_raw = M_basin.copy()
        self.Slope_raw = Slope_basin.copy()
        self.ke = ke

        # Aligned data (filled later)
        self.Evap_df = None
        self.Qsb_monthly = None
        self.PotEvap_df = None
        self.M_basin = None
        self.Slope_basin = None

        # Outputs
        self.omega_true: pd.DataFrame | None = None
        self.omega_MLR: pd.DataFrame | None = None
        self.ET_B: pd.DataFrame | None = None

        self._align_inputs()

    # -----------------------------
    # Alignment / broadcasting
    # -----------------------------
    def _align_inputs(self):
        idx = self.Evap_df_raw.index

        def reindex_df(df):
            # Keep only common columns
            cols = [c for c in df.columns if c in self.Evap_df_raw.columns]
            return df[cols].reindex(idx)

        # Align all by time, intersect columns with Evap_df
        self.Evap_df = self.Evap_df_raw.copy()
        self.Qsb_monthly = reindex_df(self.Qsb_raw)
        self.PotEvap_df = reindex_df(self.PET_raw)
        self.M_basin = reindex_df(self.M_raw)

        # Slope: 1×N -> broadcast to T×N on aligned columns
        slope_cols = [c for c in self.Slope_raw.columns if c in self.Evap_df.columns]
        slope_row = self.Slope_raw[slope_cols].iloc[0]  # 1×N
        slope_broadcast = pd.DataFrame(
            np.tile(slope_row.values, (len(idx), 1)),
            index=idx, columns=slope_cols
        )
        self.Slope_basin = slope_broadcast

        # Ensure numeric and drop columns that are entirely NaN after alignment
        for name in ["Evap_df", "Qsb_monthly", "PotEvap_df", "M_basin", "Slope_basin"]:
            df = getattr(self, name).apply(pd.to_numeric, errors='coerce')
            all_nan_cols = df.columns[df.isna().all()]
            if len(all_nan_cols) > 0:
                df = df.drop(columns=all_nan_cols)
            setattr(self, name, df)

    # -----------------------------
    # Robust ω inversion (grid search)
    # -----------------------------
    def _invert_omega_column(self, ET_ke_col, Qb_col, PET_col):
        """
        Invert Fu to get omega_true for one basin (vectorized in time).
        Uses grid search over ω in [1.01, 100] minimizing |Fu(φ, ω) - ET_ratio|.
        Returns a 1-D array length T.
        """
        T = len(ET_ke_col)
        out = np.full(T, np.nan, dtype=float)

        # P-ΔS proxy = ET + Qb (per your formulation)
        PmS = ET_ke_col + Qb_col

        # Valid mask
        valid = (PmS > 0) & np.isfinite(PmS) & np.isfinite(PET_col) & np.isfinite(ET_ke_col)
        if not np.any(valid):
            return out

        # Ratios
        ET_ratio = np.clip(ET_ke_col[valid] / PmS[valid], 0.0, 0.999999)
        phi = np.clip(PET_col[valid] / PmS[valid], 0.0, 100.0)  # aridity

        # ω grid
        # (Dense at low ω where sensitivity is high; sparser at high ω)
        w1 = np.linspace(1.01, 5.0, 400)
        w2 = np.linspace(5.0, 20.0, 200)
        w3 = np.linspace(20.0, 100.0, 80)
        omega_grid = np.concatenate([w1, w2, w3])  # length ~680

        # Evaluate Fu(φ, ω) across grid for each time (broadcast)
        # phi: (Nv,), omega_grid: (K,) -> Fu: (Nv, K)
        Fu_vals = fu_et_ratio(phi[:, None], omega_grid[None, :])
        # Choose ω minimizing absolute error to target ET ratio
        err = np.abs(Fu_vals - ET_ratio[:, None])
        idx_min = np.nanargmin(err, axis=1)
        omega_hat = omega_grid[idx_min]

        out[valid] = omega_hat
        return out

    # -----------------------------
    # Public steps
    # -----------------------------
    def compute_omega_true(self) -> pd.DataFrame:
        idx = self.Evap_df.index
        cols = self.PotEvap_df.columns  # after alignment, subset of Evap_df

        omega_true = pd.DataFrame(index=idx, columns=cols, dtype=float)

        for col in cols:
            PET_col = self.PotEvap_df[col].values
            Qb_col = self.Qsb_monthly[col].values
            ET_ke_col = self.ke * PET_col  # ET_ke = ke * PET

            # Invert ω robustly
            omega_vec = self._invert_omega_column(ET_ke_col, Qb_col, PET_col)
            omega_true[col] = omega_vec

        self.omega_true = omega_true
        return self.omega_true

    def fit_and_compute_omega_mlr(self) -> pd.DataFrame:
        if self.omega_true is None:
            self.compute_omega_true()

        # Shapes: (T, N)
        M = self.M_basin.values
        Slope = self.Slope_basin.values
        omega_true = self.omega_true.reindex(self.M_basin.index)[self.M_basin.columns].values

        T, N = M.shape
        omega_fitted = np.full((T, N), np.nan, dtype=float)

        for i in range(N):
            y = omega_true[:, i]
            xM = M[:, i]
            xS = Slope[:, i]
            valid = np.isfinite(y) & np.isfinite(xM) & np.isfinite(xS)

            if valid.sum() >= 6:
                X = np.column_stack([np.ones(valid.sum()), xM[valid], xS[valid]])
                # Ordinary least squares
                coef, _, _, _ = np.linalg.lstsq(X, y[valid], rcond=None)
                # Predict for all time
                omega_pred = coef[0] + coef[1] * xM + coef[2] * xS
                omega_fitted[:, i] = omega_pred
            else:
                # Not enough valid points to fit; leave NaN
                continue

        omega_fitted = np.clip(omega_fitted, 1.0, 10.0)
        self.omega_MLR = pd.DataFrame(omega_fitted, index=self.M_basin.index, columns=self.M_basin.columns)
        return self.omega_MLR

    def estimate_budyko_et(self) -> pd.DataFrame:
        if self.omega_MLR is None:
            self.fit_and_compute_omega_mlr()

        # P-ΔS proxy = ET + Qb; using ET from Evap_df (observed), consistent with your formulation
        PmS = (self.Evap_df + self.Qsb_monthly)
        PmS = PmS.where(PmS > 0)

        PET = self.PotEvap_df.reindex_like(PmS)
        omega = self.omega_MLR.reindex_like(PmS)

        phi = (PET / PmS).clip(lower=0.0, upper=100.0)
        E_ratio = fu_et_ratio(phi.values, omega.values)

        # ET_B = (P-ΔS) * E_ratio, with physical clipping
        ET_est = PmS.values * E_ratio
        ET_est = np.clip(ET_est, 0.0, np.minimum(PET.values, PmS.values))

        self.ET_B = pd.DataFrame(ET_est, index=PmS.index, columns=PmS.columns)
        return self.ET_B

    def OmegaTrue_OmegaMLR_BudykoET(self) -> pd.DataFrame:
        self.compute_omega_true()
        self.fit_and_compute_omega_mlr()
        self.estimate_budyko_et()
        return self.ET_B


if __name__ == '__main__':
    print("Robust BudykoModelEstimator ready.")
