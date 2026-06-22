# src/budyko.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import brentq
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
        return self.beta0 + self.beta1 * M + self.beta2 * Slope


# ---------------------------------------------------------
# Budyko Model Estimator (STATE-AWARE)
# ---------------------------------------------------------
class BudykoModelEstimator:
    """
    State-aware Budyko implementation.

    Fixes applied vs previous version
    -----------------------------------
    FIX 1  _compute_peff: P_eff now capped at P from above.
           When dS < 0 (soil draining), P - dS > P, inflating the aridity
           index φ = PET/P_eff and causing ET_B to explode.
           Physical constraint: monthly ET partitioning cannot draw on
           more water than fell as rainfall.

    FIX 2  _solve_for_omega_from_ratios: ratio_ET clipped to [0, min(1, ratio_PET)]
           before inversion.  ET_ke = Ke×PET does not respect the water-balance
           constraint ET ≤ P_eff, so ratio_ET often exceeds 1 in dry months,
           making the Fu equation have no valid root.  Clipping prevents the
           inversion from always falling back to omega_min.

    FIX 3  _solve_for_omega_from_ratios: fallback now returns np.nan instead of
           omega_min.  Returning 1.01 for every failed inversion was polluting
           the MLR training data with a degenerate lower-bound value, biasing
           the intercept β₀ toward 1.01 and destroying the MLR's ability to
           predict meaningful omega variation.

    FIX 4  estimate_budyko_et: now uses omega_MLR (the attribute-informed
           prediction) instead of omega_true clipped to ±1% of 2.2.
           The previous code completely bypassed the MLR, making the Budyko
           and BASE scenarios identical in practice.

    FIX 5  omega inversion range widened to [1.01, 20.0] and MLR clip widened
           to [1.0, 20.0].  Literature values for CAMELS basins range from
           ~1.5 (arid) to ~6–8 (dense forest).  Capping at 5.0 was truncating
           humid forested basins and biasing the MLR.
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
        self.P_df             = P_df
        self.dS_df            = dS_df
        self.PotEvap_df       = PotEvap_df
        self.M_basin          = M_basin
        self.Slope_basin      = Slope_basin
        self.calibrated_params = calibrated_params if calibrated_params is not None else {}
        self.Ke_df            = Ke_df
        self.Peff_min         = float(Peff_min)

        self.omega_true = None
        self.omega_MLR  = None
        self.ET_B       = None

    # ------------------------------------------------------------------
    # FIX 1 — P_eff bounded above by P
    # ------------------------------------------------------------------
    def _compute_peff(self, P: np.ndarray, dS: np.ndarray) -> np.ndarray:
        """
        Effective water available for monthly Budyko ET partitioning:
            P_eff = P - dS
        Bounded below by Peff_min (avoid division-by-zero).
        Bounded above by P (FIX 1): when dS < 0, P - dS > P, which
        inflates φ = PET/P_eff and causes ET_B to explode.
        """
        P   = np.asarray(P,  dtype=float)
        dS  = np.asarray(dS, dtype=float)
        Peff = P - dS
        Peff = np.where(np.isfinite(Peff), Peff, np.nan)
        # FIX 1: upper bound
        P_safe = np.where(np.isfinite(P) & (P > self.Peff_min), P, self.Peff_min)
        Peff   = np.clip(Peff, self.Peff_min, P_safe)
        return Peff

    # ------------------------------------------------------------------
    # FIX 2 + FIX 3 — robust omega inversion
    # ------------------------------------------------------------------
    def _solve_for_omega_from_ratios(
        self,
        ratio_ET: float,
        ratio_PET: float,
        omega_min: float = 1.01,
        omega_max: float = 20.0,       # FIX 5: widened from 5.0
    ) -> float:
        """
        Invert the Fu–Budyko equation to find ω such that:
            1 + φ - (1 + φ^ω)^(1/ω) = ratio_ET
        where φ = ratio_PET = PET / P_eff.

        Returns np.nan when inversion is not possible (FIX 3).
        """
        ratio_ET  = float(ratio_ET)  if np.isfinite(ratio_ET)  else np.nan
        ratio_PET = float(ratio_PET) if np.isfinite(ratio_PET) else np.nan

        if not np.isfinite(ratio_ET) or not np.isfinite(ratio_PET):
            return np.nan
        if ratio_PET <= 0:
            return np.nan

        # FIX 2: clip ratio_ET to physically admissible range.
        # ET cannot exceed min(P_eff, PET), so ET/P_eff ≤ min(1, PET/P_eff).
        # ET_ke = Ke×PET can violate this in dry months, making no valid
        # root exist and causing the fallback to fire on every such step.
        ratio_ET = float(np.clip(ratio_ET, 0.0, min(1.0, ratio_PET)))

        def f(omega):
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    val = (1.0 + ratio_PET
                           - (1.0 + ratio_PET ** omega) ** (1.0 / omega)
                           - ratio_ET)
                return float(val) if np.isfinite(val) else np.nan
            except Exception:
                return np.nan

        try:
            f_min = f(omega_min)
            f_max = f(omega_max)

            if not np.isfinite(f_min) or not np.isfinite(f_max):
                return np.nan  # FIX 3: return nan, not omega_min

            if f_min * f_max < 0:
                # Valid root exists — use Brent's method
                return float(brentq(f, omega_min, omega_max, xtol=1e-8))

            # No root in [omega_min, omega_max].
            # FIX 3: return nan so failed inversions are excluded from MLR.
            # (Previously returned omega_min=1.01, polluting the training data.)
            return np.nan

        except Exception:
            return np.nan

    # ------------------------------------------------------------------
    # Compute ω_true  (state-aware, uses P_eff = P - dS)
    # ------------------------------------------------------------------
    def compute_omega_true(self) -> pd.DataFrame:
        """
        Analytically invert ω_true such that the Fu–Budyko equation
        reproduces ET_ke / P_eff given the aridity index PET / P_eff.

        Only timesteps where the inversion succeeds are returned as finite
        values; failed timesteps are NaN and excluded from MLR fitting.
        """
        P_df   = self.P_df
        dS_df  = self.dS_df
        PET_df = self.PotEvap_df

        idx  = PET_df.index
        cols = PET_df.columns

        omega_true = pd.DataFrame(index=idx, columns=cols, dtype=float)

        P_all   = P_df.reindex(idx)[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
        dS_all  = dS_df.reindex(idx)[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
        PET_all = PET_df.reindex(idx)[cols].apply(pd.to_numeric, errors="coerce").values.astype(float)

        Peff_all = self._compute_peff(P_all, dS_all)   # uses FIX 1

        for j, basin in enumerate(cols):

            # ── Ke source ────────────────────────────────────────────
            if self.Ke_df is not None:
                if basin not in self.Ke_df.columns:
                    continue
                Ke_series = (pd.to_numeric(self.Ke_df.reindex(idx)[basin], errors="coerce")
                             .values.astype(float))
            elif basin in self.calibrated_params and "Ke" in self.calibrated_params[basin]:
                Ke_val    = float(self.calibrated_params[basin]["Ke"])
                Ke_series = np.full(len(idx), Ke_val, dtype=float)
            else:
                continue

            PET_col  = PET_all[:, j]
            Peff_col = Peff_all[:, j]
            ET_ke    = Ke_series * PET_col

            # ratio_ET and ratio_PET (NaN where denominator is invalid)
            valid = np.isfinite(ET_ke) & np.isfinite(Peff_col) & (Peff_col > 0)

            ratio_ET = np.where(valid,
                                ET_ke   / Peff_col, np.nan).astype(float)
            ratio_PET = np.where(valid,
                                 PET_col / Peff_col, np.nan).astype(float)

            # Invert ω for each timestep (FIX 2 + FIX 3 applied inside)
            omega_vals = np.array(
                [self._solve_for_omega_from_ratios(re, rp)
                 for re, rp in zip(ratio_ET, ratio_PET)],
                dtype=float
            )

            omega_true[basin] = omega_vals

        self.omega_true = omega_true
        return self.omega_true

    # ------------------------------------------------------------------
    # Fit ω_MLR from basin attributes
    # ------------------------------------------------------------------
    def fit_and_compute_omega_mlr(self) -> pd.DataFrame:
        """
        Pool OLS across all basins and time steps:
            ω_true = β₀ + β₁·M(t) + β₂·slope

        Only finite ω_true values enter the regression (NaN timesteps
        from failed inversions are automatically excluded — FIX 3 benefit).
        """
        if self.omega_true is None:
            self.compute_omega_true()

        idx  = self.omega_true.index
        cols = self.omega_true.columns

        M     = (self.M_basin.reindex(idx)[cols]
                 .apply(pd.to_numeric, errors="coerce").values.astype(float))
        Slope = (self.Slope_basin.reindex(idx)[cols]
                 .apply(pd.to_numeric, errors="coerce").values.astype(float))
        y     = (self.omega_true.reindex(idx)[cols]
                 .apply(pd.to_numeric, errors="coerce").values.astype(float))

        rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        _, B = M.shape
        for i in range(B):
            Mi  = M[:, i]
            Si  = Slope[:, i]
            yi  = y[:, i]
            # FIX 3 benefit: NaN omega_true values are excluded here
            mask = np.isfinite(Mi) & np.isfinite(Si) & np.isfinite(yi)
            if mask.any():
                rows.append(np.column_stack([np.ones(mask.sum()),
                                             Mi[mask], Si[mask]]))
                targets.append(yi[mask])

        if not rows:
            raise ValueError("No valid ω_true data to fit MLR. "
                             "Check that omega_true contains finite values.")

        X_all = np.vstack(rows)
        y_all = np.concatenate(targets)
        beta, _, _, _ = np.linalg.lstsq(X_all, y_all, rcond=None)

        _, B = M.shape
        Omega_hat = np.empty_like(M, dtype=float)
        for i in range(B):
            Omega_hat[:, i] = beta[0] + beta[1] * M[:, i] + beta[2] * Slope[:, i]

        # FIX 5: clip to [1.0, 20.0] — consistent with widened inversion range
        Omega_hat = np.clip(
            np.where(np.isfinite(Omega_hat), Omega_hat, np.nan),
            1.0, 20.0
        )

        self.omega_MLR = pd.DataFrame(Omega_hat, index=idx, columns=cols)
        return self.omega_MLR

    # ------------------------------------------------------------------
    # FIX 4 — estimate ET_B using omega_MLR (not omega_true)
    # ------------------------------------------------------------------
    def estimate_budyko_et(self) -> pd.DataFrame:
        """
        Compute ET_B from the Fu–Budyko equation using the MLR-predicted
        ω(t)_MLR.  This is the ET that enters the BUDYKO and DA scenarios.

        FIX 4: previous code used omega_true clipped to ±1% of 2.2,
        which completely bypassed the MLR and made Budyko ≡ BASE.
        Now uses omega_MLR as intended by the study design.
        """
        if self.omega_true is None:
            self.compute_omega_true()
        if self.omega_MLR is None:
            try:
                self.fit_and_compute_omega_mlr()
            except Exception:
                self.omega_MLR = pd.DataFrame(
                    np.nan,
                    index=self.omega_true.index,
                    columns=self.omega_true.columns,
                )

        idx  = self.PotEvap_df.index
        cols = self.PotEvap_df.columns

        P   = (self.P_df.reindex(idx)[cols]
               .apply(pd.to_numeric, errors="coerce").values.astype(float))
        dS  = (self.dS_df.reindex(idx)[cols]
               .apply(pd.to_numeric, errors="coerce").values.astype(float))
        PET = (self.PotEvap_df.reindex(idx)[cols]
               .apply(pd.to_numeric, errors="coerce").values.astype(float))

        # FIX 4: use omega_MLR — the attribute-informed, transferable prediction
        omega = (self.omega_MLR.reindex(idx)[cols]
                 .apply(pd.to_numeric, errors="coerce").values.astype(float))

        # P_eff with FIX 1 applied
        Peff = self._compute_peff(P, dS)

        # aridity index φ = PET / P_eff
        aridity = np.divide(
            PET, Peff,
            out=np.full_like(PET, np.nan, dtype=float),
            where=np.isfinite(PET) & np.isfinite(Peff) & (Peff > 0),
        )

        # Fu equation: ET/P_eff = 1 + φ − (1 + φ^ω)^(1/ω)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            term    = np.power(aridity, omega)
            E_ratio = 1.0 + aridity - np.power(1.0 + term, 1.0 / omega)

        E_ratio = np.where(np.isfinite(E_ratio), E_ratio, np.nan)
        ET_est  = Peff * E_ratio

        # Physical bounds: 0 ≤ ET ≤ min(P_eff, PET)
        ET_cap = np.minimum(
            np.where(np.isfinite(Peff), Peff, np.nan),
            np.where(np.isfinite(PET),  PET,  np.nan),
        )
        ET_est = np.clip(
            np.where(np.isfinite(ET_est), ET_est, np.nan),
            0.0, ET_cap,
        )

        self.ET_B = pd.DataFrame(ET_est, index=idx, columns=cols)
        return self.ET_B

