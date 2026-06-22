"""
run_simulation.py — Basin-scale hydrological simulation driver (SAC-SMA version).

Input files required (all feather, columns = basin IDs, index = time)
----------------------------------------------------------------------
PRCP.feather          precipitation
PotEvap.feather       potential ET
ET_sacsma.feather     actual ET simulated by SAC-SMA  ← replaces ET_ke / Evap
Q_SACSMA.feather      streamflow simulated by SAC-SMA ← used as reference
Q_USGS.feather        observed USGS streamflow        ← evaluation target
M.feather             vegetation / moisture index     ← omega MLR
slope.feather         terrain slope                   ← omega MLR

NOT required (removed vs NLDAS version)
----------------------------------------
EVap.feather, Qsb.feather, SoilM_0_200cm.feather

Three scenarios
---------------
BASE        two-store model forced with ET = ET_sacsma (SAC-SMA physically modelled ET)
BUDYKO      two-store model forced with ET = ET_B (Fu–Budyko equation)
BUDYKO_DA   state-only EnSRF: assimilates ET_sacsma as observation into [S, G] states
            + optionally assimilates Q_USGS as a second sequential update

Design notes
------------
* The two-store bucket model propagates [S, G] internally in the EnKF.
  S_init / G_init / Smax / Gmax come from calibrated_params.json as before.
* ET_sacsma is the assimilation OBSERVATION in DA (replaces ET_B).
  ET_B from Budyko is still computed for the BUDYKO scenario and is stored
  as a diagnostic column in DA results.
* All three EnKF fixes from the previous version are retained:
    FIX 1  P_eff capped at P for Budyko ET computation
    FIX 2  proc_S_std / proc_G_std treated as fractions of Smax/Gmax
    FIX 3  Innovation outlier rejection with relative floor
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import ModelParams, two_store_model_step
from src.budyko import BudykoModelEstimator
from src.enkf import (
    EnKFConfig,
    enkf_analysis_step,
    enkf_forecast_step_states,
)
from src.metrics import calculate_kge, calculate_nse


# =========================================================
# PROCESS-LOCAL GLOBALS
# =========================================================
_GLOBAL: dict = {
    "PET_df":            None,
    "Rainf_df":          None,
    "ET_sacsma_df":      None,   # SAC-SMA actual ET  ← NEW
    "Q_sacsma_df":       None,   # SAC-SMA streamflow ← NEW
    "Q_usgs_df":         None,
    "M_df":              None,
    "Slp_df":            None,
    "common_cols":       None,
    "idx":               None,
    "calibrated_params": None,
    "omega_mlr_beta":    None,
}


# =========================================================
# HELPERS
# =========================================================

def scenario_folder_name(scenario: str) -> str:
    s = str(scenario).strip().upper()
    if s == "BASE":   return "BASE_MODEL"
    if s == "BUDYKO": return "BUDYKO_MODEL"
    if s in ["BUDYKO_DA", "BUDYKO+DA", "DA", "ENKF"]:
        return "BUDYKO_DA"
    return f"{s}_SCENARIO"


def normalize_scenario_key(s: str) -> str:
    s = str(s).strip().upper()
    mapping = {
        "ALL": "ALL", "BASE": "BASE", "BASE_MODEL": "BASE",
        "BASE-MODEL": "BASE", "BUDYKO": "BUDYKO",
        "BUDYKO_MODEL": "BUDYKO", "BUDYKO-MODEL": "BUDYKO",
        "BUDYKO_DA": "BUDYKO_DA", "BUDYKO_DA_MODEL": "BUDYKO_DA",
        "BUDYKO+DA": "BUDYKO_DA", "DA": "BUDYKO_DA",
        "ENKF": "BUDYKO_DA", "ASSIMILATION": "BUDYKO_DA",
    }
    if s not in mapping:
        raise ValueError(f"Unknown scenario: {s!r}")
    return mapping[s]


def _stable_seed(basin_id: str) -> int:
    """Reproducible 32-bit seed from basin ID string (MD5, not hash())."""
    digest = hashlib.md5(basin_id.encode()).digest()
    return int.from_bytes(digest[:4], byteorder="little")


def load_feather_df(fname: str, ddir: str) -> pd.DataFrame:
    """
    Load a feather file and normalise the time axis to a DatetimeIndex
    named 'time'.

    Handles three layouts present in this project:
      1. 'Date' column  -> PotEvap, Rainf, Q_SACSMA, Q_USGS, Evap
      2. 'time' column  -> M
      3. DatetimeIndex already set -> slope
    """
    path = os.path.join(ddir, fname)
    if not os.path.exists(path):
        logging.warning(f"File not found: {path}. Returning empty DataFrame.")
        return pd.DataFrame()
    df = pd.read_feather(path).dropna(axis=1, how="all")
    for candidate in ("Date", "date", "time", "TIME"):
        if candidate in df.columns:
            df[candidate] = pd.to_datetime(df[candidate])
            df = df.set_index(candidate)
            df.index.name = "time"
            return df
    if isinstance(df.index, pd.DatetimeIndex):
        df.index.name = "time"
        return df
    return df


def _init_worker(global_payload: dict) -> None:
    _GLOBAL.update(global_payload)


# =========================================================
# DATA LOADING
# =========================================================

def load_all_inputs(DATA_DIR: str) -> dict:
    """
    Load all required feather files.

    Required
    --------
    PRCP.feather, PotEvap.feather, ET_sacsma.feather,
    Q_SACSMA.feather, M.feather, slope.feather

    Optional
    --------
    Q_USGS.feather  (used for evaluation; basins without gauge data get NaN)
    """
    PET_df       = load_feather_df("PotEvap.feather", DATA_DIR)
    Rainf_df     = load_feather_df("Rainf.feather",   DATA_DIR)
    ET_sacsma_df = load_feather_df("Evap.feather",    DATA_DIR)  # SAC-SMA actual ET
    Q_sacsma_df  = load_feather_df("Q_SACSMA.feather",DATA_DIR)
    Q_usgs_df    = load_feather_df("Q_USGS.feather",  DATA_DIR)
    M_df         = load_feather_df("M.feather",       DATA_DIR)
    Slp_df       = load_feather_df("slope.feather",   DATA_DIR)

    # These are strictly required
    required = {
        "PotEvap.feather": PET_df,
        "Rainf.feather":   Rainf_df,
        "Evap.feather":    ET_sacsma_df,
        "Q_SACSMA.feather":Q_sacsma_df,
        "M.feather":       M_df,
        "slope.feather":   Slp_df,
    }
    for name, df in required.items():
        if df.empty:
            raise ValueError(f"Required input is empty or missing: {name}")

    # Common basin columns across all required files
    common_cols = sorted(
        set(PET_df.columns)
        & set(Rainf_df.columns)
        & set(ET_sacsma_df.columns)
        & set(Q_sacsma_df.columns)
        & set(M_df.columns)
        & set(Slp_df.columns)
    )
    if not common_cols:
        raise ValueError(
            "No common basin columns across required inputs. "
            "Check that all feather files share the same basin ID column names."
        )

    # Common time index across all required files
    common_idx = (
        PET_df.index
        .intersection(Rainf_df.index)
        .intersection(ET_sacsma_df.index)
        .intersection(Q_sacsma_df.index)
        .intersection(M_df.index)
        .intersection(Slp_df.index)
    )
    # Q_USGS is optional — intersect only if present
    if not Q_usgs_df.empty:
        common_idx = common_idx.intersection(Q_usgs_df.index)

    if len(common_idx) == 0:
        raise ValueError(
            "No overlapping time steps across required inputs. "
            "Check that all feather files share the same time index."
        )

    # Align all frames
    PET_df       = PET_df.loc[common_idx, common_cols]
    Rainf_df     = Rainf_df.loc[common_idx, common_cols]
    ET_sacsma_df = ET_sacsma_df.loc[common_idx, common_cols]
    Q_sacsma_df  = Q_sacsma_df.loc[common_idx, common_cols]
    M_df         = M_df.loc[common_idx, common_cols]
    Slp_df       = Slp_df.loc[common_idx, common_cols]

    if Q_usgs_df.empty:
        # No gauge data at all — fill with NaN so downstream code is NaN-safe
        Q_usgs_df = pd.DataFrame(
            np.nan, index=common_idx, columns=common_cols
        )
    else:
        # Keep only basins that also appear in the required files
        qobs_cols = sorted(set(Q_usgs_df.columns) & set(common_cols))
        Q_usgs_df = Q_usgs_df.loc[common_idx, qobs_cols]

    print(f"  Loaded {len(common_cols)} basins × {len(common_idx)} time steps.")
    print(f"  Period: {common_idx[0].date()} → {common_idx[-1].date()}")

    return dict(
        PET_df       = PET_df,
        Rainf_df     = Rainf_df,
        ET_sacsma_df = ET_sacsma_df,
        Q_sacsma_df  = Q_sacsma_df,
        Q_usgs_df    = Q_usgs_df,
        M_df         = M_df,
        Slp_df       = Slp_df,
        common_cols  = common_cols,
        idx          = common_idx,
    )


# =========================================================
# DETERMINISTIC TWO-STORE MODEL RUN
# =========================================================

def run_model_deterministic(
    P: np.ndarray,
    PET: np.ndarray,
    params_cal: ModelParams,
    S_init: float,
    G_init: float,
    Gmax_cal: float,
    ET_series: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the two-store model deterministically for a full time series.

    ET_series is passed as ET_override each step.  NaN entries in ET_series
    cause the model to fall back to its internal stress-based ET calculation.

    Returns
    -------
    Q_out, S_out, G_out, dS_out : each shape (L,)
    """
    L = len(P)
    S, G = float(S_init), float(G_init)
    Q_out  = np.full(L, np.nan)
    S_out  = np.full(L, np.nan)
    G_out  = np.full(L, np.nan)
    dS_out = np.full(L, np.nan)

    for t in range(L):
        P_t   = float(P[t])         if np.isfinite(P[t])         else 0.0
        PET_t = float(PET[t])       if np.isfinite(PET[t])       else 0.0
        ET_t  = float(ET_series[t]) if np.isfinite(ET_series[t]) else None

        S, G, _, Q, _, _, _, dS_t = two_store_model_step(
            S, G, P_t, PET_t, params_cal, ET_override=ET_t,
        )
        G = float(np.clip(G, 0.0, Gmax_cal))

        Q_out[t]  = max(float(Q), 0.0)
        S_out[t]  = float(S)
        G_out[t]  = float(G)
        dS_out[t] = float(dS_t) if np.isfinite(dS_t) else np.nan

    return Q_out, S_out, G_out, dS_out


# =========================================================
# FU–BUDYKO ET  (used for BUDYKO scenario only)
# =========================================================

def fu_et_from_peff(
    P_eff: np.ndarray,
    PET: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    """
    ET/P_eff = 1 + φ − (1 + φ^ω)^(1/ω),  φ = PET / P_eff.
    Result clipped to [0, min(P_eff, PET)].
    """
    P_eff = np.asarray(P_eff, dtype=float)
    PET   = np.asarray(PET,   dtype=float)
    omega = np.asarray(omega,  dtype=float)
    ET    = np.full_like(P_eff, np.nan)

    ok = (
        np.isfinite(P_eff) & np.isfinite(PET) & np.isfinite(omega)
        & (P_eff > 0.0) & (PET >= 0.0) & (omega > 0.0)
    )
    if not np.any(ok):
        return ET

    phi = PET[ok] / P_eff[ok]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        term     = np.power(phi, omega[ok])
        e_over_p = 1.0 + phi - np.power(1.0 + term, 1.0 / omega[ok])
    e_over_p = np.where(np.isfinite(e_over_p), e_over_p, np.nan)
    ET[ok]   = np.clip(
        P_eff[ok] * e_over_p, 0.0, np.minimum(P_eff[ok], PET[ok])
    )
    return ET


# =========================================================
# GLOBAL OMEGA MLR FIT  (needed for BUDYKO scenario)
# =========================================================

def fit_global_omega_mlr(
    basins: list[str],
    idx: pd.DatetimeIndex,
    PET_df: pd.DataFrame,
    Rainf_df: pd.DataFrame,
    ET_sacsma_df: pd.DataFrame,
    M_df: pd.DataFrame,
    Slp_df: pd.DataFrame,
    calibrated_params: dict,
) -> np.ndarray:
    """
    Pool OLS: ω = β₀ + β₁·M + β₂·slope  across all basins.

    omega_true is inverted from the Fu–Budyko equation using
    ET_sacsma (SAC-SMA ET) as the target actual ET, and P_eff = P - dS
    from a warm-up two-store run driven by ET_sacsma.

    Returns
    -------
    beta : ndarray shape (3,)  [intercept, β_M, β_slope]
    """
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for basin_id in basins:
        if basin_id not in calibrated_params:
            continue

        p           = calibrated_params[basin_id]
        PET         = PET_df[basin_id].reindex(idx).to_numpy(dtype=float)
        P           = Rainf_df[basin_id].reindex(idx).to_numpy(dtype=float)
        ET_sac      = ET_sacsma_df[basin_id].reindex(idx).to_numpy(dtype=float)

        Smax_cal    = float(p.get("Smax", 50.0))
        Gmax_factor = float(p.get("Gmax_factor", 4.0))
        Gmax_cal    = Smax_cal * Gmax_factor
        S_init      = float(p.get("S_init", 0.5 * Smax_cal))
        G_init      = float(p.get("G_init", 0.5 * Gmax_cal))

        params_cal = ModelParams(
            Smax=Smax_cal,
            Kperc=float(p["Kperc"]), Kb=float(p["Kb"]),
            Ke=float(p["Ke"]),       Cqq=float(p["Cqq"]),
            Sfc_frac=0.30, beta_et=2.0,
        )

        # Warm-up run driven by ET_sacsma to get dS
        _, _, _, dS_base = run_model_deterministic(
            P=P, PET=PET, params_cal=params_cal,
            S_init=S_init, G_init=G_init, Gmax_cal=Gmax_cal,
            ET_series=ET_sac,
        )

        # FIX 1: cap P_eff at P
        P_eff_base = np.clip((P - dS_base).astype(float), 1e-6, P)

        # Use BudykoModelEstimator to invert omega_true from ET_sacsma
        bud = BudykoModelEstimator(
            P_df       = pd.DataFrame({basin_id: P},        index=idx),
            dS_df      = pd.DataFrame({basin_id: dS_base},  index=idx),
            PotEvap_df = pd.DataFrame({basin_id: PET},      index=idx),
            M_basin    = M_df[[basin_id]].reindex(idx),
            Slope_basin= Slp_df[[basin_id]].reindex(idx),
            calibrated_params = calibrated_params,
            Ke_df      = None,
        )
        omega_true = (
            bud.compute_omega_true()[basin_id]
            .reindex(idx).to_numpy(dtype=float)
        )

        M_series  = M_df[basin_id].reindex(idx).to_numpy(dtype=float)
        slope_col = Slp_df[basin_id].reindex(idx)
        slope_val = (
            float(slope_col.dropna().iloc[0])
            if slope_col.notna().any() else np.nan
        )
        S_series  = np.full_like(M_series, slope_val, dtype=float)

        mask = (
            np.isfinite(M_series)
            & np.isfinite(S_series)
            & np.isfinite(omega_true)
        )
        if np.any(mask):
            rows.append(np.column_stack([
                np.ones(mask.sum()), M_series[mask], S_series[mask]
            ]))
            targets.append(omega_true[mask])

    if not rows:
        raise ValueError("No valid data to fit global omega MLR.")

    beta, _, _, _ = np.linalg.lstsq(
        np.vstack(rows), np.concatenate(targets), rcond=None
    )
    return beta


# =========================================================
# STATE-ONLY ENKF DA
# =========================================================

def run_budyko_da(
    P: np.ndarray,
    PET: np.ndarray,
    ET_obs: np.ndarray,        # ET_sacsma — the assimilation observation
    params_cal: ModelParams,
    S_init: float,
    G_init: float,
    Smax_cal: float,
    Gmax_cal: float,
    config: EnKFConfig,
    basin_id: str,
    Q_obs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    State-only EnSRF DA loop assimilating ET_sacsma as the observation.

    Cycle per time step
    -------------------
    1. Forecast   : propagate [S, G] ensemble through two-store model
                    with process noise + forcing perturbations
    2. Analysis ET: deterministic scalar EnSRF update using ET_sacsma
                    (innovation-capped to prevent ensemble collapse)
    3. Analysis Q : optional second sequential update using Q_USGS
    4. Inflation  : multiplicative anomaly inflation
    5. Re-forecast: zero-perturbation pass from X_a → clean ET_ass, Q_ass

    Fixes applied
    -------------
    FIX 2  proc_S_std / proc_G_std treated as fractions of Smax/Gmax.
           Values < 2.0 are interpreted as fractions; ≥ 2.0 as absolute mm.
    FIX 3  Innovation cap = max(3σ_prior, 30%|ET_obs|, R_ET_std).
           The relative floor keeps the window open when ET_sacsma is large.
    """
    L    = len(P)
    nens = int(config.nens)
    rng  = np.random.default_rng(_stable_seed(basin_id))

    # ---- FIX 2: fractional → absolute process noise --------------
    proc_S_abs = (
        float(config.proc_S_std) * Smax_cal
        if float(config.proc_S_std) < 2.0
        else float(config.proc_S_std)
    )
    proc_G_abs = (
        float(config.proc_G_std) * Gmax_cal
        if float(config.proc_G_std) < 2.0
        else float(config.proc_G_std)
    )

    basin_cfg = EnKFConfig(
        nens         = config.nens,
        inflation    = config.inflation,
        R_ET_std     = config.R_ET_std,
        R_Q_std      = config.R_Q_std,
        R_ET_frac    = config.R_ET_frac,
        R_Q_frac     = config.R_Q_frac,
        proc_S_std   = proc_S_abs,
        proc_G_std   = proc_G_abs,
        P_std_frac   = config.P_std_frac,
        PET_std_frac = config.PET_std_frac,
    )

    # ---- Initial ensemble ----------------------------------------
    S0_ens = np.clip(
        S_init + rng.normal(0.0, 0.10 * Smax_cal, nens), 0.0, Smax_cal
    )
    G0_ens = np.clip(
        G_init + rng.normal(0.0, 0.10 * Gmax_cal, nens), 0.0, Gmax_cal
    )
    X = np.vstack([S0_ens, G0_ens])   # (2, nens)

    # ---- Output arrays -------------------------------------------
    S_ens_hist  = np.full((L, nens), np.nan)
    G_ens_hist  = np.full((L, nens), np.nan)
    ET_ens_hist = np.full((L, nens), np.nan)
    Q_ens_hist  = np.full((L, nens), np.nan)

    innov_ET_hist = np.full(L, np.nan)
    innov_Q_hist  = np.full(L, np.nan)
    spread_S_hist = np.full(L, np.nan)
    spread_G_hist = np.full(L, np.nan)
    spread_Q_hist = np.full(L, np.nan)

    ET_ass_mean = np.full(L, np.nan)
    Q_ass_mean  = np.full(L, np.nan)

    # Warm-start innovation cap from first ET_sacsma value
    et0 = float(ET_obs[np.isfinite(ET_obs)][0]) if np.any(np.isfinite(ET_obs)) else 30.0
    prev_et_mean   = et0
    prev_et_spread = 0.3 * et0

    for t in range(L):
        P_t   = float(P[t])   if np.isfinite(P[t])   else 0.0
        PET_t = float(PET[t]) if np.isfinite(PET[t]) else 0.0

        # ---- Observations ----------------------------------------
        y_ET_raw = (
            float(ET_obs[t])
            if ET_obs is not None and np.isfinite(ET_obs[t])
            else None
        )
        y_Q = (
            float(Q_obs[t])
            if Q_obs is not None and np.isfinite(Q_obs[t])
            else None
        )

        # ---- FIX 3: innovation outlier rejection -----------------
        if y_ET_raw is not None:
            cap = max(
                3.0 * prev_et_spread,
                0.30 * abs(y_ET_raw),
                float(basin_cfg.R_ET_std),
            )
            y_ET = float(np.clip(
                y_ET_raw,
                prev_et_mean - cap,
                prev_et_mean + cap,
            ))
        else:
            y_ET = None

        # ---- Forecast + Analysis (ET → Q) + Inflation ------------
        X_a, ET_ens_f, Q_ens_f, diag = enkf_analysis_step(
            X            = X,
            P_t          = P_t,
            PET_t        = PET_t,
            params_cal   = params_cal,
            Smax         = Smax_cal,
            Gmax         = Gmax_cal,
            rng          = rng,
            cfg          = basin_cfg,
            y_ET         = y_ET,
            y_Q          = y_Q,
            ET_override  = None,   # model computes ET from states internally
        )

        # Update running prior stats for next step's cap
        prev_et_mean   = float(np.mean(ET_ens_f))
        prev_et_spread = float(np.std(ET_ens_f, ddof=1))

        # ---- Re-forecast from posterior (zero perturbation) ------
        _, ET_ens_a, Q_ens_a = enkf_forecast_step_states(
            X            = X_a,
            P_t          = P_t,
            PET_t        = PET_t,
            params_cal   = params_cal,
            Smax         = Smax_cal,
            Gmax         = Gmax_cal,
            rng          = np.random.default_rng(),
            proc_S_std   = 0.0,
            proc_G_std   = 0.0,
            P_std_frac   = 0.0,
            PET_std_frac = 0.0,
            ET_override  = None,
        )

        # ---- Store -----------------------------------------------
        ET_ass_mean[t] = float(np.nanmean(ET_ens_a))
        Q_ass_mean[t]  = float(np.nanmean(Q_ens_a))

        ET_ens_hist[t, :] = ET_ens_a
        Q_ens_hist[t, :]  = Q_ens_a
        S_ens_hist[t, :]  = X_a[0, :]
        G_ens_hist[t, :]  = X_a[1, :]

        innov_ET_hist[t] = diag.innov_ET if diag.innov_ET is not None else np.nan
        innov_Q_hist[t]  = diag.innov_Q  if diag.innov_Q  is not None else np.nan
        spread_S_hist[t] = diag.spread_S
        spread_G_hist[t] = diag.spread_G
        spread_Q_hist[t] = diag.spread_Q

        X = X_a

    enkf_hist = dict(
        nens      = nens,
        S_ens     = S_ens_hist,
        G_ens     = G_ens_hist,
        ET_ens    = ET_ens_hist,
        Q_ens     = Q_ens_hist,
        innov_ET  = innov_ET_hist,
        innov_Q   = innov_Q_hist,
        spread_S  = spread_S_hist,
        spread_G  = spread_G_hist,
        spread_Q  = spread_Q_hist,
    )
    return ET_ass_mean, Q_ass_mean, enkf_hist


# =========================================================
# ENSEMBLE OUTPUT — SAVE
# =========================================================

def _save_enkf_ensemble(
    enkf_hist: dict,
    idx: pd.DatetimeIndex,
    basin_id: str,
    RESULT_DIR: str,
) -> None:
    nens = int(enkf_hist["nens"])

    def _summary(arr: np.ndarray, prefix: str) -> dict:
        return {
            f"{prefix}_mean": np.nanmean(arr, axis=1),
            f"{prefix}_std":  np.nanstd(arr,  axis=1),
            f"{prefix}_p05":  np.nanpercentile(arr,  5, axis=1),
            f"{prefix}_p25":  np.nanpercentile(arr, 25, axis=1),
            f"{prefix}_p50":  np.nanpercentile(arr, 50, axis=1),
            f"{prefix}_p75":  np.nanpercentile(arr, 75, axis=1),
            f"{prefix}_p95":  np.nanpercentile(arr, 95, axis=1),
        }

    summary = {
        "time": idx.values,
        **_summary(enkf_hist["ET_ens"], "ET_ens"),
        **_summary(enkf_hist["Q_ens"],  "Q_ens"),
        **_summary(enkf_hist["S_ens"],  "S_ens"),
        **_summary(enkf_hist["G_ens"],  "G_ens"),
        "innov_ET":  enkf_hist["innov_ET"],
        "innov_Q":   enkf_hist["innov_Q"],
        "spread_S":  enkf_hist["spread_S"],
        "spread_G":  enkf_hist["spread_G"],
        "spread_Q":  enkf_hist["spread_Q"],
    }
    member_cols = {
        **{f"ET_ens_{i+1:03d}": enkf_hist["ET_ens"][:, i] for i in range(nens)},
        **{f"Q_ens_{i+1:03d}":  enkf_hist["Q_ens"][:, i]  for i in range(nens)},
    }
    enkf_df = pd.concat(
        [pd.DataFrame(summary), pd.DataFrame(member_cols)], axis=1
    )
    enkf_df.to_feather(
        os.path.join(RESULT_DIR, f"enkf_ensemble_BUDYKO_DA_{basin_id}.feather")
    )


# =========================================================
# PER-BASIN SIMULATION
# =========================================================

def simulate_basin(
    basin_id: str,
    scenario: str,
    RESULT_DIR: str,
    da_cfg: dict,
) -> dict | None:

    common_cols       = _GLOBAL["common_cols"]
    calibrated_params = _GLOBAL["calibrated_params"]
    beta              = _GLOBAL["omega_mlr_beta"]
    M_df              = _GLOBAL["M_df"]
    Slp_df            = _GLOBAL["Slp_df"]
    PET_df            = _GLOBAL["PET_df"]
    Rainf_df          = _GLOBAL["Rainf_df"]
    ET_sacsma_df      = _GLOBAL["ET_sacsma_df"]
    Q_sacsma_df       = _GLOBAL["Q_sacsma_df"]
    Q_usgs_df         = _GLOBAL["Q_usgs_df"]
    idx               = _GLOBAL["idx"]

    if basin_id not in common_cols or basin_id not in calibrated_params:
        return None

    # ---- Forcings and SAC-SMA outputs ---------------------------
    PET      = PET_df[basin_id].reindex(idx).to_numpy(dtype=float)
    P        = Rainf_df[basin_id].reindex(idx).to_numpy(dtype=float)
    ET_sacsma = ET_sacsma_df[basin_id].reindex(idx).to_numpy(dtype=float)
    Q_sacsma  = Q_sacsma_df[basin_id].reindex(idx).to_numpy(dtype=float)

    Q_obs = pd.to_numeric(
        Q_usgs_df.get(basin_id, pd.Series(np.nan, index=idx)).reindex(idx),
        errors="coerce",
    ).to_numpy(dtype=float)

    # # ---- Omega MLR for Budyko scenario ---------------------------
    # M_series  = M_df[basin_id].reindex(idx).to_numpy(dtype=float)
    # slope_col = Slp_df[basin_id].reindex(idx)
    # slope_val = (
    #     float(slope_col.dropna().iloc[0]) if slope_col.notna().any() else np.nan
    # )
    # S_series  = np.full(len(idx), slope_val, dtype=float)

    # omega_mlr_raw    = beta[0] + beta[1] * M_series + beta[2] * S_series
    # omega_mlr_series = np.clip(
    #     np.where(np.isfinite(omega_mlr_raw), omega_mlr_raw, np.nan),
    #     1.01, 50.0,
    # )

    # ---- Model parameters from calibrated_params.json -----------
    p           = calibrated_params[basin_id]
    Smax_cal    = float(p.get("Smax", 50.0))
    Gmax_factor = float(p.get("Gmax_factor", 4.0))
    Gmax_cal    = Smax_cal * Gmax_factor
    S_init      = float(p.get("S_init", 0.5 * Smax_cal))
    G_init      = float(p.get("G_init", 0.5 * Gmax_cal))

    params_cal = ModelParams(
        Smax     = Smax_cal,
        Kperc    = float(p["Kperc"]),
        Kb       = float(p["Kb"]),
        Ke       = float(p["Ke"]),
        Cqq      = float(p["Cqq"]),
        Sfc_frac = 0.30,
        beta_et  = 2.0,
    )

    # ---- dS from BASE run (needed for Budyko P_eff) -------------
    # Drive the warm-up run with ET_sacsma to get a physically
    # consistent dS series for computing P_eff = P - dS.
    _, _, _, dS_tmp = run_model_deterministic(
        P=P, PET=PET, params_cal=params_cal,
        S_init=S_init, G_init=G_init, Gmax_cal=Gmax_cal,
        ET_series=ET_sacsma,
    )

    # ---- True omega (diagnostic, for BUDYKO scenario) -----------
    # REPLACE the bud / omega_true / P_eff / ET_B block with this:
    bud = BudykoModelEstimator(
        P_df        = pd.DataFrame({basin_id: P},       index=idx),
        dS_df       = pd.DataFrame({basin_id: dS_tmp},  index=idx),
        PotEvap_df  = pd.DataFrame({basin_id: PET},     index=idx),
        M_basin     = M_df[[basin_id]].reindex(idx),
        Slope_basin = Slp_df[[basin_id]].reindex(idx),
        calibrated_params = calibrated_params,
        Ke_df       = None,
    )
    omega_true_series = bud.compute_omega_true()[basin_id].to_numpy(dtype=float)
    omega_mlr_series  = bud.fit_and_compute_omega_mlr()[basin_id].to_numpy(dtype=float)
    ET_B              = bud.estimate_budyko_et()[basin_id].to_numpy(dtype=float)
    # FIX 1 already applied inside BudykoModelEstimator._compute_peff — no need to redo it here

    enkf_hist: dict | None = None

    # =============================================================
    # BASE — two-store driven by ET_sacsma
    # =============================================================
    if scenario == "BASE":
        Q_base, S_base, G_base, dS_base = run_model_deterministic(
            P=P, PET=PET, params_cal=params_cal,
            S_init=S_init, G_init=G_init, Gmax_cal=Gmax_cal,
            ET_series=ET_sacsma,        # SAC-SMA ET as forcing
        )
        results = pd.DataFrame({
            "time":        idx,
            "P":           P,
            "PET":         PET,
            "ET_ke":   ET_sacsma,
            "Q_sacsma":    Q_sacsma,
            "omega_true":  omega_true_series,
            "omega_MLR":   omega_mlr_series,
            "dS_base":     dS_base,
            "Q_obs":       Q_obs,
            "Q_base":      Q_base,
            "S_base":      S_base,
            "G_base":      G_base,
        }).set_index("time")
        qsim_name = "Q_base"

    # =============================================================
    # BUDYKO — two-store driven by ET_B (Fu–Budyko)
    # =============================================================
    elif scenario == "BUDYKO":
        Q_budyko, S_budyko, G_budyko, dS_budyko = run_model_deterministic(
            P=P, PET=PET, params_cal=params_cal,
            S_init=S_init, G_init=G_init, Gmax_cal=Gmax_cal,
            ET_series=ET_B,             # Budyko ET as forcing
        )
        results = pd.DataFrame({
            "time":        idx,
            "P":           P,
            "PET":         PET,
            "ET_ke":   ET_sacsma,
            "ET_B":        ET_B,
            "Q_sacsma":    Q_sacsma,
            "omega_true":  omega_true_series,
            "omega_MLR":   omega_mlr_series,
            "dS_budyko":   dS_budyko,
            "Q_obs":       Q_obs,
            "Q_budyko":    Q_budyko,
            "S_budyko":    S_budyko,
            "G_budyko":    G_budyko,
        }).set_index("time")
        qsim_name = "Q_budyko"

    # =============================================================
    # BUDYKO_DA — assimilate ET_sacsma into [S, G]
    # =============================================================
    elif scenario == "BUDYKO_DA":
        config = EnKFConfig(**da_cfg)

        ET_ass_mean, Q_ass_mean, enkf_hist = run_budyko_da(
            P          = P,
            PET        = PET,
            ET_obs     = ET_sacsma,     # ET_sacsma is the assimilation observation
            params_cal = params_cal,
            S_init     = S_init,
            G_init     = G_init,
            Smax_cal   = Smax_cal,
            Gmax_cal   = Gmax_cal,
            config     = config,
            basin_id   = basin_id,
            Q_obs      = Q_obs,         # NaN-safe; skipped if all NaN
        )

        S_ass  = np.nanmean(enkf_hist["S_ens"], axis=1)
        G_ass  = np.nanmean(enkf_hist["G_ens"], axis=1)
        dS_ass = np.concatenate([[np.nan], np.diff(S_ass + G_ass)])

        results = pd.DataFrame({
            "time":        idx,
            "P":           P,
            "PET":         PET,
            "ET_ke":       ET_sacsma,   # observation that was assimilated
            "ET_B":        ET_B,        # Budyko ET (diagnostic)
            "ET_ass":      ET_ass_mean, # posterior ET from re-forecast
            "Q_sacsma":    Q_sacsma,    # SAC-SMA reference
            "omega_true":  omega_true_series,
            "omega_MLR":   omega_mlr_series,
            "dS_ass":      dS_ass,
            "Q_obs":       Q_obs,
            "Q_ass":       Q_ass_mean,
            "S_ass":       S_ass,
            "G_ass":       G_ass,
        }).set_index("time")

        _save_enkf_ensemble(enkf_hist, idx, basin_id, RESULT_DIR)
        qsim_name = "Q_ass"

    else:
        raise ValueError(f"Unknown scenario: {scenario!r}")

    # ---- Save main results feather -------------------------------
    results.reset_index().to_feather(
        os.path.join(RESULT_DIR, f"results_{scenario}_{basin_id}.feather")
    )

    # ---- Metrics against Q_USGS ----------------------------------
    qobs = results["Q_obs"].to_numpy(dtype=float)
    qsim = results[qsim_name].to_numpy(dtype=float)

    return {
        "gauge_id": basin_id,
        "scenario": scenario,
        "KGE": float(calculate_kge(qobs, qsim)),
        "NSE": float(calculate_nse(qobs, qsim)),
    }


# =========================================================
# MAIN ENTRY POINT
# =========================================================

def run_simulations_from_config(cfg: dict) -> None:

    scenario         = normalize_scenario_key(cfg["scenario"])
    scenarios_to_run = (
        ["BASE", "BUDYKO", "BUDYKO_DA"] if scenario == "ALL" else [scenario]
    )

    paths           = cfg["paths"]
    DATA_DIR        = os.path.join(PROJECT_ROOT, paths["data_dir"])
    BASE_RESULT_DIR = os.path.join(PROJECT_ROOT, paths["result_dir"])
    os.makedirs(BASE_RESULT_DIR, exist_ok=True)

    with open(os.path.join(PROJECT_ROOT, paths["calibrated_params"])) as f:
        calibrated_params = json.load(f)

    # Strip "enabled" — config-level flag, not an EnKFConfig field
    da_cfg = {k: v for k, v in cfg.get("da", {}).items() if k != "enabled"}

    basin_subset = cfg.get("basins", {}).get("subset", None)
    inputs       = load_all_inputs(DATA_DIR)

    basins = (
        list(basin_subset) if basin_subset is not None
        else list(inputs["Q_sacsma_df"].columns)
    )
    common_cols = inputs["common_cols"]
    basins = [b for b in basins if b in common_cols and b in calibrated_params]

    if not basins:
        raise ValueError(
            "No basins remain after filtering by common_cols and calibrated_params. "
            "Check that basin IDs in calibrated_params.json match feather column names."
        )

    idx  = inputs["idx"]

    print(f"\nFitting global omega MLR across {len(basins)} basins ...")
    beta = fit_global_omega_mlr(
        basins       = basins,
        idx          = idx,
        PET_df       = inputs["PET_df"],
        Rainf_df     = inputs["Rainf_df"],
        ET_sacsma_df = inputs["ET_sacsma_df"],
        M_df         = inputs["M_df"],
        Slp_df       = inputs["Slp_df"],
        calibrated_params = calibrated_params,
    )
    print(f"  beta = {beta}")

    global_payload = {
        **inputs,
        "calibrated_params": calibrated_params
        # "omega_mlr_beta":    None,
    }
    _init_worker(global_payload)

    par_cfg     = cfg.get("parallel", {})
    par_enabled = bool(par_cfg.get("enabled", True))
    max_workers = int(par_cfg.get("max_workers", -1))
    if max_workers <= 0:
        max_workers = max(1, cpu_count() - 1)

    for sc in scenarios_to_run:
        RESULT_DIR = os.path.join(BASE_RESULT_DIR, scenario_folder_name(sc))
        os.makedirs(RESULT_DIR, exist_ok=True)
        all_metrics: list[dict] = []

        if par_enabled:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker,
                initargs=(global_payload,),
            ) as executor:
                futures = {
                    executor.submit(simulate_basin, b, sc, RESULT_DIR, da_cfg): b
                    for b in basins
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Running scenario={sc}",
                ):
                    try:
                        out = future.result()
                        if out is not None:
                            all_metrics.append(out)
                    except Exception as exc:
                        logging.error(
                            f"Basin {futures[future]!r} failed: {exc}",
                            exc_info=True,
                        )
        else:
            for b in tqdm(basins, desc=f"Running scenario={sc}"):
                try:
                    out = simulate_basin(b, sc, RESULT_DIR, da_cfg)
                    if out is not None:
                        all_metrics.append(out)
                except Exception as exc:
                    logging.error(f"Basin {b!r} failed: {exc}", exc_info=True)

        pd.DataFrame(all_metrics).to_csv(
            os.path.join(RESULT_DIR, f"metrics_{sc}.csv"), index=False,
        )
        print(f"\n✅  scenario={sc} complete — results in {RESULT_DIR}")


