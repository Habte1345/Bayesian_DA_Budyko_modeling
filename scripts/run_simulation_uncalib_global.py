"""
run_simulation_uncalib_global.py — Global-parameter basin simulation driver (SAC-SMA version).

What changed vs the previous version
--------------------------------------
1. load_feather_df  — now handles 'Date' column, 'time' column, and
                      DatetimeIndex, matching the actual feather layouts:
                        PotEvap / Rainf / Q_USGS / Evap / Q_SACSMA → 'Date' column
                        M                                            → 'time' column
                        slope                                        → DatetimeIndex

2. align_simulation_inputs — loads the SAC-SMA file set:
                        REQUIRED: PotEvap, Rainf, Evap (SAC-SMA ET),
                                  Q_SACSMA, Q_USGS (optional)
                        REMOVED:  Qsb (not available in SAC-SMA setup)

3. simulate_basin   — ET_ke is still computed for BASE (Ke × PET).
                      ET_B for BUDYKO / BUDYKO_DA now uses compute_fu_et
                      with the basin omega from omega_lookup (unchanged).
                      Qsb column removed from BASE results.
                      ET_sacsma added to all result DataFrames as a reference column.

Everything else is identical to the previous version:
  global params, omega JSON, train/test split, parallel execution,
  run_budyko_da, metrics, file naming.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import ModelParams, two_store_model_step
from src.enkf import (
    EnKFConfig,
    enkf_update_deterministic_scalar,
    enkf_forecast_step_states,
)
from src.metrics import calculate_kge, calculate_nse


# =========================================================
# Helpers — scenario naming / normalisation (unchanged)
# =========================================================

def scenario_folder_name(scenario: str) -> str:
    s = str(scenario).strip().upper()
    if s == "BASE":
        return "BASE_MODEL"
    elif s == "BUDYKO":
        return "BUDYKO_MODEL"
    elif s in ["BUDYKO_DA", "BUDYKO+DA", "DA", "ENKF"]:
        return "BUDYKO_DA"
    return f"{s}_SCENARIO"


def normalize_scenario_key(s: str) -> str:
    s = str(s).strip().upper()
    mapping = {
        "ALL": "ALL",
        "BASE": "BASE",        "BASE_MODEL": "BASE",   "BASE-MODEL": "BASE",
        "BUDYKO": "BUDYKO",    "BUDYKO_MODEL": "BUDYKO","BUDYKO-MODEL": "BUDYKO",
        "BUDYKO_DA": "BUDYKO_DA","BUDYKO_DA_MODEL": "BUDYKO_DA",
        "BUDYKO+DA": "BUDYKO_DA","DA": "BUDYKO_DA",
        "ENKF": "BUDYKO_DA",   "ASSIMILATION": "BUDYKO_DA",
    }
    if s not in mapping:
        raise ValueError(f"Unknown scenario: {s!r}")
    return mapping[s]


# =========================================================
# FIX 1 — load_feather_df: handle Date / time / DatetimeIndex
# =========================================================

def load_feather_df(fname: str, ddir: str) -> pd.DataFrame:
    """
    Load a feather file and normalise the time axis to a DatetimeIndex
    named 'time'.

    Handles three layouts present in this project:
      1. 'Date' column  → PotEvap, Rainf, Q_USGS, Evap, Q_SACSMA
      2. 'time' column  → M
      3. DatetimeIndex already set → slope
    """
    path = os.path.join(ddir, fname)
    if not os.path.exists(path):
        logging.warning(f"File not found: {path}. Returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.read_feather(path).dropna(axis=1, how="all")

    # Try named time columns in priority order
    for candidate in ("Date", "date", "time", "TIME"):
        if candidate in df.columns:
            df[candidate] = pd.to_datetime(df[candidate])
            df = df.set_index(candidate)
            df.index.name = "time"
            return df

    # Index is already datetime (e.g. slope.feather)
    if isinstance(df.index, pd.DatetimeIndex):
        df.index.name = "time"
        return df

    # Integer index — return as-is
    return df


# =========================================================
# Config / omega / split loaders (unchanged)
# =========================================================

def load_saved_basin_split(cfg: dict):
    paths = cfg["paths"]
    split_file = paths.get("basin_split_file", None)
    if split_file is None:
        global_calibration_dir = paths.get("global_calibration_dir", "SCE_global_params")
        split_file = os.path.join(global_calibration_dir, "basin_split.json")

    split_path = os.path.join(PROJECT_ROOT, split_file)
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Saved basin split file not found: {split_path}\n"
            f"Run global calibration first, or set paths.basin_split_file in config.yaml."
        )

    with open(split_path) as f:
        split_info = json.load(f)

    train_basins = sorted(split_info.get("train_basins", []))
    test_basins  = sorted(split_info.get("test_basins",  []))

    if not train_basins:
        raise ValueError("Saved basin split contains no training basins.")
    overlap = set(train_basins) & set(test_basins)
    if overlap:
        raise ValueError(f"Saved basin split has overlapping basins: {sorted(overlap)}")

    return train_basins, test_basins


def load_global_omega(cfg: dict):
    paths = cfg["paths"]
    omega_file = paths.get(
        "global_omega_file",
        os.path.join("SCE_global_params", "omega_global_test.json"),
    )
    omega_path = os.path.join(PROJECT_ROOT, omega_file)
    if not os.path.exists(omega_path):
        raise FileNotFoundError(
            f"Global omega file not found: {omega_path}\n"
            f"Please create the omega JSON first, or set paths.global_omega_file in config.yaml."
        )
    with open(omega_path) as f:
        omega_info = json.load(f)

    omega_test = omega_info.get("omega_test", {})
    if not isinstance(omega_test, dict) or not omega_test:
        raise ValueError(f"'omega_test' is missing or empty in {omega_path}")

    return omega_info, omega_test


# =========================================================
# FIX 2 — align_simulation_inputs: SAC-SMA file set
# =========================================================

def align_simulation_inputs(DATA_DIR: str) -> dict:
    """
    Load and align all required feather files for the SAC-SMA setup.

    Required
    --------
    PotEvap.feather   potential ET
    Rainf.feather     precipitation
    Evap.feather      SAC-SMA actual ET  (replaces computed ET_ke as reference)
    Q_SACSMA.feather  SAC-SMA streamflow (reference / diagnostic)

    Optional
    --------
    Q_USGS.feather    observed streamflow (evaluation target; NaN if absent)

    Removed vs previous version
    ---------------------------
    Qsb.feather — not available in the SAC-SMA setup
    """
    PET_df      = load_feather_df("PotEvap.feather",  DATA_DIR)
    Rainf_df    = load_feather_df("Rainf.feather",    DATA_DIR)
    ET_sacsma_df= load_feather_df("Evap.feather",     DATA_DIR)
    Q_sacsma_df = load_feather_df("Q_SACSMA.feather", DATA_DIR)
    Q_usgs_df   = load_feather_df("Q_USGS.feather",   DATA_DIR)   # optional

    required = {
        "PotEvap.feather":  PET_df,
        "Rainf.feather":    Rainf_df,
        "Evap.feather":     ET_sacsma_df,
        "Q_SACSMA.feather": Q_sacsma_df,
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
    )
    if not common_cols:
        raise ValueError("No common basin columns across simulation inputs.")

    # Common time index
    common_idx = (
        PET_df.index
        .intersection(Rainf_df.index)
        .intersection(ET_sacsma_df.index)
        .intersection(Q_sacsma_df.index)
    )
    if not Q_usgs_df.empty:
        common_idx = common_idx.intersection(Q_usgs_df.index)

    if len(common_idx) == 0:
        raise ValueError(
            "No overlapping time steps across required inputs. "
            "Check that all feather files share the same time index."
        )

    # Align
    PET_df       = PET_df.loc[common_idx, common_cols]
    Rainf_df     = Rainf_df.loc[common_idx, common_cols]
    ET_sacsma_df = ET_sacsma_df.loc[common_idx, common_cols]
    Q_sacsma_df  = Q_sacsma_df.loc[common_idx, common_cols]

    if Q_usgs_df.empty:
        Q_usgs_df = pd.DataFrame(np.nan, index=common_idx, columns=common_cols)
    else:
        qobs_cols = sorted(set(Q_usgs_df.columns) & set(common_cols))
        Q_usgs_df = Q_usgs_df.loc[common_idx, qobs_cols]

    print(f"  Loaded {len(common_cols)} basins × {len(common_idx)} time steps.")
    print(f"  Period: {common_idx[0].date()} → {common_idx[-1].date()}")

    return {
        "idx":           common_idx,
        "PET_df":        PET_df,
        "Rainf_df":      Rainf_df,
        "ET_sacsma_df":  ET_sacsma_df,
        "Q_sacsma_df":   Q_sacsma_df,
        "Q_usgs_df":     Q_usgs_df,
        "common_cols":   common_cols,
    }


# =========================================================
# Fu–Budyko ET (unchanged)
# =========================================================

def compute_fu_et(
    P: np.ndarray,
    PET: np.ndarray,
    omega: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Fu equation applied timestep-wise:
        ET = P · [1 + φ − (1 + φ^ω)^(1/ω)],  φ = PET / P
    Result clipped to [0, PET].
    """
    P     = np.asarray(P,   dtype=float)
    PET   = np.asarray(PET, dtype=float)
    omega = float(omega)
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError(f"Invalid omega: {omega}")

    ET   = np.zeros_like(P)
    mask = np.isfinite(P) & np.isfinite(PET) & (P > eps)
    if np.any(mask):
        phi     = PET[mask] / np.maximum(P[mask], eps)
        fu_term = 1.0 + phi - np.power(1.0 + np.power(phi, omega), 1.0 / omega)
        ET[mask] = P[mask] * np.clip(fu_term, 0.0, None)

    ET = np.where(np.isfinite(ET), ET, 0.0)
    return np.clip(ET, 0.0, np.maximum(PET, 0.0))


# =========================================================
# Deterministic two-store model (unchanged)
# =========================================================

def run_model_deterministic(
    P: np.ndarray,
    PET: np.ndarray,
    params_cal: ModelParams,
    S_init: float,
    G_init: float,
    Gmax_cal: float,
    ET_series: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    L = len(P)
    S, G = float(S_init), float(G_init)
    Q_out = np.full(L, np.nan)
    S_out = np.full(L, np.nan)
    G_out = np.full(L, np.nan)

    for t in range(L):
        P_t   = float(P[t])         if np.isfinite(P[t])         else 0.0
        PET_t = float(PET[t])       if np.isfinite(PET[t])       else 0.0
        et_t  = float(ET_series[t]) if np.isfinite(ET_series[t]) else None

        S, G, _, Q, *_ = two_store_model_step(
            S, G, P_t, PET_t, params_cal, ET_override=et_t,
        )
        G = float(np.clip(G, 0.0, Gmax_cal))
        Q_out[t] = max(float(Q), 0.0)
        S_out[t] = float(S)
        G_out[t] = float(G)

    return Q_out, S_out, G_out


# =========================================================
# DA run (unchanged logic, same EnKF internals)
# =========================================================

def run_budyko_da(
    P: np.ndarray,
    PET: np.ndarray,
    ET_obs: np.ndarray,
    ET_model: np.ndarray,
    params_cal: ModelParams,
    S_init: float,
    G_init: float,
    Smax_cal: float,
    Gmax_cal: float,
    config: EnKFConfig,
    basin_id: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    L    = len(P)
    nens = int(config.nens)
    # inflation and R_ET_var removed — enkf_update_deterministic_scalar
    # uses R_std (not R_var) and handles inflation internally via EnSRF beta

    rng = np.random.default_rng(hash(basin_id) % (2**32 - 1))

    S0_ens = np.clip(S_init + rng.normal(0.0, 0.05 * Smax_cal, nens), 0.0, Smax_cal)
    G0_ens = np.clip(G_init + rng.normal(0.0, 0.05 * Gmax_cal, nens), 0.0, Gmax_cal)
    X = np.vstack([S0_ens, G0_ens])

    S_ens_hist  = np.full((L, nens), np.nan)
    G_ens_hist  = np.full((L, nens), np.nan)
    ET_ens_hist = np.full((L, nens), np.nan)
    Q_ens_hist  = np.full((L, nens), np.nan)

    ET_ass_mean = np.full(L, np.nan)
    Q_ass_mean  = np.full(L, np.nan)

    for t in range(L):
        ET_override_t = float(ET_model[t]) if np.isfinite(ET_model[t]) else None
        P_t   = float(P[t])   if np.isfinite(P[t])   else 0.0
        PET_t = float(PET[t]) if np.isfinite(PET[t]) else 0.0

        X_f, ET_ens_f, Q_ens_f = enkf_forecast_step_states(
            X            = X,
            P_t          = P_t,
            PET_t        = PET_t,
            params_cal   = params_cal,
            Smax         = Smax_cal,
            Gmax         = Gmax_cal,
            rng          = rng,
            proc_S_std   = float(config.proc_S_std),
            proc_G_std   = float(config.proc_G_std),
            P_std_frac   = float(config.P_std_frac),
            PET_std_frac = float(config.PET_std_frac),
            ET_override  = ET_override_t,
        )

        ET_ens_hist[t, :] = ET_ens_f
        Q_ens_hist[t, :]  = Q_ens_f

        if np.isfinite(ET_obs[t]):
            # Use the signature from enkf.py:
            # enkf_update_deterministic_scalar(X_f, y_obs, HX, R_std, Smax, Gmax)
            R_std_eff = float(config.R_ET_std)
            X_a = enkf_update_deterministic_scalar(
                X_f   = X_f,
                y_obs = float(ET_obs[t]),
                HX    = ET_ens_f.copy(),
                R_std = R_std_eff,
                Smax  = Smax_cal,
                Gmax  = Gmax_cal,
            )
        else:
            X_a = X_f

        _, ET_ens_a, Q_ens_a = enkf_forecast_step_states(
            X            = X_a,
            P_t          = P_t,
            PET_t        = PET_t,
            params_cal   = params_cal,
            Smax         = Smax_cal,
            Gmax         = Gmax_cal,
            rng          = rng,
            proc_S_std   = 0.4,
            proc_G_std   = 0.4,
            P_std_frac   = 0.4,
            PET_std_frac = 0.5,
            ET_override  = ET_override_t,
        )

        ET_ass_mean[t] = float(np.nanmean(ET_ens_a))
        Q_ass_mean[t]  = float(np.nanmean(Q_ens_a))

        X = X_a
        S_ens_hist[t, :] = X[0, :]
        G_ens_hist[t, :] = X[1, :]

    enkf_hist = {
        "time":   np.arange(L),
        "nens":   nens,
        "S_ens":  S_ens_hist,
        "G_ens":  G_ens_hist,
        "ET_ens": ET_ens_hist,
        "Q_ens":  Q_ens_hist,
    }
    return ET_ass_mean, Q_ass_mean, enkf_hist


# =========================================================
# FIX 3 — simulate_basin: SAC-SMA data columns
# =========================================================

def simulate_basin(
    basin_id: str,
    scenario: str,
    RESULT_DIR: str,
    global_params: dict,
    omega_lookup: dict,
    da_cfg: dict,
    split_label: str,
    idx: pd.DatetimeIndex,
    P: np.ndarray,
    PET: np.ndarray,
    ET_sacsma: np.ndarray,    # SAC-SMA actual ET  ← NEW (replaces Qsb)
    Q_sacsma: np.ndarray,     # SAC-SMA streamflow ← NEW
    Q_obs: np.ndarray,
) -> dict | None:
    """
    Run one basin for one scenario.

    Changes vs previous version
    ---------------------------
    - ET_sacsma replaces Qsb as an input; included in all result DataFrames.
    - Q_sacsma (SAC-SMA streamflow) stored as a reference column.
    - BASE scenario uses ET_ke = Ke × PET as the model forcing (unchanged).
      ET_sacsma is stored as a reference column so diagnostics can compare
      the simple scaling approach against the physics-based SAC-SMA ET.
    """
    # ---- Model parameters from global_params --------------------
    Smax_cal    = float(global_params["Smax"])
    Gmax_factor = float(global_params["Gmax_factor"])
    Gmax_cal    = Smax_cal * Gmax_factor
    S_init      = float(global_params.get("S_init", global_params["fS0"] * Smax_cal))
    G_init      = float(global_params.get("G_init", global_params["fG0"] * Smax_cal))

    params_cal = ModelParams(
        Smax     = Smax_cal,
        Kperc    = float(global_params["Kperc"]),
        Kb       = float(global_params["Kb"]),
        Ke       = float(global_params["Ke"]),
        Cqq      = float(global_params["Cqq"]),
        Sfc_frac = 0.30,
        beta_et  = 2.0,
    )

    ET_ke = PET * float(params_cal.Ke)   # simple baseline ET (always computed)

    # ---- Omega / ET_B (needed for BUDYKO and BUDYKO_DA) ---------
    omega_basin = np.nan
    ET_B        = np.full(len(idx), np.nan)

    if scenario in ("BUDYKO", "BUDYKO_DA"):
        if basin_id not in omega_lookup:
            logging.warning(f"Omega not found for basin {basin_id}. Skipping.")
            return None
        omega_basin = float(omega_lookup[basin_id])
        ET_B = compute_fu_et(P=P, PET=PET, omega=omega_basin)
        if np.all(~np.isfinite(ET_B)):
            logging.warning(f"ET_B is all NaN for basin {basin_id}. Skipping.")
            return None

    # =============================================================
    # BASE
    # =============================================================
    if scenario == "BASE":
        Q_base, S_base, G_base = run_model_deterministic(
            P=P, PET=PET, params_cal=params_cal,
            S_init=S_init, G_init=G_init, Gmax_cal=Gmax_cal,
            ET_series=ET_ke,
        )
        results = pd.DataFrame({
            "time":      idx,
            "omega_MLR": np.full(len(idx), omega_basin),
            "P":         P,
            "PET":       PET,
            "ET_ke":     ET_ke,
            "ET_sacsma": ET_sacsma,    # SAC-SMA reference
            "Q_sacsma":  Q_sacsma,     # SAC-SMA reference
            "Q_obs":     Q_obs,
            "Q_base":    Q_base,
            "S_base":    S_base,
            "G_base":    G_base,
        }).set_index("time")
        qsim_name = "Q_base"

    # =============================================================
    # BUDYKO
    # =============================================================
    elif scenario == "BUDYKO":
        Q_budyko, S_budyko, G_budyko = run_model_deterministic(
            P=P, PET=PET, params_cal=params_cal,
            S_init=S_init, G_init=G_init, Gmax_cal=Gmax_cal,
            ET_series=ET_B,
        )
        results = pd.DataFrame({
            "time":      idx,
            "omega_MLR": np.full(len(idx), omega_basin),
            "P":         P,
            "PET":       PET,
            "ET_ke":     ET_ke,
            "ET_sacsma": ET_sacsma,
            "ET_B":      ET_B,
            "Q_sacsma":  Q_sacsma,
            "Q_obs":     Q_obs,
            "Q_budyko":  Q_budyko,
            "S_budyko":  S_budyko,
            "G_budyko":  G_budyko,
        }).set_index("time")
        qsim_name = "Q_budyko"

    # =============================================================
    # BUDYKO_DA
    # =============================================================
    elif scenario == "BUDYKO_DA":
        config = EnKFConfig(**da_cfg)

        ET_ass_mean, Q_ens_mean, enkf_hist = run_budyko_da(
            P          = P,
            PET        = PET,
            ET_obs     = ET_B,       # Budyko ET is the assimilation observation
            ET_model   = ET_ke,      # ET_ke drives the model inside the forecast step
            params_cal = params_cal,
            S_init     = S_init,
            G_init     = G_init,
            Smax_cal   = Smax_cal,
            Gmax_cal   = Gmax_cal,
            config     = config,
            basin_id   = basin_id,
        )

        Q_ass, S_ass, G_ass = run_model_deterministic(
            P=P, PET=PET, params_cal=params_cal,
            S_init=S_init, G_init=G_init, Gmax_cal=Gmax_cal,
            ET_series=ET_ass_mean,
        )

        results = pd.DataFrame({
            "time":      idx,
            "omega_MLR": np.full(len(idx), omega_basin),
            "P":         P,
            "PET":       PET,
            "ET_ke":     ET_ke,
            "ET_sacsma": ET_sacsma,
            "ET_B":      ET_B,
            "ET_ass":    ET_ass_mean,
            "Q_sacsma":  Q_sacsma,
            "Q_obs":     Q_obs,
            "Q_ass":     Q_ass,
            "Q_ens":     Q_ens_mean,
            "S_ass":     S_ass,
            "G_ass":     G_ass,
        }).set_index("time")

        # Save ensemble summary
        enkf_df = pd.DataFrame({
            "time":        idx,
            "ET_ens_mean": np.nanmean(enkf_hist["ET_ens"], axis=1),
            "Q_ens_mean":  np.nanmean(enkf_hist["Q_ens"],  axis=1),
            "S_ens_mean":  np.nanmean(enkf_hist["S_ens"],  axis=1),
            "G_ens_mean":  np.nanmean(enkf_hist["G_ens"],  axis=1),
        })
        enkf_df.to_feather(
            os.path.join(RESULT_DIR, f"enkf_ensemble_{scenario}_{basin_id}.feather")
        )
        qsim_name = "Q_ass"

    else:
        raise ValueError(f"Unknown scenario: {scenario!r}")

    # ---- Save main results ----------------------------------------
    results.reset_index().to_feather(
        os.path.join(RESULT_DIR, f"results_{scenario}_{basin_id}.feather")
    )

    # ---- Metrics against Q_USGS ----------------------------------
    qobs = results["Q_obs"].values
    qsim = results[qsim_name].values
    KGE  = calculate_kge(qobs, qsim)
    NSE  = calculate_nse(qobs, qsim)

    metrics: dict = {
        "gauge_id":   basin_id,
        "split":      split_label,
        "scenario":   scenario,
        "KGE":        float(KGE) if np.isfinite(KGE) else np.nan,
        "NSE":        float(NSE) if np.isfinite(NSE) else np.nan,
        "Kperc":      float(params_cal.Kperc),
        "Kb":         float(params_cal.Kb),
        "Ke":         float(params_cal.Ke),
        "Cqq":        float(params_cal.Cqq),
        "Smax":       float(params_cal.Smax),
        "S_init":     float(S_init),
        "G_init":     float(G_init),
        "Gmax":       float(Gmax_cal),
    }
    if scenario in ("BUDYKO", "BUDYKO_DA"):
        metrics["omega"] = float(omega_basin)

    return metrics


# =========================================================
# Main entry point
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

    global_params_path = os.path.join(PROJECT_ROOT, paths["global_calibrated_params"])
    if not os.path.exists(global_params_path):
        raise FileNotFoundError(
            f"Global calibrated parameter file not found: {global_params_path}"
        )
    with open(global_params_path) as f:
        global_params = json.load(f)

    _, omega_lookup = load_global_omega(cfg)
    train_basins, test_basins = load_saved_basin_split(cfg)

    # ---- Load and align inputs ----------------------------------
    aligned     = align_simulation_inputs(DATA_DIR)
    idx         = aligned["idx"]
    common_cols = set(aligned["common_cols"])

    train_basins = [b for b in train_basins if b in common_cols]
    test_basins  = [b for b in test_basins  if b in common_cols]

    target_group = str(cfg.get("prediction", {}).get("target_group", "test")).lower()
    if target_group == "train":
        basins, split_label = train_basins, "train"
    elif target_group == "all":
        basins, split_label = sorted(set(train_basins + test_basins)), "all"
    else:
        basins, split_label = test_basins, "test"

    if not basins:
        raise ValueError(
            f"No basins available for target_group='{target_group}' after alignment."
        )

    da_cfg     = cfg.get("da", {}).copy()
    da_enabled = bool(da_cfg.pop("enabled", True))

    par_cfg     = cfg.get("parallel", {})
    par_enabled = bool(par_cfg.get("enabled", True))
    max_workers = int(par_cfg.get("max_workers", -1))
    if max_workers <= 0:
        max_workers = max(1, cpu_count() - 1)

    # ---- Pre-extract per-basin arrays ---------------------------
    tasks = []
    for basin_id in basins:
        def _col(df, bid):
            return pd.to_numeric(
                df.get(bid, pd.Series(index=idx)).reindex(idx),
                errors="coerce",
            ).to_numpy(dtype=float)

        tasks.append({
            "basin_id":  basin_id,
            "idx":       idx,
            "P":         _col(aligned["Rainf_df"],     basin_id),
            "PET":       _col(aligned["PET_df"],       basin_id),
            "ET_sacsma": _col(aligned["ET_sacsma_df"], basin_id),
            "Q_sacsma":  _col(aligned["Q_sacsma_df"],  basin_id),
            "Q_obs":     _col(aligned["Q_usgs_df"],    basin_id),
        })

    # ---- Run scenarios ------------------------------------------
    for sc in scenarios_to_run:
        if sc == "BUDYKO_DA" and not da_enabled:
            continue

        RESULT_DIR = os.path.join(
            BASE_RESULT_DIR, scenario_folder_name(sc), split_label.upper()
        )
        os.makedirs(RESULT_DIR, exist_ok=True)
        all_metrics: list[dict] = []

        if par_enabled:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        simulate_basin,
                        basin_id   = task["basin_id"],
                        scenario   = sc,
                        RESULT_DIR = RESULT_DIR,
                        global_params = global_params,
                        omega_lookup  = omega_lookup,
                        da_cfg        = da_cfg,
                        split_label   = split_label,
                        idx           = task["idx"],
                        P             = task["P"],
                        PET           = task["PET"],
                        ET_sacsma     = task["ET_sacsma"],
                        Q_sacsma      = task["Q_sacsma"],
                        Q_obs         = task["Q_obs"],
                    ): task["basin_id"]
                    for task in tasks
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Running scenario={sc}, group={split_label}",
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
            for task in tqdm(tasks, desc=f"Running scenario={sc}, group={split_label}"):
                try:
                    out = simulate_basin(
                        basin_id   = task["basin_id"],
                        scenario   = sc,
                        RESULT_DIR = RESULT_DIR,
                        global_params = global_params,
                        omega_lookup  = omega_lookup,
                        da_cfg        = da_cfg,
                        split_label   = split_label,
                        idx           = task["idx"],
                        P             = task["P"],
                        PET           = task["PET"],
                        ET_sacsma     = task["ET_sacsma"],
                        Q_sacsma      = task["Q_sacsma"],
                        Q_obs         = task["Q_obs"],
                    )
                    if out is not None:
                        all_metrics.append(out)
                except Exception as exc:
                    logging.error(f"Basin {task['basin_id']!r} failed: {exc}",
                                  exc_info=True)

        pd.DataFrame(all_metrics).to_csv(
            os.path.join(RESULT_DIR, f"metrics_{sc}_{split_label}.csv"),
            index=False,
        )
        print(f"\n✅  scenario={sc}, group={split_label} — results in {RESULT_DIR}")


if __name__ == "__main__":
    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run_simulations_from_config(cfg)



