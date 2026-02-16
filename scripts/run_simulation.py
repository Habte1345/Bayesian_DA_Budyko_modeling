# scripts/run_simulation.py
import os
import sys
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ✅ DO NOT import anything from run.py (avoids circular import)

from src.model import ModelParams, two_store_model_step
from src.budyko import BudykoModelEstimator
from src.enkf import EnKFConfig, enkf_update_stochastic_scalar, enkf_forecast_step_states
from src.metrics import calculate_kge, calculate_nse


# =========================================================
# GLOBALS
# =========================================================
_GLOBAL = {
    "PET_df": None,
    "Rainf_df": None,
    "Evap_df": None,
    "Q_usgs_df": None,
    "Q_nldas_df": None,
    "Qsb_df": None,
    "M_df": None,
    "RootMoist_df": None,
    "Slp_df": None,
    "common_cols": None,
    # NOTE: state-aware Budyko is no longer precomputed globally/offline
    "calibrated_params": None,
    # ✅ NEW: global omega MLR coefficients (beta0, beta1, beta2)
    "omega_mlr_beta": None,
}


def scenario_folder_name(scenario: str) -> str:
    s = str(scenario).strip().upper()

    if s == "BASE":
        return "BASE_MODEL"
    elif s == "BUDYKO":
        return "BUDYKO_MODEL"
    elif s in ["BUDYKO_DA", "BUDYKO+DA", "DA", "ENKF"]:
        return "BUDYKO_DA"
    else:
        return f"{s}_SCENARIO"


# ---------------------------------------------------------
# Feather loader
# ---------------------------------------------------------
def load_feather_df(fname: str, ddir: str) -> pd.DataFrame:
    path = os.path.join(ddir, fname)
    if not os.path.exists(path):
        logging.warning(f"File not found: {path}. Returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.read_feather(path).dropna(axis=1, how="all")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

    return df


# ---------------------------------------------------------
# Load all inputs once
# ---------------------------------------------------------
def load_all_inputs(DATA_DIR: str) -> dict:
    PET_df = load_feather_df("PotEvap.feather", DATA_DIR)
    Rainf_df = load_feather_df("Rainf.feather", DATA_DIR)
    Evap_df = load_feather_df("EVap.feather", DATA_DIR)
    Q_usgs_df = load_feather_df("Q_USGS.feather", DATA_DIR)
    Q_nldas_df = load_feather_df("Q_nldas_mm_monthly.feather", DATA_DIR)
    Qsb_df = load_feather_df("Qsb.feather", DATA_DIR)
    M_df = load_feather_df("M.feather", DATA_DIR)
    RootMoist_df = load_feather_df("SoilM_0_200cm.feather", DATA_DIR)
    Slp_df = load_feather_df("slope.feather", DATA_DIR)

    common_cols = sorted(
        set(Evap_df.columns)
        & set(Qsb_df.columns)
        & set(PET_df.columns)
        & set(M_df.columns)
        & set(RootMoist_df.columns)
        & set(Slp_df.columns)
        & set(Rainf_df.columns)  # ✅ ensure P exists for state-aware Budyko
    )

    if len(common_cols) > 0:
        Evap_df = Evap_df[common_cols]
        Qsb_df = Qsb_df[common_cols]
        PET_df = PET_df[common_cols]
        M_df = M_df[common_cols]
        RootMoist_df = RootMoist_df[common_cols]
        Slp_df = RootMoist_df[common_cols]
        Rainf_df = Rainf_df[common_cols]

    return {
        "PET_df": PET_df,
        "Rainf_df": Rainf_df,
        "Evap_df": Evap_df,
        "Q_usgs_df": Q_usgs_df,
        "Q_nldas_df": Q_nldas_df,
        "Qsb_df": Qsb_df,
        "M_df": M_df,
        "RootMoist_df": RootMoist_df,
        "Slp_df": Slp_df,
        "common_cols": common_cols,
    }


# ---------------------------------------------------------
# Worker initializer (sets globals once per worker)
# ---------------------------------------------------------
def _init_worker(global_payload: dict):
    _GLOBAL.update(global_payload)


# # ---------------------------------------------------------
# # Deterministic single-run model (captures dS)
# # ---------------------------------------------------------
# def run_model_deterministic(
#     P: np.ndarray,
#     PET: np.ndarray,
#     params_cal: ModelParams,
#     S_init: float,
#     G_init: float,
#     Gmax_cal: float,
#     ET_series: np.ndarray,
# ):
#     L = len(P)
#     S, G = float(S_init), float(G_init)

#     Q_out = np.full(L, np.nan)
#     S_out = np.full(L, np.nan)
#     G_out = np.full(L, np.nan)
#     dS_out = np.full(L, np.nan)

#     for t in range(L):
#         P_t = float(P[t]) if np.isfinite(P[t]) else 0.0
#         PET_t = float(PET[t]) if np.isfinite(PET[t]) else 0.0

#         S, G, _, Q, *_rest = two_store_model_step(
#             S,
#             G,
#             P_t,
#             PET_t,
#             params_cal,
#             ET_override=float(ET_series[t]) if np.isfinite(ET_series[t]) else None,
#         )

#         # model.py now returns (..., Perc_t, dS) at the end
#         dS_t = np.nan
#         if len(_rest) >= 1:
#             # _rest layout from model.py:
#             # Qs_t, Qb_t, Perc_t, dS
#             if len(_rest) >= 4:
#                 dS_t = float(_rest[-1]) if np.isfinite(_rest[-1]) else np.nan

#         G = np.clip(G, 0.0, Gmax_cal)

#         Q_out[t] = max(Q, 0.0)
#         S_out[t] = S
#         G_out[t] = G
#         dS_out[t] = dS_t

#     return Q_out, S_out, G_out, dS_out

# ---------------------------------------------------------
# Deterministic single-run model (captures dS)
# ---------------------------------------------------------
def run_model_deterministic(
    P: np.ndarray,
    PET: np.ndarray,
    params_cal: ModelParams,
    S_init: float,
    G_init: float,
    Gmax_cal: float,
    ET_series: np.ndarray,
):
    L = len(P)
    S, G = float(S_init), float(G_init)

    Q_out  = np.full(L, np.nan)
    S_out  = np.full(L, np.nan)
    G_out  = np.full(L, np.nan)
    dS_out = np.full(L, np.nan)

    for t in range(L):
        P_t   = float(P[t]) if np.isfinite(P[t]) else 0.0
        PET_t = float(PET[t]) if np.isfinite(PET[t]) else 0.0

        # ✅ store previous soil state
        # store previous soil state (no longer needed for dS, but fine to keep)
        S_prev = S

        S, G, _ET, Q, Qs_t, Qb_t, Perc_t, dS_t = two_store_model_step(
            S,
            G,
            P_t,
            PET_t,
            params_cal,
            ET_override=float(ET_series[t]) if np.isfinite(ET_series[t]) else None,
        )

        G = np.clip(G, 0.0, Gmax_cal)

        Q_out[t]  = max(Q, 0.0)
        S_out[t]  = S
        G_out[t]  = G
        dS_out[t] = dS_t


    return Q_out, S_out, G_out, dS_out

# ---------------------------------------------------------
# Budyko Fu ET (state-aware via P_eff = P - dS)
# ---------------------------------------------------------
def fu_et_from_peff(P_eff: np.ndarray, PET: np.ndarray, omega: np.ndarray) -> np.ndarray:
    P_eff = np.asarray(P_eff, dtype=float)
    PET = np.asarray(PET, dtype=float)
    omega = np.asarray(omega, dtype=float)

    ET = np.full_like(P_eff, np.nan, dtype=float)

    ok = np.isfinite(P_eff) & np.isfinite(PET) & np.isfinite(omega) & (P_eff > 0) & (PET >= 0) & (omega > 0)
    if not np.any(ok):
        return ET

    phi = np.divide(PET[ok], P_eff[ok], out=np.full(np.sum(ok), np.nan), where=P_eff[ok] > 0)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        term = np.power(phi, omega[ok])
        e_over_p = 1.0 + phi - np.power(1.0 + term, 1.0 / omega[ok])

    e_over_p = np.where(np.isfinite(e_over_p), e_over_p, np.nan)
    ET_ok = P_eff[ok] * e_over_p

    # physical clipping
    ET_ok = np.clip(ET_ok, 0.0, np.minimum(P_eff[ok], PET[ok]))

    ET[ok] = ET_ok
    return ET


# ---------------------------------------------------------
# ✅ NEW: Fit global omega MLR once (across all basins & time)
# ---------------------------------------------------------
def fit_global_omega_mlr(
    basins: list,
    idx: pd.DatetimeIndex,
    PET_df: pd.DataFrame,
    Rainf_df: pd.DataFrame,
    M_df: pd.DataFrame,
    Slp_df: pd.DataFrame,
    calibrated_params: dict,
):
    rows = []
    targets = []

    for basin_id in basins:
        if basin_id not in calibrated_params:
            continue

        p = calibrated_params[basin_id]

        # Basin series
        PET = PET_df[basin_id].reindex(idx).to_numpy().ravel().astype(float)
        P = Rainf_df[basin_id].reindex(idx).to_numpy().ravel().astype(float)

        # params for BASE pass to get dS_base
        Smax_cal = float(p.get("Smax", 50.0))
        Gmax_factor = float(p.get("Gmax_factor", 4.0))
        Gmax_cal = Smax_cal * Gmax_factor
        S_init = float(p.get("S_init", 0.5 * Smax_cal))
        G_init = float(p.get("G_init", 0.5 * Gmax_cal))

        params_cal = ModelParams(
            Smax=Smax_cal,
            Kperc=p["Kperc"],
            Kb=p["Kb"],
            Ke=p["Ke"],
            Cqq=p["Cqq"],
            Sfc_frac=0.30,
            beta_et=2.0,
        )

        ET_ke = PET * float(params_cal.Ke)

        # BASE pass to get dS_base (state-aware)
        _, _, _, dS_base = run_model_deterministic(
            P=P,
            PET=PET,
            params_cal=params_cal,
            S_init=S_init,
            G_init=G_init,
            Gmax_cal=Gmax_cal,
            ET_series=ET_ke,
        )

        # Build per-basin Budyko estimator ONLY to compute omega_true
        P_df_b = pd.DataFrame({basin_id: P}, index=idx)
        dS_df_b = pd.DataFrame({basin_id: dS_base}, index=idx)
        PET_df_b = pd.DataFrame({basin_id: PET}, index=idx)

        bud = BudykoModelEstimator(
            P_df=P_df_b,
            dS_df=dS_df_b,
            PotEvap_df=PET_df_b,
            M_basin=M_df[[basin_id]].reindex(idx),
            Slope_basin=Slp_df[[basin_id]].reindex(idx),
            calibrated_params=calibrated_params,
            Ke_df=None,
        )
        omega_true_df = bud.compute_omega_true()
        omega_true = omega_true_df[basin_id].reindex(idx).to_numpy().ravel().astype(float)

        # predictors
        Mi = M_df[basin_id].reindex(idx).to_numpy().ravel().astype(float)

        # slope is constant per basin; broadcast to time
        slope_val = Slp_df[basin_id].iloc[0] if basin_id in Slp_df.columns else np.nan
        Si = np.full_like(Mi, float(slope_val) if np.isfinite(slope_val) else np.nan, dtype=float)

        mask = np.isfinite(Mi) & np.isfinite(Si) & np.isfinite(omega_true)
        if np.any(mask):
            Xi = np.column_stack([np.ones(mask.sum()), Mi[mask], Si[mask]])
            rows.append(Xi)
            targets.append(omega_true[mask])

    if not rows:
        raise ValueError("No valid data to fit GLOBAL omega MLR (check M/Slope/omega_true availability).")

    X = np.vstack(rows)
    Y = np.concatenate(targets)

    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return beta  # (beta0, beta1, beta2)


# ---------------------------------------------------------
# DA run (EnKF) -> assimilates ET_model toward ET_obs
# produces ET_ass_mean (posterior) and Q_ass_mean
# ---------------------------------------------------------
def run_budyko_da(
    P: np.ndarray,
    PET: np.ndarray,
    ET_obs: np.ndarray,     # truth (e.g., ET_B)
    ET_model: np.ndarray,   # model (e.g., ET_ke)
    params_cal: ModelParams,
    S_init: float,
    G_init: float,
    Smax_cal: float,
    Gmax_cal: float,
    config: EnKFConfig,
    basin_id: str,
):
    L = len(P)
    nens = int(config.nens)
    inflation = float(config.inflation)
    R_ET_var = float(config.R_ET_std) ** 2

    rng = np.random.default_rng(hash(basin_id) % (2**32 - 1))

    # Initial ensemble states [S,G]
    S0_ens = np.clip(
        S_init + rng.normal(0.0, 0.05 * Smax_cal, size=nens),
        0.0, Smax_cal
    )
    G0_ens = np.clip(
        G_init + rng.normal(0.0, 0.05 * Gmax_cal, size=nens),
        0.0, Gmax_cal
    )
    X = np.vstack([S0_ens, G0_ens])  # shape = (2, nens)

    # Save ensemble histories
    S_ens_hist  = np.full((L, nens), np.nan)
    G_ens_hist  = np.full((L, nens), np.nan)
    ET_ens_hist = np.full((L, nens), np.nan)
    Q_ens_hist  = np.full((L, nens), np.nan)

    # mean time series (posterior)
    ET_ass_mean = np.full(L, np.nan)
    Q_ass_mean  = np.full(L, np.nan)

    for t in range(L):

        # Forecast: ET is forced to ET_model[t] (ET_ke)
        ET_override_t = float(ET_model[t]) if np.isfinite(ET_model[t]) else None

        X_f, ET_ens_f, Q_ens_f = enkf_forecast_step_states(
            X=X,
            P_t=float(P[t]) if np.isfinite(P[t]) else 0.0,
            PET_t=float(PET[t]) if np.isfinite(PET[t]) else 0.0,
            params_cal=params_cal,
            Smax=Smax_cal,
            Gmax=Gmax_cal,
            rng=rng,

            proc_S_std=float(config.proc_S_std),
            proc_G_std=float(config.proc_G_std),
            P_std_frac=float(config.P_std_frac),
            PET_std_frac=float(config.PET_std_frac),
            ET_override=ET_override_t,
        )

        # Save prior (forecast) ET/Q
        ET_ens_hist[t, :] = ET_ens_f
        Q_ens_hist[t, :]  = Q_ens_f

        # Analysis: assimilate ET_obs[t] (ET_B) into state
        if np.isfinite(ET_obs[t]):
            X_a = enkf_update_stochastic_scalar(
                X=X_f,
                y_obs=float(ET_obs[t]),
                HX=ET_ens_f.copy(),
                R_var=R_ET_var,
                inflation=inflation,
                Smax=Smax_cal,
                Gmax=Gmax_cal,
                rng=rng,
            )
        else:
            X_a = X_f

        # Posterior: recompute ET/Q from updated states (no extra noise)
        _, ET_ens_a, Q_ens_a = enkf_forecast_step_states(
            X=X_a,
            P_t=float(P[t]) if np.isfinite(P[t]) else 0.0,
            PET_t=float(PET[t]) if np.isfinite(PET[t]) else 0.0,
            params_cal=params_cal,
            Smax=Smax_cal,
            Gmax=Gmax_cal,
            rng=rng,

            proc_S_std=0.0,
            proc_G_std=0.0,
            P_std_frac=0.0,
            PET_std_frac=0.0,

            ET_override=ET_override_t,
        )

        ET_ass_mean[t] = np.nanmean(ET_ens_a)
        Q_ass_mean[t]  = np.nanmean(Q_ens_a)

        # Continue with updated states
        X = X_a
        S_ens_hist[t, :] = X[0, :]
        G_ens_hist[t, :] = X[1, :]

    enkf_hist = {
        "time": np.arange(L),
        "nens": nens,
        "S_ens": S_ens_hist,
        "G_ens": G_ens_hist,
        "ET_ens": ET_ens_hist,   # prior ET ensemble (ET_model-driven)
        "Q_ens": Q_ens_hist,     # prior Q ensemble
    }

    return ET_ass_mean, Q_ass_mean, enkf_hist


# ---------------------------------------------------------
# Scenario normalization
# ---------------------------------------------------------
def normalize_scenario_key(s: str) -> str:
    s = str(s).strip().upper()

    mapping = {
        "ALL": "ALL",

        "BASE": "BASE",
        "BASE_MODEL": "BASE",
        "BASE-MODEL": "BASE",

        "BUDYKO": "BUDYKO",
        "BUDYKO_MODEL": "BUDYKO",
        "BUDYKO-MODEL": "BUDYKO",

        "BUDYKO_DA": "BUDYKO_DA",
        "BUDYKO_DA_MODEL": "BUDYKO_DA",
        "BUDYKO+DA": "BUDYKO_DA",
        "DA": "BUDYKO_DA",
        "ENKF": "BUDYKO_DA",
        "ASSIMILATION": "BUDYKO_DA",
    }

    if s not in mapping:
        raise ValueError(f"Unknown scenario: {s}")

    return mapping[s]


# ---------------------------------------------------------
# Main simulation per basin (state-aware Budyko)
# ---------------------------------------------------------
def simulate_basin(basin_id, scenario, DATA_DIR, RESULT_DIR, calibrated_params, da_cfg: dict):
    common_cols = _GLOBAL["common_cols"]
    if basin_id not in common_cols or basin_id not in _GLOBAL["calibrated_params"]:
        return None

    PET_df = _GLOBAL["PET_df"]
    Rainf_df = _GLOBAL["Rainf_df"]
    Evap_df = _GLOBAL["Evap_df"]
    Q_usgs_df = _GLOBAL["Q_usgs_df"]
    Q_nldas_df = _GLOBAL["Q_nldas_df"]
    Qsb_df = _GLOBAL["Qsb_df"]
    M_df = _GLOBAL["M_df"]
    Slp_df = _GLOBAL["Slp_df"]

    idx = Evap_df.index
    L = len(idx)

    p = _GLOBAL["calibrated_params"][basin_id]

    PET = PET_df[basin_id].reindex(idx).to_numpy().ravel()
    P = Rainf_df[basin_id].reindex(idx).to_numpy().ravel()
    Q_obs = Q_usgs_df.get(basin_id, pd.Series(index=idx)).reindex(idx).to_numpy().ravel()
    Q_nldas = Q_nldas_df.get(basin_id, pd.Series(index=idx)).reindex(idx).to_numpy().ravel()  # kept (even if unused)
    Qsb = Qsb_df.get(basin_id, pd.Series(index=idx)).reindex(idx).to_numpy().ravel()

    # Model params
    Smax_cal = float(p.get("Smax", 50.0))
    Gmax_factor = float(p.get("Gmax_factor", 4.0))
    Gmax_cal = Smax_cal * Gmax_factor

    S_init = float(p.get("S_init", 0.5 * Smax_cal))
    G_init = float(p.get("G_init", 0.5 * Gmax_cal))

    params_cal = ModelParams(
        Smax=Smax_cal,
        Kperc=p["Kperc"],
        Kb=p["Kb"],
        Ke=p["Ke"],
        Cqq=p["Cqq"],
        Sfc_frac=0.30,
        beta_et=2.0,
    )

    ET_ke = PET * params_cal.Ke

    # -----------------------------------------------------
    # ✅ STATE-AWARE omega_true uses dS from BASE pass (ET_ke)
    # -----------------------------------------------------
    _Q_tmp, _S_tmp, _G_tmp, dS_tmp = run_model_deterministic(
        P=P,
        PET=PET,
        params_cal=params_cal,
        S_init=S_init,
        G_init=G_init,
        Gmax_cal=Gmax_cal,
        ET_series=ET_ke,
    )
    # print(np.nanmin(P - dS_tmp), np.nanpercentile(P - dS_tmp, [1, 5, 10]))

    P_df_b = pd.DataFrame({basin_id: P}, index=idx)
    dS_df_b = pd.DataFrame({basin_id: dS_tmp}, index=idx)
    PET_df_b = pd.DataFrame({basin_id: PET}, index=idx)

    # --- compute omega_true per basin (NO per-basin MLR fit) ---
    budyko = BudykoModelEstimator(
        P_df=P_df_b,
        dS_df=dS_df_b,
        PotEvap_df=PET_df_b,
        M_basin=M_df[[basin_id]].reindex(idx),
        Slope_basin=Slp_df[[basin_id]].reindex(idx),
        calibrated_params=_GLOBAL["calibrated_params"],
        Ke_df=None,
    )
    omega_true_df = budyko.compute_omega_true()
    omega_true_all = omega_true_df[basin_id].reindex(idx).to_numpy().ravel()

    # -----------------------------------------------------
    # ✅ APPLY GLOBAL omega_MLR (beta from all basins)
    # -----------------------------------------------------
    beta = _GLOBAL.get("omega_mlr_beta", None)
    if beta is None or len(beta) != 3 or not np.all(np.isfinite(beta)):
        raise ValueError("Global omega_mlr_beta is missing/invalid. Fit it once in run_simulations_from_config().")

    M_series = M_df[basin_id].reindex(idx).to_numpy().ravel().astype(float)

    slope_val = np.nan
    if basin_id in Slp_df.columns:
        slope_val = Slp_df[basin_id].iloc[0]
    slope_val = float(slope_val) if np.isfinite(slope_val) else np.nan

    omega_MLR_all = beta[0] + beta[1] * M_series + beta[2] * slope_val
    omega_MLR_all = np.clip(omega_MLR_all, 1.0, 50.0)

    # -----------------------------------------------------
    # Compute ET_B using Fu with omega_MLR and P_eff = P - dS_tmp
    # -----------------------------------------------------
    P_eff = (P - dS_tmp).astype(float)
    ET_B = fu_et_from_peff(P_eff=P_eff, PET=PET, omega=omega_MLR_all)

    # -----------------------------------------------------
    # Run scenarios
    # -----------------------------------------------------
    enkf_hist = None
    ET_ass_mean = np.full(L, np.nan)
    Q_ass_mean = np.full(L, np.nan)

    if scenario == "BASE":
        Q_base, S_base, G_base, dS_base = run_model_deterministic(
            P=P,
            PET=PET,
            params_cal=params_cal,
            S_init=S_init,
            G_init=G_init,
            Gmax_cal=Gmax_cal,
            ET_series=ET_ke,
        )

        results = pd.DataFrame({
            "time": idx,
            "P": P,
            "PET": PET,
            "ET_ke": ET_ke,
            "dS_base": dS_base,
            "Q_bs": Qsb,
            "Q_obs": Q_obs,
            "Q_base": Q_base,
            "S_base": S_base,
            "G_base": G_base
        }).set_index("time")

    elif scenario == "BUDYKO":
        Q_budyko, S_budyko, G_budyko, dS_budyko = run_model_deterministic(
            P=P,
            PET=PET,
            params_cal=params_cal,
            S_init=S_init,
            G_init=G_init,
            Gmax_cal=Gmax_cal,
            ET_series=ET_B,
        )

        results = pd.DataFrame({
            "time": idx,
            "omega_true": omega_true_all,
            "omega_MLR": omega_MLR_all,
            "P": P,
            "PET": PET,
            "ET_B": ET_B,
            "dS_budyko": dS_budyko,
            "Q_obs": Q_obs,
            "Q_budyko": Q_budyko,
            "S_budyko": S_budyko,
            "G_budyko": G_budyko,
        }).set_index("time")

    elif scenario == "BUDYKO_DA":
        config = EnKFConfig(**da_cfg)

        ET_ass_mean, Q_ass_mean, enkf_hist = run_budyko_da(
            P=P,
            PET=PET,
            ET_obs=ET_B,     # truth (state-aware Budyko ET)
            ET_model=ET_ke,  # model
            params_cal=params_cal,
            S_init=S_init,
            G_init=G_init,
            Smax_cal=Smax_cal,
            Gmax_cal=Gmax_cal,
            config=config,
            basin_id=basin_id,
        )

        Q_ass, S_ass, G_ass, dS_ass = run_model_deterministic(
            P=P,
            PET=PET,
            params_cal=params_cal,
            S_init=S_init,
            G_init=G_init,
            Gmax_cal=Gmax_cal,
            ET_series=ET_ass_mean,
        )

        results = pd.DataFrame({
            "time": idx,
            "P": P,
            "PET": PET,
            "ET_B": ET_B,
            "ET_ass": ET_ass_mean,
            "dS_ass": dS_ass,
            "Q_obs": Q_obs,
            "Q_ass": Q_ass,
            "Q_ens": Q_ass_mean,
        }).set_index("time")

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Save results
    result_path = os.path.join(RESULT_DIR, f"results_{scenario}_{basin_id}.feather")
    results.reset_index().to_feather(result_path)

    if scenario == "BUDYKO_DA" and enkf_hist is not None:
        enkf_df = pd.DataFrame({
            "time": idx,
            "ET_ens_mean": np.nanmean(enkf_hist["ET_ens"], axis=1),
            "Q_ens_mean": np.nanmean(enkf_hist["Q_ens"], axis=1),
            "S_ens_mean": np.nanmean(enkf_hist["S_ens"], axis=1),
            "G_ens_mean": np.nanmean(enkf_hist["G_ens"], axis=1),
        })

        enkf_path = os.path.join(RESULT_DIR, f"enkf_ensemble_{scenario}_{basin_id}.feather")
        enkf_df.to_feather(enkf_path)

    # Metrics
    qobs = results["Q_obs"].values if "Q_obs" in results.columns else Q_obs

    qcol = [c for c in results.columns if c.startswith("Q_") and c != "Q_obs"]
    qsim_name = qcol[-1] if qcol else None
    qsim = results[qsim_name].values if qsim_name else None

    metrics = {
        "gauge_id": basin_id,
        "scenario": scenario,
        "KGE": calculate_kge(qobs, qsim) if qsim is not None else np.nan,
        "NSE": calculate_nse(qobs, qsim) if qsim is not None else np.nan,
    }

    return metrics


# ---------------------------------------------------------
# Run from config
# ---------------------------------------------------------
def run_simulations_from_config(cfg: dict):

    scenario = normalize_scenario_key(cfg["scenario"])
    if scenario == "ALL":
        scenarios_to_run = ["BASE", "BUDYKO", "BUDYKO_DA"]
    else:
        scenarios_to_run = [scenario]

    paths = cfg["paths"]
    DATA_DIR = os.path.join(PROJECT_ROOT, paths["data_dir"])

    BASE_RESULT_DIR = os.path.join(PROJECT_ROOT, paths["result_dir"])
    os.makedirs(BASE_RESULT_DIR, exist_ok=True)

    # calibrated params
    cal_path = os.path.join(PROJECT_ROOT, paths["calibrated_params"])
    with open(cal_path, "r") as f:
        calibrated_params = json.load(f)

    da_cfg = cfg.get("da", {})
    if "enabled" in da_cfg:
        da_cfg.pop("enabled")

    basin_subset = cfg.get("basins", {}).get("subset", None)

    # -----------------------------------------------------
    # LOAD ALL INPUTS ONCE (NO global/offline Budyko)
    # -----------------------------------------------------
    inputs = load_all_inputs(DATA_DIR)

    # Determine basins
    if basin_subset is None:
        basins = list(inputs["Evap_df"].columns)
    else:
        basins = list(basin_subset)

    # Filter basins to those present in common_cols AND calibrated_params
    common_cols = inputs["common_cols"]
    basins = [b for b in basins if (b in common_cols and b in calibrated_params)]

    # -----------------------------------------------------
    # ✅ FIT GLOBAL omega_MLR ONCE (across ALL basins)
    # -----------------------------------------------------
    idx = inputs["Evap_df"].index
    beta = fit_global_omega_mlr(
        basins=basins,
        idx=idx,
        PET_df=inputs["PET_df"],
        Rainf_df=inputs["Rainf_df"],
        M_df=inputs["M_df"],
        Slp_df=inputs["Slp_df"],
        calibrated_params=calibrated_params,
    )

    # Prepare global payload (shipped once per worker)
    global_payload = {
        "PET_df": inputs["PET_df"],
        "Rainf_df": inputs["Rainf_df"],
        "Evap_df": inputs["Evap_df"],
        "Q_usgs_df": inputs["Q_usgs_df"],
        "Q_nldas_df": inputs["Q_nldas_df"],
        "Qsb_df": inputs["Qsb_df"],
        "M_df": inputs["M_df"],
        "RootMoist_df": inputs["RootMoist_df"],
        "Slp_df": inputs["Slp_df"],
        "common_cols": common_cols,
        "calibrated_params": calibrated_params,
        "omega_mlr_beta": beta,  # ✅ NEW
    }

    # Set globals for serial mode (this process)
    _init_worker(global_payload)

    # parallel settings
    par_cfg = cfg.get("parallel", {})
    par_enabled = bool(par_cfg.get("enabled", True))
    max_workers = int(par_cfg.get("max_workers", -1))
    if max_workers == -1:
        max_workers = max(1, cpu_count() - 1)

    # -----------------------------------------------------
    # RUN EACH SCENARIO
    # -----------------------------------------------------
    for scenario in scenarios_to_run:

        scenario_dir = scenario_folder_name(scenario)
        RESULT_DIR = os.path.join(BASE_RESULT_DIR, scenario_dir)
        os.makedirs(RESULT_DIR, exist_ok=True)

        all_metrics = []

        if par_enabled:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker,
                initargs=(global_payload,),
            ) as executor:
                futures = {
                    executor.submit(
                        simulate_basin, b, scenario, DATA_DIR, RESULT_DIR, calibrated_params, da_cfg
                    ): b
                    for b in basins
                }

                for f in tqdm(as_completed(futures), total=len(futures), desc=f"Running scenario={scenario}"):
                    out = f.result()
                    if out is not None:
                        all_metrics.append(out)
        else:
            for b in tqdm(basins, desc=f"Running scenario={scenario}"):
                out = simulate_basin(b, scenario, DATA_DIR, RESULT_DIR, calibrated_params, da_cfg)
                if out is not None:
                    all_metrics.append(out)

        pd.DataFrame(all_metrics).to_csv(
            os.path.join(RESULT_DIR, f"metrics_{scenario}.csv"),
            index=False
        )

        print(f"\n✅ Completed successfully: scenario={scenario}. Results saved to {RESULT_DIR}")


































# # scripts/run_simulation.py
# import os
# import sys
# import json
# import logging
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from multiprocessing import cpu_count

# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# # ✅ DO NOT import anything from run.py (avoids circular import)

# from src.model import ModelParams, two_store_model_step
# from src.budyko import BudykoModelEstimator
# from src.enkf import EnKFConfig, enkf_update_stochastic_scalar, enkf_forecast_step_states
# from src.metrics import calculate_kge, calculate_nse


# # =========================================================
# # GLOBALS 
# # =========================================================
# _GLOBAL = {
#     "PET_df": None,
#     "Rainf_df": None,
#     "Evap_df": None,
#     "Q_usgs_df": None,
#     "Q_nldas_df": None,
#     "Qsb_df": None,
#     "M_df": None,
#     "RootMoist_df": None,
#     "Slp_df": None,
#     "common_cols": None,
#     "ET_B_df": None,
#     "omega_true_df": None,
#     "omega_MLR_df": None,
#     "calibrated_params": None,
# }


# def scenario_folder_name(scenario: str) -> str:
#     s = str(scenario).strip().upper()

#     if s == "BASE":
#         return "BASE_MODEL"
#     elif s == "BUDYKO":
#         return "BUDYKO_MODEL"
#     elif s in ["BUDYKO_DA", "BUDYKO+DA", "DA", "ENKF"]:
#         return "BUDYKO_DA"
#     else:
#         return f"{s}_SCENARIO"


# # ---------------------------------------------------------
# # Feather loader
# # ---------------------------------------------------------
# def load_feather_df(fname: str, ddir: str) -> pd.DataFrame:
#     path = os.path.join(ddir, fname)
#     if not os.path.exists(path):
#         logging.warning(f"File not found: {path}. Returning empty DataFrame.")
#         return pd.DataFrame()

#     df = pd.read_feather(path).dropna(axis=1, how="all")

#     if "time" in df.columns:
#         df["time"] = pd.to_datetime(df["time"])
#         df.set_index("time", inplace=True)

#     return df


# # ---------------------------------------------------------
# # Load all inputs once
# # ---------------------------------------------------------
# def load_all_inputs(DATA_DIR: str) -> dict:
#     PET_df = load_feather_df("PotEvap.feather", DATA_DIR)
#     Rainf_df = load_feather_df("Rainf.feather", DATA_DIR)
#     Evap_df = load_feather_df("EVap.feather", DATA_DIR)
#     Q_usgs_df = load_feather_df("Q_USGS.feather", DATA_DIR)
#     Q_nldas_df = load_feather_df("Q_nldas_mm_monthly.feather", DATA_DIR)
#     Qsb_df = load_feather_df("Qsb.feather", DATA_DIR)
#     M_df = load_feather_df("M.feather", DATA_DIR)
#     RootMoist_df = load_feather_df("SoilM_0_200cm.feather", DATA_DIR)
#     Slp_df = load_feather_df("slope.feather", DATA_DIR)

#     # Same intersection logic you had
#     common_cols = sorted(
#         set(Evap_df.columns)
#         & set(Qsb_df.columns)
#         & set(PET_df.columns)
#         & set(M_df.columns)
#         & set(RootMoist_df.columns)
#         & set(Slp_df.columns)
#     )

#     # Restrict to common columns (like your code)
#     if len(common_cols) > 0:
#         Evap_df = Evap_df[common_cols]
#         Qsb_df = Qsb_df[common_cols]
#         PET_df = PET_df[common_cols]
#         M_df = M_df[common_cols]
#         RootMoist_df = RootMoist_df[common_cols]
#         Slp_df = Slp_df[common_cols]

#     return {
#         "PET_df": PET_df,
#         "Rainf_df": Rainf_df,
#         "Evap_df": Evap_df,
#         "Q_usgs_df": Q_usgs_df,
#         "Q_nldas_df": Q_nldas_df,
#         "Qsb_df": Qsb_df,
#         "M_df": M_df,
#         "RootMoist_df": RootMoist_df,
#         "Slp_df": Slp_df,
#         "common_cols": common_cols,
#     }


# # ---------------------------------------------------------
# # Compute Budyko once for ALL basins
# # ---------------------------------------------------------
# def compute_budyko_once(inputs: dict, calibrated_params: dict) -> dict:
#     Evap_df = inputs["Evap_df"]
#     Qsb_df = inputs["Qsb_df"]
#     PET_df = inputs["PET_df"]
#     RootMoist_df = inputs["RootMoist_df"]
#     M_df = inputs["M_df"]
#     Slp_df = inputs["Slp_df"]
#     common_cols = inputs["common_cols"]

#     if Evap_df is None or len(common_cols) == 0:
#         return {
#             "ET_B_df": pd.DataFrame(index=getattr(Evap_df, "index", None)),
#             "omega_true_df": pd.DataFrame(index=getattr(Evap_df, "index", None)),
#             "omega_MLR_df": pd.DataFrame(index=getattr(Evap_df, "index", None)),
#         }

#     # IMPORTANT: keep the same arguments you were using to avoid changing results
#     budyko = BudykoModelEstimator(
#         Evap_df=Evap_df[common_cols],
#         Qsb_monthly=Qsb_df[common_cols],
#         PotEvap_df=PET_df[common_cols],
#         M_basin=M_df[common_cols],
#         Slope_basin=Slp_df[common_cols],  
#         calibrated_params=calibrated_params,
#     )

#     budyko.estimate_budyko_et()

#     # These are expected to be DataFrames (basins as columns)
#     ET_B_df = budyko.ET_B.reindex(Evap_df.index)
#     omega_true_df = budyko.omega_true.reindex(Evap_df.index)
#     omega_MLR_df = budyko.omega_MLR.reindex(Evap_df.index)

#     return {
#         "ET_B_df": ET_B_df,
#         "omega_true_df": omega_true_df,
#         "omega_MLR_df": omega_MLR_df,
#     }


# # ---------------------------------------------------------
# # Worker initializer (sets globals once per worker)
# # ---------------------------------------------------------
# def _init_worker(global_payload: dict):
#     _GLOBAL.update(global_payload)


# # ---------------------------------------------------------
# # Deterministic single-run model
# # ---------------------------------------------------------
# def run_model_deterministic(
#     P: np.ndarray,
#     PET: np.ndarray,
#     params_cal: ModelParams,
#     S_init: float,
#     G_init: float,
#     Gmax_cal: float,
#     ET_series: np.ndarray,
# ):
#     L = len(P)
#     S, G = float(S_init), float(G_init)

#     Q_out, S_out, G_out = np.full(L, np.nan), np.full(L, np.nan), np.full(L, np.nan)

#     for t in range(L):
#         P_t = float(P[t]) if np.isfinite(P[t]) else 0.0
#         PET_t = float(PET[t]) if np.isfinite(PET[t]) else 0.0

#         S, G, _, Q, *_ = two_store_model_step(
#             S,
#             G,
#             P_t,
#             PET_t,
#             params_cal,
#             ET_override=float(ET_series[t]) if np.isfinite(ET_series[t]) else None,
#         )

#         G = np.clip(G, 0.0, Gmax_cal)

#         Q_out[t] = max(Q, 0.0)
#         S_out[t] = S
#         G_out[t] = G

#     return Q_out, S_out, G_out


# # ---------------------------------------------------------
# # DA run (EnKF) -> assimilates ET_model toward ET_obs
# # produces ET_ass_mean (posterior) and Q_ass_mean
# # ---------------------------------------------------------
# def run_budyko_da(
#     P: np.ndarray,
#     PET: np.ndarray,
#     ET_obs: np.ndarray,     # truth (e.g., ET_B)
#     ET_model: np.ndarray,   # model (e.g., ET_ke)
#     params_cal: ModelParams,
#     S_init: float,
#     G_init: float,
#     Smax_cal: float,
#     Gmax_cal: float,
#     config: EnKFConfig,
#     basin_id: str,
# ):
#     L = len(P)
#     nens = int(config.nens)
#     inflation = float(config.inflation)
#     R_ET_var = float(config.R_ET_std) ** 2

#     rng = np.random.default_rng(hash(basin_id) % (2**32 - 1))

#     # Initial ensemble states [S,G]
#     S0_ens = np.clip(
#         S_init + rng.normal(0.0, 0.05 * Smax_cal, size=nens),
#         0.0, Smax_cal
#     )
#     G0_ens = np.clip(
#         G_init + rng.normal(0.0, 0.05 * Gmax_cal, size=nens),
#         0.0, Gmax_cal
#     )
#     X = np.vstack([S0_ens, G0_ens])  # shape = (2, nens)

#     # Save ensemble histories
#     S_ens_hist  = np.full((L, nens), np.nan)
#     G_ens_hist  = np.full((L, nens), np.nan)
#     ET_ens_hist = np.full((L, nens), np.nan)
#     Q_ens_hist  = np.full((L, nens), np.nan)

#     # mean time series (posterior)
#     ET_ass_mean = np.full(L, np.nan)
#     Q_ass_mean  = np.full(L, np.nan)

#     for t in range(L):

#         # Forecast: ET is forced to ET_model[t] (ET_ke)
#         ET_override_t = float(ET_model[t]) if np.isfinite(ET_model[t]) else None

#         X_f, ET_ens_f, Q_ens_f = enkf_forecast_step_states(
#             X=X,
#             P_t=float(P[t]) if np.isfinite(P[t]) else 0.0,
#             PET_t=float(PET[t]) if np.isfinite(PET[t]) else 0.0,
#             params_cal=params_cal,
#             Smax=Smax_cal,
#             Gmax=Gmax_cal,
#             rng=rng,

#             proc_S_std=float(config.proc_S_std),
#             proc_G_std=float(config.proc_G_std),
#             P_std_frac=float(config.P_std_frac),
#             PET_std_frac=float(config.PET_std_frac),
#             ET_override=ET_override_t,
#         )

#         # Save prior (forecast) ET/Q
#         ET_ens_hist[t, :] = ET_ens_f
#         Q_ens_hist[t, :]  = Q_ens_f

#         # Analysis: assimilate ET_obs[t] (ET_B) into state
#         if np.isfinite(ET_obs[t]):
#             X_a = enkf_update_stochastic_scalar(
#                 X=X_f,
#                 y_obs=float(ET_obs[t]),
#                 HX=ET_ens_f.copy(),
#                 R_var=R_ET_var,
#                 inflation=inflation,
#                 Smax=Smax_cal,
#                 Gmax=Gmax_cal,
#                 rng=rng,
#             )
#         else:
#             X_a = X_f

#         # Posterior: recompute ET/Q from updated states (no extra noise)
#         _, ET_ens_a, Q_ens_a = enkf_forecast_step_states(
#             X=X_a,
#             P_t=float(P[t]) if np.isfinite(P[t]) else 0.0,
#             PET_t=float(PET[t]) if np.isfinite(PET[t]) else 0.0,
#             params_cal=params_cal,
#             Smax=Smax_cal,
#             Gmax=Gmax_cal,
#             rng=rng,

#             proc_S_std=0.0,
#             proc_G_std=0.0,
#             P_std_frac=0.0,
#             PET_std_frac=0.0,

#             ET_override=ET_override_t,
#         )

#         ET_ass_mean[t] = np.nanmean(ET_ens_a)
#         Q_ass_mean[t]  = np.nanmean(Q_ens_a)

#         # Continue with updated states
#         X = X_a
#         S_ens_hist[t, :] = X[0, :]
#         G_ens_hist[t, :] = X[1, :]

#     enkf_hist = {
#         "time": np.arange(L),
#         "nens": nens,
#         "S_ens": S_ens_hist,
#         "G_ens": G_ens_hist,
#         "ET_ens": ET_ens_hist,   # prior ET ensemble (ET_model-driven)
#         "Q_ens": Q_ens_hist,     # prior Q ensemble
#     }

#     return ET_ass_mean, Q_ass_mean, enkf_hist


# # ---------------------------------------------------------
# # Scenario normalization
# # ---------------------------------------------------------
# def normalize_scenario_key(s: str) -> str:
#     s = str(s).strip().upper()

#     mapping = {
#         "ALL": "ALL",

#         "BASE": "BASE",
#         "BASE_MODEL": "BASE",
#         "BASE-MODEL": "BASE",

#         "BUDYKO": "BUDYKO",
#         "BUDYKO_MODEL": "BUDYKO",
#         "BUDYKO-MODEL": "BUDYKO",

#         "BUDYKO_DA": "BUDYKO_DA",
#         "BUDYKO_DA_MODEL": "BUDYKO_DA",
#         "BUDYKO+DA": "BUDYKO_DA",
#         "DA": "BUDYKO_DA",
#         "ENKF": "BUDYKO_DA",
#         "ASSIMILATION": "BUDYKO_DA",
#     }

#     if s not in mapping:
#         raise ValueError(f"Unknown scenario: {s}")

#     return mapping[s]


# # ---------------------------------------------------------
# # Main simulation per basin
# # ---------------------------------------------------------
# def simulate_basin(basin_id, scenario, DATA_DIR, RESULT_DIR, calibrated_params, da_cfg: dict):
#     # NOTE: DATA_DIR and calibrated_params are kept in signature to avoid changing your call sites,
#     # but we use the preloaded globals for speed.

#     common_cols = _GLOBAL["common_cols"]
#     if basin_id not in common_cols or basin_id not in _GLOBAL["calibrated_params"]:
#         return None

#     PET_df = _GLOBAL["PET_df"]
#     Rainf_df = _GLOBAL["Rainf_df"]
#     Evap_df = _GLOBAL["Evap_df"]
#     Q_usgs_df = _GLOBAL["Q_usgs_df"]
#     Q_nldas_df = _GLOBAL["Q_nldas_df"]
#     Qsb_df = _GLOBAL["Qsb_df"]

#     # Budyko outputs (precomputed once)
#     ET_B_df = _GLOBAL["ET_B_df"]
#     omega_true_df = _GLOBAL["omega_true_df"]
#     omega_MLR_df = _GLOBAL["omega_MLR_df"]

#     idx = Evap_df.index
#     L = len(idx)

#     p = _GLOBAL["calibrated_params"][basin_id]

#     PET = PET_df[basin_id].values
#     P = Rainf_df.get(basin_id, pd.Series(index=idx)).reindex(idx).values
#     Q_obs = Q_usgs_df.get(basin_id, pd.Series(index=idx)).reindex(idx).values
#     Q_nldas = Q_nldas_df.get(basin_id, pd.Series(index=idx)).reindex(idx).values  # kept (even if unused)
#     Qsb = Qsb_df.get(basin_id, pd.Series(index=idx)).reindex(idx).values

#     # Budyko series for this basin (already computed)
#     omega_true_all = omega_true_df[basin_id].reindex(idx).to_numpy().ravel()
#     omega_MLR_all  = omega_MLR_df[basin_id].reindex(idx).to_numpy().ravel()
#     ET_B           = ET_B_df[basin_id].reindex(idx).to_numpy().ravel()

#     # Model params
#     Smax_cal = float(p.get("Smax", 50.0))
#     Gmax_factor = float(p.get("Gmax_factor", 4.0))
#     Gmax_cal = Smax_cal * Gmax_factor

#     S_init = float(p.get("S_init", 0.5 * Smax_cal))
#     G_init = float(p.get("G_init", 0.5 * Gmax_cal))

#     params_cal = ModelParams(
#         Smax=Smax_cal,
#         Kperc=p["Kperc"],
#         Kb=p["Kb"],
#         Ke=p["Ke"],
#         Cqq=p["Cqq"],
#         Sfc_frac=0.30,
#         beta_et=2.0,
#     )

#     ET_ke = PET * params_cal.Ke

#     enkf_hist = None  # NO DA
#     ET_ass_mean = np.full(L, np.nan)
#     Q_ass_mean = np.full(L, np.nan)

#     if scenario == "BASE":
#         Q_base, S_base, G_base = run_model_deterministic(
#             P=P,
#             PET=PET,
#             params_cal=params_cal,
#             S_init=S_init,
#             G_init=G_init,
#             Gmax_cal=Gmax_cal,
#             ET_series=ET_ke,
#         )

#         results = pd.DataFrame({
#             "time": idx,
#             "P": P,
#             "PET": PET,
#             "ET_ke": ET_ke,
#             "Q_bs": Qsb,
#             "Q_obs": Q_obs,
#             "Q_base": Q_base,
#             "S_base": S_base,
#             "G_base": G_base
#         }).set_index("time")

#     elif scenario == "BUDYKO":
#         Q_budyko, S_budyko, G_budyko = run_model_deterministic(
#             P=P,
#             PET=PET,
#             params_cal=params_cal,
#             S_init=S_init,
#             G_init=G_init,
#             Gmax_cal=Gmax_cal,
#             ET_series=ET_B,
#         )

#         results = pd.DataFrame({
#             "time": idx,
#             "omega_true": omega_true_all,
#             "omega_MLR": omega_MLR_all,
#             "P": P,
#             "PET": PET,
#             "ET_B": ET_B,
#             "Q_obs": Q_obs,
#             "Q_budyko": Q_budyko,
#             "S_budyko": S_budyko,
#             "G_budyko": G_budyko,
#         }).set_index("time")

#     elif scenario == "BUDYKO_DA":
#         config = EnKFConfig(**da_cfg)

#         ET_ass_mean, Q_ass_mean, enkf_hist = run_budyko_da(
#             P=P,
#             PET=PET,
#             ET_obs=ET_B,     # truth
#             ET_model=ET_ke,  # model
#             params_cal=params_cal,
#             S_init=S_init,
#             G_init=G_init,
#             Smax_cal=Smax_cal,
#             Gmax_cal=Gmax_cal,
#             config=config,
#             basin_id=basin_id,
#         )

#         Q_ass, S_ass, G_ass = run_model_deterministic(
#             P=P,
#             PET=PET,
#             params_cal=params_cal,
#             S_init=S_init,
#             G_init=G_init,
#             Gmax_cal=Gmax_cal,
#             ET_series=ET_ass_mean,
#         )

#         results = pd.DataFrame({
#             "time": idx,
#             "P": P,
#             "PET": PET,
#             "ET_B": ET_B,
#             "ET_ass": ET_ass_mean,
#             "Q_obs": Q_obs,
#             "Q_ass": Q_ass,
#             "Q_ens": Q_ass_mean,
#         }).set_index("time")

#     else:
#         raise ValueError(f"Unknown scenario: {scenario}")

#     # Save results
#     result_path = os.path.join(RESULT_DIR, f"results_{scenario}_{basin_id}.feather")
#     results.reset_index().to_feather(result_path)

#     if scenario == "BUDYKO_DA" and enkf_hist is not None:
#         enkf_df = pd.DataFrame({
#             "time": idx,
#             "ET_ens_mean": np.nanmean(enkf_hist["ET_ens"], axis=1),
#             "Q_ens_mean": np.nanmean(enkf_hist["Q_ens"], axis=1),
#             "S_ens_mean": np.nanmean(enkf_hist["S_ens"], axis=1),
#             "G_ens_mean": np.nanmean(enkf_hist["G_ens"], axis=1),
#         })

#         enkf_path = os.path.join(RESULT_DIR, f"enkf_ensemble_{scenario}_{basin_id}.feather")
#         enkf_df.to_feather(enkf_path)

#     # Metrics
#     qobs = results["Q_obs"].values if "Q_obs" in results.columns else Q_obs

#     qcol = [c for c in results.columns if c.startswith("Q_") and c != "Q_obs"]
#     qsim_name = qcol[-1] if qcol else None
#     qsim = results[qsim_name].values if qsim_name else None

#     metrics = {
#         "gauge_id": basin_id,
#         "scenario": scenario,
#         "KGE": calculate_kge(qobs, qsim) if qsim is not None else np.nan,
#         "NSE": calculate_nse(qobs, qsim) if qsim is not None else np.nan,
#     }

#     return metrics


# # ---------------------------------------------------------
# # Run from config
# # ---------------------------------------------------------
# def run_simulations_from_config(cfg: dict):

#     scenario = normalize_scenario_key(cfg["scenario"])
#     if scenario == "ALL":
#         scenarios_to_run = ["BASE", "BUDYKO", "BUDYKO_DA"]
#     else:
#         scenarios_to_run = [scenario]

#     paths = cfg["paths"]
#     DATA_DIR = os.path.join(PROJECT_ROOT, paths["data_dir"])

#     BASE_RESULT_DIR = os.path.join(PROJECT_ROOT, paths["result_dir"])
#     os.makedirs(BASE_RESULT_DIR, exist_ok=True)

#     # calibrated params
#     cal_path = os.path.join(PROJECT_ROOT, paths["calibrated_params"])
#     with open(cal_path, "r") as f:
#         calibrated_params = json.load(f)

#     da_cfg = cfg.get("da", {})
#     if "enabled" in da_cfg:
#         da_cfg.pop("enabled")

#     basin_subset = cfg.get("basins", {}).get("subset", None)

#     # -----------------------------------------------------
#     # LOAD ALL INPUTS ONCE + COMPUTE BUDYKO ONCE
#     # -----------------------------------------------------
#     inputs = load_all_inputs(DATA_DIR)

#     # Determine basins
#     if basin_subset is None:
#         basins = list(inputs["Evap_df"].columns)
#     else:
#         basins = list(basin_subset)

#     # Filter basins to those present in common_cols AND calibrated_params
#     common_cols = inputs["common_cols"]
#     basins = [b for b in basins if (b in common_cols and b in calibrated_params)]

#     budyko_out = compute_budyko_once(inputs, calibrated_params)

#     # Prepare global payload (shipped once per worker)
#     global_payload = {
#         "PET_df": inputs["PET_df"],
#         "Rainf_df": inputs["Rainf_df"],
#         "Evap_df": inputs["Evap_df"],
#         "Q_usgs_df": inputs["Q_usgs_df"],
#         "Q_nldas_df": inputs["Q_nldas_df"],
#         "Qsb_df": inputs["Qsb_df"],
#         "M_df": inputs["M_df"],
#         "RootMoist_df": inputs["RootMoist_df"],
#         "Slp_df": inputs["Slp_df"],
#         "common_cols": common_cols,
#         "ET_B_df": budyko_out["ET_B_df"],
#         "omega_true_df": budyko_out["omega_true_df"],
#         "omega_MLR_df": budyko_out["omega_MLR_df"],
#         "calibrated_params": calibrated_params,
#     }

#     # Set globals for serial mode (this process)
#     _init_worker(global_payload)

#     # parallel settings
#     par_cfg = cfg.get("parallel", {})
#     par_enabled = bool(par_cfg.get("enabled", True))
#     max_workers = int(par_cfg.get("max_workers", -1))
#     if max_workers == -1:
#         max_workers = max(1, cpu_count() - 1)

#     # -----------------------------------------------------
#     # RUN EACH SCENARIO
#     # -----------------------------------------------------
#     for scenario in scenarios_to_run:

#         scenario_dir = scenario_folder_name(scenario)
#         RESULT_DIR = os.path.join(BASE_RESULT_DIR, scenario_dir)
#         os.makedirs(RESULT_DIR, exist_ok=True)

#         all_metrics = []

#         if par_enabled:
#             with ProcessPoolExecutor(
#                 max_workers=max_workers,
#                 initializer=_init_worker,
#                 initargs=(global_payload,),
#             ) as executor:
#                 futures = {
#                     executor.submit(
#                         simulate_basin, b, scenario, DATA_DIR, RESULT_DIR, calibrated_params, da_cfg
#                     ): b
#                     for b in basins
#                 }

#                 for f in tqdm(as_completed(futures), total=len(futures), desc=f"Running scenario={scenario}"):
#                     out = f.result()
#                     if out is not None:
#                         all_metrics.append(out)
#         else:
#             for b in tqdm(basins, desc=f"Running scenario={scenario}"):
#                 out = simulate_basin(b, scenario, DATA_DIR, RESULT_DIR, calibrated_params, da_cfg)
#                 if out is not None:
#                     all_metrics.append(out)

#         pd.DataFrame(all_metrics).to_csv(
#             os.path.join(RESULT_DIR, f"metrics_{scenario}.csv"),
#             index=False
#         )

#         print(f"\n✅ Completed successfully: scenario={scenario}. Results saved to {RESULT_DIR}")













# # scripts/run_simulation.py
# import os
# import sys
# import json
# import logging
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from multiprocessing import cpu_count

# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# # ✅ DO NOT import anything from run.py (avoids circular import)

# from src.model import ModelParams, two_store_model_step
# from src.budyko import BudykoModelEstimator
# from src.enkf import EnKFConfig, enkf_update_stochastic_scalar, enkf_forecast_step_states
# from src.metrics import calculate_kge, calculate_nse


# def scenario_folder_name(scenario: str) -> str:
#     s = str(scenario).strip().upper()

#     if s == "BASE":
#         return "BASE_MODEL"
#     elif s == "BUDYKO":
#         return "BUDYKO_MODEL"
#     elif s in ["BUDYKO_DA", "BUDYKO+DA", "DA", "ENKF"]:
#         return "BUDYKO_DA"
#     else:
#         return f"{s}_SCENARIO"


# # ---------------------------------------------------------
# # Feather loader
# # ---------------------------------------------------------
# def load_feather_df(fname: str, ddir: str) -> pd.DataFrame:
#     path = os.path.join(ddir, fname)
#     if not os.path.exists(path):
#         logging.warning(f"File not found: {path}. Returning empty DataFrame.")
#         return pd.DataFrame()

#     df = pd.read_feather(path).dropna(axis=1, how="all")

#     if "time" in df.columns:
#         df["time"] = pd.to_datetime(df["time"])
#         df.set_index("time", inplace=True)

#     return df


# # ---------------------------------------------------------
# # Deterministic single-run model
# # ---------------------------------------------------------
# def run_model_deterministic(
#     P: np.ndarray,
#     PET: np.ndarray,
#     params_cal: ModelParams,
#     S_init: float,
#     G_init: float,
#     Gmax_cal: float,
#     ET_series: np.ndarray,
# ):
#     L = len(P)
#     S, G = float(S_init), float(G_init)

#     Q_out, S_out, G_out = np.full(L, np.nan), np.full(L, np.nan), np.full(L, np.nan)

#     for t in range(L):
#         P_t = float(P[t]) if np.isfinite(P[t]) else 0.0
#         PET_t = float(PET[t]) if np.isfinite(PET[t]) else 0.0

#         S, G, _, Q, *_ = two_store_model_step(
#             S, G, P_t, PET_t, params_cal,
#             ET_override=float(ET_series[t]) if np.isfinite(ET_series[t]) else None # given  new ET, update storages (S,G) and compute Q
#         )

#         G = np.clip(G, 0.0, Gmax_cal)

#         Q_out[t] = max(Q, 0.0)
#         S_out[t] = S
#         G_out[t] = G

#     return Q_out, S_out, G_out


# # # ---------------------------------------------------------
# # # DA run (EnKF) -> produces ET_ass and Q_ass_mean
# # # ---------------------------------------------------------
# # def run_budyko_da(
# #     P: np.ndarray,
# #     PET: np.ndarray,
# #     ET_B: np.ndarray,
# #     params_cal: ModelParams,
# #     S_init: float,
# #     G_init: float,
# #     Smax_cal: float,
# #     Gmax_cal: float,
# #     config: EnKFConfig,
# #     basin_id: str,
# # ):
# #     L = len(P)
# #     nens = int(config.nens)
# #     inflation = float(config.inflation)
# #     R_ET_var = float(config.R_ET_std) ** 2

# #     rng = np.random.default_rng(hash(basin_id) % (2**32 - 1))

# #     # Initial ensemble states [S,G]
# #     S0_ens = np.clip(S_init + rng.normal(0.0, 0.05 * Smax_cal, size=nens), 0.0, Smax_cal)
# #     G0_ens = np.clip(G_init + rng.normal(0.0, 0.05 * Gmax_cal, size=nens), 0.0, Gmax_cal)
# #     X = np.vstack([S0_ens, G0_ens])  

# #     # Save ensemble histories
# #     S_ens_hist  = np.full((L, nens), np.nan)
# #     G_ens_hist  = np.full((L, nens), np.nan)
# #     ET_ens_hist = np.full((L, nens), np.nan)
# #     Q_ens_hist  = np.full((L, nens), np.nan)

# #     # mean time series
# #     ET_ass_mean = np.full(L, np.nan)
# #     Q_ass_mean  = np.full(L, np.nan)

# #     for t in range(L):

# #         # Forecast
# #         X_f, ET_ens, Q_ens = enkf_forecast_step_states(
# #             X=X,
# #             P_t=float(P[t]) if np.isfinite(P[t]) else 0.0,
# #             PET_t=float(PET[t]) if np.isfinite(PET[t]) else 0.0,
# #             params_cal=params_cal,
# #             Smax=Smax_cal,
# #             Gmax=Gmax_cal,
# #             rng=rng,

# #             proc_S_std=float(config.proc_S_std),
# #             proc_G_std=float(config.proc_G_std),
# #             P_std_frac=float(config.P_std_frac),
# #             PET_std_frac=float(config.PET_std_frac),
            

# #             ET_override=None, # This lets the state (S/G) evolve or update
# #         )

# #         ET_ens_hist[t, :] = ET_ens
# #         Q_ens_hist[t, :] = Q_ens

# #         ET_ass_mean[t] = np.nanmean(ET_ens)
# #         Q_ass_mean[t] = np.nanmean(Q_ens)

# #         # Analysis (assimilate ET_B)
# #         if np.isfinite(ET_B[t]):
# #             X = enkf_update_stochastic_scalar(
# #                 X=X_f,
# #                 y_obs=float(ET_B[t]),
# #                 HX=ET_ens.copy(),
# #                 R_var=R_ET_var,
# #                 inflation=inflation,
# #                 Smax=Smax_cal,
# #                 Gmax=Gmax_cal,
# #                 rng=rng,
# #             )
# #         else:
# #             X = X_f

# #         S_ens_hist[t, :] = X[0, :]
# #         G_ens_hist[t, :] = X[1, :]

# #     enkf_hist = {
# #         "time": np.arange(L),
# #         "nens": nens,
# #         "S_ens": S_ens_hist,
# #         "G_ens": G_ens_hist,
# #         "ET_ens": ET_ens_hist,
# #         "Q_ens": Q_ens_hist,
# #     }

# #     return ET_ass_mean, Q_ass_mean, enkf_hist

# # ---------------------------------------------------------
# # DA run (EnKF) -> assimilates ET_model toward ET_obs
# # produces ET_ass_mean (posterior) and Q_ass_mean
# # ---------------------------------------------------------
# def run_budyko_da(
#     P: np.ndarray,
#     PET: np.ndarray,
#     ET_obs: np.ndarray,     # ✅ truth (e.g., ET_B)
#     ET_model: np.ndarray,   # ✅ model (e.g., ET_ke)
#     params_cal: ModelParams,
#     S_init: float,
#     G_init: float,
#     Smax_cal: float,
#     Gmax_cal: float,
#     config: EnKFConfig,
#     basin_id: str,
# ):
#     L = len(P)
#     nens = int(config.nens)
#     inflation = float(config.inflation)
#     R_ET_var = float(config.R_ET_std) ** 2

#     rng = np.random.default_rng(hash(basin_id) % (2**32 - 1))

#     # Initial ensemble states [S,G]
#     S0_ens = np.clip(
#         S_init + rng.normal(0.0, 0.05 * Smax_cal, size=nens),
#         0.0, Smax_cal
#     )
#     G0_ens = np.clip(
#         G_init + rng.normal(0.0, 0.05 * Gmax_cal, size=nens),
#         0.0, Gmax_cal
#     )
#     X = np.vstack([S0_ens, G0_ens])  # shape = (2, nens)

#     # Save ensemble histories
#     S_ens_hist  = np.full((L, nens), np.nan)
#     G_ens_hist  = np.full((L, nens), np.nan)
#     ET_ens_hist = np.full((L, nens), np.nan)
#     Q_ens_hist  = np.full((L, nens), np.nan)

#     # mean time series (posterior)
#     ET_ass_mean = np.full(L, np.nan)
#     Q_ass_mean  = np.full(L, np.nan)

#     for t in range(L):

#         # -------------------------------------------------
#         # Forecast: ET is forced to ET_model[t] (ET_ke)
#         # -------------------------------------------------
#         ET_override_t = float(ET_model[t]) if np.isfinite(ET_model[t]) else None

#         X_f, ET_ens_f, Q_ens_f = enkf_forecast_step_states(
#             X=X,
#             P_t=float(P[t]) if np.isfinite(P[t]) else 0.0,
#             PET_t=float(PET[t]) if np.isfinite(PET[t]) else 0.0,
#             params_cal=params_cal,
#             Smax=Smax_cal,
#             Gmax=Gmax_cal,
#             rng=rng,

#             proc_S_std=float(config.proc_S_std),
#             proc_G_std=float(config.proc_G_std),
#             P_std_frac=float(config.P_std_frac),
#             PET_std_frac=float(config.PET_std_frac),
#             ET_override=ET_override_t,
#         )

#         # Save prior (forecast) ET/Q
#         ET_ens_hist[t, :] = ET_ens_f
#         Q_ens_hist[t, :]  = Q_ens_f

#         # -------------------------------------------------
#         # Analysis: assimilate ET_obs[t] (ET_B) into state
#         # -------------------------------------------------
#         if np.isfinite(ET_obs[t]):
#             X_a = enkf_update_stochastic_scalar(
#                 X=X_f,
#                 y_obs=float(ET_obs[t]),   # ✅ truth
#                 HX=ET_ens_f.copy(),       # ✅ model forecast ET
#                 R_var=R_ET_var,
#                 inflation=inflation,
#                 Smax=Smax_cal,
#                 Gmax=Gmax_cal,
#                 rng=rng,
#             )
#         else:
#             X_a = X_f

#         # -------------------------------------------------
#         # Posterior: recompute ET/Q from updated states
#         # (so ET_ass reflects assimilation)
#         # -------------------------------------------------
#         X_dummy, ET_ens_a, Q_ens_a = enkf_forecast_step_states(
#             X=X_a,
#             P_t=float(P[t]) if np.isfinite(P[t]) else 0.0,
#             PET_t=float(PET[t]) if np.isfinite(PET[t]) else 0.0,
#             params_cal=params_cal,
#             Smax=Smax_cal,
#             Gmax=Gmax_cal,
#             rng=rng,

#             proc_S_std=0.0,   # no additional noise in posterior evaluation
#             proc_G_std=0.0,
#             P_std_frac=0.0,
#             PET_std_frac=0.0,

#             # ✅ keep same ET model forcing
#             ET_override=ET_override_t,
#         )

#         # Save posterior means (this is the assimilated ET)
#         ET_ass_mean[t] = np.nanmean(ET_ens_a)
#         Q_ass_mean[t]  = np.nanmean(Q_ens_a)

#         # Store updated ensemble states and continue
#         X = X_a
#         S_ens_hist[t, :] = X[0, :]
#         G_ens_hist[t, :] = X[1, :]

#     enkf_hist = {
#         "time": np.arange(L),
#         "nens": nens,
#         "S_ens": S_ens_hist,
#         "G_ens": G_ens_hist,
#         "ET_ens": ET_ens_hist,   # prior ET ensemble (ET_model-driven)
#         "Q_ens": Q_ens_hist,     # prior Q ensemble
#     }

#     return ET_ass_mean, Q_ass_mean, enkf_hist


# # ---------------------------------------------------------
# # Scenario normalization
# # ---------------------------------------------------------
# def normalize_scenario_key(s: str) -> str:
#     s = str(s).strip().upper()

#     mapping = {
#         "ALL": "ALL",

#         "BASE": "BASE",
#         "BASE_MODEL": "BASE",
#         "BASE-MODEL": "BASE",

#         "BUDYKO": "BUDYKO",
#         "BUDYKO_MODEL": "BUDYKO",
#         "BUDYKO-MODEL": "BUDYKO",

#         "BUDYKO_DA": "BUDYKO_DA",
#         "BUDYKO_DA_MODEL": "BUDYKO_DA",
#         "BUDYKO+DA": "BUDYKO_DA",
#         "DA": "BUDYKO_DA",
#         "ENKF": "BUDYKO_DA",
#         "ASSIMILATION": "BUDYKO_DA",
#     }

#     if s not in mapping:
#         raise ValueError(f"Unknown scenario: {s}")

#     return mapping[s]


# # ---------------------------------------------------------
# # Main simulation per basin
# # ---------------------------------------------------------
# def simulate_basin(basin_id, scenario, DATA_DIR, RESULT_DIR, calibrated_params, da_cfg: dict):

#     PET_df = load_feather_df("PotEvap.feather", DATA_DIR)
#     Rainf_df = load_feather_df("Rainf.feather", DATA_DIR)
#     Evap_df = load_feather_df("EVap.feather", DATA_DIR)
#     Q_usgs_df = load_feather_df("Q_USGS.feather", DATA_DIR)
#     Q_nldas_df = load_feather_df("Q_nldas_mm_monthly.feather", DATA_DIR)
#     Qsb_df = load_feather_df("Qsb.feather", DATA_DIR)
#     M_df = load_feather_df("M.feather", DATA_DIR)
#     RootMoist = load_feather_df("SoilM_0_200cm.feather", DATA_DIR)
#     Slp_df = load_feather_df("slope.feather", DATA_DIR)


#     common_cols = sorted(
#         set(Evap_df.columns)
#         & set(Qsb_df.columns)
#         & set(PET_df.columns)
#         & set(M_df.columns)
#         & set(RootMoist.columns)
#         & set(Slp_df.columns)
#     )

#     if basin_id not in common_cols or basin_id not in calibrated_params:
#         return None

#     Evap_df, Qsb_df, PET_df, M_df, RootMoist = (
#         Evap_df[common_cols], Qsb_df[common_cols], PET_df[common_cols],
#         M_df[common_cols], RootMoist[common_cols]
#     )

#     idx = Evap_df.index
#     L = len(idx)

#     p = calibrated_params[basin_id]
#     PET = PET_df[basin_id].values
#     P = Rainf_df.get(basin_id, pd.Series(index=idx)).values
#     Q_obs = Q_usgs_df.get(basin_id, pd.Series(index=idx)).reindex(idx).values
#     Q_nldas = Q_nldas_df.get(basin_id, pd.Series(index=idx)).values
#     Qsb = Qsb_df.get(basin_id, pd.Series(index=idx)).values
#     Slp = Slp_df.get(basin_id, pd.Series(index=idx)).values
#     # ET_nldas = Evap_df[basin_id].values

#     # Budyko ET
#     M_basin = M_df.copy()
#     M_basin.index = pd.to_datetime(M_basin.index)
#     M_basin = M_basin.loc[idx]
#     Evap_df = Evap_df[common_cols]
#     Qsb_df  = Qsb_df[common_cols]
#     PET_df  = PET_df[common_cols]
#     M_df    = M_df[common_cols]
#     Slp_df    = Slp_df[common_cols]
#     RootMoist = RootMoist[common_cols]

#     budyko = BudykoModelEstimator(
#         Evap_df=Evap_df[common_cols],
#         Qsb_monthly=Qsb_df[common_cols],
#         PotEvap_df=PET_df[common_cols],
#         M_basin=RootMoist[common_cols],
#         Slope_basin=RootMoist[common_cols],  
#         calibrated_params=calibrated_params,
#     )

#     budyko.estimate_budyko_et()
#     omega_true_all = budyko.omega_true[basin_id].reindex(idx).to_numpy().ravel()
#     omega_MLR_all  = budyko.omega_MLR[basin_id].reindex(idx).to_numpy().ravel()
#     ET_B           = budyko.ET_B[basin_id].reindex(idx).to_numpy().ravel()

#     # Model params
#     Smax_cal = float(p.get("Smax", 50.0))
#     Gmax_factor = float(p.get("Gmax_factor", 4.0))
#     Gmax_cal = Smax_cal * Gmax_factor

#     S_init = float(p.get("S_init", 0.5 * Smax_cal))
#     G_init = float(p.get("G_init", 0.5 * Gmax_cal))

#     params_cal = ModelParams(
#         Smax=Smax_cal,
#         Kperc=p["Kperc"],
#         Kb=p["Kb"],
#         Ke=p["Ke"],
#         Cqq=p["Cqq"],
#         Sfc_frac=0.30,
#         beta_et=2.0,
#     )

#     ET_ke = PET * params_cal.Ke

#     enkf_hist = None # NO DA
#     ET_ass_mean = np.full(L, np.nan)
#     Q_ass_mean = np.full(L, np.nan)

#     if scenario == "BASE":
#         Q_base, S_base, G_base = run_model_deterministic(
#             P=P, 
#             PET=PET,
#             params_cal=params_cal,
#             S_init=S_init,
#             G_init=G_init,
#             Gmax_cal=Gmax_cal,
#             ET_series=ET_ke,  # Simple Scaling Based ET
#         )

#         results = pd.DataFrame({
#             "time": idx,
#             "P": P,
#             "PET": PET,
#             "ET_ke": ET_ke,
#             "Q_bs" :Qsb,
#             "Q_obs": Q_obs,
#             "Q_base": Q_budyko,
#             "S_base": S_base,
#             "G_base": G_base
#         }).set_index("time")


#     elif scenario == "BUDYKO":
#         Q_budyko, S_budyko, G_budyko = run_model_deterministic(
#             P=P, PET=PET,
#             params_cal=params_cal,
#             S_init=S_init,
#             G_init=G_init,
#             Gmax_cal=Gmax_cal,
#             ET_series=ET_B,  # Budyko ET
#         )
#         results = pd.DataFrame({
#             "time": idx,
#             "omega_true": omega_true_all,
#             "omega_MLR": omega_MLR_all,
#             "P": P,
#             "PET": PET,
#             "ET_B": ET_B,
#             "Q_obs": Q_obs,
#             "Q_budyko": Q_base,
#             "S_budyko": S_budyko,
#             "G_budyko": G_budyko,
#         }).set_index("time")

# # ET assimilation with Budyko, but Q is determined deterministically

#     elif scenario == "BUDYKO_DA":
#         config = EnKFConfig(**da_cfg)

#         ET_ass_mean, Q_ass_mean, enkf_hist = run_budyko_da(
#             P=P,
#             PET=PET,
#             ET_obs=ET_B,      # true
#             ET_model=ET_ke,   # model, assummed as model value to be corrected in the DA
#             params_cal=params_cal,
#             S_init=S_init,
#             G_init=G_init,
#             Smax_cal=Smax_cal,
#             Gmax_cal=Gmax_cal,
#             config=config,
#             basin_id=basin_id,
#         )


#         Q_ass, S_ass, G_ass = run_model_deterministic(
#             P=P, PET=PET,
#             params_cal=params_cal,
#             S_init=S_init,
#             G_init=G_init,
#             Gmax_cal=Gmax_cal,
#             ET_series=ET_ass_mean,
#         )

#         results = pd.DataFrame({
#             "time": idx,
#             "P": P,
#             "PET": PET,
#             "ET_B": ET_B,
#             "ET_ass": ET_ass_mean,
#             "Q_obs": Q_obs,
#             "Q_ass": Q_ass,
#             "Q_ens": Q_ass_mean,
            
#         }).set_index("time")

#     else:
#         raise ValueError(f"Unknown scenario: {scenario}")

#     # Save results
#     result_path = os.path.join(RESULT_DIR, f"results_{scenario}_{basin_id}.feather")
#     results.reset_index().to_feather(result_path)
#     if scenario == "BUDYKO_DA" and enkf_hist is not None:

#         # save ensemble mean as a feather file
#         enkf_df = pd.DataFrame({
#             "time": idx,
#             "ET_ens_mean": np.nanmean(enkf_hist["ET_ens"], axis=1),
#             "Q_ens_mean": np.nanmean(enkf_hist["Q_ens"], axis=1),
#             "S_ens_mean": np.nanmean(enkf_hist["S_ens"], axis=1),
#             "G_ens_mean": np.nanmean(enkf_hist["G_ens"], axis=1),
#         })

#         enkf_path = os.path.join(RESULT_DIR, f"enkf_ensemble_{scenario}_{basin_id}.feather")
#         enkf_df.to_feather(enkf_path)


#     # Metrics
#     qobs = results["Q_obs"].values if "Q_obs" in results.columns else Q_obs

#     qcol = [c for c in results.columns if c.startswith("Q_") and c != "Q_obs"]
#     qsim_name = qcol[-1] if qcol else None
#     qsim = results[qsim_name].values if qsim_name else None

#     metrics = {
#         "gauge_id": basin_id,
#         "scenario": scenario,
#         "KGE": calculate_kge(qobs, qsim) if qsim is not None else np.nan,
#         "NSE": calculate_nse(qobs, qsim) if qsim is not None else np.nan,
#     }

#     return metrics


# # ---------------------------------------------------------
# # Run from config
# # ---------------------------------------------------------
# def run_simulations_from_config(cfg: dict):

#     scenario = normalize_scenario_key(cfg["scenario"])
#     if scenario == "ALL":
#         scenarios_to_run = ["BASE", "BUDYKO", "BUDYKO_DA"]
#     else:
#         scenarios_to_run = [scenario]

#     paths = cfg["paths"]
#     DATA_DIR = os.path.join(PROJECT_ROOT, paths["data_dir"])

#     BASE_RESULT_DIR = os.path.join(PROJECT_ROOT, paths["result_dir"])
#     os.makedirs(BASE_RESULT_DIR, exist_ok=True)

#     # calibrated params
#     cal_path = os.path.join(PROJECT_ROOT, paths["calibrated_params"])
#     with open(cal_path, "r") as f:
#         calibrated_params = json.load(f)

#     da_cfg = cfg.get("da", {})
#     if "enabled" in da_cfg:
#         da_cfg.pop("enabled")

#     basin_subset = cfg.get("basins", {}).get("subset", None)

#     # Determine basins
#     if basin_subset is None:
#         tmp = load_feather_df("EVap.feather", DATA_DIR)
#         basins = list(tmp.columns)
#     else:
#         basins = list(basin_subset)

#     # parallel settings
#     par_cfg = cfg.get("parallel", {})
#     par_enabled = bool(par_cfg.get("enabled", True))
#     max_workers = int(par_cfg.get("max_workers", -1))
#     if max_workers == -1:
#         max_workers = max(1, cpu_count() - 1)

#     # -----------------------------------------------------
#     # RUN EACH SCENARIO
#     # -----------------------------------------------------
#     for scenario in scenarios_to_run:

#         scenario_dir = scenario_folder_name(scenario)
#         RESULT_DIR = os.path.join(BASE_RESULT_DIR, scenario_dir)
#         os.makedirs(RESULT_DIR, exist_ok=True)

#         all_metrics = []

#         if par_enabled:
#             with ProcessPoolExecutor(max_workers=max_workers) as executor:
#                 futures = {
#                     executor.submit(
#                         simulate_basin, b, scenario, DATA_DIR, RESULT_DIR, calibrated_params, da_cfg
#                     ): b
#                     for b in basins
#                 }

#                 for f in tqdm(as_completed(futures), total=len(futures), desc=f"Running scenario={scenario}"):
#                     out = f.result()
#                     if out is not None:
#                         all_metrics.append(out)
#         else:
#             for b in tqdm(basins, desc=f"Running scenario={scenario}"):
#                 out = simulate_basin(b, scenario, DATA_DIR, RESULT_DIR, calibrated_params, da_cfg)
#                 if out is not None:
#                     all_metrics.append(out)

#         pd.DataFrame(all_metrics).to_csv(
#             os.path.join(RESULT_DIR, f"metrics_{scenario}.csv"),
#             index=False
#         )

#         print(f"\n✅ Completed successfully: scenario={scenario}. Results saved to {RESULT_DIR}")


