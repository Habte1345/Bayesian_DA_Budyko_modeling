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
# Deterministic single-run model
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

    Q_out, S_out, G_out = np.full(L, np.nan), np.full(L, np.nan), np.full(L, np.nan)

    for t in range(L):
        P_t = float(P[t]) if np.isfinite(P[t]) else 0.0
        PET_t = float(PET[t]) if np.isfinite(PET[t]) else 0.0

        S, G, _, Q, *_ = two_store_model_step(
            S, G, P_t, PET_t, params_cal,
            ET_override=float(ET_series[t]) if np.isfinite(ET_series[t]) else None
        )

        G = np.clip(G, 0.0, Gmax_cal)

        Q_out[t] = max(Q, 0.0)
        S_out[t] = S
        G_out[t] = G

    return Q_out, S_out, G_out


# ---------------------------------------------------------
# DA run (EnKF) -> produces ET_ass and Q_ass_mean
# ---------------------------------------------------------
def run_budyko_da(
    P: np.ndarray,
    PET: np.ndarray,
    ET_B: np.ndarray,
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
    S0_ens = np.clip(S_init + rng.normal(0.0, 0.05 * Smax_cal, size=nens), 0.0, Smax_cal)
    G0_ens = np.clip(G_init + rng.normal(0.0, 0.05 * Gmax_cal, size=nens), 0.0, Gmax_cal)
    X = np.vstack([S0_ens, G0_ens])  # (2, nens)

    # Save ensemble histories
    S_ens_hist  = np.full((L, nens), np.nan)
    G_ens_hist  = np.full((L, nens), np.nan)
    ET_ens_hist = np.full((L, nens), np.nan)
    Q_ens_hist  = np.full((L, nens), np.nan)

    # mean time series
    ET_ass_mean = np.full(L, np.nan)
    Q_ass_mean  = np.full(L, np.nan)

    for t in range(L):

        # Forecast
        X_f, ET_ens, Q_ens = enkf_forecast_step_states(
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

            ET_override=None,
        )

        ET_ens_hist[t, :] = ET_ens
        Q_ens_hist[t, :] = Q_ens

        ET_ass_mean[t] = np.nanmean(ET_ens)
        Q_ass_mean[t] = np.nanmean(Q_ens)

        # Analysis (assimilate ET_B)
        if np.isfinite(ET_B[t]):
            X = enkf_update_stochastic_scalar(
                X=X_f,
                y_obs=float(ET_B[t]),
                HX=ET_ens.copy(),
                R_var=R_ET_var,
                inflation=inflation,
                Smax=Smax_cal,
                Gmax=Gmax_cal,
                rng=rng,
            )
        else:
            X = X_f

        S_ens_hist[t, :] = X[0, :]
        G_ens_hist[t, :] = X[1, :]

    enkf_hist = {
        "time": np.arange(L),
        "nens": nens,
        "S_ens_hist": S_ens_hist,
        "G_ens_hist": G_ens_hist,
        "ET_ens_hist": ET_ens_hist,
        "Q_ens_hist": Q_ens_hist,
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
# Main simulation per basin
# ---------------------------------------------------------
def simulate_basin(basin_id, scenario, DATA_DIR, RESULT_DIR, calibrated_params, da_cfg: dict):

    PET_df = load_feather_df("PotEvap.feather", DATA_DIR)
    Rainf_df = load_feather_df("Rainf.feather", DATA_DIR)
    Evap_df = load_feather_df("EVap.feather", DATA_DIR)
    Q_usgs_df = load_feather_df("Q_USGS.feather", DATA_DIR)
    Q_nldas_df = load_feather_df("Q_nldas_mm_monthly.feather", DATA_DIR)
    Qsb_df = load_feather_df("Qsb.feather", DATA_DIR)
    M_df = load_feather_df("M.feather", DATA_DIR)
    RootMoist = load_feather_df("SoilM_0_200cm.feather", DATA_DIR)

    common_cols = sorted(
        set(Evap_df.columns)
        & set(Qsb_df.columns)
        & set(PET_df.columns)
        & set(M_df.columns)
        & set(RootMoist.columns)
    )

    if basin_id not in common_cols or basin_id not in calibrated_params:
        return None

    Evap_df, Qsb_df, PET_df, M_df, RootMoist = (
        Evap_df[common_cols], Qsb_df[common_cols], PET_df[common_cols],
        M_df[common_cols], RootMoist[common_cols]
    )

    idx = Evap_df.index
    L = len(idx)

    p = calibrated_params[basin_id]
    PET = PET_df[basin_id].values
    P = Rainf_df.get(basin_id, pd.Series(index=idx)).values
    Q_obs = Q_usgs_df.get(basin_id, pd.Series(index=idx)).reindex(idx).values
    Q_nldas = Q_nldas_df.get(basin_id, pd.Series(index=idx)).values
    ET_nldas = Evap_df[basin_id].values

    # Budyko ET
    M_basin = M_df.copy()
    M_basin.index = pd.to_datetime(M_basin.index)
    M_basin = M_basin.loc[idx]

    budyko = BudykoModelEstimator(
        Evap_df=Evap_df[[basin_id]],
        Qsb_monthly=Qsb_df[[basin_id]],
        PotEvap_df=PET_df[[basin_id]],
        M_basin=M_basin[[basin_id]],
        Slope_basin=RootMoist[[basin_id]],
        calibrated_params={basin_id: p},
    )
    budyko.estimate_budyko_et()
    ET_B = budyko.ET_B[basin_id].values

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

    enkf_hist = None
    ET_ass_mean = np.full(L, np.nan)
    Q_ass_mean = np.full(L, np.nan)

    if scenario == "BASE":
        Q_base, S_base, G_base = run_model_deterministic(
            P=P, PET=PET,
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
            "Q_obs": Q_obs,
            "Q_base": Q_base,
        }).set_index("time")

    elif scenario == "BUDYKO":
        Q_budyko, S_budyko, G_budyko = run_model_deterministic(
            P=P, PET=PET,
            params_cal=params_cal,
            S_init=S_init,
            G_init=G_init,
            Gmax_cal=Gmax_cal,
            ET_series=ET_B,
        )
        results = pd.DataFrame({
            "time": idx,
            "P": P,
            "PET": PET,
            "ET_B": ET_B,
            "Q_obs": Q_obs,
            "Q_budyko": Q_budyko,
        }).set_index("time")

    elif scenario == "BUDYKO_DA":
        config = EnKFConfig(**da_cfg)

        ET_ass_mean, Q_ass_mean, enkf_hist = run_budyko_da(
            P=P, PET=PET, ET_B=ET_B,
            params_cal=params_cal,
            S_init=S_init, G_init=G_init,
            Smax_cal=Smax_cal, Gmax_cal=Gmax_cal,
            config=config,
            basin_id=basin_id,
        )

        Q_ass, S_ass, G_ass = run_model_deterministic(
            P=P, PET=PET,
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
            "Q_obs": Q_obs,
            "Q_ass": Q_ass,
            "Q_ass_forecast_mean": Q_ass_mean,
        }).set_index("time")

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Save results
    result_path = os.path.join(RESULT_DIR, f"results_{scenario}_{basin_id}.feather")
    results.reset_index().to_feather(result_path)

    # Save EnKF ensemble histories only for DA scenario
    if scenario == "BUDYKO_DA" and enkf_hist is not None:
        npz_path = os.path.join(RESULT_DIR, f"enkf_ensemble_{scenario}_{basin_id}.npz")
        np.savez_compressed(
            npz_path,
            time=idx.values.astype("datetime64[ns]"),
            nens=config.nens,
            S_ens_hist=enkf_hist["S_ens_hist"],
            G_ens_hist=enkf_hist["G_ens_hist"],
            ET_ens_hist=enkf_hist["ET_ens_hist"],
            Q_ens_hist=enkf_hist["Q_ens_hist"],
        )

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

    # Determine basins
    if basin_subset is None:
        tmp = load_feather_df("EVap.feather", DATA_DIR)
        basins = list(tmp.columns)
    else:
        basins = list(basin_subset)

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
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
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


















# # run_simulation.py
# import json
# import logging
# import os
# import sys
# from concurrent.futures import ProcessPoolExecutor, as_completed

# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# # ---------------------------------------------------------
# # Project paths
# # ---------------------------------------------------------
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
# RESULT_DIR = os.path.join(PROJECT_ROOT, "Simulation_results")
# os.makedirs(RESULT_DIR, exist_ok=True)

# sys.path.append(PROJECT_ROOT)

# # ---------------------------------------------------------
# # Imports
# # ---------------------------------------------------------
# from src.model import ModelParams, two_store_model_step
# from src.budyko import BudykoModelEstimator
# from src.enkf import (
#     EnKFConfig,
#     enkf_update_stochastic_scalar,
#     enkf_forecast_step_states,
# )
# from src.metrics import calculate_kge, calculate_nse


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
# # Load datasets
# # ---------------------------------------------------------
# PET_df = load_feather_df("PotEvap.feather", DATA_DIR)
# Rainf_df = load_feather_df("Rainf.feather", DATA_DIR)
# Evap_df = load_feather_df("EVap.feather", DATA_DIR)
# Q_usgs_df = load_feather_df("Q_USGS.feather", DATA_DIR)
# Q_nldas_df = load_feather_df("Q_nldas_mm_monthly.feather", DATA_DIR)
# Qsb_df = load_feather_df("Qsb.feather", DATA_DIR)
# M_df = load_feather_df("M.feather", DATA_DIR)
# Slope_basin = load_feather_df("slope.feather", DATA_DIR)
# RootMoist = load_feather_df("SoilM_0_200cm.feather", DATA_DIR)

# # ---------------------------------------------------------
# # Align basins across datasets
# # ---------------------------------------------------------
# common_cols = sorted(
#     set(Evap_df.columns)
#     & set(Qsb_df.columns)
#     & set(PET_df.columns)
#     & set(M_df.columns)
#     & set(Slope_basin.columns)
#     & set(RootMoist.columns)
# )

# Evap_df, Qsb_df, PET_df, M_df, Slope_basin, RootMoist = (
#     Evap_df[common_cols],
#     Qsb_df[common_cols],
#     PET_df[common_cols],
#     M_df[common_cols],
#     Slope_basin[common_cols],
#     RootMoist[common_cols],
# )

# # Ensure M_basin aligned with Evap_df index
# M_basin = M_df.copy()
# M_basin.index = pd.to_datetime(M_basin.index)
# M_basin = M_basin.loc[Evap_df.index]

# # ---------------------------------------------------------
# # Load calibrated parameters
# # ---------------------------------------------------------
# calibrated_path = os.path.join(PROJECT_ROOT, "SCE_cal_params", "final_calibrated_params.json")
# with open(calibrated_path, "r") as f:
#     calibrated_params = json.load(f)

# # ---------------------------------------------------------
# # Compute Budyko components once
# # ---------------------------------------------------------
# budyko = BudykoModelEstimator(
#     Evap_df=Evap_df,
#     Qsb_monthly=Qsb_df,
#     PotEvap_df=PET_df,
#     M_basin=M_basin,
#     Slope_basin=RootMoist,
#     calibrated_params=calibrated_params,
# )

# budyko.estimate_budyko_et()
# omega_true_all, omega_MLR_all, ET_B_all = (
#     budyko.omega_true,
#     budyko.omega_MLR,
#     budyko.ET_B,
# )


# # ---------------------------------------------------------
# # Simulation per basin
# # ---------------------------------------------------------
# def simulate_basin(basin_id):
#     if basin_id not in calibrated_params or basin_id not in Evap_df.columns:
#         return None, None

#     p = calibrated_params[basin_id]
#     idx = Evap_df.index
#     L = len(idx)

#     PET = PET_df[basin_id].values
#     P = Rainf_df.get(basin_id, pd.Series(index=idx)).values
#     Q_obs = Q_usgs_df.get(basin_id, pd.Series(index=idx)).reindex(idx).values
#     Q_nldas = Q_nldas_df.get(basin_id, pd.Series(index=idx)).values
#     ET_nldas = Evap_df[basin_id].values
#     M_series = M_basin[basin_id].values
#     Slope_series = RootMoist[basin_id].values
#     ET_B = ET_B_all[basin_id].values

#     # --- Parameter setup ---
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

#         # ET stress 
#         Sfc_frac=0.30,
#         beta_et=2.0,
#     )


#     # --- Base ET ---
#     ET_ke = PET * p["Ke"]

#     # -----------------------------------------------------
#     # EnKF setup
#     # -----------------------------------------------------
#     config = EnKFConfig()
#     nens = int(config.nens)
#     inflation = float(config.inflation)

#     # IMPORTANT: treat config.R_ET_std as std -> variance
#     R_ET_var = float(config.R_ET_std) ** 2

#     rng = np.random.default_rng(hash(basin_id) % (2**32 - 1))

#     # Initial ensemble states [S,G]
#     S0_ens = np.clip(S_init + rng.normal(0.0, 0.05 * Smax_cal, size=nens), 0.0, Smax_cal)
#     G0_ens = np.clip(G_init + rng.normal(0.0, 0.05 * Gmax_cal, size=nens), 0.0, Gmax_cal)
#     X = np.vstack([S0_ens, G0_ens])  # (2, nens)

#     # -----------------------------------------------------
#     # SAVE THESE: ensemble histories
#     # -----------------------------------------------------
#     S_ens_hist  = np.full((L, nens), np.nan)
#     G_ens_hist  = np.full((L, nens), np.nan)
#     ET_ens_hist = np.full((L, nens), np.nan)
#     Q_ens_hist  = np.full((L, nens), np.nan)

#     # Posterior mean series (for easier quick plotting/statistics)
#     ET_B_NLDAS_ass = np.full(L, np.nan)
#     Q_B_ass = np.full(L, np.nan)
#     S_B_ass = np.full(L, np.nan)
#     G_B_ass = np.full(L, np.nan)

#     # -----------------------------------------------------
#     # EnKF loop
#     # -----------------------------------------------------
#     for t in range(L):

#         # 1) Forecast
#         X_f, ET_ens, Q_ens = enkf_forecast_step_states(
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

#             ET_override=None,
#         )


#         # store forecasted outputs
#         ET_ens_hist[t, :] = ET_ens
#         Q_ens_hist[t, :] = Q_ens

#         # store mean forecast outputs
#         ET_B_NLDAS_ass[t] = np.nanmean(ET_ens)
#         Q_B_ass[t] = np.nanmean(Q_ens)

#         # 2) Analysis update (stochastic) if ET_B observation available
#         if np.isfinite(ET_B[t]):
#             X = enkf_update_stochastic_scalar(
#                 X=X_f,
#                 y_obs=float(ET_B[t]),
#                 HX=ET_ens.copy(),     # predicted observation
#                 R_var=R_ET_var,
#                 inflation=inflation,
#                 Smax=Smax_cal,
#                 Gmax=Gmax_cal,
#                 rng=rng,
#             )
#         else:
#             X = X_f

#         # store posterior ensembles (after update)
#         S_ens_hist[t, :] = X[0, :]
#         G_ens_hist[t, :] = X[1, :]

#         # store posterior mean states
#         S_B_ass[t] = X[0, :].mean()
#         G_B_ass[t] = X[1, :].mean()

#     # -----------------------------------------------------
#     # Deterministic scenarios (unchanged)
#     # -----------------------------------------------------
#     def run_model(ET_override=None):
#         S, G = S_init, G_init
#         Q_out, S_out, G_out = [], [], []

#         for t in range(L):
#             P_t = float(P[t]) if np.isfinite(P[t]) else 0.0
#             PET_t = float(PET[t]) if np.isfinite(PET[t]) else 0.0

#             S, G, _, Q, *_ = two_store_model_step(
#                 S, G, P_t, PET_t, params_cal,
#                 ET_override=ET_override[t] if ET_override is not None else None,
#             )

#             G = np.clip(G, 0, Gmax_cal)

#             Q_out.append(max(Q, 0.0))
#             S_out.append(S)
#             G_out.append(G)

#         return np.asarray(Q_out), np.asarray(S_out), np.asarray(G_out)

#     Q_ke, S_ke, G_ke = run_model(ET_ke)
#     Q_b, S_b, G_b = run_model(ET_override=ET_B)

#     # -----------------------------------------------------
#     # Consolidate results
#     # -----------------------------------------------------
#     results = pd.DataFrame(
#         {
#             "time": idx,
#             "P": P,
#             "PET": PET,
#             "Qsb": Qsb_df.get(basin_id, pd.Series(index=idx)).reindex(idx).values,

#             "ET_ke": ET_ke,
#             "ET_B": ET_B,
#             "ET_nldas": ET_nldas,

#             # EnKF outputs
#             "ET_B_ass_mean": ET_B_NLDAS_ass,
#             "Q_ass_mean": Q_B_ass,
#             "S_ass_mean": S_B_ass,
#             "G_ass_mean": G_B_ass,

#             # Observed/simulated discharge
#             "Q_obs": Q_obs,
#             "Q_ke": Q_ke,
#             "Q_b": Q_b,

#             "S_ke": S_ke,
#             "G_ke": G_ke,
#             "S_b": S_b,
#             "G_b": G_b,

#             "omega_true": omega_true_all[basin_id].values,
#             "omega_MLR": omega_MLR_all[basin_id].values,
#             "Q_nldas": Q_nldas,
#             "M": M_series,
#             "Slope": Slope_series,
#         }
#     ).set_index("time")

#     # -----------------------------------------------------
#     # Performance metrics
#     # -----------------------------------------------------
#     metrics = {
#         "Q_ke_KGE": calculate_kge(Q_obs, Q_ke),
#         "Q_ke_NSE": calculate_nse(Q_obs, Q_ke),

#         "Q_b_KGE": calculate_kge(Q_obs, Q_b),
#         "Q_b_NSE": calculate_nse(Q_obs, Q_b),

#         # EnKF filtered discharge mean
#         "Q_ass_mean_KGE": calculate_kge(Q_obs, Q_B_ass),
#         "Q_ass_mean_NSE": calculate_nse(Q_obs, Q_B_ass),

#         "Q_nldas_KGE": calculate_kge(Q_obs, Q_nldas),
#         "Q_nldas_NSE": calculate_nse(Q_obs, Q_nldas),
#     }

#     # return everything needed
#     enkf_hist = {
#         "time": idx.values.astype("datetime64[ns]"),
#         "nens": nens,
#         "S_ens_hist": S_ens_hist,
#         "G_ens_hist": G_ens_hist,
#         "ET_ens_hist": ET_ens_hist,
#         "Q_ens_hist": Q_ens_hist,
#     }

#     return results, metrics, enkf_hist


# # ---------------------------------------------------------
# # Wrapper for parallel execution
# # ---------------------------------------------------------
# def run_and_save_basin(basin_id):
#     try:
#         result_df, metrics, enkf_hist = simulate_basin(basin_id)

#         if result_df is not None:
#             # save main results
#             result_path = os.path.join(RESULT_DIR, f"results_streamflow_{basin_id}.feather")
#             result_df.reset_index().to_feather(result_path)

#             # save ensemble histories
#             npz_path = os.path.join(RESULT_DIR, f"enkf_ensemble_{basin_id}.npz")
#             np.savez_compressed(
#                 npz_path,
#                 time=enkf_hist["time"],
#                 nens=enkf_hist["nens"],
#                 S_ens_hist=enkf_hist["S_ens_hist"],
#                 G_ens_hist=enkf_hist["G_ens_hist"],
#                 ET_ens_hist=enkf_hist["ET_ens_hist"],
#                 Q_ens_hist=enkf_hist["Q_ens_hist"],
#             )

#             rows = []
#             for sc in ["Q_ke", "Q_b", "Q_ass_mean", "Q_nldas"]:
#                 rows.append(
#                     {
#                         "gauge_id": basin_id,
#                         "scenario": sc,
#                         "KGE": metrics.get(f"{sc}_KGE", np.nan),
#                         "NSE": metrics.get(f"{sc}_NSE", np.nan),
#                     }
#                 )

#             return rows

#     except Exception as e:
#         print(f"❌ Error processing {basin_id}: {e}", file=sys.stderr)

#     return []


# # ---------------------------------------------------------
# # Main
# # ---------------------------------------------------------
# if __name__ == "__main__":
#     from multiprocessing import cpu_count

#     os.makedirs(RESULT_DIR, exist_ok=True)

#     all_basins = common_cols
#     all_metrics = []

#     with ProcessPoolExecutor(max_workers=max(1, cpu_count() - 1)) as executor:
#         futures = {executor.submit(run_and_save_basin, b): b for b in all_basins}

#         for f in tqdm(
#             as_completed(futures),
#             total=len(futures),
#             desc="Running in parallel for all CAMELS basins",
#         ):
#             all_metrics.extend(f.result())

#     pd.DataFrame(all_metrics).to_csv(
#         os.path.join(RESULT_DIR, "streamflow_performance_metrics.csv"),
#         index=False
#     )

#     print("\n✅ All basin simulations completed and results saved.")






















# # run_simulation.py
# import json
# import logging
# import os
# import sys
# from concurrent.futures import ProcessPoolExecutor, as_completed

# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# # ---------------------------------------------------------
# # Project paths
# # ---------------------------------------------------------
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
# RESULT_DIR = os.path.join(PROJECT_ROOT, "Simulation_results")
# os.makedirs(RESULT_DIR, exist_ok=True)

# sys.path.append(PROJECT_ROOT)

# # ---------------------------------------------------------
# # Imports
# # ---------------------------------------------------------
# from src.model import ModelParams, two_store_model_step
# from src.budyko import BudykoModelEstimator
# from src.enkf import EnKFConfig, enkf_update
# from src.metrics import calculate_kge, calculate_nse


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
# # Load datasets
# # ---------------------------------------------------------
# PET_df = load_feather_df("PotEvap.feather", DATA_DIR)
# Rainf_df = load_feather_df("Rainf.feather", DATA_DIR)
# Evap_df = load_feather_df("EVap.feather", DATA_DIR)
# Q_usgs_df = load_feather_df("Q_USGS.feather", DATA_DIR)
# Q_nldas_df = load_feather_df("Q_nldas_mm_monthly.feather", DATA_DIR)
# Qsb_df = load_feather_df("Qsb.feather", DATA_DIR)
# M_df = load_feather_df("M.feather", DATA_DIR)
# Slope_basin = load_feather_df("slope.feather", DATA_DIR)
# RootMoist = load_feather_df("SoilM_0_200cm.feather", DATA_DIR)

# # ---------------------------------------------------------
# # Align basins across datasets
# # ---------------------------------------------------------
# common_cols = sorted(
#     set(Evap_df.columns)
#     & set(Qsb_df.columns)
#     & set(PET_df.columns)
#     & set(M_df.columns)
#     & set(Slope_basin.columns)
#     & set(RootMoist.columns)
# )

# Evap_df, Qsb_df, PET_df, M_df, Slope_basin, RootMoist = (
#     Evap_df[common_cols],
#     Qsb_df[common_cols],
#     PET_df[common_cols],
#     M_df[common_cols],
#     Slope_basin[common_cols],
#     RootMoist[common_cols],
# )

# # Ensure M_basin aligned with Evap_df index
# M_basin = M_df.copy()
# M_basin.index = pd.to_datetime(M_basin.index)
# M_basin = M_basin.loc[Evap_df.index]

# # ---------------------------------------------------------
# # Load calibrated parameters
# # ---------------------------------------------------------
# calibrated_path = os.path.join(PROJECT_ROOT, "SCE_cal_params", "final_calibrated_params.json")
# with open(calibrated_path, "r") as f:
#     calibrated_params = json.load(f)

# # ---------------------------------------------------------
# # Compute Budyko components once
# # ---------------------------------------------------------
# budyko = BudykoModelEstimator(
#     Evap_df=Evap_df,
#     Qsb_monthly=Qsb_df,
#     PotEvap_df=PET_df,
#     M_basin=M_basin,
#     Slope_basin=RootMoist,
#     calibrated_params=calibrated_params,
# )

# budyko.estimate_budyko_et()
# omega_true_all, omega_MLR_all, ET_B_all = (
#     budyko.omega_true,
#     budyko.omega_MLR,
#     budyko.ET_B,
# )


# # ---------------------------------------------------------
# # Simulation per basin
# # ---------------------------------------------------------
# def simulate_basin(basin_id):
#     if basin_id not in calibrated_params or basin_id not in Evap_df.columns:
#         return None, None

#     p = calibrated_params[basin_id]
#     idx = Evap_df.index
#     L = len(idx)

#     PET = PET_df[basin_id].values
#     P = Rainf_df.get(basin_id, pd.Series(index=idx)).values
#     Q_obs = Q_usgs_df.get(basin_id, pd.Series(index=idx)).reindex(idx).values
#     Q_nldas = Q_nldas_df.get(basin_id, pd.Series(index=idx)).values
#     ET_nldas = Evap_df[basin_id].values
#     M_series = M_basin[basin_id].values
#     Slope_series = RootMoist[basin_id].values
#     ET_B = ET_B_all[basin_id].values

#     # --- Parameter setup (auto-adaptive) ---
#     Smax_cal = p.get("Smax", 50.0)
#     Gmax_factor = p.get("Gmax_factor", 4.0)
#     Gmax_cal = Smax_cal * Gmax_factor

#     S_init = p.get("S_init", 0.5 * Smax_cal)
#     G_init = p.get("G_init", 0.5 * Gmax_cal)

#     model_params = ModelParams(
#         Smax=Smax_cal,
#         Kperc=p["Kperc"],
#         Kb=p["Kb"],
#         Ke=p["Ke"],
#         Cqq=p["Cqq"],
#     )

#     # --- Base ET ---
#     ET_ke = PET * p["Ke"]

#     # --- EnKF setup ---
#     config = EnKFConfig()
#     nens, inflation, R_ET = config.nens, config.inflation, config.R_ET

#     rng = np.random.default_rng(hash(basin_id) % (2**32 - 1))

#     X_et_bud = np.tile(ET_nldas, (nens, 1)).T + rng.normal(0, 0.05, (L, nens))
#     ET_B_NLDAS_ass = np.empty(L)

#     # -----------------------------------------------------
#     # ET assimilation loop
#     # -----------------------------------------------------
#     for t in range(L):
#         et_ens = X_et_bud[t, :]
#         ET_B_NLDAS_ass[t] = et_ens.mean()

#         if np.isfinite(ET_B[t]):
#             X_dummy = np.zeros((6, nens))
#             X_dummy[4, :] = et_ens

#             HX = X_dummy[4, :].copy()

#             X_updated = enkf_update(
#                 X_dummy,
#                 y_obs=ET_B[t],
#                 HX=HX,
#                 R=R_ET,
#                 inflation=inflation,
#                 Smax=Smax_cal,
#                 Gmax=Gmax_cal,
#             )

#             X_et_bud[t, :] = X_updated[4, :]
#             ET_B_NLDAS_ass[t] = X_updated[4, :].mean()

#     # -----------------------------------------------------
#     # Simulation core
#     # -----------------------------------------------------
#     def run_model(ET_override=None):
#         S, G = S_init, G_init
#         Q_out, S_out, G_out = [], [], []

#         for P_t, PET_t, t in zip(P, PET, range(L)):
#             S, G, _, Q, *_ = two_store_model_step(
#                 S, G, P_t, PET_t, model_params,
#                 ET_override=ET_override[t] if ET_override is not None else None,
#             )
#             G = np.clip(G, 0, Gmax_cal)
#             Q_out.append(Q)
#             S_out.append(S)
#             G_out.append(G)

#         return np.asarray(Q_out), np.asarray(S_out), np.asarray(G_out)

#     # --- Run scenarios ---
#     Q_ke, S_ke, G_ke = run_model(ET_ke)
#     Q_b, S_b, G_b = run_model(ET_override=ET_B)
#     Q_B_ass, S_B_ass, G_B_ass = run_model(ET_override=ET_B_NLDAS_ass)

#     # -----------------------------------------------------
#     # Consolidate results
#     # -----------------------------------------------------
#     results = pd.DataFrame(
#         {
#             "time": idx,
#             "P": P,
#             "PET": PET,
#             "Qsb": Qsb_df.get(basin_id, pd.Series(index=idx)).reindex(idx).values,
#             "ET_ke": ET_ke,
#             "ET_B": ET_B,
#             "ET_nldas": ET_nldas,
#             "ET_B_NLDAS_ass": ET_B_NLDAS_ass,
#             "Q_obs": Q_obs,
#             "Q_ke": Q_ke,
#             "Q_b": Q_b,
#             "Q_B_ass": Q_B_ass,
#             "S_ke": S_ke,
#             "G_ke": G_ke,
#             "S_b": S_b,
#             "G_b": G_b,
#             "S_B_ass": S_B_ass,
#             "G_B_ass": G_B_ass,
#             "omega_true": omega_true_all[basin_id].values,
#             "omega_MLR": omega_MLR_all[basin_id].values,
#             "Q_nldas": Q_nldas,
#             "M": M_series,
#             "Slope": Slope_series,
#         }
#     ).set_index("time")

#     # -----------------------------------------------------
#     # Performance metrics
#     # -----------------------------------------------------
#     metrics = {
#         "Q_ke_KGE": calculate_kge(Q_obs, Q_ke),
#         "Q_ke_NSE": calculate_nse(Q_obs, Q_ke),
#         "Q_b_KGE": calculate_kge(Q_obs, Q_b),
#         "Q_b_NSE": calculate_nse(Q_obs, Q_b),
#         "Q_B_ass_KGE": calculate_kge(Q_obs, Q_B_ass),
#         "Q_B_ass_NSE": calculate_nse(Q_obs, Q_B_ass),
#         "Q_nldas_KGE": calculate_kge(Q_obs, Q_nldas),
#         "Q_nldas_NSE": calculate_nse(Q_obs, Q_nldas),
#     }

#     return results, metrics


# # ---------------------------------------------------------
# # Wrapper for parallel execution
# # ---------------------------------------------------------
# def run_and_save_basin(basin_id):
#     try:
#         result_df, metrics = simulate_basin(basin_id)

#         if result_df is not None:
#             result_path = os.path.join(RESULT_DIR, f"results_streamflow_{basin_id}.feather")
#             result_df.reset_index().to_feather(result_path)

#             rows = [
#                 {
#                     "gauge_id": basin_id,
#                     "scenario": sc,
#                     "KGE": metrics.get(f"{sc}_KGE", np.nan),
#                     "NSE": metrics.get(f"{sc}_NSE", np.nan),
#                 }
#                 for sc in ["Q_ke", "Q_b", "Q_B_ass", "Q_nldas"]
#             ]
#             return rows

#     except Exception as e:
#         print(f"❌ Error processing {basin_id}: {e}", file=sys.stderr)

#     return []


# # ---------------------------------------------------------
# # Main
# # ---------------------------------------------------------
# if __name__ == "__main__":
#     from multiprocessing import cpu_count

#     os.makedirs(RESULT_DIR, exist_ok=True)

#     all_basins = common_cols
#     all_metrics = []

#     with ProcessPoolExecutor(max_workers=max(1, cpu_count() - 1)) as executor:
#         futures = {executor.submit(run_and_save_basin, b): b for b in all_basins}

#         for f in tqdm(
#             as_completed(futures),
#             total=len(futures),
#             desc="Running in parallel for all CAMELS basins",
#         ):
#             all_metrics.extend(f.result())

#     pd.DataFrame(all_metrics).to_csv(
#         os.path.join(RESULT_DIR, "streamflow_performance_metrics.csv"), index=False
#     )

#     print("\n✅ All basin simulations completed and results saved.")