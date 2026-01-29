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
            ET_override=float(ET_series[t]) if np.isfinite(ET_series[t]) else None # given  new ET, update storages (S,G) and compute Q
        )

        G = np.clip(G, 0.0, Gmax_cal)

        Q_out[t] = max(Q, 0.0)
        S_out[t] = S
        G_out[t] = G

    return Q_out, S_out, G_out


# # ---------------------------------------------------------
# # DA run (EnKF) -> produces ET_ass and Q_ass_mean
# # ---------------------------------------------------------
# def run_budyko_da(
#     P: np.ndarray,
#     PET: np.ndarray,
#     ET_B: np.ndarray,
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
#     S0_ens = np.clip(S_init + rng.normal(0.0, 0.05 * Smax_cal, size=nens), 0.0, Smax_cal)
#     G0_ens = np.clip(G_init + rng.normal(0.0, 0.05 * Gmax_cal, size=nens), 0.0, Gmax_cal)
#     X = np.vstack([S0_ens, G0_ens])  

#     # Save ensemble histories
#     S_ens_hist  = np.full((L, nens), np.nan)
#     G_ens_hist  = np.full((L, nens), np.nan)
#     ET_ens_hist = np.full((L, nens), np.nan)
#     Q_ens_hist  = np.full((L, nens), np.nan)

#     # mean time series
#     ET_ass_mean = np.full(L, np.nan)
#     Q_ass_mean  = np.full(L, np.nan)

#     for t in range(L):

#         # Forecast
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
            

#             ET_override=None, # This lets the state (S/G) evolve or update
#         )

#         ET_ens_hist[t, :] = ET_ens
#         Q_ens_hist[t, :] = Q_ens

#         ET_ass_mean[t] = np.nanmean(ET_ens)
#         Q_ass_mean[t] = np.nanmean(Q_ens)

#         # Analysis (assimilate ET_B)
#         if np.isfinite(ET_B[t]):
#             X = enkf_update_stochastic_scalar(
#                 X=X_f,
#                 y_obs=float(ET_B[t]),
#                 HX=ET_ens.copy(),
#                 R_var=R_ET_var,
#                 inflation=inflation,
#                 Smax=Smax_cal,
#                 Gmax=Gmax_cal,
#                 rng=rng,
#             )
#         else:
#             X = X_f

#         S_ens_hist[t, :] = X[0, :]
#         G_ens_hist[t, :] = X[1, :]

#     enkf_hist = {
#         "time": np.arange(L),
#         "nens": nens,
#         "S_ens": S_ens_hist,
#         "G_ens": G_ens_hist,
#         "ET_ens": ET_ens_hist,
#         "Q_ens": Q_ens_hist,
#     }

#     return ET_ass_mean, Q_ass_mean, enkf_hist

# ---------------------------------------------------------
# DA run (EnKF) -> assimilates ET_model toward ET_obs
# produces ET_ass_mean (posterior) and Q_ass_mean
# ---------------------------------------------------------
def run_budyko_da(
    P: np.ndarray,
    PET: np.ndarray,
    ET_obs: np.ndarray,     # ✅ truth (e.g., ET_B)
    ET_model: np.ndarray,   # ✅ model (e.g., ET_ke)
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

        # -------------------------------------------------
        # Forecast: ET is forced to ET_model[t] (ET_ke)
        # -------------------------------------------------
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

        # -------------------------------------------------
        # Analysis: assimilate ET_obs[t] (ET_B) into state
        # -------------------------------------------------
        if np.isfinite(ET_obs[t]):
            X_a = enkf_update_stochastic_scalar(
                X=X_f,
                y_obs=float(ET_obs[t]),   # ✅ truth
                HX=ET_ens_f.copy(),       # ✅ model forecast ET
                R_var=R_ET_var,
                inflation=inflation,
                Smax=Smax_cal,
                Gmax=Gmax_cal,
                rng=rng,
            )
        else:
            X_a = X_f

        # -------------------------------------------------
        # Posterior: recompute ET/Q from updated states
        # (so ET_ass reflects assimilation)
        # -------------------------------------------------
        X_dummy, ET_ens_a, Q_ens_a = enkf_forecast_step_states(
            X=X_a,
            P_t=float(P[t]) if np.isfinite(P[t]) else 0.0,
            PET_t=float(PET[t]) if np.isfinite(PET[t]) else 0.0,
            params_cal=params_cal,
            Smax=Smax_cal,
            Gmax=Gmax_cal,
            rng=rng,

            proc_S_std=0.0,   # no additional noise in posterior evaluation
            proc_G_std=0.0,
            P_std_frac=0.0,
            PET_std_frac=0.0,

            # ✅ keep same ET model forcing
            ET_override=ET_override_t,
        )

        # Save posterior means (this is the assimilated ET)
        ET_ass_mean[t] = np.nanmean(ET_ens_a)
        Q_ass_mean[t]  = np.nanmean(Q_ens_a)

        # Store updated ensemble states and continue
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
    Qsb = Qsb_df.get(basin_id, pd.Series(index=idx)).values

    ET_nldas = Evap_df[basin_id].values

    # Budyko ET
    M_basin = M_df.copy()
    M_basin.index = pd.to_datetime(M_basin.index)
    M_basin = M_basin.loc[idx]
    Evap_df = Evap_df[common_cols]
    Qsb_df  = Qsb_df[common_cols]
    PET_df  = PET_df[common_cols]
    M_df    = M_df[common_cols]
    RootMoist = RootMoist[common_cols]

    budyko = BudykoModelEstimator(
        Evap_df=Evap_df[common_cols],
        Qsb_monthly=Qsb_df[common_cols],
        PotEvap_df=PET_df[common_cols],
        M_basin=M_df[common_cols],
        Slope_basin=RootMoist[common_cols],  # but this should be slope, not soil moisture
        calibrated_params=calibrated_params,
    )

    budyko.estimate_budyko_et()
    omega_true_all = budyko.omega_true[basin_id].reindex(idx).to_numpy().ravel()
    omega_MLR_all  = budyko.omega_MLR[basin_id].reindex(idx).to_numpy().ravel()
    ET_B           = budyko.ET_B[basin_id].reindex(idx).to_numpy().ravel()

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

    enkf_hist = None # for saving ensemble history if DA is run
    ET_ass_mean = np.full(L, np.nan)
    Q_ass_mean = np.full(L, np.nan)

    if scenario == "BASE":
        Q_base, S_base, G_base = run_model_deterministic(
            P=P, 
            PET=PET,
            params_cal=params_cal,
            S_init=S_init,
            G_init=G_init,
            Gmax_cal=Gmax_cal,
            ET_series=ET_ke,  # Simple Scaling Based ET
        )

        results = pd.DataFrame({
            "time": idx,
            "omega_true": omega_true_all,
            "omega_MLR": omega_MLR_all,
            "P": P,
            "PET": PET,
            "ET_ke": ET_ke,
            "Q_bs" :Qsb,
            "Q_obs": Q_obs,
            "Q_base": Q_base,
            "S_base": S_base,
            "G_base": G_base
        }).set_index("time")


    elif scenario == "BUDYKO":
        Q_budyko, S_budyko, G_budyko = run_model_deterministic(
            P=P, PET=PET,
            params_cal=params_cal,
            S_init=S_init,
            G_init=G_init,
            Gmax_cal=Gmax_cal,
            ET_series=ET_B,  # Budyko ET
        )
        results = pd.DataFrame({
            "time": idx,
            "P": P,
            "PET": PET,
            "ET_B": ET_B,
            "Q_obs": Q_obs,
            "Q_budyko": Q_budyko,
            "S_budyko": S_budyko,
            "G_budyko": G_budyko,
        }).set_index("time")

# ET assimilation with Budyko, but Q is determined deterministically
    elif scenario == "BUDYKO_DA":
        config = EnKFConfig(**da_cfg)

        ET_ass_mean, Q_ass_mean, enkf_hist = run_budyko_da(
            P=P,
            PET=PET,
            ET_obs=ET_B,      # truth (observation)
            ET_model=ET_ke,   # model
            params_cal=params_cal,
            S_init=S_init,
            G_init=G_init,
            Smax_cal=Smax_cal,
            Gmax_cal=Gmax_cal,
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
            "Q_ens": Q_ass_mean,
            
        }).set_index("time")

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Save results
    result_path = os.path.join(RESULT_DIR, f"results_{scenario}_{basin_id}.feather")
    results.reset_index().to_feather(result_path)
    if scenario == "BUDYKO_DA" and enkf_hist is not None:

        # save ensemble mean as a feather file
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

