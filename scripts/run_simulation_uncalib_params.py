# scripts/run_simulation_uncalib_params.py

import os
import sys
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import heapq

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import ModelParams, two_store_model_step
from src.budyko import BudykoModelEstimator
from src.enkf import EnKFConfig, enkf_update_stochastic_scalar, enkf_forecast_step_states
from src.metrics import calculate_kge, calculate_nse


# =========================================================
# Scenario helpers
# =========================================================
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


# =========================================================
# IO
# =========================================================
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


# =========================================================
# Random parameters (UNCALIB)
# =========================================================
def sample_random_params_for_basin(basin_id: str, bounds: dict, seed: int = 1234) -> dict:
    basin_seed = (hash(str(basin_id)) + int(seed)) % (2**32 - 1)
    rng = np.random.default_rng(basin_seed)

    def uniform(name: str, default=(0.1, 0.99)):
        lo, hi = bounds.get(name, default)
        return float(rng.uniform(float(lo), float(hi)))

    Kperc = uniform("Kperc")
    Kb    = uniform("Kb")
    Ke    = uniform("Ke")
    Cqq   = uniform("Cqq")

    Smax        = uniform("Smax")
    Gmax_factor = uniform("Gmax_factor")
    Gmax        = Smax * Gmax_factor

    fS0 = uniform("fS0", (0.1, 0.8))
    fG0 = uniform("fG0", (0.1, 0.8))

    return {
        "Kperc": Kperc,
        "Kb": Kb,
        "Ke": Ke,
        "Cqq": Cqq,
        "Smax": Smax,
        "Gmax_factor": Gmax_factor,
        "S_init": fS0 * Smax,
        "G_init": fG0 * Gmax,
        "fS0": fS0,
        "fG0": fG0,
    }


# =========================================================
# Deterministic model (MATCH CALIBRATED OUTPUT FORMAT)
# =========================================================
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

    Q_out = np.full(L, np.nan)
    S_out = np.full(L, np.nan)
    G_out = np.full(L, np.nan)

    for t in range(L):
        P_t   = float(P[t])   if np.isfinite(P[t])   else 0.0
        PET_t = float(PET[t]) if np.isfinite(PET[t]) else 0.0
        et_t  = float(ET_series[t]) if np.isfinite(ET_series[t]) else None

        S, G, _, Q, *_ = two_store_model_step(
            S, G, P_t, PET_t, params_cal,
            ET_override=et_t
        )

        G = np.clip(G, 0.0, Gmax_cal)

        Q_out[t] = max(float(Q), 0.0)
        S_out[t] = float(S)
        G_out[t] = float(G)

    return Q_out, S_out, G_out


# =========================================================
# DA run (EnKF) -> MATCH CALIBRATED FORMAT
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
):
    L = len(P)
    nens = int(config.nens)
    inflation = float(config.inflation)
    R_ET_var = float(config.R_ET_std) ** 2

    rng = np.random.default_rng(hash(basin_id) % (2**32 - 1))

    S0_ens = np.clip(S_init + rng.normal(0.0, 0.05 * Smax_cal, size=nens), 0.0, Smax_cal)
    G0_ens = np.clip(G_init + rng.normal(0.0, 0.05 * Gmax_cal, size=nens), 0.0, Gmax_cal)
    X = np.vstack([S0_ens, G0_ens])

    S_ens_hist  = np.full((L, nens), np.nan)
    G_ens_hist  = np.full((L, nens), np.nan)
    ET_ens_hist = np.full((L, nens), np.nan)
    Q_ens_hist  = np.full((L, nens), np.nan)

    ET_ass_mean = np.full(L, np.nan)
    Q_ass_mean  = np.full(L, np.nan)

    for t in range(L):
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

        # save prior
        ET_ens_hist[t, :] = ET_ens_f
        Q_ens_hist[t, :]  = Q_ens_f

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

        # posterior ET/Q from updated states (no extra noise)
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

        X = X_a
        S_ens_hist[t, :] = X[0, :]
        G_ens_hist[t, :] = X[1, :]

    enkf_hist = {
        "time": np.arange(L),
        "nens": nens,
        "S_ens": S_ens_hist,
        "G_ens": G_ens_hist,
        "ET_ens": ET_ens_hist,
        "Q_ens": Q_ens_hist,
    }

    return ET_ass_mean, Q_ass_mean, enkf_hist


# =========================================================
# One trial (DOES NOT WRITE unless write_outputs=True)
# =========================================================
def _run_one_trial(
    basin_id: str,
    scenario: str,
    idx,
    P,
    PET,
    Q_obs,
    Qsb,
    Evap_df,
    Qsb_df,
    PET_df,
    RootMoist_df,
    Slp_df,
    calibrated_params: dict,
    da_cfg: dict,
    use_calibrated: bool,
    random_bounds: dict,
    base_seed: int,
    trial_id: int,
    RESULT_DIR: str,
    write_outputs: bool,
):
    # choose params
    if use_calibrated:
        p = calibrated_params[basin_id]
    else:
        p = sample_random_params_for_basin(
            basin_id=basin_id,
            bounds=random_bounds,
            seed=int(base_seed) + int(trial_id),
        )

    # model params
    Smax_cal = float(p.get("Smax", 50.0))
    Gmax_factor = float(p.get("Gmax_factor", 4.0))
    Gmax_cal = Smax_cal * Gmax_factor

    S_init = float(p.get("S_init", 0.5 * Smax_cal))
    G_init = float(p.get("G_init", 0.5 * Gmax_cal))

    params_cal = ModelParams(
        Smax=Smax_cal,
        Kperc=float(p["Kperc"]),
        Kb=float(p["Kb"]),
        Ke=float(p["Ke"]),
        Cqq=float(p["Cqq"]),
        Sfc_frac=0.30,
        beta_et=2.0,
    )

    ET_ke = PET * float(params_cal.Ke)

    # Budyko
    omega_true_all = np.full(len(idx), np.nan)
    omega_MLR_all  = np.full(len(idx), np.nan)
    ET_B           = np.full(len(idx), np.nan)

    if scenario in ["BUDYKO", "BUDYKO_DA"]:
        # >>> THIS is the critical fix: give Budyko the UNCALIB Ke via Ke_df
        Ke_df = None
        if not use_calibrated:
            Ke_df = pd.DataFrame(index=idx, data={basin_id: float(p["Ke"])})

        budyko = BudykoModelEstimator(
            Evap_df=Evap_df[[basin_id]],
            Qsb_monthly=Qsb_df[[basin_id]],
            PotEvap_df=PET_df[[basin_id]],
            M_basin=RootMoist_df[[basin_id]],
            Slope_basin=Slp_df[[basin_id]],   # <<< correct slope
            calibrated_params=calibrated_params if use_calibrated else None,
            Ke_df=Ke_df,
        )

        budyko.estimate_budyko_et()

        if hasattr(budyko, "omega_true"):
            omega_true_all = budyko.omega_true[basin_id].reindex(idx).to_numpy(dtype=float).ravel()
        if hasattr(budyko, "omega_MLR"):
            omega_MLR_all = budyko.omega_MLR[basin_id].reindex(idx).to_numpy(dtype=float).ravel()

        ET_B = budyko.ET_B[basin_id].reindex(idx).to_numpy(dtype=float).ravel()

        # if still NaN, trial is invalid
        if np.all(~np.isfinite(ET_B)):
            return None, None

    enkf_hist = None
    ET_ass_mean = np.full(len(idx), np.nan)
    Q_ens_mean  = np.full(len(idx), np.nan)

    # Run scenario and match CALIBRATED columns
    if scenario == "BASE":
        Q_base, S_base, G_base = run_model_deterministic(
            P=P, PET=PET,
            params_cal=params_cal,
            S_init=S_init, G_init=G_init,
            Gmax_cal=Gmax_cal,
            ET_series=ET_ke,
        )
        results = pd.DataFrame({
            "time": idx,
            "P": P,
            "PET": PET,
            "ET_ke": ET_ke,
            "Q_bs": Qsb,
            "Q_obs": Q_obs,
            "Q_base": Q_base,
            "S_base": S_base,
            "G_base": G_base,
        }).set_index("time")

    elif scenario == "BUDYKO":
        Q_budyko, S_budyko, G_budyko = run_model_deterministic(
            P=P, PET=PET,
            params_cal=params_cal,
            S_init=S_init, G_init=G_init,
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
            "Q_obs": Q_obs,
            "Q_budyko": Q_budyko,
            "S_budyko": S_budyko,
            "G_budyko": G_budyko,
        }).set_index("time")

    elif scenario == "BUDYKO_DA":
        config = EnKFConfig(**da_cfg)

        ET_ass_mean, Q_ens_mean, enkf_hist = run_budyko_da(
            P=P,
            PET=PET,
            ET_obs=ET_B,
            ET_model=ET_ke,
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
            S_init=S_init, G_init=G_init,
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
            "Q_ens": Q_ens_mean,
        }).set_index("time")

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # metrics
    qobs = results["Q_obs"].values if "Q_obs" in results.columns else Q_obs
    qcol = [c for c in results.columns if c.startswith("Q_") and c != "Q_obs"]
    qsim_name = qcol[-1] if qcol else None
    qsim = results[qsim_name].values if qsim_name else None

    KGE = calculate_kge(qobs, qsim) if qsim is not None else np.nan
    NSE = calculate_nse(qobs, qsim) if qsim is not None else np.nan

    metrics = {
        "gauge_id": basin_id,
        "scenario": scenario,
        "trial": int(trial_id),
        "KGE": float(KGE) if np.isfinite(KGE) else np.nan,
        "NSE": float(NSE) if np.isfinite(NSE) else np.nan,
        "Kperc": float(params_cal.Kperc),
        "Kb": float(params_cal.Kb),
        "Ke": float(params_cal.Ke),
        "Cqq": float(params_cal.Cqq),
        "Smax": float(params_cal.Smax),
        "S_init": float(S_init),
        "G_init": float(G_init),
        "Gmax": float(Gmax_cal),
    }

    # write outputs ONLY if requested
    if write_outputs:
        # results_{scenario}_{basin}.feather  (MATCH CALIBRATED)
        results.reset_index().to_feather(os.path.join(RESULT_DIR, f"results_{scenario}_{basin_id}.feather"))

        if scenario == "BUDYKO_DA" and enkf_hist is not None:
            enkf_df = pd.DataFrame({
                "time": idx,
                "ET_ens_mean": np.nanmean(enkf_hist["ET_ens"], axis=1),
                "Q_ens_mean": np.nanmean(enkf_hist["Q_ens"], axis=1),
                "S_ens_mean": np.nanmean(enkf_hist["S_ens"], axis=1),
                "G_ens_mean": np.nanmean(enkf_hist["G_ens"], axis=1),
            })
            enkf_df.to_feather(os.path.join(RESULT_DIR, f"enkf_ensemble_{scenario}_{basin_id}.feather"))

    return metrics, results


# =========================================================
# Per-basin simulation (pick BEST trial, write BEST only)
# =========================================================
def simulate_basin(
    basin_id,
    scenario,
    DATA_DIR,
    RESULT_DIR,
    calibrated_params,
    da_cfg: dict,
    use_calibrated: bool,
    random_bounds: dict,
    random_seed: int,
    n_trials: int,
    save_top_k: int,   # kept for signature compatibility
):
    PET_df      = load_feather_df("PotEvap.feather", DATA_DIR)
    Rainf_df    = load_feather_df("Rainf.feather", DATA_DIR)
    Evap_df     = load_feather_df("EVap.feather", DATA_DIR)
    Q_usgs_df   = load_feather_df("Q_USGS.feather", DATA_DIR)
    Qsb_df      = load_feather_df("Qsb.feather", DATA_DIR)
    RootMoist_df = load_feather_df("SoilM_0_200cm.feather", DATA_DIR)
    Slp_df      = load_feather_df("slope.feather", DATA_DIR)

    # intersection (match calibrated behavior)
    common_cols = sorted(
        set(Evap_df.columns)
        & set(Qsb_df.columns)
        & set(PET_df.columns)
        & set(Rainf_df.columns)
        & set(RootMoist_df.columns)
        & set(Slp_df.columns)
    )

    if basin_id not in common_cols:
        return None

    if use_calibrated and basin_id not in calibrated_params:
        return None

    # restrict
    Evap_df      = Evap_df[common_cols]
    Qsb_df       = Qsb_df[common_cols]
    PET_df       = PET_df[common_cols]
    Rainf_df     = Rainf_df[common_cols]
    RootMoist_df = RootMoist_df[common_cols]
    Slp_df       = Slp_df[common_cols]

    idx = Evap_df.index

    PET = pd.to_numeric(PET_df[basin_id], errors="coerce").to_numpy(dtype=float)
    P   = pd.to_numeric(Rainf_df[basin_id], errors="coerce").to_numpy(dtype=float)

    Q_obs = pd.to_numeric(
        Q_usgs_df.get(basin_id, pd.Series(index=idx)).reindex(idx),
        errors="coerce"
    ).to_numpy(dtype=float)

    Qsb = pd.to_numeric(
        Qsb_df.get(basin_id, pd.Series(index=idx)).reindex(idx),
        errors="coerce"
    ).to_numpy(dtype=float)

    n_trials = int(max(1, n_trials))

    best = None
    best_kge = -np.inf
    best_trial_id = None

    # run trials (NO writing)
    for trial in range(n_trials):
        m, _ = _run_one_trial(
            basin_id=basin_id,
            scenario=scenario,
            idx=idx,
            P=P,
            PET=PET,
            Q_obs=Q_obs,
            Qsb=Qsb,
            Evap_df=Evap_df,
            Qsb_df=Qsb_df,
            PET_df=PET_df,
            RootMoist_df=RootMoist_df,
            Slp_df=Slp_df,
            calibrated_params=calibrated_params,
            da_cfg=da_cfg,
            use_calibrated=use_calibrated,
            random_bounds=random_bounds,
            base_seed=random_seed,
            trial_id=trial,
            RESULT_DIR=RESULT_DIR,
            write_outputs=False,
        )
        if m is None:
            continue

        kge = m.get("KGE", np.nan)
        if np.isfinite(kge) and kge > best_kge:
            best_kge = kge
            best = m
            best_trial_id = trial

    if best is None:
        return None

    # write ONLY the best trial outputs with standard filenames
    _best_metrics, _best_results = _run_one_trial(
        basin_id=basin_id,
        scenario=scenario,
        idx=idx,
        P=P,
        PET=PET,
        Q_obs=Q_obs,
        Qsb=Qsb,
        Evap_df=Evap_df,
        Qsb_df=Qsb_df,
        PET_df=PET_df,
        RootMoist_df=RootMoist_df,
        Slp_df=Slp_df,
        calibrated_params=calibrated_params,
        da_cfg=da_cfg,
        use_calibrated=use_calibrated,
        random_bounds=random_bounds,
        base_seed=random_seed,
        trial_id=int(best_trial_id),
        RESULT_DIR=RESULT_DIR,
        write_outputs=True,
    )

    best["rank"] = 1
    best["n_trials"] = int(n_trials)
    best["save_top_k"] = int(save_top_k)

    return best


# =========================================================
# Main
# =========================================================
def run_simulations_from_config(cfg: dict):
    scenario = normalize_scenario_key(cfg["scenario"])
    scenarios_to_run = ["BASE", "BUDYKO", "BUDYKO_DA"] if scenario == "ALL" else [scenario]

    paths = cfg["paths"]
    DATA_DIR = os.path.join(PROJECT_ROOT, paths["data_dir"])

    BASE_RESULT_DIR = os.path.join(PROJECT_ROOT, paths["result_dir"])
    os.makedirs(BASE_RESULT_DIR, exist_ok=True)

    # params mode
    params_cfg = cfg.get("params", {})
    if "UNCALIBRATED" in params_cfg:
        params_cfg = params_cfg["UNCALIBRATED"]

    use_calibrated = bool(params_cfg.get("use_calibrated", False))
    random_seed = int(params_cfg.get("random_seed", 1234))
    random_bounds = params_cfg.get("bounds", {})

    n_trials = int(params_cfg.get("n_trials", 1))
    save_top_k = int(params_cfg.get("save_top_k", 1))

    calibrated_params = {}
    if use_calibrated:
        cal_path = os.path.join(PROJECT_ROOT, paths["calibrated_params"])
        with open(cal_path, "r") as f:
            calibrated_params = json.load(f)

    da_cfg = cfg.get("da", {})
    da_cfg.pop("enabled", None)

    basin_subset = cfg.get("basins", {}).get("subset", None)
    if basin_subset is None:
        tmp = load_feather_df("EVap.feather", DATA_DIR)
        basins = list(tmp.columns)
    else:
        basins = list(basin_subset)

    par_cfg = cfg.get("parallel", {})
    par_enabled = bool(par_cfg.get("enabled", True))
    max_workers = int(par_cfg.get("max_workers", -1))
    if max_workers == -1:
        max_workers = max(1, cpu_count() - 1)

    for sc in scenarios_to_run:
        sc_dir = scenario_folder_name(sc)
        RESULT_DIR = os.path.join(BASE_RESULT_DIR, sc_dir)
        os.makedirs(RESULT_DIR, exist_ok=True)

        all_metrics = []

        if par_enabled:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        simulate_basin,
                        b, sc, DATA_DIR, RESULT_DIR,
                        calibrated_params, da_cfg,
                        use_calibrated, random_bounds, random_seed,
                        n_trials, save_top_k
                    ): b
                    for b in basins
                }
                for f in tqdm(as_completed(futures), total=len(futures), desc=f"Running scenario={sc}"):
                    out = f.result()
                    if out is not None:
                        all_metrics.append(out)
        else:
            for b in tqdm(basins, desc=f"Running scenario={sc}"):
                out = simulate_basin(
                    b, sc, DATA_DIR, RESULT_DIR,
                    calibrated_params, da_cfg,
                    use_calibrated, random_bounds, random_seed,
                    n_trials, save_top_k
                )
                if out is not None:
                    all_metrics.append(out)

        pd.DataFrame(all_metrics).to_csv(
            os.path.join(RESULT_DIR, f"metrics_{sc}.csv"),
            index=False
        )

        print(f"\n✅ Completed successfully: scenario={sc}. Results saved to {RESULT_DIR}")


if __name__ == "__main__":
    import yaml
    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    run_simulations_from_config(cfg)










# # scripts/run_simulation_uncalib_params.py

# import os
# import sys
# import json
# import logging
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from multiprocessing import cpu_count
# import heapq

# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# from src.model import ModelParams, two_store_model_step
# from src.budyko import BudykoModelEstimator
# from src.enkf import EnKFConfig, enkf_update_stochastic_scalar, enkf_forecast_step_states
# from src.metrics import calculate_kge, calculate_nse


# # ---------------------------------------------------------
# # scenario folder
# # ---------------------------------------------------------
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
# # Random param generator
# # ---------------------------------------------------------
# def sample_random_params_for_basin(basin_id: str, bounds: dict, seed: int = 1234) -> dict:
#     """
#     Sample random parameter set within bounds.
#     Returned dict uses same keys expected downstream.
#     """
#     basin_seed = (hash(str(basin_id)) + int(seed)) % (2**32 - 1)
#     rng = np.random.default_rng(basin_seed)

#     def uniform(name: str, default=(0.1, 0.99)):
#         lo, hi = bounds.get(name, default)
#         return float(rng.uniform(float(lo), float(hi)))

#     Kperc = uniform("Kperc")
#     Kb    = uniform("Kb")
#     Ke    = uniform("Ke")
#     Cqq   = uniform("Cqq")

#     Smax        = uniform("Smax")
#     Gmax_factor = uniform("Gmax_factor")
#     Gmax        = Smax * Gmax_factor

#     fS0 = uniform("fS0", (0.1, 0.8))
#     fG0 = uniform("fG0", (0.1, 0.8))
#     S_init = fS0 * Smax
#     G_init = fG0 * Gmax

#     return {
#         "Kperc": Kperc,
#         "Kb": Kb,
#         "Ke": Ke,
#         "Cqq": Cqq,
#         "Smax": Smax,
#         "Gmax_factor": Gmax_factor,
#         "S_init": S_init,
#         "G_init": G_init,
#         "fS0": fS0,
#         "fG0": fG0,
#     }


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
#         P_t   = float(P[t])   if np.isfinite(P[t])   else 0.0
#         PET_t = float(PET[t]) if np.isfinite(PET[t]) else 0.0

#         et_t = float(ET_series[t]) if np.isfinite(ET_series[t]) else None

#         S, G, _, Q, *_ = two_store_model_step(
#             S, G, P_t, PET_t, params_cal,
#             ET_override=et_t
#         )

#         G = np.clip(G, 0.0, Gmax_cal)

#         Q_out[t] = max(Q, 0.0)
#         S_out[t] = S
#         G_out[t] = G

#     return Q_out, S_out, G_out


# # ---------------------------------------------------------
# # DA run (EnKF) -> assimilates ET_model toward ET_obs
# # ---------------------------------------------------------
# def run_budyko_da(
#     P: np.ndarray,
#     PET: np.ndarray,
#     ET_obs: np.ndarray,
#     ET_model: np.ndarray,
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

#     S0_ens = np.clip(S_init + rng.normal(0.0, 0.05 * Smax_cal, size=nens), 0.0, Smax_cal)
#     G0_ens = np.clip(G_init + rng.normal(0.0, 0.05 * Gmax_cal, size=nens), 0.0, Gmax_cal)
#     X = np.vstack([S0_ens, G0_ens])

#     S_ens_hist  = np.full((L, nens), np.nan)
#     G_ens_hist  = np.full((L, nens), np.nan)
#     ET_ens_hist = np.full((L, nens), np.nan)
#     Q_ens_hist  = np.full((L, nens), np.nan)

#     ET_ass_mean = np.full(L, np.nan)
#     Q_ass_mean  = np.full(L, np.nan)

#     for t in range(L):

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

#         ET_ens_hist[t, :] = ET_ens_f
#         Q_ens_hist[t, :]  = Q_ens_f

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

#         X_dummy, ET_ens_a, Q_ens_a = enkf_forecast_step_states(
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

#         X = X_a
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


# # ---------------------------------------------------------
# # One trial run (return metrics + optionally write results)
# # ---------------------------------------------------------
# def _run_one_trial(
#     basin_id: str,
#     scenario: str,
#     idx,
#     P,
#     PET,
#     Q_obs,
#     Qsb,
#     Evap_df,
#     Qsb_df,
#     PET_df,
#     RootMoist,
#     Slp_df,
#     calibrated_params,
#     da_cfg,
#     use_calibrated,
#     random_bounds,
#     base_seed,
#     trial_id,
#     RESULT_DIR,
#     write_outputs: bool,
# ):
#     # -----------------------------------------
#     # choose params
#     # -----------------------------------------
#     if use_calibrated:
#         p = calibrated_params[basin_id]
#     else:
#         # ✅ change seed each trial (THIS WAS MISSING BEFORE)
#         p = sample_random_params_for_basin(
#             basin_id=basin_id,
#             bounds=random_bounds,
#             seed=int(base_seed) + int(trial_id)
#         )

#     # model params
#     Smax_cal    = float(p.get("Smax", 50.0))
#     Gmax_factor = float(p.get("Gmax_factor", 4.0))
#     Gmax_cal    = Smax_cal * Gmax_factor

#     S_init = float(p.get("S_init", 0.5 * Smax_cal))
#     G_init = float(p.get("G_init", 0.5 * Gmax_cal))

#     params_cal = ModelParams(
#         Smax=Smax_cal,
#         Kperc=float(p["Kperc"]),
#         Kb=float(p["Kb"]),
#         Ke=float(p["Ke"]),
#         Cqq=float(p["Cqq"]),
#         Sfc_frac=0.30,
#         beta_et=2.0,
#     )

#     ET_ke = PET * float(params_cal.Ke)

#     # -----------------------------------------
#     # Budyko ET if needed
#     # -----------------------------------------
#     omega_true_all = np.full(len(idx), np.nan)
#     omega_MLR_all  = np.full(len(idx), np.nan)
#     ET_B           = np.full(len(idx), np.nan)

#     if scenario in ["BUDYKO", "BUDYKO_DA"]:
#         Ke_df = None
#         if not use_calibrated:
#             Ke_df = pd.DataFrame(index=idx, data={basin_id: float(p["Ke"])})

#         budyko = BudykoModelEstimator(
#             Evap_df=Evap_df[[basin_id]],
#             Qsb_monthly=Qsb_df[[basin_id]],
#             PotEvap_df=PET_df[[basin_id]],
#             M_basin=RootMoist[[basin_id]],
#             Slope_basin=Slp_df[[basin_id]],
#             calibrated_params=calibrated_params if use_calibrated else None,
#             Ke_df=Ke_df,
#         )

#         budyko.estimate_budyko_et()

#         omega_true_all = budyko.omega_true[basin_id].reindex(idx).to_numpy().astype(float).ravel()
#         omega_MLR_all  = budyko.omega_MLR[basin_id].reindex(idx).to_numpy().astype(float).ravel()
#         ET_B           = budyko.ET_B[basin_id].reindex(idx).to_numpy().astype(float).ravel()

#         if np.all(~np.isfinite(ET_B)):
#             return None

#     # -----------------------------------------
#     # Run scenario
#     # -----------------------------------------
#     enkf_hist = None
#     ET_ass_mean = np.full(len(idx), np.nan)
#     Q_ass_mean  = np.full(len(idx), np.nan)

#     if scenario == "BASE":
#         Q_base, S_base, G_base = run_model_deterministic(
#             P=P, PET=PET,
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
#             "G_base": G_base,
#         }).set_index("time")

#     elif scenario == "BUDYKO":
#         Q_budyko, S_budyko, G_budyko = run_model_deterministic(
#             P=P, PET=PET,
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
#             ET_obs=ET_B,
#             ET_model=ET_ke,
#             params_cal=params_cal,
#             S_init=S_init,
#             G_init=G_init,
#             Smax_cal=Smax_cal,
#             Gmax_cal=Gmax_cal,
#             config=config,
#             basin_id=basin_id,
#         )

#         Q_ass, _, _ = run_model_deterministic(
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

#     # -----------------------------------------
#     # Metrics
#     # -----------------------------------------
#     qobs = results["Q_obs"].values if "Q_obs" in results.columns else Q_obs
#     qcol = [c for c in results.columns if c.startswith("Q_") and c != "Q_obs"]
#     qsim_name = qcol[-1] if qcol else None
#     qsim = results[qsim_name].values if qsim_name else None

#     KGE = calculate_kge(qobs, qsim) if qsim is not None else np.nan
#     NSE = calculate_nse(qobs, qsim) if qsim is not None else np.nan

#     # Optionally write outputs (ONLY for top-k or best)
#     if write_outputs:
#         fname = f"results_{scenario}_{basin_id}_trial{trial_id:04d}.feather"
#         results.reset_index().to_feather(os.path.join(RESULT_DIR, fname))

#         if scenario == "BUDYKO_DA" and enkf_hist is not None:
#             enkf_df = pd.DataFrame({
#                 "time": idx,
#                 "ET_ens_mean": np.nanmean(enkf_hist["ET_ens"], axis=1),
#                 "Q_ens_mean": np.nanmean(enkf_hist["Q_ens"], axis=1),
#                 "S_ens_mean": np.nanmean(enkf_hist["S_ens"], axis=1),
#                 "G_ens_mean": np.nanmean(enkf_hist["G_ens"], axis=1),
#             })
#             enkf_path = os.path.join(RESULT_DIR, f"enkf_ensemble_{scenario}_{basin_id}_trial{trial_id:04d}.feather")
#             enkf_df.to_feather(enkf_path)

#     metrics = {
#         "gauge_id": basin_id,
#         "scenario": scenario,
#         "trial": int(trial_id),
#         "KGE": float(KGE) if np.isfinite(KGE) else np.nan,
#         "NSE": float(NSE) if np.isfinite(NSE) else np.nan,
#         "Kperc": float(params_cal.Kperc),
#         "Kb": float(params_cal.Kb),
#         "Ke": float(params_cal.Ke),
#         "Cqq": float(params_cal.Cqq),
#         "Smax": float(params_cal.Smax),
#         "S_init": float(S_init),
#         "G_init": float(G_init),
#         "Gmax": float(Gmax_cal),
#     }

#     return metrics


# # ---------------------------------------------------------
# # Main simulation per basin
# # Implements n_trials + save_top_k (FIXED)
# # ---------------------------------------------------------
# def simulate_basin(
#     basin_id,
#     scenario,
#     DATA_DIR,
#     RESULT_DIR,
#     calibrated_params,
#     da_cfg: dict,
#     use_calibrated: bool,
#     random_bounds: dict,
#     random_seed: int,
#     n_trials: int,
#     save_top_k: int,
# ):

#     PET_df      = load_feather_df("PotEvap.feather", DATA_DIR)
#     Rainf_df    = load_feather_df("Rainf.feather", DATA_DIR)
#     Evap_df     = load_feather_df("EVap.feather", DATA_DIR)
#     Q_usgs_df   = load_feather_df("Q_USGS.feather", DATA_DIR)
#     Qsb_df      = load_feather_df("Qsb.feather", DATA_DIR)
#     RootMoist   = load_feather_df("SoilM_0_200cm.feather", DATA_DIR)
#     Slp_df      = load_feather_df("slope.feather", DATA_DIR)

#     common_cols = sorted(
#         set(Evap_df.columns)
#         & set(Qsb_df.columns)
#         & set(PET_df.columns)
#         & set(Rainf_df.columns)
#         & set(RootMoist.columns)
#         & set(Slp_df.columns)
#     )

#     if basin_id not in common_cols:
#         return None

#     if use_calibrated and basin_id not in calibrated_params:
#         return None

#     Evap_df    = Evap_df[common_cols]
#     Qsb_df     = Qsb_df[common_cols]
#     PET_df     = PET_df[common_cols]
#     Rainf_df   = Rainf_df[common_cols]
#     RootMoist  = RootMoist[common_cols]
#     Slp_df     = Slp_df[common_cols]

#     idx = Evap_df.index

#     PET = pd.to_numeric(PET_df[basin_id], errors="coerce").to_numpy(dtype=float)
#     P   = pd.to_numeric(Rainf_df[basin_id], errors="coerce").to_numpy(dtype=float)

#     Q_obs = pd.to_numeric(
#         Q_usgs_df.get(basin_id, pd.Series(index=idx)).reindex(idx),
#         errors="coerce"
#     ).to_numpy(dtype=float)

#     Qsb = pd.to_numeric(
#         Qsb_df.get(basin_id, pd.Series(index=idx)),
#         errors="coerce"
#     ).to_numpy(dtype=float)

#     # -------------------------------------------------
#     # CALIBRATED MODE: run once
#     # -------------------------------------------------
#     if use_calibrated:
#         out = _run_one_trial(
#             basin_id=basin_id,
#             scenario=scenario,
#             idx=idx,
#             P=P,
#             PET=PET,
#             Q_obs=Q_obs,
#             Qsb=Qsb,
#             Evap_df=Evap_df,
#             Qsb_df=Qsb_df,
#             PET_df=PET_df,
#             RootMoist=RootMoist,
#             Slp_df=Slp_df,
#             calibrated_params=calibrated_params,
#             da_cfg=da_cfg,
#             use_calibrated=True,
#             random_bounds=random_bounds,
#             base_seed=random_seed,
#             trial_id=0,
#             RESULT_DIR=RESULT_DIR,
#             write_outputs=True,
#         )
#         if out is None:
#             return None

#         # Save standard filenames expected by downstream code
#         # (overwrite previous behavior)
#         trial_file = os.path.join(RESULT_DIR, f"results_{scenario}_{basin_id}_trial0000.feather")
#         final_file = os.path.join(RESULT_DIR, f"results_{scenario}_{basin_id}.feather")
#         if os.path.exists(trial_file):
#             os.replace(trial_file, final_file)

#         return out

#     # -------------------------------------------------
#     # UNCALIBRATED MODE: n_trials search
#     # -------------------------------------------------
#     n_trials = int(max(1, n_trials))
#     save_top_k = int(max(1, save_top_k))
#     save_top_k = min(save_top_k, n_trials)

#     # keep top-k by KGE
#     # heap stores (KGE, metrics)
#     top_heap = []

#     for trial in range(n_trials):
#         m = _run_one_trial(
#             basin_id=basin_id,
#             scenario=scenario,
#             idx=idx,
#             P=P,
#             PET=PET,
#             Q_obs=Q_obs,
#             Qsb=Qsb,
#             Evap_df=Evap_df,
#             Qsb_df=Qsb_df,
#             PET_df=PET_df,
#             RootMoist=RootMoist,
#             Slp_df=Slp_df,
#             calibrated_params=calibrated_params,
#             da_cfg=da_cfg,
#             use_calibrated=False,
#             random_bounds=random_bounds,
#             base_seed=random_seed,
#             trial_id=trial,
#             RESULT_DIR=RESULT_DIR,
#             write_outputs=False,
#         )
#         if m is None:
#             continue

#         kge = m["KGE"]
#         if not np.isfinite(kge):
#             continue

#         if len(top_heap) < save_top_k:
#             heapq.heappush(top_heap, (kge, m))
#         else:
#             if kge > top_heap[0][0]:
#                 heapq.heapreplace(top_heap, (kge, m))

#     if len(top_heap) == 0:
#         return None

#     # sort best->worst
#     top_heap.sort(key=lambda x: x[0], reverse=True)

#     # # write outputs only for top_k trials
#     # written_metrics = []
#     # for rank, (kge, m) in enumerate(top_heap, start=1):
#     #     trial_id = int(m["trial"])
#     #     _ = _run_one_trial(
#     #         basin_id=basin_id,
#     #         scenario=scenario,
#     #         idx=idx,
#     #         P=P,
#     #         PET=PET,
#     #         Q_obs=Q_obs,
#     #         Qsb=Qsb,
#     #         Evap_df=Evap_df,
#     #         Qsb_df=Qsb_df,
#     #         PET_df=PET_df,
#     #         RootMoist=RootMoist,
#     #         Slp_df=Slp_df,
#     #         calibrated_params=calibrated_params,
#     #         da_cfg=da_cfg,
#     #         use_calibrated=False,
#     #         random_bounds=random_bounds,
#     #         base_seed=random_seed,
#     #         trial_id=trial_id,
#     #         RESULT_DIR=RESULT_DIR,
#     #         write_outputs=True,
#     #     )
#     #     m2 = dict(m)
#     #     m2["rank"] = rank
#     #     written_metrics.append(m2)
#     # write outputs ONLY for the best trial
#     best_trial_id = int(top_heap[0][1]["trial"])

#     _ = _run_one_trial(
#         basin_id=basin_id,
#         scenario=scenario,
#         idx=idx,
#         P=P,
#         PET=PET,
#         Q_obs=Q_obs,
#         Qsb=Qsb,
#         Evap_df=Evap_df,
#         Qsb_df=Qsb_df,
#         PET_df=PET_df,
#         RootMoist=RootMoist,
#         Slp_df=Slp_df,
#         calibrated_params=calibrated_params,
#         da_cfg=da_cfg,
#         use_calibrated=False,
#         random_bounds=random_bounds,
#         base_seed=random_seed,
#         trial_id=best_trial_id,
#         RESULT_DIR=RESULT_DIR,
#         write_outputs=True,
#     )
#     best_trial_file = os.path.join(
#         RESULT_DIR, f"results_{scenario}_{basin_id}_trial{best_trial_id:04d}.feather"
#     )
#     final_file = os.path.join(RESULT_DIR, f"results_{scenario}_{basin_id}.feather")

#     if os.path.exists(best_trial_file):
#         os.replace(best_trial_file, final_file)



#     # the best trial outputs should also be saved with standard filename
#     best_trial_id = int(top_heap[0][1]["trial"])
#     best_trial_file = os.path.join(RESULT_DIR, f"results_{scenario}_{basin_id}_trial{best_trial_id:04d}.feather")
#     final_file = os.path.join(RESULT_DIR, f"results_{scenario}_{basin_id}.feather")
#     if os.path.exists(best_trial_file):
#         os.replace(best_trial_file, final_file)

#     # return best metrics (rank=1)
#     best_metrics = dict(top_heap[0][1])
#     best_metrics["rank"] = 1
#     best_metrics["n_trials"] = int(n_trials)
#     best_metrics["save_top_k"] = int(save_top_k)

#     return best_metrics


# # ---------------------------------------------------------
# # Run from config
# # ---------------------------------------------------------
# def run_simulations_from_config(cfg: dict):

#     scenario = normalize_scenario_key(cfg["scenario"])
#     scenarios_to_run = ["BASE", "BUDYKO", "BUDYKO_DA"] if scenario == "ALL" else [scenario]

#     paths = cfg["paths"]
#     DATA_DIR = os.path.join(PROJECT_ROOT, paths["data_dir"])

#     BASE_RESULT_DIR = os.path.join(PROJECT_ROOT, paths["result_dir"])
#     os.makedirs(BASE_RESULT_DIR, exist_ok=True)

#     # parameter mode (DO NOT CHANGE)
#     params_cfg = cfg.get("params", {})
#     if "UNCALIBRATED" in params_cfg:
#         params_cfg = params_cfg["UNCALIBRATED"]

#     use_calibrated = bool(params_cfg.get("use_calibrated", False))
#     random_seed = int(params_cfg.get("random_seed", 1234))
#     random_bounds = params_cfg.get("bounds", {})

#     # ✅ NOW THESE ACTUALLY MATTER
#     n_trials = int(params_cfg.get("n_trials", 1))
#     save_top_k = int(params_cfg.get("save_top_k", 1))

#     calibrated_params = {}
#     if use_calibrated:
#         cal_path = os.path.join(PROJECT_ROOT, paths["calibrated_params"])
#         with open(cal_path, "r") as f:
#             calibrated_params = json.load(f)

#     da_cfg = cfg.get("da", {})
#     if "enabled" in da_cfg:
#         da_cfg.pop("enabled")

#     basin_subset = cfg.get("basins", {}).get("subset", None)

#     if basin_subset is None:
#         tmp = load_feather_df("EVap.feather", DATA_DIR)
#         basins = list(tmp.columns)
#     else:
#         basins = list(basin_subset)

#     par_cfg = cfg.get("parallel", {})
#     par_enabled = bool(par_cfg.get("enabled", True))
#     max_workers = int(par_cfg.get("max_workers", -1))
#     if max_workers == -1:
#         max_workers = max(1, cpu_count() - 1)

#     for scenario in scenarios_to_run:

#         scenario_dir = scenario_folder_name(scenario)
#         RESULT_DIR = os.path.join(BASE_RESULT_DIR, scenario_dir)
#         os.makedirs(RESULT_DIR, exist_ok=True)

#         all_metrics = []

#         if par_enabled:
#             with ProcessPoolExecutor(max_workers=max_workers) as executor:
#                 futures = {
#                     executor.submit(
#                         simulate_basin,
#                         b, scenario, DATA_DIR, RESULT_DIR,
#                         calibrated_params, da_cfg,
#                         use_calibrated, random_bounds, random_seed,
#                         n_trials, save_top_k
#                     ): b
#                     for b in basins
#                 }

#                 for f in tqdm(as_completed(futures), total=len(futures), desc=f"Running scenario={scenario}"):
#                     out = f.result()
#                     if out is not None:
#                         all_metrics.append(out)
#         else:
#             for b in tqdm(basins, desc=f"Running scenario={scenario}"):
#                 out = simulate_basin(
#                     b, scenario, DATA_DIR, RESULT_DIR,
#                     calibrated_params, da_cfg,
#                     use_calibrated, random_bounds, random_seed,
#                     n_trials, save_top_k
#                 )
#                 if out is not None:
#                     all_metrics.append(out)

#         pd.DataFrame(all_metrics).to_csv(
#             os.path.join(RESULT_DIR, f"metrics_{scenario}.csv"),
#             index=False
#         )

#         print(f"\n✅ Completed successfully: scenario={scenario}. Results saved to {RESULT_DIR}")


# if __name__ == "__main__":
#     import yaml
#     cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
#     with open(cfg_path, "r") as f:
#         cfg = yaml.safe_load(f)
#     run_simulations_from_config(cfg)























# # scripts/run_simulation_uncalib_params.py

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

# from src.model import ModelParams, two_store_model_step
# from src.budyko import BudykoModelEstimator
# from src.enkf import EnKFConfig, enkf_update_stochastic_scalar, enkf_forecast_step_states
# from src.metrics import calculate_kge, calculate_nse


# # ---------------------------------------------------------
# # scenario folder
# # ---------------------------------------------------------
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
# # Random param generator
# # ---------------------------------------------------------
# def sample_random_params_for_basin(basin_id: str, bounds: dict, seed: int = 1234) -> dict:
#     """
#     Sample random parameter set within bounds.
#     Returned dict uses same keys expected downstream.
#     """
#     basin_seed = (hash(str(basin_id)) + int(seed)) % (2**32 - 1)
#     rng = np.random.default_rng(basin_seed)

#     def uniform(name: str, default=(0.0, 1.0)):
#         lo, hi = bounds.get(name, default)
#         return float(rng.uniform(float(lo), float(hi)))

#     Kperc = uniform("Kperc")
#     Kb    = uniform("Kb")
#     Ke    = uniform("Ke")
#     Cqq   = uniform("Cqq")

#     Smax        = uniform("Smax")
#     Gmax_factor = uniform("Gmax_factor")
#     Gmax        = Smax * Gmax_factor

#     fS0 = uniform("fS0", (0.1, 0.9))
#     fG0 = uniform("fG0", (0.1, 0.9))
#     S_init = fS0 * Smax
#     G_init = fG0 * Gmax

#     return {
#         "Kperc": Kperc,
#         "Kb": Kb,
#         "Ke": Ke,
#         "Cqq": Cqq,
#         "Smax": Smax,
#         "Gmax_factor": Gmax_factor,
#         "S_init": S_init,
#         "G_init": G_init,
#         "fS0": fS0,
#         "fG0": fG0,
#     }


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

#         et_t = float(ET_series[t]) if np.isfinite(ET_series[t]) else None

#         S, G, _, Q, *_ = two_store_model_step(
#             S, G, P_t, PET_t, params_cal,
#             ET_override=et_t
#         )

#         G = np.clip(G, 0.0, Gmax_cal)

#         Q_out[t] = max(Q, 0.0)
#         S_out[t] = S
#         G_out[t] = G

#     return Q_out, S_out, G_out


# # ---------------------------------------------------------
# # DA run (EnKF) -> assimilates ET_model toward ET_obs
# # ---------------------------------------------------------
# def run_budyko_da(
#     P: np.ndarray,
#     PET: np.ndarray,
#     ET_obs: np.ndarray,
#     ET_model: np.ndarray,
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

#     S0_ens = np.clip(S_init + rng.normal(0.0, 0.05 * Smax_cal, size=nens), 0.0, Smax_cal)
#     G0_ens = np.clip(G_init + rng.normal(0.0, 0.05 * Gmax_cal, size=nens), 0.0, Gmax_cal)
#     X = np.vstack([S0_ens, G0_ens])

#     S_ens_hist  = np.full((L, nens), np.nan)
#     G_ens_hist  = np.full((L, nens), np.nan)
#     ET_ens_hist = np.full((L, nens), np.nan)
#     Q_ens_hist  = np.full((L, nens), np.nan)

#     ET_ass_mean = np.full(L, np.nan)
#     Q_ass_mean  = np.full(L, np.nan)

#     for t in range(L):

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

#         ET_ens_hist[t, :] = ET_ens_f
#         Q_ens_hist[t, :]  = Q_ens_f

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

#         X_dummy, ET_ens_a, Q_ens_a = enkf_forecast_step_states(
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

#         X = X_a
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


# # ---------------------------------------------------------
# # Main simulation per basin
# # ✅ Budyko computed basin-by-basin (robust)
# # ---------------------------------------------------------
# def simulate_basin(
#     basin_id,
#     scenario,
#     DATA_DIR,
#     RESULT_DIR,
#     calibrated_params,
#     da_cfg: dict,
#     use_calibrated: bool,
#     random_bounds: dict,
#     random_seed: int,
# ):

#     PET_df      = load_feather_df("PotEvap.feather", DATA_DIR)
#     Rainf_df    = load_feather_df("Rainf.feather", DATA_DIR)
#     Evap_df     = load_feather_df("EVap.feather", DATA_DIR)
#     Q_usgs_df   = load_feather_df("Q_USGS.feather", DATA_DIR)
#     Qsb_df      = load_feather_df("Qsb.feather", DATA_DIR)
#     RootMoist   = load_feather_df("SoilM_0_200cm.feather", DATA_DIR)
#     Slp_df      = load_feather_df("slope.feather", DATA_DIR)

#     common_cols = sorted(
#         set(Evap_df.columns)
#         & set(Qsb_df.columns)
#         & set(PET_df.columns)
#         & set(Rainf_df.columns)
#         & set(RootMoist.columns)
#         & set(Slp_df.columns)
#     )

#     if basin_id not in common_cols:
#         return None

#     # ✅ only enforce calibrated_params existence if use_calibrated
#     if use_calibrated and basin_id not in calibrated_params:
#         return None

#     # restrict all DF to common_cols (required by BudykoEstimator)
#     Evap_df    = Evap_df[common_cols]
#     Qsb_df     = Qsb_df[common_cols]
#     PET_df     = PET_df[common_cols]
#     Rainf_df   = Rainf_df[common_cols]
#     RootMoist  = RootMoist[common_cols]
#     Slp_df     = Slp_df[common_cols]

#     idx = Evap_df.index
#     L = len(idx)

#     # ✅ force numeric arrays (fix isfinite error)
#     PET   = pd.to_numeric(PET_df[basin_id], errors="coerce").to_numpy(dtype=float)
#     P     = pd.to_numeric(Rainf_df[basin_id], errors="coerce").to_numpy(dtype=float)
#     Q_obs = pd.to_numeric(Q_usgs_df.get(basin_id, pd.Series(index=idx)).reindex(idx), errors="coerce").to_numpy(dtype=float)
#     Qsb   = pd.to_numeric(Qsb_df.get(basin_id, pd.Series(index=idx)), errors="coerce").to_numpy(dtype=float)

#     # parameters
#     if use_calibrated:
#         p = calibrated_params[basin_id]
#     else:
#         p = sample_random_params_for_basin(
#             basin_id=basin_id,
#             bounds=random_bounds,
#             seed=random_seed
#         )

#     # Budyko ET
#     omega_true_all = np.full(L, np.nan)
#     omega_MLR_all  = np.full(L, np.nan)
#     ET_B           = np.full(L, np.nan)

#     if scenario in ["BUDYKO", "BUDYKO_DA"]:
#         try:
#             # ✅ Build Ke_df for UNCALIBRATED mode
#             Ke_df = None
#             if not use_calibrated:
#                 Ke_df = pd.DataFrame(index=idx, data={basin_id: float(p["Ke"])})

#             budyko = BudykoModelEstimator(
#                 Evap_df=Evap_df[[basin_id]],
#                 Qsb_monthly=Qsb_df[[basin_id]],
#                 PotEvap_df=PET_df[[basin_id]],
#                 M_basin=M_[[basin_id]],
#                 Slope_basin=Slp_df[[basin_id]],

#                 # ✅ calibrated_params only if calibrated mode
#                 calibrated_params=calibrated_params if use_calibrated else None,

#                 # ✅ Ke_df only if uncalibrated mode
#                 Ke_df=Ke_df,
#             )

#             budyko.estimate_budyko_et()

#             omega_true_all = budyko.omega_true[basin_id].reindex(idx).to_numpy().astype(float).ravel()
#             omega_MLR_all  = budyko.omega_MLR[basin_id].reindex(idx).to_numpy().astype(float).ravel()
#             ET_B           = budyko.ET_B[basin_id].reindex(idx).to_numpy().astype(float).ravel()

#             if np.all(~np.isfinite(ET_B)):
#                 print(f"❌ Budyko failed basin={basin_id}: ET_B is all NaN")
#                 return None

#         except Exception as e:
#             print(f"❌ Budyko failed for basin={basin_id} scenario={scenario}: {e}")
#             return None

#     # model params
#     Smax_cal    = float(p.get("Smax", 50.0))
#     Gmax_factor = float(p.get("Gmax_factor", 4.0))
#     Gmax_cal    = Smax_cal * Gmax_factor

#     S_init = float(p.get("S_init", 0.5 * Smax_cal))
#     G_init = float(p.get("G_init", 0.5 * Gmax_cal))

#     params_cal = ModelParams(
#         Smax=Smax_cal,
#         Kperc=float(p["Kperc"]),
#         Kb=float(p["Kb"]),
#         Ke=float(p["Ke"]),
#         Cqq=float(p["Cqq"]),
#         Sfc_frac=0.30,
#         beta_et=2.0,
#     )

#     ET_ke = PET * float(params_cal.Ke)

#     enkf_hist = None
#     ET_ass_mean = np.full(L, np.nan)
#     Q_ass_mean  = np.full(L, np.nan)

#     # run scenarios
#     if scenario == "BASE":
#         Q_base, S_base, G_base = run_model_deterministic(
#             P=P, PET=PET,
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
#             "G_base": G_base,
#         }).set_index("time")

#     elif scenario == "BUDYKO":
#         Q_budyko, S_budyko, G_budyko = run_model_deterministic(
#             P=P, PET=PET,
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
#             ET_obs=ET_B,
#             ET_model=ET_ke,
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

#     # save
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

#     # metrics
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

#     metrics.update({
#         "Kperc": float(params_cal.Kperc),
#         "Kb": float(params_cal.Kb),
#         "Ke": float(params_cal.Ke),
#         "Cqq": float(params_cal.Cqq),
#         "Smax": float(params_cal.Smax),
#         "S_init": float(S_init),
#         "G_init": float(G_init),
#         "Gmax": float(Gmax_cal),
#     })

#     return metrics


# # ---------------------------------------------------------
# # Run from config
# # ---------------------------------------------------------
# def run_simulations_from_config(cfg: dict):

#     scenario = normalize_scenario_key(cfg["scenario"])
#     scenarios_to_run = ["BASE", "BUDYKO", "BUDYKO_DA"] if scenario == "ALL" else [scenario]

#     paths = cfg["paths"]
#     DATA_DIR = os.path.join(PROJECT_ROOT, paths["data_dir"])

#     BASE_RESULT_DIR = os.path.join(PROJECT_ROOT, paths["result_dir"])
#     os.makedirs(BASE_RESULT_DIR, exist_ok=True)

#     # parameter mode
#     params_cfg = cfg.get("params", {})

#     # ✅ support BOTH structures:
#     # (A) params: {use_calibrated: false, ...}
#     # (B) params: {UNCALIBRATED: {...}, CALIBRATED: {...}}
#     if "UNCALIBRATED" in params_cfg:
#         params_cfg = params_cfg["UNCALIBRATED"]

#     use_calibrated = bool(params_cfg.get("use_calibrated", False))
#     random_seed = int(params_cfg.get("random_seed", 1234))
#     random_bounds = params_cfg.get("bounds", {})

#     calibrated_params = {}
#     if use_calibrated:
#         cal_path = os.path.join(PROJECT_ROOT, paths["calibrated_params"])
#         with open(cal_path, "r") as f:
#             calibrated_params = json.load(f)

#     da_cfg = cfg.get("da", {})
#     if "enabled" in da_cfg:
#         da_cfg.pop("enabled")

#     basin_subset = cfg.get("basins", {}).get("subset", None)

#     # determine basins
#     if basin_subset is None:
#         tmp = load_feather_df("EVap.feather", DATA_DIR)
#         basins = list(tmp.columns)
#     else:
#         basins = list(basin_subset)

#     # parallel
#     par_cfg = cfg.get("parallel", {})
#     par_enabled = bool(par_cfg.get("enabled", True))
#     max_workers = int(par_cfg.get("max_workers", -1))
#     if max_workers == -1:
#         max_workers = max(1, cpu_count() - 1)

#     for scenario in scenarios_to_run:

#         scenario_dir = scenario_folder_name(scenario)
#         RESULT_DIR = os.path.join(BASE_RESULT_DIR, scenario_dir)
#         os.makedirs(RESULT_DIR, exist_ok=True)

#         all_metrics = []

#         if par_enabled:
#             with ProcessPoolExecutor(max_workers=max_workers) as executor:
#                 futures = {
#                     executor.submit(
#                         simulate_basin,
#                         b, scenario, DATA_DIR, RESULT_DIR,
#                         calibrated_params, da_cfg,
#                         use_calibrated, random_bounds, random_seed
#                     ): b
#                     for b in basins
#                 }

#                 for f in tqdm(as_completed(futures), total=len(futures), desc=f"Running scenario={scenario}"):
#                     out = f.result()
#                     if out is not None:
#                         all_metrics.append(out)
#         else:
#             for b in tqdm(basins, desc=f"Running scenario={scenario}"):
#                 out = simulate_basin(
#                     b, scenario, DATA_DIR, RESULT_DIR,
#                     calibrated_params, da_cfg,
#                     use_calibrated, random_bounds, random_seed
#                 )
#                 if out is not None:
#                     all_metrics.append(out)

#         pd.DataFrame(all_metrics).to_csv(
#             os.path.join(RESULT_DIR, f"metrics_{scenario}.csv"),
#             index=False
#         )

#         print(f"\n✅ Completed successfully: scenario={scenario}. Results saved to {RESULT_DIR}")


# if __name__ == "__main__":
#     import yaml
#     cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
#     with open(cfg_path, "r") as f:
#         cfg = yaml.safe_load(f)
#     run_simulations_from_config(cfg)

















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
                
# from src.model import ModelParams, two_store_model_step
# from src.budyko import BudykoModelEstimator
# from src.enkf import EnKFConfig, enkf_update_stochastic_scalar, enkf_forecast_step_states
# from src.metrics import calculate_kge, calculate_nse


# # ---------------------------------------------------------
# # scenario folder
# # ---------------------------------------------------------
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
# # Random param generator
# # ---------------------------------------------------------
# def sample_random_params_for_basin(basin_id: str, bounds: dict, seed: int = 1234) -> dict:
#     """
#     Sample random parameter set within bounds.
#     Returned dict uses same keys expected downstream.
#     """
#     # basin-specific reproducible RNG
#     basin_seed = (hash(str(basin_id)) + int(seed)) % (2**32 - 1)
#     rng = np.random.default_rng(basin_seed)

#     def uniform(name: str, default=(0.0, 1.0)):
#         lo, hi = bounds.get(name, default)
#         return float(rng.uniform(float(lo), float(hi)))

#     # ---- model params ----
#     Kperc = uniform("Kperc")
#     Kb    = uniform("Kb")
#     Ke    = uniform("Ke")
#     Cqq   = uniform("Cqq")

#     # ---- storage params ----
#     Smax         = uniform("Smax")
#     Gmax_factor  = uniform("Gmax_factor")
#     Gmax         = Smax * Gmax_factor

#     # initial conditions as fractions
#     fS0 = uniform("fS0", (0.1, 0.9))
#     fG0 = uniform("fG0", (0.1, 0.9))
#     S_init = fS0 * Smax
#     G_init = fG0 * Gmax

#     return {
#         "Kperc": Kperc,
#         "Kb": Kb,
#         "Ke": Ke,
#         "Cqq": Cqq,
#         "Smax": Smax,
#         "Gmax_factor": Gmax_factor,
#         "S_init": S_init,
#         "G_init": G_init,
#         "fS0": fS0,
#         "fG0": fG0,
#     }


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
#             ET_override=float(ET_series[t]) if np.isfinite(ET_series[t]) else None
#         )

#         G = np.clip(G, 0.0, Gmax_cal)

#         Q_out[t] = max(Q, 0.0)
#         S_out[t] = S
#         G_out[t] = G

#     return Q_out, S_out, G_out


# # ---------------------------------------------------------
# # DA run (EnKF) -> assimilates ET_model toward ET_obs
# # ---------------------------------------------------------
# def run_budyko_da(
#     P: np.ndarray,
#     PET: np.ndarray,
#     ET_obs: np.ndarray,
#     ET_model: np.ndarray,
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
#     X = np.vstack([S0_ens, G0_ens])  # (2, nens)

#     # histories
#     S_ens_hist  = np.full((L, nens), np.nan)
#     G_ens_hist  = np.full((L, nens), np.nan)
#     ET_ens_hist = np.full((L, nens), np.nan)
#     Q_ens_hist  = np.full((L, nens), np.nan)

#     ET_ass_mean = np.full(L, np.nan)
#     Q_ass_mean  = np.full(L, np.nan)

#     for t in range(L):

#         ET_override_t = float(ET_model[t]) if np.isfinite(ET_model[t]) else None

#         # forecast
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

#         ET_ens_hist[t, :] = ET_ens_f
#         Q_ens_hist[t, :]  = Q_ens_f

#         # analysis
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

#         # posterior evaluation
#         X_dummy, ET_ens_a, Q_ens_a = enkf_forecast_step_states(
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

#         X = X_a
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


# # ---------------------------------------------------------
# # Main simulation per basin
# # ---------------------------------------------------------
# def simulate_basin(
#     basin_id,
#     scenario,
#     DATA_DIR,
#     RESULT_DIR,
#     calibrated_params,
#     da_cfg: dict,
#     use_calibrated: bool,
#     random_bounds: dict,
#     random_seed: int,
# ):

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
#         & set(Rainf_df.columns)      
#     )

#     # basin must exist in the data
#     if basin_id not in common_cols:
#         return None

#     # if using calibrated, basin must also exist in params JSON
#     if use_calibrated and basin_id not in calibrated_params:
#         return None

#     Evap_df     = Evap_df[common_cols]
#     Qsb_df      = Qsb_df[common_cols]
#     PET_df      = PET_df[common_cols]
#     M_df        = M_df[common_cols]
#     RootMoist   = RootMoist[common_cols]
#     Slp_df      = Slp_df[common_cols]        # ✅ IMPORTANT
#     Rainf_df    = Rainf_df[common_cols]      # ✅ IMPORTANT


#     idx = Evap_df.index
#     L = len(idx)

#     PET = PET_df[basin_id].values
#     P = Rainf_df.get(basin_id, pd.Series(index=idx)).values
#     Q_obs = Q_usgs_df.get(basin_id, pd.Series(index=idx)).reindex(idx).values
#     Q_nldas = Q_nldas_df.get(basin_id, pd.Series(index=idx)).values
#     Qsb = Qsb_df.get(basin_id, pd.Series(index=idx)).values
#     Slp = Slp_df.get(basin_id, pd.Series(index=idx)).values

#     # -------------------------------------------------
#     # get parameter set
#     # -------------------------------------------------
#     if use_calibrated:
#         p = calibrated_params[basin_id]
#     else:
#         p = sample_random_params_for_basin(
#             basin_id=basin_id,
#             bounds=random_bounds,
#             seed=random_seed
#         )

#     # -------------------------------------------------
#     # Budyko ET
#     # -------------------------------------------------

#     omega_true_all = np.full(L, np.nan)
#     omega_MLR_all  = np.full(L, np.nan)
#     ET_B           = np.full(L, np.nan)

#     if scenario in ["BUDYKO", "BUDYKO_DA"]:
#         try:
#             budyko = BudykoModelEstimator(
#                 Evap_df=Evap_df[common_cols],
#                 Qsb_monthly=Qsb_df[common_cols],
#                 PotEvap_df=PET_df[common_cols],
#                 M_basin=RootMoist[common_cols],
#                 Slope_basin=Slp_df[common_cols],
#                 calibrated_params=calibrated_params if use_calibrated else {},
#             )

#             budyko.estimate_budyko_et()

#             omega_true_all = budyko.omega_true[basin_id].reindex(idx).to_numpy().ravel()
#             omega_MLR_all  = budyko.omega_MLR[basin_id].reindex(idx).to_numpy().ravel()
#             ET_B           = budyko.ET_B[basin_id].reindex(idx).to_numpy().ravel()

#         except Exception as e:
#             print(f"❌ Budyko failed for basin={basin_id} scenario={scenario}: {e}")
#             return None



#     # -------------------------------------------------
#     # Model params
#     # -------------------------------------------------
#     Smax_cal = float(p.get("Smax", 50.0))
#     Gmax_factor = float(p.get("Gmax_factor", 4.0))
#     Gmax_cal = Smax_cal * Gmax_factor

#     S_init = float(p.get("S_init", 0.5 * Smax_cal))
#     G_init = float(p.get("G_init", 0.5 * Gmax_cal))

#     params_cal = ModelParams(
#         Smax=Smax_cal,
#         Kperc=float(p["Kperc"]),
#         Kb=float(p["Kb"]),
#         Ke=float(p["Ke"]),
#         Cqq=float(p["Cqq"]),
#         Sfc_frac=0.30,
#         beta_et=2.0,
#     )

#     ET_ke = PET * params_cal.Ke

#     enkf_hist = None
#     ET_ass_mean = np.full(L, np.nan)
#     Q_ass_mean = np.full(L, np.nan)

#     # -------------------------------------------------
#     # Run scenarios
#     # -------------------------------------------------
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
#             "G_base": G_base,
#         }).set_index("time")

#     elif scenario == "BUDYKO":
#         Q_budyko, S_budyko, G_budyko = run_model_deterministic(
#             P=P, PET=PET,
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
#             ET_obs=ET_B,
#             ET_model=ET_ke,
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

#     # -------------------------------------------------
#     # Save
#     # -------------------------------------------------
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
    
#     # -------------------------------------------------
#     # Metrics
#     # -------------------------------------------------
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

#     # add params used (important for debugging!)
#     metrics.update({
#         "Kperc": float(params_cal.Kperc),
#         "Kb": float(params_cal.Kb),
#         "Ke": float(params_cal.Ke),
#         "Cqq": float(params_cal.Cqq),
#         "Smax": float(params_cal.Smax),
#         "S_init": float(S_init),
#         "G_init": float(G_init),
#         "Gmax": float(Gmax_cal),
#     })

#     return metrics


# # ---------------------------------------------------------
# # Run from config
# # ---------------------------------------------------------
# def run_simulations_from_config(cfg: dict):

#     scenario = normalize_scenario_key(cfg["scenario"])
#     scenarios_to_run = ["BASE", "BUDYKO", "BUDYKO_DA"] if scenario == "ALL" else [scenario]

#     paths = cfg["paths"]
#     DATA_DIR = os.path.join(PROJECT_ROOT, paths["data_dir"])

#     BASE_RESULT_DIR = os.path.join(PROJECT_ROOT, paths["result_dir"])
#     os.makedirs(BASE_RESULT_DIR, exist_ok=True)

#     # -------------------------------------------------
#     # parameter mode
#     # -------------------------------------------------
#     params_cfg = cfg.get("params", {})
#     use_calibrated = bool(params_cfg.get("use_calibrated", True))
#     random_seed = int(params_cfg.get("random_seed", 1234))

#     random_bounds = params_cfg.get("bounds", {})

#     # load calibrated params ONLY if needed
#     calibrated_params = {}
#     if use_calibrated:
#         cal_path = os.path.join(PROJECT_ROOT, paths["calibrated_params"])
#         with open(cal_path, "r") as f:
#             calibrated_params = json.load(f)

#     # DA config
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
#                         simulate_basin,
#                         b, scenario, DATA_DIR, RESULT_DIR,
#                         calibrated_params, da_cfg,
#                         use_calibrated, random_bounds, random_seed
#                     ): b
#                     for b in basins
#                 }

#                 for f in tqdm(as_completed(futures), total=len(futures), desc=f"Running scenario={scenario}"):
#                     out = f.result()
#                     if out is not None:
#                         all_metrics.append(out)
#         else:
#             for b in tqdm(basins, desc=f"Running scenario={scenario}"):
#                 out = simulate_basin(
#                     b, scenario, DATA_DIR, RESULT_DIR,
#                     calibrated_params, da_cfg,
#                     use_calibrated, random_bounds, random_seed
#                 )
#                 if out is not None:
#                     all_metrics.append(out)

#         pd.DataFrame(all_metrics).to_csv(
#             os.path.join(RESULT_DIR, f"metrics_{scenario}.csv"),
#             index=False
#         )

#         print(f"\n✅ Completed successfully: scenario={scenario}. Results saved to {RESULT_DIR}")
