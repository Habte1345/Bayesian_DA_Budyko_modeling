# calibration.py
import os
import sys

# Ensure project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import warnings
import numpy as np
import pandas as pd
import spotpy

from typing import Dict, Any, Tuple, Optional
from multiprocessing import cpu_count, get_context

from src.model import ModelParams, two_store_model_step
from src.metrics import calculate_nse  # calculate_kge not used anymore (we use kge_2012)

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in true_divide"
)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Repetitions for SCE-UA (adaptive strategy)
REPS_FAST = 10000
REPS_SLOW = 10000

# Parameter bounds
Smax_min, Smax_max = 0.02, 500
fS0_min, fS0_max = 0.01, 0.99
fG0_min, fG0_max = 0.00, 0.99
Gmaxfac_min, Gmaxfac_max = 1.0, 100.0


# --------------------------------------------------------------------------------------
# Utility: Ensure JSON-safe serialization
# --------------------------------------------------------------------------------------
def clean_dict_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for k, v in data.items():
        if isinstance(v, dict):
            cleaned[k] = clean_dict_for_json(v)
        elif isinstance(v, (np.float32, np.float64)):
            cleaned[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            cleaned[k] = int(v)
        else:
            cleaned[k] = v
    return cleaned


# --------------------------------------------------------------------------------------
# KGE (2012) — robust implementation
# --------------------------------------------------------------------------------------
def kge_2012(evaluation, simulation, eps=1e-12):
    evaluation = np.asarray(evaluation, dtype=np.float64)
    simulation = np.asarray(simulation, dtype=np.float64)

    obs_mean = np.mean(evaluation)
    sim_mean = np.mean(simulation)

    obs_std = np.std(evaluation)
    sim_std = np.std(simulation)

    # correlation
    r_num = np.sum((simulation - sim_mean) * (evaluation - obs_mean))
    r_den = np.sqrt(
        np.sum((simulation - sim_mean) ** 2) *
        np.sum((evaluation - obs_mean) ** 2)
    )
    r = r_num / (r_den + eps)

    # bias ratio
    beta = (sim_mean + eps) / (obs_mean + eps)

    # variability ratio (CV ratio)
    gamma = (sim_std / (sim_mean + eps)) / ((obs_std / (obs_mean + eps)) + eps)

    return 1.0 - np.sqrt((r - 1.0) ** 2 + (gamma - 1.0) ** 2 + (beta - 1.0) ** 2)


# --------------------------------------------------------------------------------------
# Forward model
# --------------------------------------------------------------------------------------
def run_forward_model(
    P_data: np.ndarray,
    PET_data: np.ndarray,
    Q_obs: np.ndarray,
    initial_state: tuple,
    params: ModelParams,
    gmax_factor: float,
) -> tuple:
    """
    Run monthly forward simulation.
    Returns simulated + observed series with spin-up removed and valid-mask applied.
    """
    nmonths = len(P_data)
    S_curr, G_curr = initial_state
    Q_sim = np.zeros(nmonths)

    S_cap = max(params.Smax, 1e-9)
    G_cap = max(gmax_factor * params.Smax, 1e-9)

    for t in range(nmonths):
        P_t = np.nan_to_num(max(P_data[t], 0.0))
        PET_t = np.nan_to_num(max(PET_data[t], 0.0))

        try:
            S_next, G_next, _, Q_t, *_ = two_store_model_step(
                S_curr, G_curr, P_t, PET_t, params
            )
        except Exception:
            S_next, G_next, Q_t = S_curr, G_curr, 0.0

        S_curr = np.clip(S_next, 0.0, S_cap)
        G_curr = np.clip(G_next, 0.0, G_cap)
        Q_sim[t] = max(Q_t, 0.0)

    spinup = max(20, min(60, nmonths // 8))

    Q_sim_clean = Q_sim[spinup:]
    Q_obs_clean = Q_obs[spinup:]

    # ✅ consistent mask: based ONLY on observed Q (already improved)
    mask = np.isfinite(Q_obs_clean)

    if mask.sum() < 12:
        return np.zeros(1), np.zeros(1)

    return Q_sim_clean[mask], Q_obs_clean[mask]


# --------------------------------------------------------------------------------------
# SpotPY Model Class
# --------------------------------------------------------------------------------------
class TwoStoreModel_SCE:
    def __init__(self, P_data, PET_data, Q_obs, target_basin):
        self.P_data = P_data
        self.PET_data = PET_data
        self.Q_obs = Q_obs
        self.target_basin = target_basin

    def parameters(self):
        return spotpy.parameter.generate([
            spotpy.parameter.Uniform(0.01, 1.0, name="Kperc"),
            spotpy.parameter.Uniform(0.01, 1.0, name="Kb"),
            spotpy.parameter.Uniform(0.01, 1.0, name="Ke"),
            spotpy.parameter.Uniform(0.01, 10.0, name="Cqq"),
            spotpy.parameter.Uniform(Smax_min, Smax_max, name="Smax"),
            spotpy.parameter.Uniform(fS0_min, fS0_max, name="fS0"),
            spotpy.parameter.Uniform(fG0_min, fG0_max, name="fG0"),
            spotpy.parameter.Uniform(Gmaxfac_min, Gmaxfac_max, name="Gmax_factor"),
        ])

    @staticmethod
    def _mk_params(d: Dict[str, float]) -> ModelParams:
        return ModelParams(
            Smax=d["Smax"],
            Kperc=d["Kperc"],
            Kb=d["Kb"],
            Ke=d["Ke"],
            Cqq=d["Cqq"],
        )

    def simulation(self, params: Dict[str, float]):
        S0 = params["fS0"] * params["Smax"]
        G0 = params["fG0"] * params["Smax"]
        mp = self._mk_params(params)

        Q_sim, _ = run_forward_model(
            self.P_data, self.PET_data, self.Q_obs,
            (S0, G0), mp, params["Gmax_factor"]
        )

        # Keep behavior as you had it: if something goes wrong, return mean
        return Q_sim.tolist() if len(Q_sim) > 0 else [np.nanmean(self.Q_obs)]

    def evaluation(self):
        Q = np.array(self.Q_obs, dtype=float)

        nmonths = len(Q)
        spinup = max(20, min(60, nmonths // 8))

        Q_clean = Q[spinup:]
        mask = np.isfinite(Q_clean)

        if mask.sum() < 12:
            return [np.nanmean(Q)]

        return Q_clean[mask].tolist()

    # -----------------------------------------------------
    # Objective: maximize KGE-2012 on raw Q (no mixing)
    # -----------------------------------------------------
    def objectivefunction(self, evaluation, simulation):
        evaluation = np.asarray(evaluation, dtype=np.float64)
        simulation = np.asarray(simulation, dtype=np.float64)

        # ✅ consistency checks (evaluation AND simulation)
        if len(evaluation) < 12 or len(simulation) != len(evaluation):
            return 9999.0
        if not (np.all(np.isfinite(evaluation)) and np.all(np.isfinite(simulation))):
            return 9999.0

        eps = 1e-6
        qobs = np.clip(evaluation, eps, None)
        qsim = np.clip(simulation, eps, None)

        kge = kge_2012(qobs, qsim)

        if not np.isfinite(kge) or kge < -1.0:
            return 9999.0

        return -kge  # SpotPY minimizes, so maximize KGE


# --------------------------------------------------------------------------------------
# Basin Calibration Worker
# --------------------------------------------------------------------------------------
def worker_calibrate_basin(
    target_basin: str,
    P_data: np.ndarray,
    PET_data: np.ndarray,
    Q_usgs: np.ndarray,
    reps_fast: int,
    reps_slow: int,
) -> Tuple[str, Optional[Dict[str, Any]]]:

    print(f"\n---> Calibrating Basin: {target_basin} (PID: {os.getpid()})...")

    # NOTE: This check is very strict; keeping your logic but it's a common skip reason.
    # If Q has any NaN, it will skip. If you want, change to "if np.all(~np.isfinite(Q_usgs))".
    if not np.all(np.isfinite(Q_usgs)) or np.all(np.isnan(P_data)):
        print(f" > Basin {target_basin}: invalid data. Skipping.")
        return target_basin, None

    model = TwoStoreModel_SCE(P_data, PET_data, Q_usgs, target_basin)
    db_path = os.path.join(PROJECT_ROOT, "SCE_cal_params", f"sceua_{target_basin}")

    def _run(reps):
        sampler = spotpy.algorithms.sceua(
            model, dbname=db_path, dbformat="csv", save_sim=False
        )
        sampler.sample(repetitions=reps, ngs=70, kstop=30, peps=1e-4, pcento=1e-4)
        res = sampler.getdata()

        if res is None or len(res) == 0:
            return None, None

        idx, _ = spotpy.analyser.get_minlikeindex(res)
        return res, idx

    try:
        res, idx = _run(reps_fast)
        if res is None:
            return target_basin, None

        like = float(res["like1"][idx][0])
        if like > -0.60:
            res, idx = _run(reps_slow)
            if res is None:
                return target_basin, None

        best = {
            k: float(res[k][idx][0])
            for k in res.dtype.names
            if k.startswith(("par", "like"))
        }

        par = {
            "Kperc": best["parKperc"],
            "Kb": best["parKb"],
            "Ke": best["parKe"],
            "Cqq": best["parCqq"],
            "Smax": best["parSmax"],
            "fS0": best["parfS0"],
            "fG0": best["parfG0"],
            "Gmax_factor": best["parGmax_factor"],
        }

        S0 = par["fS0"] * par["Smax"]
        G0 = par["fG0"] * par["Smax"]

        mp = ModelParams(
            Smax=par["Smax"],
            Kperc=par["Kperc"],
            Kb=par["Kb"],
            Ke=par["Ke"],
            Cqq=par["Cqq"],
        )

        qsim, qobs = run_forward_model(
            P_data, PET_data, Q_usgs, (S0, G0), mp, par["Gmax_factor"]
        )

        eps = 1e-6
        qobs_c = np.clip(qobs, eps, None)
        qsim_c = np.clip(qsim, eps, None)

        # ✅ consistent metrics: KGE-2012 + NSE (reported only)
        KGE = kge_2012(qobs_c, qsim_c)
        NSE = calculate_nse(qobs_c, qsim_c)

        output = {
            **par,
            "S_init": S0,
            "G_init": G0,
            "KGE": float(KGE),
            "NSE": float(NSE),
        }

        return target_basin, clean_dict_for_json(output)

    except Exception as e:
        print(f"❌ Calibration failed for {target_basin}: {e}")
        return target_basin, None

    finally:
        csv_path = db_path + ".csv"
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
            except Exception:
                pass


# --------------------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("🔄 Loading and aligning input data...")

    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

    Rainf_df = pd.read_feather(os.path.join(DATA_DIR, "Rainf.feather")).set_index("time")
    PotEvap_df = pd.read_feather(os.path.join(DATA_DIR, "PotEvap.feather")).set_index("time")
    Q_usgs_df = pd.read_feather(os.path.join(DATA_DIR, "Q_USGS.feather")).set_index("time")

    common = list(set(Rainf_df.columns) & set(PotEvap_df.columns) & set(Q_usgs_df.columns))

    Rainf_df = Rainf_df[common]
    PotEvap_df = PotEvap_df[common]
    Q_usgs_df = Q_usgs_df[common]

    aligned = Rainf_df.index.intersection(PotEvap_df.index).intersection(Q_usgs_df.index)
    Rainf_df = Rainf_df.loc[aligned]
    PotEvap_df = PotEvap_df.loc[aligned]
    Q_usgs_df = Q_usgs_df.loc[aligned]

    TARGET_BASINS = common
    NUM_CORES = max(1, cpu_count() - 1)
    tasks = []

    for basin in TARGET_BASINS:
        try:
            tasks.append((
                basin,
                Rainf_df[basin].values.astype(float),
                PotEvap_df[basin].values.astype(float),
                Q_usgs_df[basin].values.astype(float),
                REPS_FAST,
                REPS_SLOW,
            ))
        except KeyError:
            print(f"⚠️ Skipping basin {basin} due to missing aligned data.")
            continue

    results = {}
    os.makedirs(os.path.join(PROJECT_ROOT, "SCE_cal_params"), exist_ok=True)

    with get_context("spawn").Pool(NUM_CORES) as pool:
        for basin, out in pool.starmap(worker_calibrate_basin, tasks):
            if out:
                results[basin] = out

    output_json = os.path.join(PROJECT_ROOT, "SCE_cal_params", "final_calibrated_params.json")
    with open(output_json, "w") as f:
        json.dump(clean_dict_for_json(results), f, indent=2)

    # -----------------------------------------------------
    # Save summary table
    # -----------------------------------------------------
    result_rows = [
        {"Basin": k, **v} for k, v in results.items() if v.get("KGE") is not None
    ]

    if result_rows:
        df = pd.DataFrame(result_rows).round(3).sort_values("KGE", ascending=False)
        csv_summary = os.path.join(PROJECT_ROOT, "SCE_cal_params", "final_calibrated_params_with_KGE.csv")

        df.to_csv(csv_summary, index=False)

        print("\n" + "=" * 80)
        print("✅ FINAL CALIBRATED PARAMETERS WITH KGE\n")
        print(df.to_markdown(index=False))
        print(f"\n💾 JSON Saved: {output_json}")
        print(f"📄 Summary Saved: {csv_summary}")

        total = len(TARGET_BASINS)
        good = int((df["KGE"] > 0.5).sum())
        print(f"\n🏆 SUCCESS: {good}/{total} basins with KGE > 0.5")
    else:
        print("No successful calibrations to summarize.")




















# # calibration.py
# import os
# import sys

# # Ensure project root path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import json
# import warnings
# import numpy as np
# import pandas as pd
# import spotpy

# from typing import Dict, Any, Tuple, Optional
# from multiprocessing import cpu_count, get_context

# from src.model import ModelParams, two_store_model_step
# from src.metrics import calculate_nse  # calculate_kge not used anymore (we use kge_2012)

# warnings.filterwarnings(
#     "ignore",
#     category=RuntimeWarning,
#     message="invalid value encountered in true_divide"
# )

# # --------------------------------------------------------------------------------------
# # Configuration
# # --------------------------------------------------------------------------------------
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# # Repetitions for SCE-UA (adaptive strategy)
# REPS_FAST = 1000
# REPS_SLOW = 2000

# # Parameter bounds
# Smax_min, Smax_max = 0.02, 1000
# fS0_min, fS0_max = 0.01, 0.99
# fG0_min, fG0_max = 0.00, 0.99
# Gmaxfac_min, Gmaxfac_max = 1.0, 500.0


# # --------------------------------------------------------------------------------------
# # Utility: Ensure JSON-safe serialization
# # --------------------------------------------------------------------------------------
# def clean_dict_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
#     cleaned = {}
#     for k, v in data.items():
#         if isinstance(v, dict):
#             cleaned[k] = clean_dict_for_json(v)
#         elif isinstance(v, (np.float32, np.float64)):
#             cleaned[k] = float(v)
#         elif isinstance(v, (np.int32, np.int64)):
#             cleaned[k] = int(v)
#         else:
#             cleaned[k] = v
#     return cleaned


# # --------------------------------------------------------------------------------------
# # KGE (2012) — robust implementation
# # --------------------------------------------------------------------------------------
# def kge_2012(evaluation, simulation, eps=1e-12):
#     evaluation = np.asarray(evaluation, dtype=np.float64)
#     simulation = np.asarray(simulation, dtype=np.float64)

#     obs_mean = np.mean(evaluation)
#     sim_mean = np.mean(simulation)

#     obs_std = np.std(evaluation)
#     sim_std = np.std(simulation)

#     # correlation
#     r_num = np.sum((simulation - sim_mean) * (evaluation - obs_mean))
#     r_den = np.sqrt(
#         np.sum((simulation - sim_mean) ** 2) *
#         np.sum((evaluation - obs_mean) ** 2)
#     )
#     r = r_num / (r_den + eps)

#     # bias ratio
#     beta = (sim_mean + eps) / (obs_mean + eps)

#     # variability ratio (CV ratio)
#     gamma = (sim_std / (sim_mean + eps)) / ((obs_std / (obs_mean + eps)) + eps)

#     return 1.0 - np.sqrt((r - 1.0) ** 2 + (gamma - 1.0) ** 2 + (beta - 1.0) ** 2)


# # --------------------------------------------------------------------------------------
# # Forward model
# # --------------------------------------------------------------------------------------
# def run_forward_model(
#     P_data: np.ndarray,
#     PET_data: np.ndarray,
#     Q_obs: np.ndarray,
#     initial_state: tuple,
#     params: ModelParams,
#     gmax_factor: float,
# ) -> tuple:
#     """
#     Run monthly forward simulation.
#     Returns simulated + observed series with spin-up removed and valid-mask applied.
#     """
#     nmonths = len(P_data)
#     S_curr, G_curr = initial_state
#     Q_sim = np.zeros(nmonths)

#     S_cap = max(params.Smax, 1e-9)
#     G_cap = max(gmax_factor * params.Smax, 1e-9)

#     for t in range(nmonths):
#         P_t = np.nan_to_num(max(P_data[t], 0.0))
#         PET_t = np.nan_to_num(max(PET_data[t], 0.0))

#         try:
#             S_next, G_next, _, Q_t, *_ = two_store_model_step(
#                 S_curr, G_curr, P_t, PET_t, params
#             )
#         except Exception:
#             S_next, G_next, Q_t = S_curr, G_curr, 0.0

#         S_curr = np.clip(S_next, 0.0, S_cap)
#         G_curr = np.clip(G_next, 0.0, G_cap)
#         Q_sim[t] = max(Q_t, 0.0)

#     spinup = max(20, min(60, nmonths // 8))

#     Q_sim_clean = Q_sim[spinup:]
#     Q_obs_clean = Q_obs[spinup:]

#     # ✅ consistent mask: based ONLY on observed Q (already improved)
#     mask = np.isfinite(Q_obs_clean)

#     if mask.sum() < 12:
#         return np.zeros(1), np.zeros(1)

#     return Q_sim_clean[mask], Q_obs_clean[mask]


# # --------------------------------------------------------------------------------------
# # SpotPY Model Class
# # --------------------------------------------------------------------------------------
# class TwoStoreModel_SCE:
#     def __init__(self, P_data, PET_data, Q_obs, target_basin):
#         self.P_data = P_data
#         self.PET_data = PET_data
#         self.Q_obs = Q_obs
#         self.target_basin = target_basin

#     def parameters(self):
#         return spotpy.parameter.generate([
#             spotpy.parameter.Uniform(0.1, 0.99, name="Kperc"),
#             spotpy.parameter.Uniform(0.1, 0.99, name="Kb"),
#             spotpy.parameter.Uniform(0.3, 1.0, name="Ke"),
#             spotpy.parameter.Uniform(0.1, 1.0, name="Cqq"),
#             spotpy.parameter.Uniform(Smax_min, Smax_max, name="Smax"),
#             spotpy.parameter.Uniform(fS0_min, fS0_max, name="fS0"),
#             spotpy.parameter.Uniform(fG0_min, fG0_max, name="fG0"),
#             spotpy.parameter.Uniform(Gmaxfac_min, Gmaxfac_max, name="Gmax_factor"),
#         ])

#     @staticmethod
#     def _mk_params(d: Dict[str, float]) -> ModelParams:
#         return ModelParams(
#             Smax=d["Smax"],
#             Kperc=d["Kperc"],
#             Kb=d["Kb"],
#             Ke=d["Ke"],
#             Cqq=d["Cqq"],
#         )

#     def simulation(self, params: Dict[str, float]):
#         S0 = params["fS0"] * params["Smax"]
#         G0 = params["fG0"] * params["Smax"]
#         mp = self._mk_params(params)

#         Q_sim, _ = run_forward_model(
#             self.P_data, self.PET_data, self.Q_obs,
#             (S0, G0), mp, params["Gmax_factor"]
#         )

#         # Keep behavior as you had it: if something goes wrong, return mean
#         return Q_sim.tolist() if len(Q_sim) > 0 else [np.nanmean(self.Q_obs)]

#     def evaluation(self):
#         Q = np.array(self.Q_obs, dtype=float)

#         nmonths = len(Q)
#         spinup = max(20, min(60, nmonths // 8))

#         Q_clean = Q[spinup:]
#         mask = np.isfinite(Q_clean)

#         if mask.sum() < 12:
#             return [np.nanmean(Q)]

#         return Q_clean[mask].tolist()

#     # -----------------------------------------------------
#     # Objective: maximize KGE-2012 on raw Q (no mixing)
#     # -----------------------------------------------------
#     def objectivefunction(self, evaluation, simulation):
#         evaluation = np.asarray(evaluation, dtype=np.float64)
#         simulation = np.asarray(simulation, dtype=np.float64)

#         # ✅ consistency checks (evaluation AND simulation)
#         if len(evaluation) < 12 or len(simulation) != len(evaluation):
#             return 9999.0
#         if not (np.all(np.isfinite(evaluation)) and np.all(np.isfinite(simulation))):
#             return 9999.0

#         eps = 1e-6
#         qobs = np.clip(evaluation, eps, None)
#         qsim = np.clip(simulation, eps, None)

#         kge = kge_2012(qobs, qsim)

#         if not np.isfinite(kge) or kge < -1.0:
#             return 9999.0

#         return -kge  # SpotPY minimizes, so maximize KGE


# # --------------------------------------------------------------------------------------
# # Basin Calibration Worker
# # --------------------------------------------------------------------------------------
# def worker_calibrate_basin(
#     target_basin: str,
#     P_data: np.ndarray,
#     PET_data: np.ndarray,
#     Q_usgs: np.ndarray,
#     reps_fast: int,
#     reps_slow: int,
# ) -> Tuple[str, Optional[Dict[str, Any]]]:

#     print(f"\n---> Calibrating Basin: {target_basin} (PID: {os.getpid()})...")
#     if not np.all(np.isfinite(Q_usgs)) or np.all(np.isnan(P_data)):
#         print(f" > Basin {target_basin}: invalid data. Skipping.")
#         return target_basin, None

#     model = TwoStoreModel_SCE(P_data, PET_data, Q_usgs, target_basin)
#     db_path = os.path.join(PROJECT_ROOT, "SCE_cal_params", f"sceua_{target_basin}")

#     def _run(reps):
#         sampler = spotpy.algorithms.sceua(
#             model, dbname=db_path, dbformat="csv", save_sim=False
#         )
#         sampler.sample(repetitions=reps, ngs=70, kstop=30, peps=1e-4, pcento=1e-4)
#         res = sampler.getdata()

#         if res is None or len(res) == 0:
#             return None, None

#         idx, _ = spotpy.analyser.get_minlikeindex(res)
#         return res, idx

#     try:
#         res, idx = _run(reps_fast)
#         if res is None:
#             return target_basin, None

#         like = float(res["like1"][idx][0])
#         if like > -0.60:
#             res, idx = _run(reps_slow)
#             if res is None:
#                 return target_basin, None

#         best = {
#             k: float(res[k][idx][0])
#             for k in res.dtype.names
#             if k.startswith(("par", "like"))
#         }

#         par = {
#             "Kperc": best["parKperc"],
#             "Kb": best["parKb"],
#             "Ke": best["parKe"],
#             "Cqq": best["parCqq"],
#             "Smax": best["parSmax"],
#             "fS0": best["parfS0"],
#             "fG0": best["parfG0"],
#             "Gmax_factor": best["parGmax_factor"],
#         }

#         S0 = par["fS0"] * par["Smax"]
#         G0 = par["fG0"] * par["Smax"]

#         mp = ModelParams(
#             Smax=par["Smax"],
#             Kperc=par["Kperc"],
#             Kb=par["Kb"],
#             Ke=par["Ke"],
#             Cqq=par["Cqq"],
#         )

#         qsim, qobs = run_forward_model(
#             P_data, PET_data, Q_usgs, (S0, G0), mp, par["Gmax_factor"]
#         )

#         eps = 1e-6
#         qobs_c = np.clip(qobs, eps, None)
#         qsim_c = np.clip(qsim, eps, None)


#         KGE = kge_2012(qobs_c, qsim_c)
#         NSE = calculate_nse(qobs_c, qsim_c)

#         output = {
#             **par,
#             "S_init": S0,
#             "G_init": G0,
#             "KGE": float(KGE),
#             "NSE": float(NSE),
#         }

#         return target_basin, clean_dict_for_json(output)

#     except Exception as e:
#         print(f"❌ Calibration failed for {target_basin}: {e}")
#         return target_basin, None

#     finally:
#         csv_path = db_path + ".csv"
#         if os.path.exists(csv_path):
#             try:
#                 os.remove(csv_path)
#             except Exception:
#                 pass


# # --------------------------------------------------------------------------------------
# # Main Execution
# # --------------------------------------------------------------------------------------
# if __name__ == "__main__":
#     print("🔄 Loading and aligning input data...")

#     DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

#     Rainf_df = pd.read_feather(os.path.join(DATA_DIR, "Rainf.feather")).set_index("time")
#     PotEvap_df = pd.read_feather(os.path.join(DATA_DIR, "PotEvap.feather")).set_index("time")
#     Q_usgs_df = pd.read_feather(os.path.join(DATA_DIR, "Q_USGS.feather")).set_index("time")

#     common = list(set(Rainf_df.columns) & set(PotEvap_df.columns) & set(Q_usgs_df.columns))

#     Rainf_df = Rainf_df[common]
#     PotEvap_df = PotEvap_df[common]
#     Q_usgs_df = Q_usgs_df[common]

#     aligned = Rainf_df.index.intersection(PotEvap_df.index).intersection(Q_usgs_df.index)
#     Rainf_df = Rainf_df.loc[aligned]
#     PotEvap_df = PotEvap_df.loc[aligned]
#     Q_usgs_df = Q_usgs_df.loc[aligned]

#     TARGET_BASINS = common
#     NUM_CORES = max(1, cpu_count() - 1)
#     tasks = []

#     for basin in TARGET_BASINS:
#         try:
#             tasks.append((
#                 basin,
#                 Rainf_df[basin].values.astype(float),
#                 PotEvap_df[basin].values.astype(float),
#                 Q_usgs_df[basin].values.astype(float),
#                 REPS_FAST,
#                 REPS_SLOW,
#             ))
#         except KeyError:
#             print(f"⚠️ Skipping basin {basin} due to missing aligned data.")
#             continue

#     results = {}
#     os.makedirs(os.path.join(PROJECT_ROOT, "SCE_cal_params"), exist_ok=True)

#     with get_context("spawn").Pool(NUM_CORES) as pool:
#         for basin, out in pool.starmap(worker_calibrate_basin, tasks):
#             if out:
#                 results[basin] = out

#     output_json = os.path.join(PROJECT_ROOT, "SCE_cal_params", "final_calibrated_params.json")
#     with open(output_json, "w") as f:
#         json.dump(clean_dict_for_json(results), f, indent=2)

#     # -----------------------------------------------------
#     # Save summary table
#     # -----------------------------------------------------
#     result_rows = [
#         {"Basin": k, **v} for k, v in results.items() if v.get("KGE") is not None
#     ]

#     if result_rows:
#         df = pd.DataFrame(result_rows).round(3).sort_values("KGE", ascending=False)
#         csv_summary = os.path.join(PROJECT_ROOT, "SCE_cal_params", "final_calibrated_params_with_KGE.csv")

#         df.to_csv(csv_summary, index=False)

#         print("\n" + "=" * 80)
#         print("✅ FINAL CALIBRATED PARAMETERS WITH KGE\n")
#         print(df.to_markdown(index=False))
#         print(f"\n💾 JSON Saved: {output_json}")
#         print(f"📄 Summary Saved: {csv_summary}")

#         total = len(TARGET_BASINS)
#         good = int((df["KGE"] > 0.2).sum())
#         print(f"\n🏆 SUCCESS: {good}/{total} basins with KGE > 0.5")
#     else:
#         print("No successful calibrations to summarize.")





































# # calibration.py
# import os
# import sys

# # Ensure project root path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import json
# import warnings
# import numpy as np
# import pandas as pd
# import spotpy

# from typing import Dict, Any, Tuple, Optional
# from multiprocessing import cpu_count, get_context

# from src.model import ModelParams, two_store_model_step
# from src.metrics import calculate_kge, calculate_nse

# warnings.filterwarnings(
#     "ignore",
#     category=RuntimeWarning,
#     message="invalid value encountered in true_divide"
# )

# # --------------------------------------------------------------------------------------
# # Configuration
# # --------------------------------------------------------------------------------------
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# # Repetitions for SCE-UA (adaptive strategy)
# REPS_FAST = 2000
# REPS_SLOW = 3000

# # Parameter bounds
# Smax_min, Smax_max = 0.02, 50
# fS0_min, fS0_max = 0.01, 0.99
# fG0_min, fG0_max = 0.00, 0.99
# Gmaxfac_min, Gmaxfac_max = 1.0, 100.0


# # --------------------------------------------------------------------------------------
# # Utility: Ensure JSON-safe serialization
# # --------------------------------------------------------------------------------------
# def clean_dict_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
#     cleaned = {}
#     for k, v in data.items():
#         if isinstance(v, dict):
#             cleaned[k] = clean_dict_for_json(v)
#         elif isinstance(v, (np.float32, np.float64)):
#             cleaned[k] = float(v)
#         elif isinstance(v, (np.int32, np.int64)):
#             cleaned[k] = int(v)
#         else:
#             cleaned[k] = v
#     return cleaned

# def kge_2012(evaluation, simulation, eps=1e-12):
#     evaluation = np.asarray(evaluation, dtype=np.float64)
#     simulation = np.asarray(simulation, dtype=np.float64)

#     obs_mean = np.mean(evaluation)
#     sim_mean = np.mean(simulation)

#     obs_std = np.std(evaluation)
#     sim_std = np.std(simulation)

#     # correlation
#     r_num = np.sum((simulation - sim_mean) * (evaluation - obs_mean))
#     r_den = np.sqrt(
#         np.sum((simulation - sim_mean) ** 2) *
#         np.sum((evaluation - obs_mean) ** 2)
#     )
#     r = r_num / (r_den + eps)

#     # bias ratio
#     beta = (sim_mean + eps) / (obs_mean + eps)

#     # variability ratio (CV ratio)
#     gamma = (sim_std / (sim_mean + eps)) / ((obs_std / (obs_mean + eps)) + eps)

#     return 1.0 - np.sqrt((r - 1.0) ** 2 + (gamma - 1.0) ** 2 + (beta - 1.0) ** 2)

# # --------------------------------------------------------------------------------------
# # Forward model
# # --------------------------------------------------------------------------------------
# def run_forward_model(
#     P_data: np.ndarray,
#     PET_data: np.ndarray,
#     Q_obs: np.ndarray,
#     initial_state: tuple,
#     params: ModelParams,
#     gmax_factor: float,
# ) -> tuple:
#     """
#     Run monthly forward simulation.
#     Returns simulated + observed series with spin-up removed and valid-mask applied.
#     """
#     nmonths = len(P_data)
#     S_curr, G_curr = initial_state
#     Q_sim = np.zeros(nmonths)

#     S_cap = max(params.Smax, 1e-9)
#     G_cap = max(gmax_factor * params.Smax, 1e-9)

#     for t in range(nmonths):
#         P_t = np.nan_to_num(max(P_data[t], 0.0))
#         PET_t = np.nan_to_num(max(PET_data[t], 0.0))

#         try:
#             S_next, G_next, _, Q_t, *_ = two_store_model_step(
#                 S_curr, G_curr, P_t, PET_t, params
#             )
#         except Exception:
#             S_next, G_next, Q_t = S_curr, G_curr, 0.0

#         S_curr = np.clip(S_next, 0.0, S_cap)
#         G_curr = np.clip(G_next, 0.0, G_cap)
#         Q_sim[t] = max(Q_t, 0.0)

#     spinup = max(20, min(60, nmonths // 8))

#     Q_sim_clean = Q_sim[spinup:]
#     Q_obs_clean = Q_obs[spinup:]

#     # mask = ~(np.isnan(Q_sim_clean) | np.isnan(Q_obs_clean))
#     mask = np.isfinite(Q_obs_clean)

#     if mask.sum() < 12:
#         return np.zeros(1), np.zeros(1)

#     return Q_sim_clean[mask], Q_obs_clean[mask]


# # --------------------------------------------------------------------------------------
# # SpotPY Model Class
# # --------------------------------------------------------------------------------------
# class TwoStoreModel_SCE:
#     def __init__(self, P_data, PET_data, Q_obs, target_basin):
#         self.P_data = P_data
#         self.PET_data = PET_data
#         self.Q_obs = Q_obs
#         self.target_basin = target_basin

#     def parameters(self):
#         return spotpy.parameter.generate([
#             spotpy.parameter.Uniform(0.01, 1.0, name="Kperc"),
#             spotpy.parameter.Uniform(0.01, 1.0, name="Kb"),
#             spotpy.parameter.Uniform(0.01, 1.0, name="Ke"),
#             spotpy.parameter.Uniform(0.01, 1.0, name="Cqq"),
#             spotpy.parameter.Uniform(Smax_min, Smax_max, name="Smax"),
#             spotpy.parameter.Uniform(fS0_min, fS0_max, name="fS0"),
#             spotpy.parameter.Uniform(fG0_min, fG0_max, name="fG0"),
#             spotpy.parameter.Uniform(Gmaxfac_min, Gmaxfac_max, name="Gmax_factor"),
#         ])

#     @staticmethod
#     def _mk_params(d: Dict[str, float]) -> ModelParams:
#         return ModelParams(
#             Smax=d["Smax"],
#             Kperc=d["Kperc"],
#             Kb=d["Kb"],
#             Ke=d["Ke"],
#             Cqq=d["Cqq"],
#         )

#     def simulation(self, params: Dict[str, float]):
#         S0 = params["fS0"] * params["Smax"]
#         G0 = params["fG0"] * params["Smax"]
#         mp = self._mk_params(params)

#         Q_sim, _ = run_forward_model(
#             self.P_data, self.PET_data, self.Q_obs,
#             (S0, G0), mp, params["Gmax_factor"]
#         )

#         return Q_sim.tolist() if len(Q_sim) > 0 else [np.nanmean(self.Q_obs)]

#     def evaluation(self):
#         Q = np.array(self.Q_obs, dtype=float)

#         nmonths = len(Q)
#         spinup = max(20, min(60, nmonths // 8))

#         Q_clean = Q[spinup:]
#         mask = np.isfinite(Q_clean)

#         if mask.sum() < 12:
#             return [np.nanmean(Q)]

#         return Q_clean[mask].tolist()

#     # -----------------------------------------------------
#     # Mixed objective: 0.6*KGE(sqrt(Q)) + 0.4*NSE(Q)
#     # -----------------------------------------------------
#     def objectivefunction(self, evaluation, simulation):
#         evaluation = np.asarray(evaluation, dtype=np.float64)
#         simulation = np.asarray(simulation, dtype=np.float64)

#         if len(evaluation) < 12 or len(simulation) != len(evaluation):
#             return 9999.0
#         if not (np.all(np.isfinite(evaluation)) and np.all(np.isfinite(simulation))):
#             return 9999.0

#         eps = 1e-6
#         qobs = np.clip(evaluation, eps, None)
#         qsim = np.clip(simulation, eps, None)

#         kge = kge_2012(qobs, qsim)

#         if not np.isfinite(kge) or kge < -1.0:
#             return 9999.0

#         return -kge    # SpotPY minimizes, so maximize KGE



# # --------------------------------------------------------------------------------------
# # Basin Calibration Worker
# # --------------------------------------------------------------------------------------
# def worker_calibrate_basin(
#     target_basin: str,
#     P_data: np.ndarray,
#     PET_data: np.ndarray,
#     Q_usgs: np.ndarray,
#     reps_fast: int,
#     reps_slow: int,
# ) -> Tuple[str, Optional[Dict[str, Any]]]:

#     print(f"\n---> Calibrating Basin: {target_basin} (PID: {os.getpid()})...")

#     if not np.all(np.isfinite(Q_usgs)) or np.all(np.isnan(P_data)):
#         print(f" > Basin {target_basin}: invalid data. Skipping.")
#         return target_basin, None

#     model = TwoStoreModel_SCE(P_data, PET_data, Q_usgs, target_basin)
#     db_path = os.path.join(PROJECT_ROOT, "SCE_cal_params", f"sceua_{target_basin}")

#     def _run(reps):
#         sampler = spotpy.algorithms.sceua(
#             model, dbname=db_path, dbformat="csv", save_sim=False
#         )
#         sampler.sample(repetitions=reps, ngs=70, kstop=30, peps=1e-4, pcento=1e-4)
#         res = sampler.getdata()

#         if res is None or len(res) == 0:
#             return None, None

#         idx, _ = spotpy.analyser.get_minlikeindex(res)
#         return res, idx

#     try:
#         res, idx = _run(reps_fast)
#         if res is None:
#             return target_basin, None

#         like = float(res["like1"][idx][0])
#         if like > -0.60:
#             res, idx = _run(reps_slow)
#             if res is None:
#                 return target_basin, None

#         best = {
#             k: float(res[k][idx][0])
#             for k in res.dtype.names
#             if k.startswith(("par", "like"))
#         }

#         par = {
#             "Kperc": best["parKperc"],
#             "Kb": best["parKb"],
#             "Ke": best["parKe"],
#             "Cqq": best["parCqq"],
#             "Smax": best["parSmax"],
#             # "fS0": best["parfS0"],
#             # "fG0": best["parfG0"],
#             "Gmax_factor": best["parGmax_factor"],
#         }

#         S0 = par["fS0"] * par["Smax"]
#         G0 = par["fG0"] * par["Smax"]

#         mp = ModelParams(
#             Smax=par["Smax"],
#             Kperc=par["Kperc"],
#             Kb=par["Kb"],
#             Ke=par["Ke"],
#             Cqq=par["Cqq"],
#         )

#         qsim, qobs = run_forward_model(
#             P_data, PET_data, Q_usgs, (S0, G0), mp, par["Gmax_factor"]
#         )
#         eps = 1e-6
#         qobs_c = np.clip(qobs, eps, None)
#         qsim_c = np.clip(qsim, eps, None)

#         KGE = kge_2012(qobs_c, qsim_c)
#         NSE = calculate_nse(qobs_c, qsim_c)


#         output = {
#             **par,
#             "S_init": S0,
#             "G_init": G0,
#             "KGE": float(KGE),
#             "NSE": float(NSE),
#         }


#         return target_basin, clean_dict_for_json(output)

#     except Exception as e:
#         print(f"❌ Calibration failed for {target_basin}: {e}")
#         return target_basin, None

#     finally:
#         csv_path = db_path + ".csv"
#         if os.path.exists(csv_path):
#             try:
#                 os.remove(csv_path)
#             except Exception:
#                 pass


# # --------------------------------------------------------------------------------------
# # Main Execution
# # --------------------------------------------------------------------------------------
# if __name__ == "__main__":
#     print("🔄 Loading and aligning input data...")

#     DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

#     Rainf_df = pd.read_feather(os.path.join(DATA_DIR, "Rainf.feather")).set_index("time")
#     PotEvap_df = pd.read_feather(os.path.join(DATA_DIR, "PotEvap.feather")).set_index("time")
#     Q_usgs_df = pd.read_feather(os.path.join(DATA_DIR, "Q_USGS.feather")).set_index("time")

#     common = list(set(Rainf_df.columns) & set(PotEvap_df.columns) & set(Q_usgs_df.columns))

#     Rainf_df = Rainf_df[common]
#     PotEvap_df = PotEvap_df[common]
#     Q_usgs_df = Q_usgs_df[common]

#     aligned = Rainf_df.index.intersection(PotEvap_df.index).intersection(Q_usgs_df.index)
#     Rainf_df = Rainf_df.loc[aligned]
#     PotEvap_df = PotEvap_df.loc[aligned]
#     Q_usgs_df = Q_usgs_df.loc[aligned]

#     TARGET_BASINS = common
#     NUM_CORES = max(1, cpu_count() - 1)
#     tasks = []

#     for basin in TARGET_BASINS:
#         try:
#             tasks.append((
#                 basin,
#                 Rainf_df[basin].values.astype(float),
#                 PotEvap_df[basin].values.astype(float),
#                 Q_usgs_df[basin].values.astype(float),
#                 REPS_FAST,
#                 REPS_SLOW,
#             ))
#         except KeyError:
#             print(f"⚠️ Skipping basin {basin} due to missing aligned data.")
#             continue

#     results = {}
#     os.makedirs(os.path.join(PROJECT_ROOT, "SCE_cal_params"), exist_ok=True)

#     with get_context("spawn").Pool(NUM_CORES) as pool:
#         for basin, out in pool.starmap(worker_calibrate_basin, tasks):
#             if out:
#                 results[basin] = out

#     output_json = os.path.join(PROJECT_ROOT, "SCE_cal_params", "final_calibrated_params.json")
#     with open(output_json, "w") as f:
#         json.dump(clean_dict_for_json(results), f, indent=2)

#     # -----------------------------------------------------
#     # Save summary table
#     # -----------------------------------------------------
#     result_rows = [
#         {"Basin": k, **v} for k, v in results.items() if v.get("KGE") is not None
#     ]

#     if result_rows:
#         df = pd.DataFrame(result_rows).round(3).sort_values("KGE", ascending=False)
#         csv_summary = os.path.join(PROJECT_ROOT, "SCE_cal_params", "final_calibrated_params_with_KGE.csv")

#         df.to_csv(csv_summary, index=False)

#         print("\n" + "=" * 80)
#         print("✅ FINAL CALIBRATED PARAMETERS WITH KGE\n")
#         print(df.to_markdown(index=False))
#         print(f"\n💾 JSON Saved: {output_json}")
#         print(f"📄 Summary Saved: {csv_summary}")

#         total = len(TARGET_BASINS)
#         good = int((df["KGE"] > 0.5).sum())
#         print(f"\n🏆 SUCCESS: {good}/{total} basins with KGE > 0.5")
#     else:
#         print("No successful calibrations to summarize.")
