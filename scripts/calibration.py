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
from src.metrics import calculate_kge, calculate_nse

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
REPS_FAST = 1000
REPS_SLOW = 2000

# Parameter bounds
Smax_min, Smax_max = 0.02, 1000
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

    mask = ~(np.isnan(Q_sim_clean) | np.isnan(Q_obs_clean))
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
            spotpy.parameter.Uniform(0.01, 1.0, name="Cqq"),
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

        return Q_sim.tolist() if len(Q_sim) > 0 else [np.nanmean(self.Q_obs)]

    def evaluation(self):
        mid = 0.5 * (Smax_min + Smax_max)
        S0, G0 = 0.5 * mid, 0.1 * mid

        mp = ModelParams(Smax=mid, Kperc=0.3, Kb=0.3, Ke=0.3, Cqq=0.5)

        _, Q_clean = run_forward_model(
            self.P_data, self.PET_data, self.Q_obs, (S0, G0), mp, gmax_factor=3.0
        )
        return Q_clean.tolist() if len(Q_clean) > 0 else [np.nanmean(self.Q_obs)]

    # -----------------------------------------------------
    # Mixed objective: 0.6*KGE(sqrt(Q)) + 0.4*NSE(Q)
    # -----------------------------------------------------
    def objectivefunction(self, evaluation, simulation):
        evaluation = np.array(evaluation)
        simulation = np.array(simulation)

        if not np.all(np.isfinite(simulation)) or len(evaluation) < 12:
            return 9999.0

        eps = 1e-6
        eval_c = np.clip(evaluation, eps, None)
        sim_c = np.clip(simulation, eps, None)

        kge_sr = calculate_kge(np.sqrt(eval_c), np.sqrt(sim_c))
        nse_q = calculate_nse(eval_c, sim_c)

        kge_sr = 0.0 if not np.isfinite(kge_sr) else kge_sr
        nse_q = 0.0 if not np.isfinite(nse_q) else nse_q

        if kge_sr < -1.0 or nse_q < -1.0:
            return 9999.0

        return -(0.6 * kge_sr + 0.4 * nse_q)


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
        KGE = calculate_kge(qobs, qsim) if len(qsim) > 0 else 0.0

        output = {
            **par,
            "S_init": S0,
            "G_init": G0,
            "KGE": float(KGE),
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
