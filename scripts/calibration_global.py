# calibration.py
import os
import sys
import json
import warnings
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import spotpy
import yaml
from tqdm import tqdm
import contextlib
import io


# Ensure project root path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import ModelParams, two_store_model_step
from src.metrics import calculate_nse

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in true_divide"
)


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def clean_dict_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for k, v in data.items():
        if isinstance(v, dict):
            cleaned[k] = clean_dict_for_json(v)
        elif isinstance(v, list):
            cleaned[k] = [
                float(x) if isinstance(x, (np.float32, np.float64)) else
                int(x) if isinstance(x, (np.int32, np.int64)) else x
                for x in v
            ]
        elif isinstance(v, (np.float32, np.float64)):
            cleaned[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            cleaned[k] = int(v)
        else:
            cleaned[k] = v
    return cleaned


def load_feather_df(fname: str, ddir: str) -> pd.DataFrame:
    path = os.path.join(ddir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_feather(path).dropna(axis=1, how="all")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

    return df


def kge_2012(evaluation, simulation, eps=1e-12):
    evaluation = np.asarray(evaluation, dtype=np.float64)
    simulation = np.asarray(simulation, dtype=np.float64)

    obs_mean = np.mean(evaluation)
    sim_mean = np.mean(simulation)

    obs_std = np.std(evaluation)
    sim_std = np.std(simulation)

    r_num = np.sum((simulation - sim_mean) * (evaluation - obs_mean))
    r_den = np.sqrt(
        np.sum((simulation - sim_mean) ** 2) *
        np.sum((evaluation - obs_mean) ** 2)
    )
    r = r_num / (r_den + eps)

    beta = (sim_mean + eps) / (obs_mean + eps)
    gamma = (sim_std / (sim_mean + eps)) / ((obs_std / (obs_mean + eps)) + eps)

    return 1.0 - np.sqrt((r - 1.0) ** 2 + (gamma - 1.0) ** 2 + (beta - 1.0) ** 2)


def align_common_data(data_dir: str):
    rainf = load_feather_df("Rainf.feather", data_dir)
    pet = load_feather_df("PotEvap.feather", data_dir)
    q_usgs = load_feather_df("Q_USGS.feather", data_dir)

    common_cols = sorted(set(rainf.columns) & set(pet.columns) & set(q_usgs.columns))
    if not common_cols:
        raise ValueError("No common basins across Rainf, PotEvap, and Q_USGS.")

    common_idx = rainf.index.intersection(pet.index).intersection(q_usgs.index)

    rainf = rainf.loc[common_idx, common_cols]
    pet = pet.loc[common_idx, common_cols]
    q_usgs = q_usgs.loc[common_idx, common_cols]

    return rainf, pet, q_usgs


def resolve_basin_split(all_basins: List[str], cfg: dict) -> Tuple[List[str], List[str]]:
    split_cfg = cfg.get("basins", {}).get("split", {})

    train_basins = split_cfg.get("train_basins", None)
    test_basins = split_cfg.get("test_basins", None)

    if train_basins is not None and test_basins is not None:
        train_basins = sorted([b for b in train_basins if b in all_basins])
        test_basins = sorted([b for b in test_basins if b in all_basins])

        overlap = set(train_basins) & set(test_basins)
        if overlap:
            raise ValueError(f"Train/test split overlaps: {sorted(overlap)}")

        if len(train_basins) == 0:
            raise ValueError("No valid training basins remain after filtering.")
        return train_basins, test_basins

    seed = int(split_cfg.get("seed", 42))
    train_fraction = float(split_cfg.get("train_fraction", 0.7))

    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(all_basins), dtype=object)
    rng.shuffle(shuffled)

    n_train = max(1, int(round(train_fraction * len(shuffled))))
    train_basins = sorted(shuffled[:n_train].tolist())
    test_basins = sorted(shuffled[n_train:].tolist())

    return train_basins, test_basins


def get_bounds_from_config(cfg: dict) -> dict:
    default_bounds = {
        "Kperc": (0.01, 1.0),
        "Kb": (0.01, 1.0),
        "Ke": (0.01, 1.0),
        "Cqq": (0.01, 1.0),
        "Smax": (0.02, 500.0),
        "fS0": (0.01, 0.99),
        "fG0": (0.00, 0.99),
        "Gmax_factor": (1.0, 100.0),
    }

    user_bounds = cfg.get("global_calibration", {}).get("bounds", {})
    for k, v in user_bounds.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            default_bounds[k] = (float(v[0]), float(v[1]))

    return default_bounds


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
    min_valid_months: int = 12,
) -> tuple:
    """
    Run monthly forward simulation.
    Returns simulated + observed series with spin-up removed and valid-mask applied.
    """
    nmonths = len(P_data)
    S_curr, G_curr = initial_state
    Q_sim = np.zeros(nmonths, dtype=float)

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

    mask = np.isfinite(Q_obs_clean)

    if mask.sum() < min_valid_months:
        return np.zeros(1, dtype=float), np.zeros(1, dtype=float)

    return Q_sim_clean[mask], Q_obs_clean[mask]


# --------------------------------------------------------------------------------------
# Global calibration model
# --------------------------------------------------------------------------------------
class GlobalTwoStoreModelSCE:
    """
    One shared parameter set calibrated over all TRAIN basins.

    Objective:
        maximize mean KGE across training basins
        while penalizing parameter sets that fail on many basins.
    """

    INVALID_SCORE = -999.0

    def __init__(
        self,
        basin_data: Dict[str, Dict[str, np.ndarray]],
        bounds: dict,
        min_valid_months: int = 12,
        min_valid_basin_fraction: float = 0.8,
        invalid_basin_penalty: float = 1.0,
    ):
        self.basin_data = basin_data
        self.train_basins = sorted(list(basin_data.keys()))
        self.bounds = bounds
        self.min_valid_months = int(min_valid_months)
        self.min_valid_basin_fraction = float(min_valid_basin_fraction)
        self.invalid_basin_penalty = float(invalid_basin_penalty)

    def parameters(self):
        b = self.bounds
        return spotpy.parameter.generate([
            spotpy.parameter.Uniform(*b["Kperc"], name="Kperc"),
            spotpy.parameter.Uniform(*b["Kb"], name="Kb"),
            spotpy.parameter.Uniform(*b["Ke"], name="Ke"),
            spotpy.parameter.Uniform(*b["Cqq"], name="Cqq"),
            spotpy.parameter.Uniform(*b["Smax"], name="Smax"),
            spotpy.parameter.Uniform(*b["fS0"], name="fS0"),
            spotpy.parameter.Uniform(*b["fG0"], name="fG0"),
            spotpy.parameter.Uniform(*b["Gmax_factor"], name="Gmax_factor"),
        ])

    @staticmethod
    def _mk_params(d: Dict[str, float]) -> ModelParams:
        return ModelParams(
            Smax=float(d["Smax"]),
            Kperc=float(d["Kperc"]),
            Kb=float(d["Kb"]),
            Ke=float(d["Ke"]),
            Cqq=float(d["Cqq"]),
        )

    def _evaluate_parameter_set(self, params: Dict[str, float]):
        mp = self._mk_params(params)

        S0 = float(params["fS0"]) * float(params["Smax"])
        G0 = float(params["fG0"]) * float(params["Smax"])

        basin_kges = np.full(len(self.train_basins), self.INVALID_SCORE, dtype=float)
        basin_nses = np.full(len(self.train_basins), np.nan, dtype=float)
        basin_nvalid = np.zeros(len(self.train_basins), dtype=int)

        for i, basin in enumerate(self.train_basins):
            P = self.basin_data[basin]["P"]
            PET = self.basin_data[basin]["PET"]
            Q = self.basin_data[basin]["Q"]

            qsim, qobs = run_forward_model(
                P_data=P,
                PET_data=PET,
                Q_obs=Q,
                initial_state=(S0, G0),
                params=mp,
                gmax_factor=float(params["Gmax_factor"]),
                min_valid_months=self.min_valid_months,
            )

            if len(qsim) < self.min_valid_months or len(qobs) != len(qsim):
                continue
            if not (np.all(np.isfinite(qsim)) and np.all(np.isfinite(qobs))):
                continue

            eps = 1e-6
            qobs_c = np.clip(qobs, eps, None)
            qsim_c = np.clip(qsim, eps, None)

            kge = kge_2012(qobs_c, qsim_c)
            nse = calculate_nse(qobs_c, qsim_c)
  # The main core idea of this step is to assign           
            if np.isfinite(kge):
                basin_kges[i] = float(kge)
                basin_nses[i] = float(nse) if np.isfinite(nse) else np.nan
                basin_nvalid[i] = len(qsim_c)

        return basin_kges, basin_nses, basin_nvalid

    def simulation(self, params: Dict[str, float]):
        basin_kges, _, _ = self._evaluate_parameter_set(params)
        return basin_kges.tolist()

    def evaluation(self):
        return [1.0] * len(self.train_basins)

    def objectivefunction(self, evaluation, simulation):
        sim = np.asarray(simulation, dtype=float)

        valid_mask = sim > self.INVALID_SCORE / 10.0
        n_valid = int(valid_mask.sum())
        n_total = len(sim)

        if n_total == 0:
            return 9999.0

        valid_fraction = n_valid / n_total
        if valid_fraction < self.min_valid_basin_fraction:
            return 9999.0

        sim_valid = sim[valid_mask]
        mean_kge = float(np.mean(sim_valid))

        if not np.isfinite(mean_kge):
            return 9999.0

        missing_fraction = 1.0 - valid_fraction
        penalized_score = mean_kge - self.invalid_basin_penalty * missing_fraction

        return -penalized_score


# --------------------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------------------
def evaluate_global_params_on_basins(
    params: Dict[str, float],
    basin_data: Dict[str, Dict[str, np.ndarray]],
    min_valid_months: int = 12,
) -> pd.DataFrame:
    mp = ModelParams(
        Smax=float(params["Smax"]),
        Kperc=float(params["Kperc"]),
        Kb=float(params["Kb"]),
        Ke=float(params["Ke"]),
        Cqq=float(params["Cqq"]),
    )

    S0 = float(params["fS0"]) * float(params["Smax"])
    G0 = float(params["fG0"]) * float(params["Smax"])
    gmax_factor = float(params["Gmax_factor"])

    rows = []

    for basin, d in basin_data.items():
        qsim, qobs = run_forward_model(
            P_data=d["P"],
            PET_data=d["PET"],
            Q_obs=d["Q"],
            initial_state=(S0, G0),
            params=mp,
            gmax_factor=gmax_factor,
            min_valid_months=min_valid_months,
        )

        if len(qsim) < min_valid_months or len(qsim) != len(qobs):
            rows.append({
                "basin": basin,
                "KGE": np.nan,
                "NSE": np.nan,
                "n_valid": 0,
            })
            continue

        eps = 1e-6
        qobs_c = np.clip(qobs, eps, None)
        qsim_c = np.clip(qsim, eps, None)

        kge = kge_2012(qobs_c, qsim_c)
        nse = calculate_nse(qobs_c, qsim_c)

        rows.append({
            "basin": basin,
            "KGE": float(kge) if np.isfinite(kge) else np.nan,
            "NSE": float(nse) if np.isfinite(nse) else np.nan,
            "n_valid": int(len(qsim_c)),
        })

    return pd.DataFrame(rows).sort_values("KGE", ascending=False)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = os.path.join(PROJECT_ROOT, cfg["paths"]["data_dir"])
    # output_dir = os.path.join(PROJECT_ROOT, cfg["paths"]["global_calibration_dir"])
    output_dir = os.path.join(PROJECT_ROOT, cfg["paths"].get("global_calibration_dir", "SCE_global_params"))
    
    os.makedirs(output_dir, exist_ok=True)

    gcfg = cfg.get("global_calibration", {})
    reps_fast = int(gcfg.get("reps_fast", 1000))             # chnage the repetitions for the fast stage to 10 for testing, can be set back to 100 or more for final runs
    reps_slow = int(gcfg.get("reps_slow", 1000))
    min_valid_months = int(gcfg.get("min_valid_months", 12))
    min_valid_basin_fraction = float(gcfg.get("min_valid_basin_fraction", 0.8))
    invalid_basin_penalty = float(gcfg.get("invalid_basin_penalty", 1.0))
    refine_threshold = float(gcfg.get("refine_threshold", -0.60))

    bounds = get_bounds_from_config(cfg)

    print("Loading aligned data...")
    rainf_df, pet_df, q_df = align_common_data(data_dir)

    all_basins = sorted(list(rainf_df.columns))
    train_basins, test_basins = resolve_basin_split(all_basins, cfg)

    print(f"Total basins : {len(all_basins)}")
    print(f"Train basins : {len(train_basins)}")
    print(f"Test basins  : {len(test_basins)}")

    basin_data_train = {
        b: {
            "P": rainf_df[b].values.astype(float),
            "PET": pet_df[b].values.astype(float),
            "Q": q_df[b].values.astype(float),
        }
        for b in train_basins
    }

    model = GlobalTwoStoreModelSCE(
        basin_data=basin_data_train,
        bounds=bounds,
        min_valid_months=min_valid_months,
        min_valid_basin_fraction=min_valid_basin_fraction,
        invalid_basin_penalty=invalid_basin_penalty,
    )

    db_path = os.path.join(output_dir, "global_sceua")


    def _run_sce(reps: int):
        sampler = spotpy.algorithms.sceua(
            model,
            dbname=db_path,
            dbformat="csv",
            save_sim=False,
        )

        # Suppress SpotPY prints (no chunking!)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            sampler.sample(
                repetitions=reps,
                ngs=70,
                kstop=30,
                peps=1e-4,
                pcento=1e-4,
            )

        # Clean progress indicator (after completion)
        with tqdm(total=reps, desc="SCE-UA Calibration", ncols=100) as pbar:
            pbar.update(reps)

        # Retrieve results
        res = sampler.getdata()
        if res is None or len(res) == 0:
            raise RuntimeError("No results returned from SCE-UA.")

        idx, _ = spotpy.analyser.get_minlikeindex(res)

        return res, idx

    print("Running global calibration...")
    res, idx = _run_sce(reps_fast)
    # like = float(res["like1"][idx][0])
    like = float(res["like1"][idx])

    if like > refine_threshold:
        print("Refining with second SCE-UA stage...")
        res, idx = _run_sce(reps_slow)

    best = {
        k: float(res[k][idx])
        for k in res.dtype.names
        if k.startswith(("par", "like"))
    }

    global_params = {
        "Kperc": best["parKperc"],
        "Kb": best["parKb"],
        "Ke": best["parKe"],
        "Cqq": best["parCqq"],
        "Smax": best["parSmax"],
        "fS0": best["parfS0"],
        "fG0": best["parfG0"],
        "Gmax_factor": best["parGmax_factor"],
    }
    global_params["S_init"] = global_params["fS0"] * global_params["Smax"]
    global_params["G_init"] = global_params["fG0"] * global_params["Smax"]

    train_eval = evaluate_global_params_on_basins(
        params=global_params,
        basin_data=basin_data_train,
        min_valid_months=min_valid_months,
    )

    basin_data_test = {
        b: {
            "P": rainf_df[b].values.astype(float),
            "PET": pet_df[b].values.astype(float),
            "Q": q_df[b].values.astype(float),
        }
        for b in test_basins
    }

    if len(test_basins) > 0:
        test_eval = evaluate_global_params_on_basins(
            params=global_params,
            basin_data=basin_data_test,
            min_valid_months=min_valid_months,
        )
    else:
        test_eval = pd.DataFrame(columns=["basin", "KGE", "NSE", "n_valid"])

    split_info = {
        "train_basins": train_basins,
        "test_basins": test_basins,
        "n_train": len(train_basins),
        "n_test": len(test_basins),
    }

    summary = {
        "mean_train_kge": float(train_eval["KGE"].mean()) if len(train_eval) else np.nan,
        "median_train_kge": float(train_eval["KGE"].median()) if len(train_eval) else np.nan,
        "mean_test_kge": float(test_eval["KGE"].mean()) if len(test_eval) else np.nan,
        "median_test_kge": float(test_eval["KGE"].median()) if len(test_eval) else np.nan,
        "n_train": len(train_basins),
        "n_test": len(test_basins),
        "reps_fast": reps_fast,
        "reps_slow": reps_slow,
        "min_valid_months": min_valid_months,
        "min_valid_basin_fraction": min_valid_basin_fraction,
        "invalid_basin_penalty": invalid_basin_penalty,
    }

    with open(os.path.join(output_dir, "global_calibrated_params.json"), "w") as f:
        json.dump(clean_dict_for_json(global_params), f, indent=2)

    with open(os.path.join(output_dir, "basin_split.json"), "w") as f:
        json.dump(clean_dict_for_json(split_info), f, indent=2)

    with open(os.path.join(output_dir, "global_calibration_summary.json"), "w") as f:
        json.dump(clean_dict_for_json(summary), f, indent=2)

    train_eval.to_csv(os.path.join(output_dir, "train_basin_metrics.csv"), index=False)
    test_eval.to_csv(os.path.join(output_dir, "test_basin_metrics.csv"), index=False)

    csv_path = db_path + ".csv"
    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("GLOBAL CALIBRATION COMPLETED")
    print("=" * 80)
    print(json.dumps(clean_dict_for_json(summary), indent=2))
    print(f"\nSaved global params to: {os.path.join(output_dir, 'global_calibrated_params.json')}")