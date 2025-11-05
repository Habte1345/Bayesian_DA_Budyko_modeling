import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Tuple, Optional
from multiprocessing import cpu_count, get_context
import warnings
import spotpy

from src.model import ModelParams, two_store_model_step
from src.metrics import calculate_kge

warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in true_divide')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

S_MAX_CEILING = 2500.0
G_MAX_CEILING = S_MAX_CEILING * 3.0
REPS = 5000
Smax_heu = 0.05
S_init = 0.35
G_init = 0.15


def run_forward_model(P_data, PET_data, Q_obs, initial_state: tuple, params: ModelParams) -> tuple:
    nmonths = len(P_data)
    S_curr, G_curr = initial_state
    Q_sim = np.zeros(nmonths)

    for t in range(nmonths):
        P_t, PET_t = np.nan_to_num(max(P_data[t], 0.0)), np.nan_to_num(max(PET_data[t], 0.0))
        try:
            S_next, G_next, _, Q_t, _, _, _ = two_store_model_step(S_curr, G_curr, P_t, PET_t, params)
        except Exception:
            S_next, G_next, Q_t = S_curr, G_curr, 0.0

        S_curr = np.clip(S_next, 0.0, params.Smax)
        G_curr = np.clip(G_next, 0.0, G_MAX_CEILING)
        Q_sim[t] = max(Q_t, 0.0)

    spin_up = 20
    Q_sim_clean = Q_sim[spin_up:]
    Q_obs_clean = Q_obs[spin_up:]
    valid_mask = ~(np.isnan(Q_sim_clean) | np.isnan(Q_obs_clean))
    if valid_mask.sum() < 12:
        return np.zeros(1), np.zeros(1)

    return Q_sim_clean[valid_mask], Q_obs_clean[valid_mask]


def clean_dict_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
    cleaned_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            cleaned_data[key] = clean_dict_for_json(value)
        elif isinstance(value, (np.float32, np.float64)):
            cleaned_data[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            cleaned_data[key] = int(value)
        else:
            cleaned_data[key] = value
    return cleaned_data


class TwoStoreModel_SCE:
    def __init__(self, P_data, PET_data, Q_obs, Smax, initial_S, initial_G, target_basin):
        self.P_data = P_data
        self.PET_data = PET_data
        self.Q_obs = Q_obs
        self.Smax = Smax
        self.initial_S = initial_S
        self.initial_G = initial_G
        self.target_basin = target_basin

    def parameters(self):
        return spotpy.parameter.generate([
            spotpy.parameter.Uniform(0.01, 1.0, name='Kperc'),
            spotpy.parameter.Uniform(0.01, 1.0, name='Kb'),
            spotpy.parameter.Uniform(0.01, 1.0, name='Ke'),
            spotpy.parameter.Uniform(0.1, 1.0, name='Cqq'),
        ])

    def simulation(self, params):
        model_params = ModelParams(Smax=self.Smax, Kperc=params['Kperc'], Kb=params['Kb'],
                                   Ke=params['Ke'], Cqq=params['Cqq'])
        initial_state = (self.initial_S, self.initial_G)
        Q_sim, _ = run_forward_model(self.P_data, self.PET_data, self.Q_obs, initial_state, model_params)
        return Q_sim.tolist() if len(Q_sim) > 0 else [np.nanmean(self.Q_obs)]

    def evaluation(self):
        _, Q_obs_clean = run_forward_model(self.P_data, self.PET_data, self.Q_obs,
                                           (self.initial_S, self.initial_G),
                                           ModelParams(Smax=self.Smax))
        return Q_obs_clean.tolist() if len(Q_obs_clean) > 0 else [np.nanmean(self.Q_obs)]

    def objectivefunction(self, evaluation, simulation):
        evaluation, simulation = np.array(evaluation), np.array(simulation)
        if not np.all(np.isfinite(simulation)) or len(evaluation) < 12:
            return 9999.0
        epsilon = 1e-6
        kge_sr = calculate_kge(np.sqrt(np.clip(evaluation, epsilon, None)),
                               np.sqrt(np.clip(simulation, epsilon, None)))
        return -kge_sr if np.isfinite(kge_sr) else 9999.0


def worker_calibrate_basin(target_basin: str, P_data: np.ndarray, PET_data: np.ndarray, Q_nldas: np.ndarray,
                           S_init: float, G_init: float, Smax: float, reps: int) -> Tuple[str, Optional[Dict[str, Any]]]:

    print(f"\n---> Calibrating Basin: {target_basin} (PID: {os.getpid()})...")

    if not np.all(np.isfinite(Q_nldas)) or np.all(np.isnan(P_data)):
        print(f" > Basin {target_basin} has invalid data. Skipping calibration.")
        return target_basin, None

    model = TwoStoreModel_SCE(P_data, PET_data, Q_nldas, Smax, S_init, G_init, target_basin)
    db_path = os.path.join(PROJECT_ROOT, 'SCE_cal_params', f'sceua_{target_basin}')

    try:
        sampler = spotpy.algorithms.sceua(model, dbname=db_path, dbformat='csv', save_sim=False)
        sampler.sample(repetitions=reps, ngs=70, kstop=30, peps=1e-4, pcento=1e-4)
        res = sampler.getdata()
        if res is None or len(res) == 0:
            return target_basin, None

        best_idx, _ = spotpy.analyser.get_minlikeindex(res)
        best_params = {
            'Kperc': float(res['parKperc'][best_idx][0]),
            'Kb': float(res['parKb'][best_idx][0]),
            'Ke': float(res['parKe'][best_idx][0]),
            'Cqq': float(res['parCqq'][best_idx][0]),
            'S_init': S_init,
            'G_init': G_init,
            'Smax': Smax
        }

        mp = ModelParams(Smax=Smax, **{k: best_params[k] for k in ['Kperc', 'Kb', 'Ke', 'Cqq']})
        qsim, qobs = run_forward_model(P_data, PET_data, Q_nldas, (S_init, G_init), mp)
        best_params['KGE'] = calculate_kge(qobs, qsim) if len(qsim) > 0 else 0.0

        return target_basin, clean_dict_for_json(best_params)

    except Exception as e:
        print(f"❌ Calibration failed for {target_basin}: {e}")
        return target_basin, None
    finally:
        if os.path.exists(db_path + '.csv'):
            os.remove(db_path + '.csv')


if __name__ == '__main__':
    print("🔄 Loading input data...")

    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    Rainf_df = pd.read_feather(os.path.join(DATA_DIR, "Rainf.feather")).set_index('time')
    PotEvap_df = pd.read_feather(os.path.join(DATA_DIR, "PotEvap.feather")).set_index('time')
    Q_nldas_df = pd.read_feather(os.path.join(DATA_DIR, "Q_nldas_mm_monthly.feather")).set_index('time')

    TARGET_BASINS = list(Q_nldas_df.columns)
    NUM_CORES = max(1, cpu_count() - 1)
    tasks = []

    for basin in TARGET_BASINS:
        try:
            tasks.append((
                basin,
                Rainf_df[basin].values,
                PotEvap_df[basin].values,
                Q_nldas_df[basin].values,
                S_init,
                G_init,
                Smax_heu,
                REPS
            ))
        except KeyError:
            print(f"⚠️ Skipping basin {basin} due to missing data.")
            continue

    results = {}
    os.makedirs(os.path.join(PROJECT_ROOT, 'SCE_cal_params'), exist_ok=True)
    with get_context("spawn").Pool(NUM_CORES) as pool:
        for basin, param_dict in pool.starmap(worker_calibrate_basin, tasks):
            if param_dict:
                results[basin] = param_dict

    output_path = os.path.join(PROJECT_ROOT, 'SCE_cal_params', 'final_calibrated_params.json')
    with open(output_path, 'w') as f:
        json.dump(clean_dict_for_json(results), f, indent=2)
    # Save JSON of calibrated parameters
    output_path = os.path.join(PROJECT_ROOT, 'SCE_cal_params', 'final_calibrated_params.json')
    with open(output_path, 'w') as f:
        json.dump(clean_dict_for_json(results), f, indent=2)

    # ------------------------- Save and Print Calibrated Parameters with KGE -------------------------

    # Clean results into DataFrame
    results_list = [{'Basin': k, **v} for k, v in results.items() if v.get('KGE') is not None]
    df = pd.DataFrame(results_list).round(3).sort_values('KGE', ascending=False)

    # Save to CSV
    summary_csv_path = os.path.join(PROJECT_ROOT, 'SCE_cal_params', 'final_calibrated_params_with_KGE.csv')
    df.to_csv(summary_csv_path, index=False)

    # Print to console
    print("\n" + "="*80)
    print("✅ FINAL CALIBRATED PARAMETERS WITH KGE")
    print(df.to_markdown(index=False))
    print(f"\n💾 Saved: {output_path}")
    print(f"📄 Summary Table Saved: {summary_csv_path}")
    print(f"\n🏆 SUCCESS: {len(df[df['KGE'] > 0.5])}/{len(TARGET_BASINS)} basins with KGE > 0.5")
