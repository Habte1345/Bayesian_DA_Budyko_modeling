"""
run_enkf.py  –  Main entry point
=================================
1. Load NLDAS Rainf, PotEvap (kg/m²/s → mm/month) and USGS Q observations
2. Align all three DataFrames on a common time × basin grid
3. Run EnKF for every basin  ← NO plots saved here
4. Save metrics CSV + all posterior time series to one Parquet
5. After all basins finish, plot:
     • top-N basin summary panels  (N_PLOT_BASINS, default 12)
     • global summary figures  (distributions, scatter, improvement)
"""

import os, sys, warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← edit these
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = r"F:\Github_repos\Bayesian_DA_Budyko_modeling"
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_DIR      = Path(PROJECT_ROOT) / "results" / "enkf"

RAINF_PATH   = os.path.join(DATA_DIR, "Rainf.feather")
POTEVAP_PATH = os.path.join(DATA_DIR, "PotEvap.feather")
Q_USGS_PATH  = os.path.join(DATA_DIR, "Q_USGS.feather")

# ── Model parameters (shared across all basins) ───────────────────────────
DEFAULT_PARAMS = dict(
    Smax     = 150.0,
    Kperc    = 0.05,
    Kb       = 0.04,
    Ke       = 0.80,
    Cqq      = 0.85,
    Sfc_frac = 0.30,
    beta_et  = 2.0,
)

# ── EnKF hyperparameters ──────────────────────────────────────────────────
ENSEMBLE_SIZE     = 50
OBS_NOISE_STD     = 5.0    # mm/month
INFLATION_STD     = 1.0    # mm/month
RANDOM_SEED       = 42

# ── Subset of basins to run (None = all) ─────────────────────────────────
BASIN_IDS = None

# ── Plotting: how many top basins get individual summary panels ───────────
N_PLOT_BASINS = 12          # e.g. top-12 by EnKF NSE
PLOT_RANK_BY  = "NSE_enkf"  # column in metrics CSV to rank on

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

sys.path.append(str(Path(__file__).parent))
sys.path.append(PROJECT_ROOT)

from model       import ModelParams
from enkf_engine import enkf_loop, compute_metrics
from plot_enkf   import plot_top_basins_grid, plot_global_summary


# ─────────────────────────────────────────────────────────────────────────────
# Unit conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_forcing(df: pd.DataFrame) -> pd.DataFrame:
    """kg/m²/s → mm/month, using actual days per calendar month."""
    seconds = (df.index.days_in_month.values * 86_400.0).reshape(-1, 1)
    return pd.DataFrame(df.values * seconds, index=df.index, columns=df.columns)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & alignment
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading feather files …")
    Rainf_df   = pd.read_feather(RAINF_PATH).set_index("time")
    PotEvap_df = pd.read_feather(POTEVAP_PATH).set_index("time")
    Q_df       = pd.read_feather(Q_USGS_PATH).set_index("time")

    for df, name in [(Rainf_df, "Rainf"), (PotEvap_df, "PotEvap"), (Q_df, "Q_USGS")]:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"{name}: index must be DatetimeIndex after set_index('time')")

    print("Converting units: kg/m²/s → mm/month …")
    Rainf_mm   = convert_forcing(Rainf_df)
    PotEvap_mm = convert_forcing(PotEvap_df)

    # Q assumed already in mm/month.
    # If your Q is in m³/s, add: Q_df = Q_df * (seconds_per_month / basin_area_m2 * 1000)
    Q_mm = Q_df.copy()

    print(f"  Rainf   {Rainf_mm.shape}  [{Rainf_mm.index[0].date()} – {Rainf_mm.index[-1].date()}]")
    print(f"  PotEvap {PotEvap_mm.shape}")
    print(f"  Q_USGS  {Q_mm.shape}")
    return Rainf_mm, PotEvap_mm, Q_mm


def align_data(Rainf_mm, PotEvap_mm, Q_mm, basin_ids=None):
    t_common = (Rainf_mm.index
                .intersection(PotEvap_mm.index)
                .intersection(Q_mm.index))
    if len(t_common) == 0:
        raise ValueError("No overlapping time steps. Check that all feather files share the same time range.")

    basins_all = set(Rainf_mm.columns) & set(PotEvap_mm.columns) & set(Q_mm.columns)
    if basin_ids is not None:
        basins_all &= set(basin_ids)
    if not basins_all:
        raise ValueError("No common basin IDs across all three files. Check column names.")

    basins = sorted(basins_all)
    print(f"\nAligned: {len(t_common)} time steps  ×  {len(basins)} basins.")
    return (Rainf_mm.loc[t_common, basins],
            PotEvap_mm.loc[t_common, basins],
            Q_mm.loc[t_common, basins],
            basins, t_common)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_initial_ensemble(params: ModelParams, N: int, rng: np.random.Generator):
    S0 = rng.uniform(0.1 * params.Smax, 0.6 * params.Smax, size=N)
    G0 = rng.uniform(5.0, 50.0, size=N)
    return np.stack([S0, G0], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    # ── 1. Load & align ───────────────────────────────────────────────────
    Rainf_mm, PotEvap_mm, Q_mm = load_data()
    P, PET, Q_obs, basins, time_index = align_data(
        Rainf_mm, PotEvap_mm, Q_mm, basin_ids=BASIN_IDS
    )

    params   = ModelParams(**DEFAULT_PARAMS)
    n_basins = len(basins)

    # ── 2. EnKF loop ──────────────────────────────────────────────────────
    # No plots here. Just accumulate metrics + lightweight result dicts.
    all_metrics   = []
    basin_results = {}   # basin_id → (Q_arr, P_arr, PET_arr, result_dict)

    print(f"\nRunning EnKF on {n_basins} basins …  (progress every 50)\n")

    for b_idx, basin_id in enumerate(basins):
        P_arr   = P[basin_id].values.astype(float)
        PET_arr = PET[basin_id].values.astype(float)
        Q_arr   = Q_obs[basin_id].values.astype(float)

        P_arr[~np.isfinite(P_arr)]     = 0.0
        PET_arr[~np.isfinite(PET_arr)] = 0.0
        # Q NaNs are kept → enkf_engine skips analysis at those steps

        init_ens = make_initial_ensemble(params, ENSEMBLE_SIZE, rng)

        result = enkf_loop(
            initial_ensemble    = init_ens,
            obs_Q_series        = Q_arr,
            P_series            = P_arr,
            PET_series          = PET_arr,
            params              = params,
            obs_noise_std       = OBS_NOISE_STD,
            inflation_noise_std = INFLATION_STD,
            rng                 = rng,
        )

        m_enkf = compute_metrics(Q_arr, result["post_Q_mean"])
        m_ol   = compute_metrics(Q_arr, result["openloop_Q"])

        if b_idx == 0 or (b_idx + 1) % 50 == 0 or b_idx == n_basins - 1:
            print(f"  [{b_idx+1:4d}/{n_basins}]  {basin_id}  "
                  f"OL-NSE={m_ol['NSE']:+.3f}  "
                  f"EnKF-NSE={m_enkf['NSE']:+.3f}  "
                  f"EnKF-KGE={m_enkf['KGE']:+.3f}")

        all_metrics.append(dict(
            basin_id   = basin_id,
            NSE_ol     = m_ol["NSE"],    KGE_ol    = m_ol["KGE"],
            RMSE_ol    = m_ol["RMSE"],   Bias_ol   = m_ol["Bias_pct"],
            NSE_enkf   = m_enkf["NSE"],  KGE_enkf  = m_enkf["KGE"],
            RMSE_enkf  = m_enkf["RMSE"], Bias_enkf = m_enkf["Bias_pct"],
            NSE_gain   = round(m_enkf["NSE"] - m_ol["NSE"], 4),
            n_obs      = m_enkf["n"],
        ))

        basin_results[basin_id] = (Q_arr, P_arr, PET_arr, result)

    # ── 3. Metrics CSV ────────────────────────────────────────────────────
    metrics_df   = pd.DataFrame(all_metrics)
    metrics_path = OUT_DIR / "all_basins_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Metrics  → {metrics_path}")

    # ── 4. Posterior time series → single Parquet ─────────────────────────
    print("Saving posterior time series …")
    records = []
    for basin_id, (Q_arr, P_arr, PET_arr, result) in basin_results.items():
        records.append(pd.DataFrame({
            "time"        : time_index,
            "basin_id"    : basin_id,
            "obs_Q"       : Q_arr,
            "enkf_Q_mean" : result["post_Q_mean"],
            "enkf_Q_lo95" : result["post_Q_ci"][:, 0],
            "enkf_Q_hi95" : result["post_Q_ci"][:, 1],
            "openloop_Q"  : result["openloop_Q"],
            "enkf_S_mean" : result["post_S_mean"],
            "enkf_G_mean" : result["post_G_mean"],
            "P"           : P_arr,
            "PET"         : PET_arr,
        }))

    parquet_path = OUT_DIR / "all_basins_posterior.parquet"
    pd.concat(records, ignore_index=True).to_parquet(parquet_path, index=False)
    print(f"✓ Posterior → {parquet_path}")

    # ── 5. Aggregate stats ────────────────────────────────────────────────
    print("\n─── Aggregate performance ───────────────────────────────────────")
    for col in ["NSE_ol", "NSE_enkf", "KGE_ol", "KGE_enkf", "NSE_gain"]:
        v = metrics_df[col].dropna()
        print(f"  {col:12s}  median={v.median():+.3f}  mean={v.mean():+.3f}  "
              f"[{v.quantile(0.1):+.3f}–{v.quantile(0.9):+.3f}] (p10–p90)")
    print("─────────────────────────────────────────────────────────────────\n")

    # ── 6. Plots: top-N basins + global summary ───────────────────────────
    # Select top-N basins by PLOT_RANK_BY metric
    top_ids = (metrics_df
               .nlargest(N_PLOT_BASINS, PLOT_RANK_BY)["basin_id"]
               .tolist())

    top_data = {
        bid: (basin_results[bid][0],          # Q_arr
              basin_results[bid][3],          # result dict
              metrics_df[metrics_df.basin_id == bid].iloc[0])
        for bid in top_ids
    }

    print(f"Plotting top-{N_PLOT_BASINS} basins (ranked by {PLOT_RANK_BY}) …")
    plot_top_basins_grid(top_data, time_index, plot_dir)

    print("Plotting global summary …")
    plot_global_summary(metrics_df, basin_results, time_index, plot_dir)

    print(f"\n✓ All done.  Plots → {plot_dir}")


if __name__ == "__main__":
    main()
