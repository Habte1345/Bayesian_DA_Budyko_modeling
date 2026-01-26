import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Robust PROJECT_ROOT resolution
# ---------------------------------------------------------
THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(THIS_FILE)  # plot.py is inside the project root
RESULT_DIR = os.path.join(PROJECT_ROOT, "Simulation_results")

print(f"\nPROJECT_ROOT = {PROJECT_ROOT}")
print(f"RESULT_DIR   = {RESULT_DIR}")


def list_available_basins(result_dir: str):
    feather_files = glob.glob(os.path.join(result_dir, "results_streamflow_*.feather"))
    basins = [
        os.path.basename(f).replace("results_streamflow_", "").replace(".feather", "")
        for f in feather_files
    ]
    return sorted(basins)


def load_basin_outputs(basin_id: str, result_dir: str):
    feather_path = os.path.join(result_dir, f"results_streamflow_{basin_id}.feather")
    npz_path = os.path.join(result_dir, f"enkf_ensemble_{basin_id}.npz")

    if not os.path.exists(feather_path) or not os.path.exists(npz_path):
        basins = list_available_basins(result_dir)
        raise FileNotFoundError(
            f"\nMissing files for basin: {basin_id}\n\n"
            f"Expected:\n  {feather_path}\n  {npz_path}\n\n"
            f"Available basins (first 30):\n" + "\n".join(basins[:30])
        )

    df = pd.read_feather(feather_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    z = np.load(npz_path, allow_pickle=True)

    time = pd.to_datetime(z["time"])
    S_ens = z["S_ens_hist"]
    G_ens = z["G_ens_hist"]
    ET_ens = z["ET_ens_hist"]
    Q_ens = z["Q_ens_hist"]

    return df, time, S_ens, G_ens, ET_ens, Q_ens


def ens_stats(ens_hist):
    mean = np.nanmean(ens_hist, axis=1)
    lo = np.nanpercentile(ens_hist, 2.5, axis=1)
    hi = np.nanpercentile(ens_hist, 97.5, axis=1)
    return mean, lo, hi


def plot_states(time, S_ens, G_ens, basin_id):
    S_mean, S_lo, S_hi = ens_stats(S_ens)
    G_mean, G_lo, G_hi = ens_stats(G_ens)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    ax.plot(time, S_mean, linestyle="dashed", label="Posterior Mean S")
    ax.fill_between(time, S_lo, S_hi, alpha=0.2, label="95% CI S")

    ax.plot(time, G_mean, linestyle="dashed", label="Posterior Mean G")
    ax.fill_between(time, G_lo, G_hi, alpha=0.2, label="95% CI G")

    ax.set_title(f"EnKF States (S,G) — {basin_id}")
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_et(df, time, ET_ens, basin_id):
    ET_mean, ET_lo, ET_hi = ens_stats(ET_ens)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    ax.plot(df.index, df["ET_B"], label="Observation ET_B")
    ax.plot(time, ET_mean, linestyle="dashed", label="Posterior Mean ET")
    ax.fill_between(time, ET_lo, ET_hi, alpha=0.2, label="95% CI ET")

    ax.set_title(f"ET during EnKF — {basin_id}")
    ax.set_xlabel("Time")
    ax.set_ylabel("ET")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_q(df, time, Q_ens, basin_id):
    Q_mean, Q_lo, Q_hi = ens_stats(Q_ens)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    ax.plot(df.index, df["Q_obs"], label="Observation Q")
    ax.plot(time, Q_mean, linestyle="dashed", label="Posterior Mean Q")
    ax.fill_between(time, Q_lo, Q_hi, alpha=0.2, label="95% CI Q")

    ax.set_title(f"Streamflow during EnKF — {basin_id}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Q")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    basins = list_available_basins(RESULT_DIR)

    if not basins:
        raise FileNotFoundError(
            f"No results_streamflow_*.feather found in:\n{RESULT_DIR}\n\n"
            "Run run_simulation.py first."
        )

    print(f"\n✅ Found {len(basins)} basins.")
    print("First 15 basins:")
    for b in basins[:15]:
        print(" -", b)

    basin_id = basins[0]  # pick the first saved basin automatically
    # basin_id = "camels_01022500"

    df, time, S_ens, G_ens, ET_ens, Q_ens = load_basin_outputs(basin_id, RESULT_DIR)

    plot_states(time, S_ens, G_ens, basin_id)
    plot_et(df, time, ET_ens, basin_id)
    plot_q(df, time, Q_ens, basin_id)
