"""
plot_enkf.py  –  All plotting for EnKF + Two-Store results
===========================================================
Two public entry points called from run_enkf.py:

  plot_top_basins_grid(top_data, time_index, out_dir)
      One multi-panel PNG with N_PLOT_BASINS rows (default 12).
      Each row = one basin, 3 columns: Q time series | S time series | Q scatter.
      Saved as:  plots/top_basins_grid.png

  plot_global_summary(metrics_df, basin_results, time_index, out_dir)
      Four-panel figure summarising ALL basins:
        1. NSE distribution: open-loop vs EnKF (violin + box)
        2. KGE distribution: open-loop vs EnKF (violin + box)
        3. NSE gain histogram  (EnKF NSE − OL NSE)
        4. Pooled obs vs EnKF scatter  (sample of points across all basins)
      Saved as:  plots/global_summary.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

COLORS = dict(
    obs      = "#222222",
    enkf     = "#1f77b4",
    openloop = "#d62728",
    S        = "#2ca02c",
    G        = "#ff7f0e",
    ci_enkf  = "#1f77b4",
    ci_S     = "#2ca02c",
)


def _date_axis(ax, freq="2Y"):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)


def _save(fig, out_dir: Path, filename: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Top-N basins grid
#     One figure, N rows × 3 cols
#     Col 0: Q time series (obs / open-loop / EnKF + CI)
#     Col 1: S time series (open-loop / EnKF + CI)
#     Col 2: obs vs EnKF scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_basins_grid(
    top_data: dict,          # basin_id → (Q_arr, result_dict, metrics_row)
    time_index: pd.DatetimeIndex,
    out_dir: Path,
):
    """
    Parameters
    ----------
    top_data   : ordered dict, length = N_PLOT_BASINS
    time_index : shared DatetimeIndex
    out_dir    : where to save
    """
    n = len(top_data)
    if n == 0:
        print("  No basins to plot.")
        return

    fig_h = max(4, n * 2.8)
    fig, axes = plt.subplots(n, 3, figsize=(18, fig_h),
                             gridspec_kw={"wspace": 0.35, "hspace": 0.55})

    # ensure axes is always 2-D even for n=1
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (basin_id, (Q_arr, result, mrow)) in enumerate(top_data.items()):

        nse_enkf = mrow["NSE_enkf"]
        nse_ol   = mrow["NSE_ol"]
        kge_enkf = mrow["KGE_enkf"]

        # ── Col 0: Streamflow ─────────────────────────────────────────────
        ax = axes[row, 0]
        ax.fill_between(time_index,
                        result["post_Q_ci"][:, 0], result["post_Q_ci"][:, 1],
                        alpha=0.20, color=COLORS["ci_enkf"])
        ax.plot(time_index, Q_arr,
                color=COLORS["obs"], lw=1.0, label="Obs", zorder=4)
        ax.plot(time_index, result["openloop_Q"],
                color=COLORS["openloop"], lw=0.9, linestyle="--",
                label=f"OL NSE={nse_ol:.2f}")
        ax.plot(time_index, result["post_Q_mean"],
                color=COLORS["enkf"], lw=1.2,
                label=f"EnKF NSE={nse_enkf:.2f}")
        ax.set_ylabel("Q (mm/mo)", fontsize=8)
        ax.set_title(f"{basin_id}  KGE={kge_enkf:.2f}", fontsize=8, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.7)
        ax.grid(alpha=0.25)
        _date_axis(ax)

        # ── Col 1: Soil moisture ──────────────────────────────────────────
        ax = axes[row, 1]
        ax.fill_between(time_index,
                        result["post_S_ci"][:, 0], result["post_S_ci"][:, 1],
                        alpha=0.20, color=COLORS["ci_S"])
        ax.plot(time_index, result["openloop_S"],
                color=COLORS["openloop"], lw=0.9, linestyle="--", label="OL")
        ax.plot(time_index, result["post_S_mean"],
                color=COLORS["S"], lw=1.2, label="EnKF")
        ax.set_ylabel("S (mm)", fontsize=8)
        ax.set_title("Soil moisture", fontsize=8)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.7)
        ax.grid(alpha=0.25)
        _date_axis(ax)

        # ── Col 2: Scatter ────────────────────────────────────────────────
        ax = axes[row, 2]
        mask = np.isfinite(Q_arr) & np.isfinite(result["post_Q_mean"])
        lim  = max(np.nanmax(Q_arr[mask]), np.nanmax(result["post_Q_mean"][mask])) * 1.05
        ax.scatter(Q_arr[mask], result["post_Q_mean"][mask],
                   s=6, alpha=0.5, color=COLORS["enkf"])
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Obs Q", fontsize=7)
        ax.set_ylabel("EnKF Q", fontsize=7)
        ax.set_title(f"NSE={nse_enkf:.2f}", fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)

    # Column headers (only on first row)
    for ax, title in zip(axes[0], ["Streamflow", "Soil Moisture", "Obs vs EnKF"]):
        ax.set_title(title + "\n" + axes[0, list(axes[0]).index(ax)].get_title(),
                     fontsize=9, fontweight="bold")

    fig.suptitle(f"Top-{n} basins by EnKF NSE", fontsize=13, fontweight="bold", y=1.002)
    _save(fig, out_dir, "top_basins_grid.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Global summary  (all basins, aggregated)
# ─────────────────────────────────────────────────────────────────────────────

def plot_global_summary(
    metrics_df: pd.DataFrame,
    basin_results: dict,        # basin_id → (Q_arr, P_arr, PET_arr, result)
    time_index: pd.DatetimeIndex,
    out_dir: Path,
    scatter_sample: int = 5000,
):
    """
    Four-panel figure over ALL basins:
      [0,0] NSE distribution violin  open-loop vs EnKF
      [0,1] KGE distribution violin  open-loop vs EnKF
      [1,0] NSE gain histogram
      [1,1] Pooled obs vs EnKF scatter (random sample across all basins)
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                             gridspec_kw={"hspace": 0.40, "wspace": 0.35})

    # ── Panel [0,0]: NSE violin ───────────────────────────────────────────
    ax = axes[0, 0]
    nse_ol   = metrics_df["NSE_ol"].dropna().values
    nse_enkf = metrics_df["NSE_enkf"].dropna().values

    vp = ax.violinplot([nse_ol, nse_enkf], positions=[1, 2],
                       showmedians=True, showextrema=True)
    vp["bodies"][0].set_facecolor(COLORS["openloop"]); vp["bodies"][0].set_alpha(0.6)
    vp["bodies"][1].set_facecolor(COLORS["enkf"]);     vp["bodies"][1].set_alpha(0.6)
    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        vp[part].set_color("black"); vp[part].set_linewidth(1.0)

    # overlay boxplot for quartile detail
    bp = ax.boxplot([nse_ol, nse_enkf], positions=[1, 2],
                    widths=0.12, patch_artist=False,
                    medianprops=dict(color="black", lw=2),
                    whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
                    flierprops=dict(marker=".", ms=2, alpha=0.3))

    ax.axhline(0, color="grey", lw=0.8, linestyle="--")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Open-loop", "EnKF"], fontsize=10)
    ax.set_ylabel("NSE", fontsize=11); ax.set_title("NSE Distribution", fontsize=11, fontweight="bold")
    ax.text(1, np.nanmedian(nse_ol)   + 0.02, f"med={np.nanmedian(nse_ol):.3f}",
            ha="center", fontsize=8, color=COLORS["openloop"])
    ax.text(2, np.nanmedian(nse_enkf) + 0.02, f"med={np.nanmedian(nse_enkf):.3f}",
            ha="center", fontsize=8, color=COLORS["enkf"])
    ax.grid(axis="y", alpha=0.3)

    # ── Panel [0,1]: KGE violin ───────────────────────────────────────────
    ax = axes[0, 1]
    kge_ol   = metrics_df["KGE_ol"].dropna().values
    kge_enkf = metrics_df["KGE_enkf"].dropna().values

    vp2 = ax.violinplot([kge_ol, kge_enkf], positions=[1, 2],
                        showmedians=True, showextrema=True)
    vp2["bodies"][0].set_facecolor(COLORS["openloop"]); vp2["bodies"][0].set_alpha(0.6)
    vp2["bodies"][1].set_facecolor(COLORS["enkf"]);     vp2["bodies"][1].set_alpha(0.6)
    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        vp2[part].set_color("black"); vp2[part].set_linewidth(1.0)
    ax.boxplot([kge_ol, kge_enkf], positions=[1, 2],
               widths=0.12, patch_artist=False,
               medianprops=dict(color="black", lw=2),
               whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
               flierprops=dict(marker=".", ms=2, alpha=0.3))
    ax.axhline(0, color="grey", lw=0.8, linestyle="--")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Open-loop", "EnKF"], fontsize=10)
    ax.set_ylabel("KGE", fontsize=11); ax.set_title("KGE Distribution", fontsize=11, fontweight="bold")
    ax.text(1, np.nanmedian(kge_ol)   + 0.02, f"med={np.nanmedian(kge_ol):.3f}",
            ha="center", fontsize=8, color=COLORS["openloop"])
    ax.text(2, np.nanmedian(kge_enkf) + 0.02, f"med={np.nanmedian(kge_enkf):.3f}",
            ha="center", fontsize=8, color=COLORS["enkf"])
    ax.grid(axis="y", alpha=0.3)

    # ── Panel [1,0]: NSE gain histogram ──────────────────────────────────
    ax = axes[1, 0]
    gains = metrics_df["NSE_gain"].dropna().values
    n_pos = (gains > 0).sum()
    n_neg = (gains <= 0).sum()
    ax.hist(gains, bins=40, color=COLORS["enkf"], alpha=0.75, edgecolor="white", lw=0.4)
    ax.axvline(0,             color="black", lw=1.0, linestyle="--")
    ax.axvline(np.median(gains), color="red", lw=1.5, linestyle="-",
               label=f"Median gain = {np.median(gains):+.3f}")
    ax.set_xlabel("NSE gain  (EnKF − Open-loop)", fontsize=11)
    ax.set_ylabel("Number of basins", fontsize=11)
    ax.set_title("NSE Improvement", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.text(0.97, 0.95, f"Improved: {n_pos}/{len(gains)} basins",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color="green" if n_pos > n_neg else "red")
    ax.grid(axis="y", alpha=0.3)

    # ── Panel [1,1]: Pooled scatter (random sample) ───────────────────────
    ax = axes[1, 1]
    rng  = np.random.default_rng(0)
    all_obs, all_sim = [], []
    for bid, (Q_arr, _, _, result) in basin_results.items():
        mask = np.isfinite(Q_arr) & np.isfinite(result["post_Q_mean"])
        all_obs.append(Q_arr[mask])
        all_sim.append(result["post_Q_mean"][mask])
    all_obs = np.concatenate(all_obs)
    all_sim = np.concatenate(all_sim)

    if len(all_obs) > scatter_sample:
        idx = rng.choice(len(all_obs), scatter_sample, replace=False)
        all_obs = all_obs[idx]; all_sim = all_sim[idx]

    lim = max(np.nanmax(all_obs), np.nanmax(all_sim)) * 1.02
    ax.scatter(all_obs, all_sim, s=3, alpha=0.2, color=COLORS["enkf"])
    ax.plot([0, lim], [0, lim], "k--", lw=1.0, label="1:1")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Observed Q (mm/month)", fontsize=11)
    ax.set_ylabel("EnKF Q (mm/month)", fontsize=11)
    ax.set_title(f"Pooled Obs vs EnKF  (n={len(all_obs):,} samples)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Global EnKF Performance — All Basins", fontsize=14, fontweight="bold")
    _save(fig, out_dir, "global_summary.png")