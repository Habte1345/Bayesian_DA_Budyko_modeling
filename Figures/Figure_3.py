"""
plot_basin_dashboard_6rows.py
==============================
Six-basin dashboard — one row per basin, four panels per row.

Layout per row:  [ts (wide)] [scatter] [FDC] [seasonal]
                  col 0        col 1    col 2   col 3
                  width_ratio  1.8       1       1       1.2

All panel functions and styling are identical to plot_basin_Params.
Only the figure layout changes: 6 rows × 4 columns instead of 2 rows.

Call
----
    top6 = (attrs_df.dropna(subset=["KGE_DA"])
                    .sort_values("KGE_DA", ascending=False)
                    .head(6).reset_index(drop=True))
    plot_top6_dashboard(top6, result_dir_base, result_dir_da, out_dir)
"""

from __future__ import annotations
import os, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "legend.framealpha": 0.93, "legend.edgecolor": "0.65",
    "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 1.0, "xtick.major.size": 4,
    "ytick.major.width": 1.0, "ytick.major.size": 4,
    "grid.linewidth": 0.5, "grid.color": "0.87",
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

C  = dict(obs="#111111", base="#E34B0F", bud="#1565C0", da="#10A718", ribbon="#08450B")
LW = dict(obs=2.0, base=1.5, bud=1.5, da=2.0)
LS = dict(obs="-.", base="--", bud=":", da="-")
MON = ["O","N","D","J","F","M","A","M","J","J","A","S"]


# ── Helpers (unchanged) ───────────────────────────────────────────

def _load(path):
    df = pd.read_feather(path)
    for col in ("time", "Date", "date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            return df.set_index(col)
    return df

def _get(df, *cols):
    for c in cols:
        if c in df.columns:
            return df[c].values.astype(float)
    return np.full(len(df), np.nan)

def _align(v, si, ti):
    return pd.Series(v, index=si).reindex(ti).values.astype(float)

def _kge(o, s):
    m = np.isfinite(o) & np.isfinite(s); o, s = o[m], s[m]
    if len(o) < 3 or o.std() == 0: return np.nan
    r = np.corrcoef(o, s)[0, 1]
    return float(1 - np.sqrt((r-1)**2 + (s.std()/o.std()-1)**2 + (s.mean()/o.mean()-1)**2))

def _nse(o, s):
    m = np.isfinite(o) & np.isfinite(s); o, s = o[m], s[m]
    d = ((o - o.mean())**2).sum()
    return float(1 - ((o-s)**2).sum()/d) if d > 0 and len(o) > 2 else np.nan


# ── Panel functions (unchanged) ───────────────────────────────────

def _ts(ax, t, obs, base, bud, da, lo, hi, first_row=False):
    ax.fill_between(t, lo, hi, color=C["ribbon"], alpha=0.20, zorder=1)
    ax.plot(t, base, color=C["base"], lw=LW["base"], ls=LS["base"], zorder=2)
    ax.plot(t, bud,  color=C["bud"],  lw=LW["bud"],  ls=LS["bud"],  zorder=3)
    ax.plot(t, da,   color=C["da"],   lw=LW["da"],   ls=LS["da"],   zorder=4)
    ax.plot(t, obs,  color=C["obs"],  lw=LW["obs"],  ls=LS["obs"],  zorder=5)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(bottom=0)
    ax.set_ylabel("Q (mm/month)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y")
    if not first_row:
        plt.setp(ax.get_xticklabels(), visible=False)


def _scatter(ax, obs, base, bud, da, kge_b, kge_bud, kge_da):
    lim = max(np.nanmax([obs, base, bud, da]) * 1.06, 1.0)
    for sim, color, marker in [
        (base, C["bud"], "o"),
        (bud,  C["base"],  "s"),
        (da,   C["da"],   "^"),
    ]:
        m = np.isfinite(obs) & np.isfinite(sim)
        ax.scatter(obs[m], sim[m], s=30, color=color, marker=marker,
                   alpha=0.85, linewidths=0, zorder=3)
    ax.plot([0, lim], [0, lim], color="0.25", lw=1.0, ls="--", zorder=5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal"); ax.grid(True)
    ax.set_xlabel("Obs Q"); ax.set_ylabel("Sim Q")
    for txt, col, y0 in [
        (f"KGE={kge_bud:.2f}", C["base"],  0.97),
        (f"KGE={kge_b:.2f}",   C["bud"], 0.87),
        (f"KGE={kge_da:.2f}",  C["da"],   0.77),
    ]:
        ax.text(0.04, y0, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, color=col, fontfamily="monospace", fontweight="bold")


def _fdc(ax, obs, base, bud, da):
    def _f(x):
        v = np.sort(x[np.isfinite(x)])[::-1]
        return np.arange(1, len(v)+1)/(len(v)+1)*100, v
    for sim, color, ls, lw in [
        (obs,  C["obs"],  "-.", 2.0),
        (base, C["bud"], "--", 1.5),
        (bud,  C["base"],  ":",  1.5),
        (da,   C["da"],   "-",  2.0),
    ]:
        p, v = _f(sim)
        if len(v): ax.semilogy(p, v, color=color, lw=lw, ls=ls, zorder=3)
    ax.set_xlabel("Exceedance (%)"); ax.set_ylabel("Q (log)")
    ax.set_xlim(0, 100); ax.grid(True, which="both", alpha=0.35)


def _clim(ax, t, obs, base, bud, da):
    mon  = np.array([d.month for d in t])
    xpos = np.array([(m - 10) % 12 for m in mon])
    for sim, color, ls, lw, marker, label in [
        (obs,  C["obs"],  "-.", 2.0, "o",  "Observed"),
        (base, C["bud"], "--", 1.5, None, "BASE"),
        (bud,  C["base"],  ":",  1.5, None, "Budyko"),
        (da,   C["da"],   "-",  2.0, "^",  "BUDYKO-DA"),
    ]:
        cl = np.array([np.nanmean(sim[xpos == x]) for x in range(12)])
        ax.plot(range(12), cl, color=color, lw=lw, ls=ls,
                marker=marker, ms=4, label=label, zorder=3)
    ax.set_xticks(range(12)); ax.set_xticklabels(MON, fontsize=10)
    ax.set_xlim(-0.5, 11.5); ax.set_ylim(bottom=0)
    ax.set_xlabel("Month"); ax.set_ylabel("Mean Q")
    ax.grid(axis="y")
    # ax.legend(loc="upper right", fontsize=8, frameon=False, handlelength=1.5)



# ── Multi-basin figure ────────────────────────────────────────────

def plot_top6_dashboard(
    top6_df: pd.DataFrame,
    result_dir_base:   str,
    result_dir_budyko: str,
    result_dir_da:     str,
    out_dir: str,
    start_year: int = 2005,
    end_year:   int = 2014,
    fname: str = "fig_basin_dashboard_6rows",
):
    """
    One row per basin, four panels per row.
    Filters basins where KGE >= 0.5 in ALL three scenarios.
    top6_df must have columns: gauge_id, KGE_Base, KGE_Budyko, KGE_DA,
    and optionally: AI_ass/aridity, EI_base/EI_ass, dom_land_cover_short.
    """
    n_rows = len(top6_df)

    fig = plt.figure(figsize=(18.0, n_rows * 2.6 + 0.6))

    gs = gridspec.GridSpec(
        n_rows, 4, figure=fig,
        left=0.06, right=0.99,
        top=0.97,  bottom=0.05,
        hspace=0.38, wspace=0.28,
        width_ratios=[2.2, 1, 1, 1.2],
    )

    # Shared legend on the very first row time-series panel
    legend_drawn = False

    for row_i, (_, mrow) in enumerate(top6_df.iterrows()):
        gid = str(mrow["gauge_id"]).zfill(8)

        # Load data
        try:
            db  = _load(os.path.join(result_dir_base,   f"results_BASE_{gid}.feather"))
            dd  = _load(os.path.join(result_dir_budyko, f"results_BUDYKO_{gid}.feather"))
            da  = _load(os.path.join(result_dir_da,     f"results_BUDYKO_DA_{gid}.feather"))
            ens = _load(os.path.join(result_dir_da,     f"enkf_ensemble_BUDYKO_DA_{gid}.feather"))
        except FileNotFoundError as e:
            print(f"  ⚠  {gid}: {e}"); continue

        t_all = ens.loc[f"{start_year}":f"{end_year}-12-31"].index
        obs   = _align(_get(db,  "Q_obs"),              db.index,  t_all)
        base  = _align(_get(db,  "Q_base", "Q_ke"),     db.index,  t_all)
        bud_q = _align(_get(dd,  "Q_budyko", "Q_B"),    dd.index,  t_all)
        da_q  = _align(_get(da,  "Q_ass"),               da.index,  t_all)
        q_lo  = _align(_get(ens, "Q_ens_p05"),          ens.index, t_all)
        q_hi  = _align(_get(ens, "Q_ens_p95"),          ens.index, t_all)

        kge_b   = float(mrow.get("KGE_Base",   _kge(obs, base)))
        kge_bud = float(mrow.get("KGE_Budyko", _kge(obs, bud_q)))
        kge_da  = float(mrow.get("KGE_DA",     _kge(obs, da_q)))
        nse_b   = _nse(obs, base)
        nse_da  = _nse(obs, da_q)

        # Axes
        ax_ts  = fig.add_subplot(gs[row_i, 0])
        ax_sc  = fig.add_subplot(gs[row_i, 1])
        ax_fdc = fig.add_subplot(gs[row_i, 2])
        ax_cl  = fig.add_subplot(gs[row_i, 3])

        # Draw panels
        is_last = (row_i == n_rows - 1)
        _ts(ax_ts, t_all, obs, base, bud_q, da_q, q_lo, q_hi, first_row=is_last)
        _scatter(ax_sc, obs, base, bud_q, da_q, kge_b, kge_bud, kge_da)
        _fdc(ax_fdc, obs, base, bud_q, da_q)
        _clim(ax_cl, t_all, obs, base, bud_q, da_q)

        # Row title: basin ID, vegetation, AI — then colour-coded KGE per scenario
        ai  = mrow.get("AI_ass",  mrow.get("aridity",  np.nan))
        veg = str(mrow.get("dom_land_cover_short",
                            mrow.get("dom_land_cover", "—")))
        ai_s = f"{float(ai):.2f}" if pd.notna(ai) else "—"

        # Plain text prefix
        ax_ts.set_title(
            f"USGS: {gid}  |  Veg: {veg}  |  AI: {ai_s}",
            loc="left", fontsize=10, fontweight="bold", pad=3, color="0.15",
        )
        # Colour-coded KGE values — no subscripts, color identifies scenario
        for x_off, kge_val, col, txt in [
            (0.50, kge_bud, C["base"], f"KGE = {kge_bud:.2f}"),
            (0.65, kge_b,   C["bud"],  f"| KGE = {kge_b:.2f}"),
            (0.80, kge_da,  C["da"],   f"| KGE = {kge_da:.2f}"),
        ]:
            ax_ts.text(
                x_off, 1.04, txt,
                transform=ax_ts.transAxes,
                fontsize=9.5, fontweight="bold",
                color=col, va="bottom", ha="left", clip_on=False
            )
        # Panel labels (a–d) on first row only
        if row_i == 0:
            for ax, lbl in zip([ax_ts, ax_sc, ax_fdc, ax_cl],
                                ["(a)", "(b)", "(c)", "(d)"]):
                ax.set_title(lbl, loc="right", fontsize=16,
                             fontweight="bold", pad=3, color="blue")

        # Shared legend on first row time-series panel
        if row_i == 1 and not legend_drawn:
            ax_ts.legend(
                handles=[
                    Line2D([0],[0], color=C["obs"],  lw=2.0, ls="-.",
                           label="Observed"),
                    Line2D([0],[0], color=C["bud"],  lw=1.5, ls=":",
                           label="Budyko"),
                    Line2D([0],[0], color=C["base"], lw=1.5, ls="--",
                           label="BASE"),
                    Line2D([0],[0], color=C["da"],   lw=2.0, ls="-",
                           label="BUDYKO-DA"),
                    Patch(fc=C["ribbon"], alpha=0.35, label="DA 5–95%"),
                ],
                loc="upper center", ncol=2, fontsize=10,
                handlelength=1.8, frameon=False, edgecolor="0.65",
            )
            legend_drawn = True

        print(f"  ✓  row {row_i+1}  {gid}")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{fname}.png")
    fig.savefig(path, dpi=300)
    print(f"\n  ✓  {path}")
    plt.show()
    return fig


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":

    ROOT = r"F:\Github_repos\Bayesian_DA_Budyko_modeling"
    SIM  = os.path.join(ROOT, "Simulation_results",
                        "Simulation_with_BASIN_CALIB_PARAMS")

    df = attrs_all_simu_mean_505_basins_BASIN_CAL
    top6 = (
        df.dropna(subset=["KGE_Base","KGE_Budyko","KGE_DA"])
        .loc[
            (df["KGE_Base"]   >= 0.50) &
            (df["KGE_Budyko"] >= 0.50) &
            (df["KGE_DA"]     >= 0.50)
        ]
        .sort_values("KGE_DA", ascending=False)
        .head(6)
        .reset_index(drop=True)
    )
    print(f"Basins with KGE >= 0.50 in all scenarios: {len(top6)}")

    plot_top6_dashboard(
        top6_df           = top6,
        result_dir_base   = os.path.join(SIM, "BASE_MODEL"),
        result_dir_budyko = os.path.join(SIM, "BUDYKO_MODEL"),
        result_dir_da     = os.path.join(SIM, "BUDYKO_DA"),
        out_dir           = os.path.join(SIM, "figures"),
    )