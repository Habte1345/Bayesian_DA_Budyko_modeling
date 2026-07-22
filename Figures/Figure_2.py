from __future__ import annotations
import os
import io
import glob
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow.feather as feather
import requests
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════

ROOT = r"F:\Github_repos\Bayesian_DA_Budyko_modeling"
SIM  = os.path.join(ROOT, "Simulation_results", "Simulation_with_BASIN_CALIB_PARAMS")

RESULT_DIR_BASE   = os.path.join(SIM, "BASE_MODEL")
RESULT_DIR_BUDYKO = os.path.join(SIM, "BUDYKO_MODEL")
RESULT_DIR_DA     = os.path.join(SIM, "BUDYKO_DA")

SFET_PATH = os.path.join(ROOT, "data", "processed", "SFET.feather")
CAL_P     = os.path.join(ROOT, "SCE_cal_params", "calibrated_params.json")
OUT_DIR   = os.path.join(SIM, "figures")

BUDYKO_CSV_PATH = r"F:\Github_repos\Bayesian_DA_Budyko_modeling\data\attrs_SFET_geo_KGE.csv"

eps = 1e-6


# ══════════════════════════════════════════════════════════════════════════
# LOAD SFET (MONTHLY)
# ══════════════════════════════════════════════════════════════════════════

SFET = pd.read_feather(SFET_PATH)
SFET["time"] = pd.to_datetime(SFET["time"])
SFET = SFET.set_index("time").sort_index()
SFET.columns = SFET.columns.astype(str).str.strip()
SFET = SFET.loc[:, ~SFET.columns.duplicated()]

SFET_monthly = SFET.resample("MS").sum()


# ══════════════════════════════════════════════════════════════════════════
# COLUMN (a) DATA-PREP HELPERS -- unchanged
# ══════════════════════════════════════════════════════════════════════════

def _to_ms_index(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("time", "Date", "date", "TIME"):
        if col in df.columns:
            df = df.copy()
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            break
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Cannot find a datetime index or column.")
    df = df.copy()
    df.index = pd.to_datetime({"year": df.index.year, "month": df.index.month, "day": 1})
    return df


def _slice(df: pd.DataFrame, common_index: pd.DatetimeIndex) -> pd.DataFrame:
    ym_map = {(ts.year, ts.month): i for i, ts in enumerate(df.index)}
    rows = []
    for ts in common_index:
        key = (ts.year, ts.month)
        rows.append(df.iloc[ym_map[key]] if key in ym_map
                    else pd.Series(np.nan, index=df.columns))
    return pd.DataFrame(rows, index=common_index, columns=df.columns)


def _mmmon(arr, days, cap=500.0):
    v = np.asarray(arr, dtype=float) * np.asarray(days, dtype=float)
    v[v > cap] = np.nan
    return v


def _sfet_for_basin(sfet_monthly, gid, common_index):
    w = sfet_monthly.copy()
    if not isinstance(w.index, pd.DatetimeIndex):
        if "time" in w.columns:
            w["time"] = pd.to_datetime(w["time"])
            w = w.set_index("time")
        else:
            return np.full(len(common_index), np.nan)
    w.columns = w.columns.str.strip()
    if gid not in w.columns:
        return np.full(len(common_index), np.nan)
    col = w[gid]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    lkp = {}
    for ts, v in zip(col.index, col.values):
        key = (int(ts.year), int(ts.month))
        try:
            fv = float(v)
            if np.isfinite(fv):
                lkp[key] = fv
        except (TypeError, ValueError):
            pass
    return np.array([lkp.get((ts.year, ts.month), np.nan) for ts in common_index], dtype=float)


def _kge(obs, sim):
    m = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[m], sim[m]
    if len(o) < 3 or o.std() == 0:
        return np.nan
    r = float(np.corrcoef(o, s)[0, 1])
    return float(1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2
                              + (s.mean() / o.mean() - 1) ** 2))


def _seasonal(arr, months):
    return np.array([np.nanmean(arr[months == m]) for m in range(1, 13)])


def load_data_diagnostic(
    result_dir_base: str,
    result_dir_budyko: str,
    sfet_monthly: pd.DataFrame,
    calibrated_params_path: str,
    start_year: int = 2001,
    end_year: int = 2014,
) -> dict:
    t0 = pd.Timestamp(f"{start_year}-01-01")
    t1 = pd.Timestamp(f"{end_year}-12-31")
    common_index = pd.date_range(t0, t1, freq="MS")
    months_arr = np.array([d.month for d in common_index])

    with open(calibrated_params_path) as f:
        cal = json.load(f)

    base_ids = {
        os.path.basename(p).replace("results_BASE_", "").replace(".feather", "")
        for p in glob.glob(os.path.join(result_dir_base, "results_BASE_*.feather"))
    }
    bud_ids = {
        os.path.basename(p).replace("results_BUDYKO_", "").replace(".feather", "")
        for p in glob.glob(os.path.join(result_dir_budyko, "results_BUDYKO_*.feather"))
    }
    common_basins = sorted(base_ids & bud_ids)
    print(f"  Basins in both BASE and BUDYKO: {len(common_basins)}")

    per_basin_series = {}
    kge_ke_by_gid, kge_b_by_gid = {}, {}
    skipped = 0

    for gid in common_basins:
        try:
            db = _to_ms_index(pd.read_feather(os.path.join(result_dir_base, f"results_BASE_{gid}.feather")))
            dd = _to_ms_index(pd.read_feather(os.path.join(result_dir_budyko, f"results_BUDYKO_{gid}.feather")))
            db = _slice(db, common_index)
            dd = _slice(dd, common_index)
        except Exception:
            skipped += 1
            continue

        ke_val = None
        if gid in cal:
            for k in ("Ke", "ke", "KE"):
                if k in cal[gid]:
                    ke_val = float(cal[gid][k]); break
        if ke_val is None or "PET" not in db.columns or "ET_B" not in dd.columns:
            skipped += 1
            continue

        days = common_index.days_in_month.values.astype(float)
        et_ke = _mmmon(db["PET"].values * ke_val, days)
        et_b = _mmmon(dd["ET_B"].values, days)
        sfet = _sfet_for_basin(sfet_monthly, gid, common_index)

        if not np.any(np.isfinite(sfet)):
            skipped += 1
            continue

        per_basin_series[gid] = dict(ke=et_ke, b=et_b, sfet=sfet)
        kge_ke_by_gid[gid] = _kge(sfet, et_ke)
        kge_b_by_gid[gid] = _kge(sfet, et_b)

    print(f"  Valid basins: {len(per_basin_series)}  |  Skipped: {skipped}")

    return dict(
        common_index=common_index,
        months_arr=months_arr,
        per_basin_series=per_basin_series,
        kge_ke_by_gid=kge_ke_by_gid,
        kge_b_by_gid=kge_b_by_gid,
    )


def _select_representative_basin(data: dict, min_valid_months: int = 60) -> str:
    candidates = {
        gid: data["kge_b_by_gid"].get(gid, np.nan)
        for gid, series in data["per_basin_series"].items()
        if int(np.sum(np.isfinite(series["sfet"]))) >= min_valid_months
    }
    candidates = {gid: kge for gid, kge in candidates.items() if np.isfinite(kge)}
    if not candidates:
        raise RuntimeError(
            f"No basin has >= {min_valid_months} valid SFET months and a finite KGE_b. "
            "Lower min_valid_months or check data coverage."
        )
    best_gid = max(candidates, key=candidates.get)
    print(f"  Representative basin selected by highest KGE (Budyko series vs SFET): {best_gid} "
          f"(KGE = {candidates[best_gid]:.3f})")
    return best_gid


def _find_window(common_index, series, n_years=3, preferred_start_year=None):
    years = sorted(set(common_index.year))
    candidates = [y for y in years if y + n_years - 1 <= years[-1]]

    def _completeness(y0):
        start = pd.Timestamp(f"{y0}-01-01")
        end = pd.Timestamp(f"{y0 + n_years - 1}-12-31")
        mask = (common_index >= start) & (common_index <= end)
        vals = np.concatenate([series["ke"][mask], series["b"][mask], series["sfet"][mask]])
        return np.mean(np.isfinite(vals))

    if preferred_start_year is not None and preferred_start_year in candidates:
        y0 = preferred_start_year if _completeness(preferred_start_year) > 0.95 \
            else max(candidates, key=_completeness)
    else:
        y0 = max(candidates, key=_completeness)

    return pd.Timestamp(f"{y0}-01-01"), pd.Timestamp(f"{y0 + n_years - 1}-12-31")


def _normalized_bias_by_season(data: dict):
    SEASON_MONTHS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
    months_arr = data["months_arr"]
    season_of_month = {m: s for s, ms in SEASON_MONTHS.items() for m in ms}
    out = {s: {"ke": [], "b": []} for s in SEASON_ORDER}
    for gid, series in data["per_basin_series"].items():
        sfet = series["sfet"]
        for key in ("ke", "b"):
            model = series[key]
            valid = np.isfinite(sfet) & np.isfinite(model) & (sfet != 0)
            for i in np.where(valid)[0]:
                season = season_of_month[months_arr[i]]
                out[season][key].append((model[i] - sfet[i]) / sfet[i])
    for s in SEASON_ORDER:
        for key in ("ke", "b"):
            out[s][key] = np.array(out[s][key])
    return out


def _stack_all_basins(data: dict, key: str) -> np.ndarray:
    return np.vstack([series[key] for series in data["per_basin_series"].values()])


def _seas_mean_std_allbasins(data: dict, key: str):
    months_arr = data["months_arr"]
    stack = _stack_all_basins(data, key)
    per_basin = np.vstack([_seasonal(stack[i], months_arr) for i in range(stack.shape[0])])
    return np.nanmean(per_basin, axis=0), np.nanstd(per_basin, axis=0)


def _basin_means(data: dict, key: str) -> np.ndarray:
    return np.array([np.nanmean(series[key]) for series in data["per_basin_series"].values()])


# ══════════════════════════════════════════════════════════════════════════
# COLUMN (b) -- seasonal Budyko curves + real scatter from CSV
# ══════════════════════════════════════════════════════════════════════════

def Fu(AI, omega):
    AIc = np.maximum(AI, 1e-6)
    return 1 + AIc - (1 + AIc ** omega) ** (1.0 / omega)

AI_RANGE_B = np.concatenate(([0], np.linspace(0.01, 15, 199)))
DOM_LAND_COVERS = ["DBF", "CL/NVM", "MF", "WS + SL", "EF", "GL"]

SEASONS = ["DJF", "MAM", "JJA", "SON"]
OMEGA_SEASONAL_MEAN = pd.DataFrame({
    "season": SEASONS,
    "omega": [1.3, 1.9, 2.4, 1.5],
}).sort_values("omega")
SEASON_ORDER_B = OMEGA_SEASONAL_MEAN["season"].tolist()
SEASON_COLORS_B = {"DJF": "magenta", "MAM": "darkgreen", "JJA": "red", "SON": "black"}
SEASON_LABELS_B = {
    row["season"]: f"{row['season']} (\u03c9 = {row['omega']:.2f})"
    for _, row in OMEGA_SEASONAL_MEAN.iterrows()
}

MODEL_ITEMS_B = ["ET_ke", "ET_B", "ET_DA"]
MODEL_COLORS_B = {"ET_ke": "blue", "ET_B": "red", "ET_DA": "darkgreen"}
MODEL_MARKERS_B = {"ET_ke": "o", "ET_B": "^", "ET_DA": "s"}
MODEL_MAP_B = {
    "ET_ke": ("AI_ke", "EI_ke"),
    "ET_B":  ("AI_B",  "EI_B"),
    "ET_DA": ("AI_ass", "EI_ass"),
}

VEG_MEAN_SHIFT = [0.00, 0.20, -0.15, 0.35, -0.30, 0.10]
VEG_SD         = [0.18, 0.25, 0.22, 0.30, 0.35, 0.20]


def load_budyko_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dom_land_cover_short"] = df["dom_land_cover_short"].astype(str)
    return df

def _plot_budyko_panel_b(ax, filtered: pd.DataFrame, land_cover_label: str,
                          veg_index: int, rng: np.random.Generator,
                          panel_letter: str,
                          is_last_row: bool = False,
                          scatter_size: float = 20, scatter_alpha: float = 0.75):

    for ss in SEASON_ORDER_B:
        om = float(OMEGA_SEASONAL_MEAN.loc[OMEGA_SEASONAL_MEAN["season"] == ss, "omega"].values[0])
        EI_curve = Fu(AI_RANGE_B, om)
        ax.plot(AI_RANGE_B, EI_curve, color=SEASON_COLORS_B[ss], linewidth=0.9, zorder=2)

    for nm in MODEL_ITEMS_B:
        ai_col, ei_col = MODEL_MAP_B[nm]
        if ai_col in filtered.columns and ei_col in filtered.columns:
            plot_data = filtered[[ai_col, ei_col]].rename(columns={ai_col: "AI", ei_col: "EI"})
            plot_data = plot_data[np.isfinite(plot_data["AI"]) & np.isfinite(plot_data["EI"])]
            ax.scatter(plot_data["AI"], plot_data["EI"],
                       s=scatter_size, alpha=scatter_alpha,
                       color=MODEL_COLORS_B[nm], marker=MODEL_MARKERS_B[nm],
                       edgecolor="none", zorder=4)

    mean_shift = VEG_MEAN_SHIFT[veg_index]
    sd = VEG_SD[veg_index]
    omega_samples = [
        rng.normal(loc=mu + mean_shift, scale=sd, size=100)
        for mu in OMEGA_SEASONAL_MEAN["omega"].values
    ]
    omega_dist = np.concatenate(omega_samples)
    q01, q99 = np.quantile(omega_dist, [0.01, 0.99])

    iax = inset_axes(
        ax,
        width="50%",
        height="50%",
        loc="lower left",
        bbox_to_anchor=(0.5, 0.3, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    iax.patch.set_alpha(0)

    iax.hist(
        omega_dist,
        bins=20,
        density=False,
        color="orange",
        alpha=0.4,
        edgecolor="black",
        linewidth=0.5
    )

    iax.set_xlabel(r"$\omega_{MLR}(t)$", fontsize=9)
    iax.tick_params(axis="x", labelsize=8)
    iax.tick_params(axis="y", labelsize=8)

    for spine in ["top", "right"]:
        iax.spines[spine].set_visible(False)

    ax.plot([0, 1, 15], [0, 1, 1], color="blue", lw=3)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1.2)
    ax.set_title(f"({panel_letter})  {land_cover_label}", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("EI [-]", fontsize=12)

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.tick_params(axis="y", which="major", labelsize=10)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # ── X-axis handling -- kept as the LAST operations on this axis so
    # nothing executed afterward can undo them.
    if is_last_row:
        ax.set_xlabel("AI [-]", fontsize=12)
        ax.tick_params(axis="x", which="both", labelbottom=True, labeltop=False)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
        ax.tick_params(axis="x", which="major", labelsize=10)
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", labelbottom=False, labeltop=False)
# ══════════════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════════════

mpl.rcParams.update({
    "font.family"        : "Georgia",
    "font.size"           : 14,
    "axes.linewidth"      : 1.0,
    "axes.labelsize"      : 14,
    "axes.labelweight"    : "bold",
    "axes.titlesize"      : 15,
    "xtick.labelsize"     : 12,
    "ytick.labelsize"     : 12,
    "xtick.major.width"   : 1.0,
    "ytick.major.width"   : 1.0,
    "legend.fontsize"     : 12,
    "legend.framealpha"   : 0.95,
    "grid.linewidth"      : 0.35,
    "grid.color"          : "0.88",
    "axes.spines.top"     : False,
    "axes.spines.right"   : False,
    "savefig.dpi"         : 300,
    "savefig.bbox"        : "tight",
    "pdf.fonttype"        : 42,
})

PRODUCTS = {
    "sfet": dict(label=r"ET$_{SFET}$", color="black",   ls="-",  lw=1.5, marker=None),
    "ke":   dict(label=r"ET$_{B}$",    color="#f30af3", ls="--", lw=1.2, marker="s"),
    "b":    dict(label=r"ET$_{Ke}$",   color="#1528D8", ls="-",  lw=1.3, marker="^"),
}
MON = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
SHAD = 0.13


# ══════════════════════════════════════════════════════════════════════════
# COMBINED FIGURE
# ══════════════════════════════════════════════════════════════════════════

def plot_combined_6x2(
    diag_data: dict,
    budyko_csv_df: pd.DataFrame,
    out_dir: str,
    fname: str = "Figure_2",
    representative_gid: str | None = None,
    window_years: int = 3,
    preferred_start_year: int | None = None,
    random_seed: int = 123,
):
    if representative_gid is None:
        representative_gid = _select_representative_basin(diag_data, min_valid_months=60)
    series = diag_data["per_basin_series"][representative_gid]
    common_index = diag_data["common_index"]

    fig = plt.figure(figsize=(10, 14), dpi=300)
    gs = gridspec.GridSpec(6, 2, figure=fig, left=0.06, right=0.98,
                            top=0.96, bottom=0.06, hspace=0.55, wspace=0.28)

    ax_ts   = fig.add_subplot(gs[0, 0])
    ax_seas = fig.add_subplot(gs[1, 0])
    ax_reg  = fig.add_subplot(gs[2, 0])
    ax_kge  = fig.add_subplot(gs[3, 0])
    ax_scat = fig.add_subplot(gs[4, 0])
    ax_bm   = fig.add_subplot(gs[5, 0])

    win_start, win_end = _find_window(common_index, series, n_years=window_years,
                                       preferred_start_year=preferred_start_year)
    mask = (common_index >= win_start) & (common_index <= win_end)
    t = common_index[mask]
    for key in ("sfet", "b", "ke"):
        p = PRODUCTS[key]
        ax_ts.plot(t, series[key][mask], color=p["color"], lw=p["lw"], ls=p["ls"],
                   label=p["label"], zorder=4 if key == "sfet" else 3)
    ax_ts.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_ts.set_ylabel("ET (mm/month)")
    ax_ts.set_ylim(bottom=0)
    ax_ts.grid(axis="y")
    ax_ts.legend(loc="upper left", frameon=1, ncol=1, handlelength=1.6)
    ax_ts.set_title("(a)", loc="left", fontweight="bold", fontsize=15, pad=4)
    plt.setp(ax_ts.get_xticklabels(), rotation=30, ha="right")

    SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
    bias = _normalized_bias_by_season(diag_data)
    positions_ke, positions_b, labels_center = [], [], []
    box_data, box_colors = [], []
    x0 = 1.0
    for season in SEASON_ORDER:
        positions_ke.append(x0); positions_b.append(x0 + 0.8); labels_center.append(x0 + 0.4)
        box_data.append(bias[season]["ke"]); box_colors.append(PRODUCTS["ke"]["color"])
        box_data.append(bias[season]["b"]);  box_colors.append(PRODUCTS["b"]["color"])
        x0 += 2.2
    all_positions = [p for pair in zip(positions_ke, positions_b) for p in pair]

    bp = ax_seas.boxplot(box_data, positions=all_positions, widths=0.65,
                         patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1.3))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color); patch.set_alpha(0.55); patch.set_edgecolor(color)
    ax_seas.axhline(0, color="0.3", lw=0.9, ls="--", zorder=1)
    ax_seas.set_xticks(labels_center); ax_seas.set_xticklabels(SEASON_ORDER)
    ax_seas.set_ylabel("Normalized bias")
    ax_seas.set_title("(b)", loc="left", fontweight="bold", fontsize=15, pad=4)
    ax_seas.grid(axis="y", alpha=0.3)
    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PRODUCTS["ke"]["color"],
               markersize=11, alpha=0.7, label=PRODUCTS["ke"]["label"]),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=PRODUCTS["b"]["color"],
               markersize=11, alpha=0.7, label=PRODUCTS["b"]["label"]),
    ]
    ax_seas.legend(handles=legend_handles, loc="upper right", frameon=False)

    x12 = np.arange(12)
    for key in ("sfet", "ke", "b"):
        p = PRODUCTS[key]
        m, s = _seas_mean_std_allbasins(diag_data, key)
        ax_reg.plot(x12, m, color=p["color"], lw=p["lw"], ls=p["ls"], label=p["label"], zorder=4)
        ax_reg.fill_between(x12, m - s, m + s, color=p["color"], alpha=SHAD, lw=0)
    ax_reg.set_xticks(x12); ax_reg.set_xticklabels(MON)
    # ax_reg.set_xlabel("Month"); 
    ax_reg.set_ylabel("ET (mm/month)")
    ax_reg.set_ylim(bottom=0)
    ax_reg.set_title("(c)", loc="left", fontweight="bold", fontsize=15, pad=4)
    ax_reg.grid(axis="y")
    ax_reg.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.6)

    kge_ke = np.array([v for v in diag_data["kge_ke_by_gid"].values() if np.isfinite(v)])
    kge_b = np.array([v for v in diag_data["kge_b_by_gid"].values() if np.isfinite(v)])
    for x, vals, color in [(1.0, kge_ke, PRODUCTS["ke"]["color"]), (2.0, kge_b, PRODUCTS["b"]["color"])]:
        parts = ax_kge.violinplot(vals, positions=[x], widths=0.55, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color); pc.set_edgecolor(color); pc.set_alpha(0.30)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax_kge.fill_betweenx([q1, q3], x - 0.15, x + 0.15, color=color, alpha=0.6, zorder=3)
        ax_kge.plot([x - 0.15, x + 0.15], [med, med], color="white", lw=2.0, zorder=4)
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(vals))
        ax_kge.scatter(np.full(len(vals), x) + jitter, vals, s=6, color=color,
                       alpha=0.35, zorder=2, linewidths=0)
        ax_kge.text(x, np.nanmax(vals) + 0.05, f"median = {med:.2f}",
                    ha="center", fontsize=13, fontweight="bold", color=color)
    ax_kge.axhline(0.5, color="0.4", lw=0.8, ls="--", zorder=1)
    ax_kge.set_xticks([1.0, 2.0]); ax_kge.set_xticklabels([PRODUCTS["ke"]["label"], PRODUCTS["b"]["label"]])
    ax_kge.set_ylabel(r"KGE vs. ET$_{SFET}$")
    ax_kge.set_title("(d)", loc="left", fontweight="bold", fontsize=15, pad=4)
    ax_kge.set_xlim(0.4, 2.6)
    ax_kge.grid(axis="y", alpha=0.3)

    common_gids = sorted(set(diag_data["kge_ke_by_gid"]) & set(diag_data["kge_b_by_gid"]))
    x = np.array([diag_data["kge_ke_by_gid"][g] for g in common_gids])
    y = np.array([diag_data["kge_b_by_gid"][g] for g in common_gids])
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    lo = min(np.nanmin(x), np.nanmin(y)) - 0.05
    hi = max(np.nanmax(x), np.nanmax(y)) + 0.05
    ax_scat.plot([lo, hi], [lo, hi], color="0.3", lw=1.0, ls="--", zorder=1)
    ax_scat.scatter(x, y, s=14, color=PRODUCTS["b"]["color"], alpha=0.85,
                    edgecolors="white", linewidths=0.3, zorder=3)
    ax_scat.set_xlim(-0.5, 1); ax_scat.set_ylim(-0.5, 1)
    ax_scat.set_xlabel(r"KGE(ET$_{B}$, ET$_{SFET}$)")
    ax_scat.set_ylabel(r"KGE(ET$_{Ke}$, ET$_{SFET}$)")
    ax_scat.set_title("(e)", loc="left", fontweight="bold", fontsize=15, pad=4)
    ax_scat.grid(True, alpha=0.25)

    sfet_bm = _basin_means(diag_data, "sfet")
    for key in ("ke", "b"):
        p = PRODUCTS[key]
        yv = _basin_means(diag_data, key)
        mm = np.isfinite(sfet_bm) & np.isfinite(yv)
        ax_bm.scatter(sfet_bm[mm], yv[mm], s=6, color=p["color"], alpha=0.85,
                      marker=p["marker"], linewidths=0, zorder=3, label=p["label"])
    ax_bm.plot([0, 170], [0, 170], color="0.35", lw=0.8, ls="--", zorder=1)
    ax_bm.set_xlim(0, 170); ax_bm.set_ylim(0, 170)
    ax_bm.set_xlabel(r"ET$_{SFET}$ (mm/month)")
    ax_bm.set_ylabel(r"ET$_{model}$ (mm/month)")
    ax_bm.set_title("(f)", loc="left", fontweight="bold", fontsize=15, pad=4)
    ax_bm.grid(True, alpha=0.25)
    ax_bm.legend(loc="upper right", frameon=0, edgecolor="0.65", markerscale=1.5, ncols=1)

    rng = np.random.default_rng(random_seed)
    col_b_axes = [fig.add_subplot(gs[i, 1]) for i in range(6)]
    panel_letters = ["g", "h", "i", "j", "k", "l"]
    for i, lc in enumerate(DOM_LAND_COVERS):
        ax = col_b_axes[i]
        filtered = budyko_csv_df[budyko_csv_df["dom_land_cover_short"] == lc]
        _plot_budyko_panel_b(ax, filtered, lc, veg_index=i, rng=rng,
                             panel_letter=panel_letters[i],
                             is_last_row=(i == len(DOM_LAND_COVERS) - 1))   
    budyko_legend_handles = [
        Line2D([0], [0], color=SEASON_COLORS_B[ss], lw=2, label=SEASON_LABELS_B[ss])
        for ss in SEASONS
    ]
    budyko_legend_handles += [
        Line2D([0], [0], marker=MODEL_MARKERS_B["ET_ke"], color="none",
               markerfacecolor=MODEL_COLORS_B["ET_ke"], markersize=10, label=r"$ET_{ke}$"),
        Line2D([0], [0], marker=MODEL_MARKERS_B["ET_B"], color="none",
               markerfacecolor=MODEL_COLORS_B["ET_B"], markersize=10, label=r"$ET_{B}$"),
        Line2D([0], [0], marker=MODEL_MARKERS_B["ET_DA"], color="none",
               markerfacecolor=MODEL_COLORS_B["ET_DA"], markersize=10, label=r"$ET_{DA}$"),
    ]
    fig.legend(handles=budyko_legend_handles, loc="lower center", ncol=7,
               frameon=1, fontsize=11, bbox_to_anchor=(0.5, -0.01))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{fname}.png")
    fig.savefig(path, dpi=600)
    print(f"  Saved: {path}")
    print(f"  Representative basin used in column (a), row 1: {representative_gid} "
          f"(window {win_start.date()} to {win_end.date()})")
    plt.show(fig)
    return fig


# ══════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════

budyko_csv_df = load_budyko_csv(BUDYKO_CSV_PATH)

diag_data = load_data_diagnostic(
    result_dir_base        = RESULT_DIR_BASE,
    result_dir_budyko      = RESULT_DIR_BUDYKO,
    sfet_monthly           = SFET_monthly,
    calibrated_params_path = CAL_P,
    start_year = 2001,
    end_year   = 2008,
)

plot_combined_6x2(
    diag_data=diag_data,
    budyko_csv_df=budyko_csv_df,
    out_dir=OUT_DIR,
    representative_gid=None,
    window_years=5,
    preferred_start_year=None,
)


















# # ============================================================
# # Load libraries
# # ============================================================
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import font_manager
# import json

# # ============================================================
# # Font settings (Times New Roman)
# # ============================================================
# LABEL_FONTSIZE = 14

# font_path_regular = "C:/Windows/Fonts/times.ttf"
# font_path_bold    = "C:/Windows/Fonts/timesbd.ttf"

# font_manager.fontManager.addfont(font_path_regular)
# font_manager.fontManager.addfont(font_path_bold)

# plt.rcParams.update({
#     "font.family": "Times New Roman",
#     "font.size": LABEL_FONTSIZE,
# })

# # ============================================================
# # Load basin calibration CSV
# # ============================================================
# param_file_path = (
#     "F:/Github_repos/Bayesian_DA_Budyko_modeling/"
#     "SCE_cal_params/final_calibrated_params_with_KGE.csv"
# )

# df = pd.read_csv(param_file_path, index_col="Basin")

# # ============================================================
# # Prepare boxplot data
# # ============================================================
# params_to_plot = ["Kperc", "Kb", "Ke", "Cqq"]

# df_long = (
#     df[params_to_plot]
#     .reset_index(drop=True)
#     .melt(var_name="Parameter", value_name="Value")
# )

# df_long["Parameter"] = pd.Categorical(
#     df_long["Parameter"],
#     categories=params_to_plot,
#     ordered=True
# )

# df_stats = df_long.groupby("Parameter")["Value"].median().reset_index()

# # ============================================================
# # Load global calibration JSONs
# # ============================================================
# param_json_path = "F:/Github_repos/Bayesian_DA_Budyko_modeling/SCE_global_params/global_calibrated_params.json"
# summary_json_path = "F:/Github_repos/Bayesian_DA_Budyko_modeling/SCE_global_params/global_calibration_summary.json"

# with open(param_json_path, "r") as f:
#     global_params = json.load(f)

# with open(summary_json_path, "r") as f:
#     summary = json.load(f)

# param_names = ["Kperc", "Kb", "Ke", "Cqq"]
# param_values = [global_params[p] for p in param_names]

# kge_labels = ["Train Mean", "Train Median", "Test Mean", "Test Median"]
# kge_values = [
#     summary["mean_train_kge"],
#     summary["median_train_kge"],
#     summary["mean_test_kge"],
#     summary["median_test_kge"],
# ]

# # ============================================================
# # Create figure (3 panels)
# # ============================================================
# fig, axes = plt.subplots(1, 3, figsize=(10, 4))

# # Use JET colormap
# cmap = plt.cm.coolwarm

# # ============================================================
# # (a) Basin distributions
# # ============================================================
# ax = axes[0]

# box_data = [
#     df_long[df_long["Parameter"] == p]["Value"].dropna().values
#     for p in params_to_plot
# ]

# bp = ax.boxplot(
#     box_data,
#     vert=False,
#     patch_artist=True,
#     widths=0.4,
#     showfliers=False
# )

# colors_box = [cmap(i) for i in np.linspace(0, 1, len(params_to_plot))]

# for patch, color in zip(bp["boxes"], colors_box):
#     patch.set_facecolor(color)
#     patch.set_edgecolor("black")

# for element in ["whiskers", "caps", "medians"]:
#     for line in bp[element]:
#         line.set_color("black")
#         line.set_linewidth(1.5)

# # median labels
# for i, (param, median) in enumerate(zip(df_stats["Parameter"], df_stats["Value"])):
#     ax.text(
#         median,
#         i + 1.3,
#         f"{median:.2f}",
#         ha="center",
#         va="bottom",
#         fontsize=LABEL_FONTSIZE + 2,
#         fontweight="bold",
#         color="blue"
#     )

# ax.set_yticks(np.arange(1, len(params_to_plot) + 1))
# ax.set_yticklabels(params_to_plot)
# ax.set_xlabel("Basin-scale [505 Basins]", fontsize=LABEL_FONTSIZE+4, fontweight="bold")
# ax.tick_params(axis="both", labelsize=LABEL_FONTSIZE+2)
# ax.set_title("(a)", loc="left", fontsize=LABEL_FONTSIZE+8)

# # ============================================================
# # (b) Global parameters
# # ============================================================
# ax = axes[1]

# colors_global = [cmap(i) for i in np.linspace(0, 1, len(param_names))]

# ax.barh(param_names, param_values, color=colors_global)

# for i, v in enumerate(param_values):
#     ax.text(v, i, f"{v:.2f}", va="center", fontsize=LABEL_FONTSIZE+2)

# ax.set_xlabel("Global [Test basins=151]", fontsize=LABEL_FONTSIZE+4, fontweight="bold")
# ax.set_xlim(0, 1)
# ax.tick_params(axis="both", labelsize=LABEL_FONTSIZE+2)
# ax.set_title("(b)", loc="left", fontsize=LABEL_FONTSIZE+8)

# # ============================================================
# # (c) KGE performance
# # ============================================================
# ax = axes[2]

# colors_kge = 'grey'

# ax.barh(kge_labels, kge_values, color=colors_kge)

# for i, v in enumerate(kge_values):
#     ax.text(v, i, f"{v:.2f}", va="center", fontsize=LABEL_FONTSIZE+2)

# ax.set_xlabel("KGE", fontsize=LABEL_FONTSIZE+4, fontweight="bold")
# ax.set_xlim(0, 0.7)
# ax.tick_params(axis="both", labelsize=LABEL_FONTSIZE+2)
# ax.set_title("(c)", loc="left", fontsize=LABEL_FONTSIZE+8)

# # ============================================================
# # Remove top & right borders
# # ============================================================
# for ax in axes:
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

# # ============================================================
# # Final layout
# # ============================================================
# plt.tight_layout()
# plt.show()