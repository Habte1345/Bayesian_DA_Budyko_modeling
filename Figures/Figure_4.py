"""
plot_et_validation_mean505.py
==============================
Mean-of-505-basins ET validation figure.

Layout  (1 row × 4 panels)
--------------------------
(a) Mean monthly time series  2001–2014   ± 1σ shading
(b) Mean seasonal regime curve Jan–Dec    ± 1σ shading
(c) Scatter: 12 monthly means vs SFET     one point per calendar month
(d) KGE violin + box: all basins          three ET products vs SFET

ET products
-----------
ET_sac  – SAC-SMA actual ET  → column "ET_ke" in BASE feather   (mm/day × days)
ET_ke   – Ke × PET baseline  → cal["Ke"] × PET column           (mm/day × days)
ET_B    – Budyko ET           → column "ET_B"  in BUDYKO feather (mm/day × days)
SFET    – reference           → SFTET_monthly (DatetimeIndex, mm/month already)

Requirements
------------
SFTET_monthly : pd.DataFrame
    DatetimeIndex (MS), basin IDs as columns, values in mm/month.
    No trailing spaces on columns. Starts March 2000 or earlier.
    Build it as:
        SFTET = pd.read_feather(...SFET.feather...)
        SFTET.columns = SFTET.columns.str.strip()
        SFTET["time"] = pd.to_datetime(SFTET["time"])
        SFTET = SFTET.set_index("time").sort_index()
        SFTET_monthly = SFTET.loc["2000-01-01":"2014-12-31"].resample("MS").sum(min_count=20)
        # Do NOT reset_index — keep DatetimeIndex

Call
----
    data = load_data_mean505(
        result_dir_base        = ...,
        result_dir_budyko      = ...,
        sfet_monthly           = SFTET_monthly,
        calibrated_params_path = ...,
        start_year = 2001,
        end_year   = 2014,
    )
    plot_et_validation_mean505(data, out_dir=...)
"""

from __future__ import annotations
import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

import pandas as pd

SFTET = pd.read_feather(
    r"F:\Github_repos\Bayesian_DA_Budyko_modeling\data\processed\SFET.feather"
)

# Convert time to datetime
SFTET["time"] = pd.to_datetime(SFTET["time"])

# Set time as index
SFTET = SFTET.set_index("time").sort_index()

# Keep only 2000–2014
SFTET_2000_2014 = SFTET.loc["2000-01-01":"2014-12-31"]

# Aggregate daily ET to monthly ET
SFTET_monthly = SFTET_2000_2014.resample("MS").sum()

# Force complete monthly time index: Jan 2000 to Dec 2014
monthly_index = pd.date_range("2000-01-01", "2014-12-01", freq="MS")
SFTET_monthly = SFTET_monthly.reindex(monthly_index)

# Put time back as a column
SFTET_monthly = SFTET_monthly.reset_index().rename(columns={"index": "time"})

SFTET_monthly.columns

warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family"       : "sans-serif",
    "font.sans-serif"   : ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size"         : 8,
    "axes.titlesize"    : 9,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 8,
    "xtick.labelsize"   : 7.5,
    "ytick.labelsize"   : 7.5,
    "legend.fontsize"   : 7.5,
    "legend.framealpha" : 0.95,
    "axes.linewidth"    : 0.6,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "xtick.major.width" : 0.6,
    "xtick.major.size"  : 2.5,
    "ytick.major.width" : 0.6,
    "ytick.major.size"  : 2.5,
    "grid.linewidth"    : 0.35,
    "grid.color"        : "0.88",
    "savefig.dpi"       : 300,
    "savefig.bbox"      : "tight",
    "pdf.fonttype"      : 42,
})

C  = dict(sfet="black", sac="#D84315", ke="darkgreen", b="blue")
LW = dict(sfet=1.4,       sac=0.95,     ke=0.95,      b=1.1)
LS = dict(sfet="-",       sac="--",     ke="-.",       b="-")
SHAD = 0.11
MON  = ["J","F","M","A","M","J","J","A","S","O","N","D"]


# ══════════════════════════════════════════════════════════════════════════════
# PURE HELPER FUNCTIONS  (no side-effects, no file I/O)
# ══════════════════════════════════════════════════════════════════════════════

def _to_ms_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a monthly MS DatetimeIndex.
    Handles:
      * 'time' / 'Date' / 'date' column  -> set as index
      * Already a DatetimeIndex           -> floor to day-1 of month
    Uses pd.to_datetime(year/month/day dict) to avoid to_period('MS')
    which is unsupported on some pandas versions.
    """
    for col in ("time", "Date", "date", "TIME"):
        if col in df.columns:
            df = df.copy()
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            break
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Cannot find a datetime index or column.")
    # Floor to first day of each month — works on all pandas versions
    df = df.copy()
    df.index = pd.to_datetime(
        {"year": df.index.year, "month": df.index.month, "day": 1}
    )
    return df


def _slice(df: pd.DataFrame,
           common_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Align df to common_index using (year, month) matching.
    Returns a DataFrame with exactly len(common_index) rows.
    Missing months are NaN.
    """
    # Build (year,month) → row mapping
    ym_map = {(ts.year, ts.month): i for i, ts in enumerate(df.index)}
    rows = []
    for ts in common_index:
        key = (ts.year, ts.month)
        if key in ym_map:
            rows.append(df.iloc[ym_map[key]])
        else:
            rows.append(pd.Series(np.nan, index=df.columns))
    return pd.DataFrame(rows, index=common_index, columns=df.columns)


def _mmmon(arr: np.ndarray, days: np.ndarray, cap: float = 500.0) -> np.ndarray:
    """mm/day × days_in_month → mm/month. Values > cap → NaN."""
    v = np.asarray(arr, dtype=float) * np.asarray(days, dtype=float)
    v[v > cap] = np.nan
    return v


def _sfet_for_basin(sfet_monthly: pd.DataFrame,
                    gid: str,
                    common_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Extract SFET mm/month series for basin gid aligned to common_index.
    Uses (year, month) lookup — immune to any timestamp offset or duplicates.
    """
    w = sfet_monthly.copy()

    # Ensure DatetimeIndex
    if not isinstance(w.index, pd.DatetimeIndex):
        if "time" in w.columns:
            w["time"] = pd.to_datetime(w["time"])
            w = w.set_index("time")
        else:
            return np.full(len(common_index), np.nan)

    w.columns = w.columns.str.strip()

    if gid not in w.columns:
        return np.full(len(common_index), np.nan)

    # Extract scalar values safely — squeeze to Series, iterate as floats
    col = w[gid]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]   # drop duplicate column

    # Build (year, month) -> float lookup; skip NaN rows
    lkp = {}
    for ts, v in zip(col.index, col.values):
        key = (int(ts.year), int(ts.month))
        try:
            fv = float(v)
            if np.isfinite(fv):
                lkp[key] = fv
        except (TypeError, ValueError):
            pass

    return np.array(
        [lkp.get((ts.year, ts.month), np.nan) for ts in common_index],
        dtype=float,
    )


def _kge(obs: np.ndarray, sim: np.ndarray) -> float:
    m = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[m], sim[m]
    if len(o) < 3 or o.std() == 0:
        return np.nan
    r = float(np.corrcoef(o, s)[0, 1])
    return float(1 - np.sqrt((r - 1)**2
                             + (s.std() / o.std() - 1)**2
                             + (s.mean() / o.mean() - 1)**2))


def _seasonal(arr: np.ndarray, months: np.ndarray) -> np.ndarray:
    """12-element mean seasonal cycle (Jan=1 … Dec=12)."""
    return np.array([np.nanmean(arr[months == m]) for m in range(1, 13)])


def _mean_std(stack: np.ndarray):
    """Column-wise mean and std across basins (axis=0)."""
    return np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)


def _seas_mean_std(stack: np.ndarray, months: np.ndarray):
    """
    Basin-average seasonal cycle mean and std.
    stack: (n_basins, L)
    Returns two (12,) arrays.
    """
    per_basin = np.vstack([_seasonal(stack[i], months)
                           for i in range(stack.shape[0])])
    return np.nanmean(per_basin, axis=0), np.nanstd(per_basin, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_data_mean505(
    result_dir_base: str,
    result_dir_budyko: str,
    sfet_monthly: pd.DataFrame,
    calibrated_params_path: str,
    start_year: int = 2001,
    end_year:   int = 2014,
) -> dict:
    """
    Load all matching BASE + BUDYKO feathers and compute cross-basin statistics.

    Returns a dict consumed by plot_et_validation_mean505().
    """
    t0 = pd.Timestamp(f"{start_year}-01-01")
    t1 = pd.Timestamp(f"{end_year}-12-31")
    common_index = pd.date_range(t0, t1, freq="MS")
    L = len(common_index)
    months_arr = np.array([d.month for d in common_index])

    with open(calibrated_params_path) as f:
        cal = json.load(f)

    # ── Find basins present in both scenario folders ───────────────────────
    base_ids = {
        os.path.basename(p).replace("results_BASE_", "").replace(".feather", "")
        for p in glob.glob(os.path.join(result_dir_base, "results_BASE_*.feather"))
    }
    bud_ids = {
        os.path.basename(p).replace("results_BUDYKO_", "").replace(".feather", "")
        for p in glob.glob(os.path.join(result_dir_budyko, "results_BUDYKO_*.feather"))
    }
    common_basins = sorted(base_ids & bud_ids)
    # print(f"  Basins in both BASE and BUDYKO: {len(common_basins)}")

    # ── SFET diagnostic (first basin only) ────────────────────────────────
    _test_gid = common_basins[0]
    _test_sfet = _sfet_for_basin(sfet_monthly, _test_gid, common_index)
    # print(f"  SFET diagnostic — basin {_test_gid}: "
    #       f"{int(np.sum(np.isfinite(_test_sfet)))}/{L} months, "
    #       f"mean={np.nanmean(_test_sfet):.2f} mm/month")

    # ── Per-basin loop ─────────────────────────────────────────────────────
    sac_stack   = []
    ke_stack    = []
    b_stack     = []
    sfet_stack  = []
    kge_sac_all = []
    kge_ke_all  = []
    kge_b_all   = []
    skipped = 0
    errors  = {}   # gid → error message, for first 5 failures

    for gid in common_basins:

        # ── Load and normalise both feathers ──────────────────────────
        try:
            db_raw = pd.read_feather(
                os.path.join(result_dir_base, f"results_BASE_{gid}.feather"))
            dd_raw = pd.read_feather(
                os.path.join(result_dir_budyko, f"results_BUDYKO_{gid}.feather"))
        except Exception as e:
            errors[gid] = f"read_feather: {e}"
            skipped += 1
            continue

        try:
            db = _to_ms_index(db_raw)
            dd = _to_ms_index(dd_raw)
        except Exception as e:
            errors[gid] = f"_to_ms_index: {e}"
            skipped += 1
            continue

        # ── Align to common_index via (year,month) matching ───────────
        try:
            db = _slice(db, common_index)
            dd = _slice(dd, common_index)
        except Exception as e:
            errors[gid] = f"_slice: {e}"
            skipped += 1
            continue

        # ── Build ET arrays ───────────────────────────────────────────
        days = common_index.days_in_month.values.astype(float)

        try:
            # ET_sac: SAC-SMA actual ET stored as "ET_ke" in BASE feather
            if "ET_ke" in db.columns:
                et_sac = _mmmon(db["ET_ke"].values, days)
            else:
                et_sac = np.full(L, np.nan)

            # ET_ke: Ke × PET  (calibrated Ke from params JSON)
            ke_val = None
            if gid in cal:
                for k in ("Ke", "ke", "KE"):
                    if k in cal[gid]:
                        ke_val = float(cal[gid][k])
                        break
            if ke_val is not None and "PET" in db.columns:
                et_ke = _mmmon(db["PET"].values * ke_val, days)
            else:
                et_ke = et_sac.copy()   # fallback: same as ET_sac

            # ET_B: Budyko ET from BUDYKO feather
            if "ET_B" in dd.columns:
                et_b = _mmmon(dd["ET_B"].values, days)
            else:
                et_b = np.full(L, np.nan)

        except Exception as e:
            errors[gid] = f"ET arrays: {e}"
            skipped += 1
            continue

        # ── SFET for this basin ───────────────────────────────────────
        sfet = _sfet_for_basin(sfet_monthly, gid, common_index)

        if not np.any(np.isfinite(sfet)):
            errors[gid] = "SFET all-NaN"
            skipped += 1
            continue

        # ── Accumulate ────────────────────────────────────────────────
        sac_stack.append(et_sac)
        ke_stack.append(et_ke)
        b_stack.append(et_b)
        sfet_stack.append(sfet)

        kge_sac_all.append(_kge(sfet, et_sac))
        kge_ke_all.append(_kge(sfet, et_ke))
        kge_b_all.append(_kge(sfet, et_b))

    # ── Report ─────────────────────────────────────────────────────────────
    n_valid = len(sfet_stack)
    # # print(f"  Valid basins: {n_valid}  |  Skipped: {skipped}")
    # if errors:
    #     # print(f"  First failure reasons (up to 5):")
    #     for gid, msg in list(errors.items())[:5]:
    #         print(f"    {gid}: {msg}")

    if n_valid == 0:
        raise RuntimeError(
            "No valid basins loaded. "
            "Check result_dir paths, feather column names, and SFET coverage."
        )

    # ── Stack → (n_valid, L) ───────────────────────────────────────────────
    sac_arr  = np.vstack(sac_stack)
    ke_arr   = np.vstack(ke_stack)
    b_arr    = np.vstack(b_stack)
    sfet_arr = np.vstack(sfet_stack)

    # ── Cross-basin statistics ─────────────────────────────────────────────
    sac_mean,  sac_std  = _mean_std(sac_arr)
    ke_mean,   ke_std   = _mean_std(ke_arr)
    b_mean,    b_std    = _mean_std(b_arr)
    sfet_mean, sfet_std = _mean_std(sfet_arr)

    sac_seas_m,  sac_seas_s  = _seas_mean_std(sac_arr,  months_arr)
    ke_seas_m,   ke_seas_s   = _seas_mean_std(ke_arr,   months_arr)
    b_seas_m,    b_seas_s    = _seas_mean_std(b_arr,    months_arr)
    sfet_seas_m, sfet_seas_s = _seas_mean_std(sfet_arr, months_arr)

    def _clean_kge(lst):
        return np.array([v for v in lst if np.isfinite(v)])

    return dict(
        common_index = common_index,
        n_valid      = n_valid,
        # Time series basin-mean (L,)
        sac_mean=sac_mean,   sac_std=sac_std,
        ke_mean=ke_mean,     ke_std=ke_std,
        b_mean=b_mean,       b_std=b_std,
        sfet_mean=sfet_mean, sfet_std=sfet_std,
        # Seasonal (12,)
        sac_seas_m=sac_seas_m,   sac_seas_s=sac_seas_s,
        ke_seas_m=ke_seas_m,     ke_seas_s=ke_seas_s,
        b_seas_m=b_seas_m,       b_seas_s=b_seas_s,
        sfet_seas_m=sfet_seas_m, sfet_seas_s=sfet_seas_s,
        # Per-basin mean (n_basins,) — one value per basin for scatter
        sac_basin =np.nanmean(sac_arr,  axis=1),
        ke_basin  =np.nanmean(ke_arr,   axis=1),
        b_basin   =np.nanmean(b_arr,    axis=1),
        sfet_basin=np.nanmean(sfet_arr, axis=1),
        # KGE per basin (1-D)
        kge_sac=_clean_kge(kge_sac_all),
        kge_ke =_clean_kge(kge_ke_all),
        kge_b  =_clean_kge(kge_b_all),
    )


# ══════════════════════════════════════════════════════════════════════════════
# VIOLIN + BOX HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _violin_box(ax, vals: np.ndarray, x: float,
                color: str, width: float = 0.52):
    """Draw filled violin + IQR box + whiskers + median label at position x."""
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return

    # Violin
    parts = ax.violinplot(vals, positions=[x], widths=width,
                          showmeans=False, showmedians=False,
                          showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.30)

    # Quartiles and whiskers
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    iqr  = q3 - q1
    wlo  = max(vals[vals >= q1 - 1.5 * iqr].min(), vals.min())
    whi  = min(vals[vals <= q3 + 1.5 * iqr].max(), vals.max())
    hw   = width * 0.28   # half box width

    # IQR box
    ax.fill_betweenx([q1, q3], x - hw, x + hw,
                     color=color, alpha=0.60, zorder=3)
    # Median line
    ax.plot([x - hw, x + hw], [med, med],
            color="white", lw=2.0, zorder=4)
    # Whisker lines
    ax.plot([x, x], [wlo, q1], color=color, lw=1.0, zorder=3)
    ax.plot([x, x], [q3, whi], color=color, lw=1.0, zorder=3)
    # Whisker caps
    cap = hw * 0.5
    ax.plot([x - cap, x + cap], [wlo, wlo], color=color, lw=0.9, zorder=3)
    ax.plot([x - cap, x + cap], [whi, whi], color=color, lw=0.9, zorder=3)

    # Median value label above violin
    ax.text(x, whi + 0.03, f"{med:.2f}",
            ha="center", va="bottom", fontsize=7,
            color=color, fontweight="bold", zorder=5)
    # # Sample size below
    # ax.text(x, wlo - 0.05, f"n={len(vals)}",
    #         ha="center", va="top", fontsize=6,
    #         color="0.50", zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def plot_et_validation_mean505(
    data: dict,
    out_dir: str,
    fname: str = "fig_ET_validation_mean505",
):
    idx = data["common_index"]
    n   = data["n_valid"]

    fig = plt.figure(figsize=(12, 4))
    # Row 0: panel (a) time series spans full width
    # Row 1: panels (b) regime wider, (c) scatter, (d) KGE
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.065, right=0.985,
        top=0.93,   bottom=0.09,
        hspace=0.40, wspace=0.22,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.6, 1.0, 1.0],
    )
    ax_ts  = fig.add_subplot(gs[0, :])      # row 0, all 3 columns
    ax_reg = fig.add_subplot(gs[1, 0])      # row 1, col 0
    ax_sc  = fig.add_subplot(gs[1, 1])      # row 1, col 1
    ax_kge = fig.add_subplot(gs[1, 2])      # row 1, col 2

    # Shared legend handles
    handles = [
        Line2D([0],[0], color=C["sfet"], lw=LW["sfet"], ls=LS["sfet"],
               label="SFET"),
        Line2D([0],[0], color=C["sac"],  lw=LW["sac"],  ls=LS["sac"],
               label=r"ET$_B$"),
        Line2D([0],[0], color=C["ke"],   lw=LW["ke"],   ls=LS["ke"],
               label=r"ET$_{\rm sac}$ "),
        Line2D([0],[0], color=C["b"],    lw=LW["b"],    ls=LS["b"],
               label=r"ET$_{ke}$"),
    ]

    def _shade(ax, t, m, s, col):
        ax.fill_between(t, m - s, m + s, color=col, alpha=SHAD, lw=0)

    # ══════════════════════════════════════════════════════════
    # (a)  Mean time series
    # ══════════════════════════════════════════════════════════
    for key in ("sfet", "sac", "ke", "b"):
        m = data[f"{key}_mean"]
        s = data[f"{key}_std"]
        ax_ts.plot(idx, m, color=C[key], lw=LW[key], ls=LS[key], zorder=4)
        _shade(ax_ts, idx, m, s, C[key])

    ax_ts.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ts.set_xlim(idx[0], idx[-1])
    ax_ts.set_ylim(bottom=0)
    ax_ts.set_ylabel("ET (mm/month)", fontsize=14)
    ax_ts.set_xlabel(" ", fontsize=14)
    ax_ts.grid(axis="y")
    ax_ts.legend(handles=handles, loc="upper right",
                 ncol=4, fontsize=12, handlelength=1.8,
                 columnspacing=0.8, frameon=0, edgecolor="0.65")
    ax_ts.set_title(f"(a) ",
                    loc="left", fontsize=14, fontweight="bold", pad=4)

    # ══════════════════════════════════════════════════════════
    # (b)  Mean seasonal regime curve
    # ══════════════════════════════════════════════════════════
    x12 = np.arange(12)
    for key in ("sfet", "sac", "ke", "b"):
        m = data[f"{key}_seas_m"]
        s = data[f"{key}_seas_s"]
        ax_reg.plot(x12, m, color=C[key], lw=LW[key], ls=LS[key])
        ax_reg.fill_between(x12, m - s, m + s,
                            color=C[key], alpha=SHAD, lw=0)

    ax_reg.set_xticks(x12)
    ax_reg.set_xticklabels(MON, fontsize=7)
    ax_reg.set_xlabel("Month", fontsize=14)
    ax_reg.set_ylabel("ET (mm/month)", fontsize=14)
    ax_reg.set_ylim(bottom=0)
    ax_reg.grid(axis="y")
    ax_reg.set_title("(b)", loc="left", fontsize=14,
                     fontweight="bold", pad=4)

    # ══════════════════════════════════════════════════════════
    # (c)  Scatter: one point per basin, three ET products
    # ══════════════════════════════════════════════════════════
    sfet_b = data["sfet_basin"]
    all_v  = np.concatenate([sfet_b,
                             data["sac_basin"],
                             data["ke_basin"],
                             data["b_basin"]])
    lim = float(np.nanmax(all_v[np.isfinite(all_v)])) * 1.08

    ax_sc.plot([0, lim], [0, lim], color="0.35", lw=0.8,
               ls="--", zorder=1)
    markers = dict(sac="o", ke="s", b="^")
    n = data["n_valid"]
    for key in ("sac", "ke", "b"):
        x = sfet_b
        y = data[f"{key}_basin"]
        m = np.isfinite(x) & np.isfinite(y)
        ax_sc.scatter(x[m], y[m],
                      s=10, color=C[key], alpha=0.8,
                      linewidths=0.0, edgecolors="white",
                      zorder=3, marker=markers[key],
                      label=f"ET$_{{{key}}}$")

    ax_sc.set_xlim(0, 100)
    ax_sc.set_ylim(0, 100)
    ax_sc.set_aspect("equal", adjustable="box")
    ax_sc.set_xlabel("SFET (mm/month)", fontsize=14)
    ax_sc.set_ylabel("Simulated ET (mm/month)", fontsize=14)
    ax_sc.grid(True, alpha=0.25)
    ax_sc.legend(fontsize=8, loc="upper left",ncol=1,
                 handletextpad=0.4, frameon=1, edgecolor="0.65",
                 markerscale=1.8)
    ax_sc.set_title("(c)", loc="left", fontsize=14,
                    fontweight="bold", pad=4)

    # ══════════════════════════════════════════════════════════
    # (d)  KGE violin + box
    # ══════════════════════════════════════════════════════
    positions = {"sac": 1.0, "ke": 2.0, "b": 3.0}
    for key, xi in positions.items():
        _violin_box(ax_kge, data[f"kge_{key}"], xi, C[key], width=0.52)

    ax_kge.axhline(0.5, color="0.40", lw=0.8, ls="--",
                   zorder=1)
    ax_kge.set_xticks([1, 2, 3])
    ax_kge.set_xticklabels(
        [r"ET$_B$", r"ET$_{\rm sac}$", r"ET$_{ke}$"], fontsize=12
    )
    ax_kge.set_ylabel("KGE", fontsize=12)
    ax_kge.set_ylim(-0.55, 1.08)
    ax_kge.set_xlim(0.35, 3.65)
    ax_kge.grid(axis="y", alpha=0.30)
    # ax_kge.legend(fontsize=7, loc="lower right",
    #               frameon=True, edgecolor="0.65")
    ax_kge.set_title("(d)", loc="left", fontsize=14,
                     fontweight="bold", pad=4)

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{fname}.png")
    fig.savefig(path, dpi=300)
    print(f"  ✓  {path}")
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    ROOT    = r"F:\Github_repos\Bayesian_DA_Budyko_modeling"
    SIM     = os.path.join(ROOT, "Simulation_results",
                           "Simulation_with_BASIN_CALIB_PARAMS")
    OUT_DIR = os.path.join(SIM, "figures")
    CAL_P   = os.path.join(ROOT, "SCE_cal_params", "calibrated_params.json")


    data = load_data_mean505(
        result_dir_base        = os.path.join(SIM, "BASE_MODEL"),
        result_dir_budyko      = os.path.join(SIM, "BUDYKO_MODEL"),
        sfet_monthly           = SFTET_monthly,
        calibrated_params_path = CAL_P,
        start_year = 2001,
        end_year   = 2014,
    )

    plot_et_validation_mean505(data, out_dir=OUT_DIR)