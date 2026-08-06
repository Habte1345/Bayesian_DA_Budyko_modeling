"""
Figure_3_streamflow_skill.py
==============================
Streamflow prediction skill improvement: Q_Ke -> Q_B -> Q_DA.

Layout (6 rows x 2 columns)
----------------------------
(a) Representative hydrograph, Q_obs vs Q_Ke/Q_B/Q_DA
(b) Seasonal normalized bias boxplots, three scenarios
(c) Mean seasonal regime curve, all basins, +/-1 sigma
(d) KGE distributions, basin-wise vs global calibration, three scenarios
(e) Basin-by-basin scatter, KGE(Q_B) vs KGE(Q_Ke), basin-wise calibration
(f) % of basins with KGE >= 0.5, global calibration
(g) Spatial map, DeltaKGE (Budyko - Base), global calibration
(h) Spatial map, DeltaKGE (DA - Budyko), global calibration
(i) Flow duration curves, representative basin (basin-wise data)
(j) CDF of KGE, basin-wise vs global calibration
(k) Peak-magnitude error boxplots, three scenarios (basin-wise data)
(l) Basin-wise vs global KGE scatter, Q_B scenario

Confirmed real paths/columns:
  Basin-wise (SIM_BASINWISE):
    BASE_MODEL/results_BASE_<gid>.feather        -> Q_obs, Q_base
    BUDYKO_MODEL/results_BUDYKO_<gid>.feather     -> Q_obs, Q_budyko
    BUDYKO_DA/results_BUDYKO_DA_<gid>.feather     -> Q_obs, Q_ass

  Global (SIM_GLOBAL) -- TEST subfolder, 150 held-out test basins:
    BASE_MODEL/TEST/results_BASE_<gid>.feather        -> Q_obs, Q_base
    BUDYKO_MODEL/TEST/results_BUDYKO_<gid>.feather     -> Q_obs, Q_budyko
    BUDYKO_DA/TEST/enkf_ensemble_BUDYKO_DA_<gid>.feather
        -> Q_ens_mean only, NO Q_obs (borrowed from BUDYKO_MODEL/TEST)

  CAMELS attributes geometry column is basin-boundary MultiPolygon, not a
  gauge point -- centroid is used as a representative basin location for
  the spatial maps (g)/(h).
"""

from __future__ import annotations
import os
import io
import glob
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# PATHS -- confirmed
# ══════════════════════════════════════════════════════════════════════════

ROOT = r"F:\Github_repos\Bayesian_DA_Budyko_modeling"
SIM_BASINWISE = os.path.join(ROOT, "Simulation_results", "Simulation_with_BASIN_CALIB_PARAMS")
SIM_GLOBAL    = os.path.join(ROOT, "Simulation_results", "Simulation_with_GLOBAL_CAL_PARAMS")
OUT_DIR       = os.path.join(SIM_BASINWISE, "figures")

Q_COL_MAP = {
    "ke": ("BASE_MODEL",     "results_BASE_",      "Q_base"),
    "b":  ("BUDYKO_MODEL",   "results_BUDYKO_",     "Q_budyko"),
    "da": ("BUDYKO_DA",      "results_BUDYKO_DA_",  "Q_ass"),
}


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

PRODUCTS_Q = {
    "obs": dict(label=r"Q$_{obs}$", color="black",   ls="-",  lw=1.5),
    "ke":  dict(label=r"Q$_{Ke}$",  color="#1528D8", ls="-.", lw=1.2),
    "b":   dict(label=r"Q$_{B}$",   color="#f30af3", ls="--", lw=1.3),
    "da":  dict(label=r"Q$_{DA}$",  color="darkgreen", ls="-", lw=1.2),
}
SCEN_ORDER = ["ke", "b", "da"]
SEASON_MONTHS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
MON = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


# ══════════════════════════════════════════════════════════════════════════
# CAMELS GEOMETRY LOADER (minimal -- only gauge_id + geometry needed for
# panels g/h; avoids re-running the full ET data-prep pipeline)
# ══════════════════════════════════════════════════════════════════════════

def load_camels_geometry() -> pd.DataFrame:
    r = requests.get(
        "https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/"
        "data/contents/camels_attributes_v2.0.feather"
    )
    attrs_geo = gpd.read_feather(io.BytesIO(r.content)).reset_index(drop=False)
    attrs_geo["gauge_id"] = attrs_geo["gauge_id"].astype(str).str.zfill(8)
    out = attrs_geo[["gauge_id", "geometry"]].copy()
    print(f"  Loaded {len(out)} basins with geometry")
    return out


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _to_ms_index(df):
    for col in ("time", "Date", "date", "TIME"):
        if col in df.columns:
            df = df.copy()
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            break
    df = df.copy()
    df.index = pd.to_datetime({"year": df.index.year, "month": df.index.month, "day": 1})
    return df


def _slice(df, common_index):
    ym_map = {(ts.year, ts.month): i for i, ts in enumerate(df.index)}
    rows = []
    for ts in common_index:
        key = (ts.year, ts.month)
        rows.append(df.iloc[ym_map[key]] if key in ym_map else pd.Series(np.nan, index=df.columns))
    return pd.DataFrame(rows, index=common_index, columns=df.columns)


def _kge(obs, sim):
    m = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[m], sim[m]
    if len(o) < 3 or o.std() == 0 or o.mean() == 0:
        return np.nan
    r = float(np.corrcoef(o, s)[0, 1])
    return float(1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2
                              + (s.mean() / o.mean() - 1) ** 2))


def _seasonal(arr, months):
    return np.array([np.nanmean(arr[months == m]) for m in range(1, 13)])


# ══════════════════════════════════════════════════════════════════════════
# STREAMFLOW LOADER -- handles both basin-wise and global (TEST-subfolder,
# enkf_ensemble naming) structures
# ══════════════════════════════════════════════════════════════════════════
def load_streamflow_scenarios(sim_root: str, start_year=2001, end_year=2014,
                               is_global: bool = False) -> dict:
    t0 = pd.Timestamp(f"{start_year}-01-01")
    t1 = pd.Timestamp(f"{end_year}-12-31")
    common_index = pd.date_range(t0, t1, freq="MS")
    months_arr = np.array([d.month for d in common_index])

    def _result_dir(folder):
        base = os.path.join(sim_root, folder)
        return os.path.join(base, "TEST") if is_global else base

    # --- ke (BASE) and b (BUDYKO): same convention in both structures ---
    per_scenario_basins = {}
    for key in ("ke", "b"):
        folder, prefix, qcol = Q_COL_MAP[key]
        result_dir = _result_dir(folder)
        files = glob.glob(os.path.join(result_dir, f"{prefix}*.feather"))
        basins = {}
        for fpath in files:
            gid = os.path.basename(fpath).replace(prefix, "").replace(".feather", "")
            try:
                df = _to_ms_index(pd.read_feather(fpath))
                df = _slice(df, common_index)
                if "Q_obs" not in df.columns or qcol not in df.columns:
                    continue
                # basins[gid] = dict(obs=df["Q_obs"].values.astype(float),
                #                     sim=np.clip(df[qcol].values.astype(float), 0, None))

                q_obs_raw = df["Q_obs"].values.astype(float)
                q_obs_raw[q_obs_raw < -10] = np.nan
                basins[gid] = dict(obs=np.clip(q_obs_raw, 0, None),
                                    sim=np.clip(df[qcol].values.astype(float), 0, None))

            except Exception:
                continue
        per_scenario_basins[key] = basins
        print(f"  Loaded {len(basins)} basins for scenario '{key}' from {folder}"
              + ("/TEST" if is_global else ""))

    # --- da (BUDYKO_DA): different file/column convention under global calibration ---
    da_basins = {}
    if is_global:
        da_dir = _result_dir("BUDYKO_DA")
        da_files = glob.glob(os.path.join(da_dir, "enkf_ensemble_BUDYKO_DA_*.feather"))
        budyko_obs = per_scenario_basins["b"]  # source of Q_obs, keyed by gid
        for fpath in da_files:
            gid = os.path.basename(fpath).replace("enkf_ensemble_BUDYKO_DA_", "").replace(".feather", "")
            if gid not in budyko_obs:
                continue  # no Q_obs available for this basin
            try:
                df = _to_ms_index(pd.read_feather(fpath))
                df = _slice(df, common_index)
                if "Q_ens_mean" not in df.columns:
                    continue
                da_basins[gid] = dict(obs=budyko_obs[gid]["obs"],
                                       sim=np.clip(df["Q_ens_mean"].values.astype(float), 0, None))
            except Exception:
                continue
        print(f"  Loaded {len(da_basins)} basins for scenario 'da' from BUDYKO_DA/TEST "
              f"(Q_obs borrowed from BUDYKO_MODEL/TEST, sim = Q_ens_mean)")
    else:
        folder, prefix, qcol = Q_COL_MAP["da"]
        result_dir = _result_dir(folder)
        files = glob.glob(os.path.join(result_dir, f"{prefix}*.feather"))
        for fpath in files:
            gid = os.path.basename(fpath).replace(prefix, "").replace(".feather", "")
            try:
                df = _to_ms_index(pd.read_feather(fpath))
                df = _slice(df, common_index)
                if "Q_obs" not in df.columns or qcol not in df.columns:
                    continue
                # da_basins[gid] = dict(obs=df["Q_obs"].values.astype(float),
                #                        sim=np.clip(df[qcol].values.astype(float), 0, None))
                q_obs_raw = df["Q_obs"].values.astype(float)
                q_obs_raw[q_obs_raw < -10] = np.nan
                da_basins[gid] = dict(obs=np.clip(q_obs_raw, 0, None),
                                    sim=np.clip(df[qcol].values.astype(float), 0, None))
                                
            except Exception:
                continue
        print(f"  Loaded {len(da_basins)} basins for scenario 'da' from {folder}")

    per_scenario_basins["da"] = da_basins

    common_gids = set.intersection(*[set(v.keys()) for v in per_scenario_basins.values()])
    print(f"  Basins common to all three scenarios: {len(common_gids)}")

    out = {}
    for gid in sorted(common_gids):
        out[gid] = dict(
            obs=per_scenario_basins["ke"][gid]["obs"],
            ke=per_scenario_basins["ke"][gid]["sim"],
            b=per_scenario_basins["b"][gid]["sim"],
            da=per_scenario_basins["da"][gid]["sim"],
        )
    return dict(common_index=common_index, months_arr=months_arr, per_basin=out)


def compute_kge_all(data: dict) -> dict:
    out = {k: {} for k in SCEN_ORDER}
    for gid, series in data["per_basin"].items():
        obs = series["obs"]
        for k in SCEN_ORDER:
            out[k][gid] = _kge(obs, series[k])
    return out


def _select_representative_basin(kge_by_scenario: dict, min_kge_b: float = 0.4) -> str:
    candidates = {gid: kge for gid, kge in kge_by_scenario["b"].items() if np.isfinite(kge) and kge >= min_kge_b}
    if not candidates:
        candidates = {gid: kge for gid, kge in kge_by_scenario["b"].items() if np.isfinite(kge)}
    best_gid = max(candidates, key=candidates.get)
    print(f"  Representative basin: {best_gid} (KGE_B = {candidates[best_gid]:.3f})")
    return best_gid


def normalized_bias_by_season(data: dict) -> dict:
    months_arr = data["months_arr"]
    season_of_month = {m: s for s, ms in SEASON_MONTHS.items() for m in ms}
    out = {s: {k: [] for k in SCEN_ORDER} for s in SEASON_ORDER}
    for gid, series in data["per_basin"].items():
        obs = series["obs"]
        for k in SCEN_ORDER:
            sim = series[k]
            valid = np.isfinite(obs) & np.isfinite(sim) & (obs != 0)
            for i in np.where(valid)[0]:
                season = season_of_month[months_arr[i]]
                out[season][k].append((sim[i] - obs[i]) / obs[i])
    for s in SEASON_ORDER:
        for k in SCEN_ORDER:
            out[s][k] = np.array(out[s][k])
    return out


def seasonal_regime_all(data: dict) -> dict:
    months_arr = data["months_arr"]
    gids = list(data["per_basin"].keys())
    out = {}
    for k in ["obs"] + SCEN_ORDER:
        stack = np.vstack([data["per_basin"][gid][k] for gid in gids])
        per_basin = np.vstack([_seasonal(stack[i], months_arr) for i in range(stack.shape[0])])
        out[k] = (np.nanmean(per_basin, axis=0), np.nanstd(per_basin, axis=0))
    return out


# ══════════════════════════════════════════════════════════════════════════
# MAIN FIGURE -- 6 rows x 2 columns
# ══════════════════════════════════════════════════════════════════════════

def plot_streamflow_skill_figure(
    data_basinwise: dict,
    data_global: dict,
    attrs_longterm_df: pd.DataFrame,
    out_dir: str,
    fname: str = "Figure_3_streamflow_skill",
    representative_gid: str | None = None,
    window_years: int = 5,
):
    kge_bw = compute_kge_all(data_basinwise)
    kge_gl = compute_kge_all(data_global)

    if representative_gid is None:
        representative_gid = _select_representative_basin(kge_bw)

    fig = plt.figure(figsize=(12, 15), dpi=300)
    gs = gridspec.GridSpec(6, 2, figure=fig, left=0.07, right=0.98,
                            top=0.96, bottom=0.06, hspace=0.55, wspace=0.28)

    ax_hydro = fig.add_subplot(gs[0, 0])
    ax_bias  = fig.add_subplot(gs[1, 0])
    ax_reg   = fig.add_subplot(gs[2, 0])
    ax_kge   = fig.add_subplot(gs[3, 0])
    ax_scat  = fig.add_subplot(gs[4, 0])
    ax_thresh = fig.add_subplot(gs[5, 0])

    ax_mapA  = fig.add_subplot(gs[0, 1])
    ax_mapB  = fig.add_subplot(gs[1, 1])
    # ax_fdc   = fig.add_subplot(gs[2, 1])
    # ax_cdf   = fig.add_subplot(gs[3, 1])
    ax_cdf   = fig.add_subplot(gs[2, 1])
    ax_fdc   = fig.add_subplot(gs[3, 1])
    ax_peak  = fig.add_subplot(gs[4, 1])
    ax_bwgl  = fig.add_subplot(gs[5, 1])

    # ── (a) Representative hydrograph ───────────────────────────────────
    common_index = data_basinwise["common_index"]
    series = data_basinwise["per_basin"][representative_gid]
    years = sorted(set(common_index.year))
    y0 = years[len(years) // 2 - window_years // 2]
    win = (common_index >= f"{y0}-01-01") & (common_index < f"{y0+window_years}-01-01")
    t = common_index[win]
    for k in ["obs", "ke", "b", "da"]:
        p = PRODUCTS_Q[k]
        ax_hydro.plot(t, series[k][win], color=p["color"], lw=p["lw"], ls=p["ls"], label=p["label"])
    ax_hydro.xaxis.set_major_locator(mdates.YearLocator())
    ax_hydro.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_hydro.set_ylabel("Q (mm/month)")
    ax_hydro.legend(loc="upper right", ncol=2, frameon=False, fontsize=12)
    ax_hydro.set_title("(a) Hydrograph", loc="left", fontweight="bold")
    plt.setp(ax_hydro.get_xticklabels(), rotation=30, ha="right")

    # ── (b) Seasonal normalized bias ────────────────────────────────────
    bias = normalized_bias_by_season(data_basinwise)
    positions, box_data, box_colors, centers = [], [], [], []
    x0 = 1.0
    for season in SEASON_ORDER:
        for j, k in enumerate(SCEN_ORDER):
            positions.append(x0 + j * 0.6)
            box_data.append(bias[season][k])
            box_colors.append(PRODUCTS_Q[k]["color"])
        centers.append(x0 + 0.6)
        x0 += 2.4
    bp = ax_bias.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    ax_bias.axhline(0, color="0.3", ls="--", lw=0.9)
    ax_bias.set_xticks(centers); ax_bias.set_xticklabels(SEASON_ORDER)
    ax_bias.set_ylabel("Normalized bias")
    ax_bias.set_title("(b) Seasonal Variations", loc="left", fontweight="bold")

    # ── (c) Mean seasonal regime ─────────────────────────────────────
    regime = seasonal_regime_all(data_basinwise)
    x12 = np.arange(12)
    for k in ["obs", "ke", "b", "da"]:
        p = PRODUCTS_Q[k]
        m, s = regime[k]
        ax_reg.plot(x12, m, color=p["color"], lw=p["lw"], ls=p["ls"], label=p["label"])
        ax_reg.fill_between(x12, np.maximum(m - s, 0), m + s, color=p["color"], alpha=0.1, lw=0)
    ax_reg.set_xticks(x12); ax_reg.set_xticklabels(MON)
    ax_reg.set_ylabel("Q (mm/month)")
    ax_reg.set_title("(c) Mean seasonal regime", loc="left", fontweight="bold")
    ax_reg.set_ylim(0, 5)

    # ── (d) KGE distributions, basin-wise vs global ─────────────────────
    pos = 1.0
    xt, xtl = [], []
    for cal_label, kge_dict in [("BW", kge_bw), ("GL", kge_gl)]:
        for k in SCEN_ORDER:
            vals = np.array([v for v in kge_dict[k].values() if np.isfinite(v)])
            parts = ax_kge.violinplot(vals, positions=[pos], widths=0.6, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(PRODUCTS_Q[k]["color"]); pc.set_alpha(0.4)
            med = np.median(vals)
            ax_kge.plot([pos-0.15, pos+0.15], [med, med], color="black", lw=1.5)
            xt.append(pos); xtl.append(f"{PRODUCTS_Q[k]['label']} {cal_label}")
            pos += 1.0
        pos += 0.8
    ax_kge.set_xticks(xt); ax_kge.set_xticklabels(xtl, fontsize=9)
    ax_kge.axhline(0.5, color="0.4", ls="--", lw=0.8)
    ax_kge.set_ylim(-5, 1)
    ax_kge.set_ylabel("KGE vs. Q$_{obs}$")
    ax_kge.set_title("(d) KGE distributions, basin-wise vs global", loc="left", fontweight="bold")

    # ── (e) Basin-wise KGE(Budyko vs Base) ────────────
    gids = sorted(set(kge_bw["ke"]) & set(kge_bw["b"]))
    x = np.array([kge_bw["ke"][g] for g in gids])
    y = np.array([kge_bw["b"][g] for g in gids])
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax_scat.plot([lo, hi], [lo, hi], color="0.3", ls="--", lw=1.0)
    ax_scat.scatter(x, y, s=12, color=PRODUCTS_Q["b"]["color"], alpha=0.6, edgecolors="none")
    pct_above = 100 * np.mean(y > x)
    ax_scat.text(0.03, 0.95, f"{pct_above:.1f}% above 1:1", transform=ax_scat.transAxes,
                 va="top", fontsize=12)
    ax_scat.set_xlabel(r"KGE(Q$_{Ke}$, Q$_{obs}$)"); ax_scat.set_ylabel(r"KGE(Q$_{B}$, Q$_{obs}$)")
    ax_scat.set_title("(e) Basin-wise KGE(Budyko vs Base)", loc="left", fontweight="bold")
    ax_scat.set_ylim(-2, 1)
    ax_scat.set_xlim(-2, 1)


    # ── (f) Success rate: % basins KGE>=0.5, global calibration ─────────
    rates = []
    for k in SCEN_ORDER:
        vals = np.array([v for v in kge_gl[k].values() if np.isfinite(v)])
        rates.append(100 * np.mean(vals >= 0.5))
    bars = ax_thresh.bar([PRODUCTS_Q[k]["label"] for k in SCEN_ORDER], rates,
                         color=[PRODUCTS_Q[k]["color"] for k in SCEN_ORDER], alpha=0.7)
    for b, r in zip(bars, rates):
        ax_thresh.text(b.get_x() + b.get_width()/2, r + 1, f"{r:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax_thresh.set_ylabel("% basins KGE \u2265 0.5")
    ax_thresh.set_title("(f) % basins KGE>=0.5, global calibration", loc="left", fontweight="bold")

    # ── (g), (h) Spatial maps: DeltaKGE Budyko-Base, DA-Budyko ──────────

    latlon = attrs_longterm_df.set_index("gauge_id")["geometry"].apply(
        lambda g: (g.centroid.y, g.centroid.x)
    )
    for ax_map, (k2, k1), title in [(ax_mapA, ("b", "ke"), "(g) Spatial maps (Budyko-Base)"), (ax_mapB, ("da", "b"), "(h) Spatial maps (DA-Budyko")]:
        gids2 = sorted(set(kge_gl[k1]) & set(kge_gl[k2]) & set(latlon.index))
        d = np.array([kge_gl[k2][g] - kge_gl[k1][g] for g in gids2])
        lat = np.array([latlon[g][0] for g in gids2])
        lon = np.array([latlon[g][1] for g in gids2])
        vmax = np.nanpercentile(np.abs(d), 95) if len(d) else 1.0
        sca = ax_map.scatter(lon, lat, c=d, cmap="jet", vmin=-5, vmax=1, s=18, edgecolors="none")
        plt.colorbar(sca, ax=ax_map, fraction=0.046, pad=0.04, label="\u0394KGE")
        ax_map.set_title(title, loc="left", fontweight="bold")
        ax_map.set_xlabel("Lon"); ax_map.set_ylabel("Lat")
        # ax_map.set_xlim(-0.5, 1)
        # ax_map.set_ylim(-5, 1)

# ── (i) Flow duration curves, representative basin (basin-wise) ─────
    for k in ["obs", "ke", "b", "da"]:
        p = PRODUCTS_Q[k]
        q = np.sort(series[k][np.isfinite(series[k])])[::-1]
        exceed = np.arange(1, len(q) + 1) / (len(q) + 1) * 100
        ax_fdc.plot(exceed, q, color=p["color"], lw=p["lw"], ls=p["ls"], label=p["label"])
    ax_fdc.set_yscale("log")
    ax_fdc.set_xlabel("Exc. probability (%)"); ax_fdc.set_ylabel("Q (mm/month, log)")
    ax_fdc.set_title("(j) Flow duration curves", loc="left", fontweight="bold")
    ax_fdc.legend(loc="lower left", ncol=2, frameon=False, fontsize=12)

    # ── (j) CDF of KGE, basin-wise vs global ────────────────────────────
    for cal_label, kge_dict, ls in [("BW", kge_bw, "-"), ("GL", kge_gl, "--")]:
        for k in SCEN_ORDER:
            vals = np.sort(np.array([v for v in kge_dict[k].values() if np.isfinite(v)]))
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax_cdf.plot(vals, cdf, color=PRODUCTS_Q[k]["color"], ls=ls, lw=1.3,
                       label=f"{PRODUCTS_Q[k]['label']} ({cal_label})")
    ax_cdf.set_xlabel("KGE"); ax_cdf.set_ylabel("Cum. fraction")
    ax_cdf.legend(fontsize=12, ncol=2, frameon=False)
    ax_cdf.set_title("(i) CDF of KGE", loc="left", fontweight="bold")
    ax_cdf.set_xlim(-15, 1)

    # ── (k) Peak-magnitude error (basin-wise) ───────────────────────────
    mag_err = {k: [] for k in SCEN_ORDER}
    yrs = common_index.year.values
    for gid, s in data_basinwise["per_basin"].items():
        for yr in np.unique(yrs):
            mask = yrs == yr
            obs_y = s["obs"][mask]
            if not np.any(np.isfinite(obs_y)):
                continue
            obs_peak_i = np.nanargmax(obs_y)
            for k in SCEN_ORDER:
                sim_y = s[k][mask]
                if not np.any(np.isfinite(sim_y)):
                    continue
                sim_peak_i = np.nanargmax(sim_y)
                if obs_y[obs_peak_i] != 0:
                    mag_err[k].append((sim_y[sim_peak_i] - obs_y[obs_peak_i]) / obs_y[obs_peak_i])
    bp2 = ax_peak.boxplot([mag_err[k] for k in SCEN_ORDER], positions=[1, 2, 3], showfliers=False,
                          patch_artist=True, widths=0.5)
    for patch, k in zip(bp2["boxes"], SCEN_ORDER):
        patch.set_facecolor(PRODUCTS_Q[k]["color"]); patch.set_alpha(0.55)
    ax_peak.axhline(0, color="0.3", ls="--", lw=0.8)
    ax_peak.set_xticks([1, 2, 3]); ax_peak.set_xticklabels([PRODUCTS_Q[k]["label"] for k in SCEN_ORDER])
    ax_peak.set_ylabel("Error (fraction)")
    ax_peak.set_title("(k) Peak-magnitude error (basin-wise)", loc="left", fontweight="bold")


    # ── (l) Basin-wise vs global KGE scatter, Q_B scenario ──────────────
    gids3 = sorted(set(kge_bw["b"]) & set(kge_gl["b"]))
    xb = np.array([kge_bw["b"][g] for g in gids3])
    yb = np.array([kge_gl["b"][g] for g in gids3])
    mm = np.isfinite(xb) & np.isfinite(yb)
    xb, yb = xb[mm], yb[mm]
    lo2, hi2 = min(xb.min(), yb.min()), max(xb.max(), yb.max())
    ax_bwgl.plot([lo2, hi2], [lo2, hi2], color="0.3", ls="--", lw=1.0)
    ax_bwgl.scatter(xb, yb, s=12, color=PRODUCTS_Q["b"]["color"], alpha=0.6, edgecolors="none")
    ax_bwgl.set_xlabel(r"KGE(Q$_{B}$), basin-wise"); ax_bwgl.set_ylabel(r"KGE(Q$_{B}$), global")
    ax_bwgl.set_title("(l)Basin-wise vs global KGE scatter, Q$_{B}$", loc="left", fontweight="bold")
    ax_bwgl.set_xlim(-2, 0.85)
    ax_bwgl.set_ylim(-2, 0.85)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{fname}.png")
    fig.savefig(path, dpi=600)
    print(f"  Saved: {path}")
    plt.show(fig)
    return fig


# ══════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════

attrs_all_simu_mean_505_basins_longterm = load_camels_geometry()

data_basinwise = load_streamflow_scenarios(SIM_BASINWISE, start_year=2005, end_year=2014, is_global=False)
data_global    = load_streamflow_scenarios(SIM_GLOBAL,    start_year=2005, end_year=2014, is_global=True)

plot_streamflow_skill_figure(
    data_basinwise=data_basinwise,
    data_global=data_global,
    attrs_longterm_df=attrs_all_simu_mean_505_basins_longterm,
    out_dir=OUT_DIR,
    window_years=5,
)






















# """
# plot_basin_dashboard_6rows.py
# ==============================
# Six-basin dashboard — one row per basin, four panels per row.

# Layout per row:  [ts (wide)] [scatter] [FDC] [seasonal]
#                   col 0        col 1    col 2   col 3
#                   width_ratio  1.8       1       1       1.2

# All panel functions and styling are identical to plot_basin_Params.
# Only the figure layout changes: 6 rows × 4 columns instead of 2 rows.

# Call
# ----
#     top6 = (attrs_df.dropna(subset=["KGE_DA"])
#                     .sort_values("KGE_DA", ascending=False)
#                     .head(6).reset_index(drop=True))
#     plot_top6_dashboard(top6, result_dir_base, result_dir_da, out_dir)
# """

# from __future__ import annotations
# import os, warnings
# import numpy as np
# import pandas as pd
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.dates as mdates
# from matplotlib.lines import Line2D
# from matplotlib.patches import Patch

# warnings.filterwarnings("ignore")

# mpl.rcParams.update({
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
#     "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
#     "xtick.labelsize": 10, "ytick.labelsize": 10,
#     "legend.fontsize": 10, "legend.framealpha": 0.93, "legend.edgecolor": "0.65",
#     "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
#     "xtick.major.width": 1.0, "xtick.major.size": 4,
#     "ytick.major.width": 1.0, "ytick.major.size": 4,
#     "grid.linewidth": 0.5, "grid.color": "0.87",
#     "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
# })

# C  = dict(obs="#111111", base="#E34B0F", bud="#1565C0", da="#10A718", ribbon="#08450B")
# LW = dict(obs=2.0, base=1.5, bud=1.5, da=2.0)
# LS = dict(obs="-.", base="--", bud=":", da="-")
# MON = ["O","N","D","J","F","M","A","M","J","J","A","S"]


# # ── Helpers (unchanged) ───────────────────────────────────────────

# def _load(path):
#     df = pd.read_feather(path)
#     for col in ("time", "Date", "date"):
#         if col in df.columns:
#             df[col] = pd.to_datetime(df[col])
#             return df.set_index(col)
#     return df

# def _get(df, *cols):
#     for c in cols:
#         if c in df.columns:
#             return df[c].values.astype(float)
#     return np.full(len(df), np.nan)

# def _align(v, si, ti):
#     return pd.Series(v, index=si).reindex(ti).values.astype(float)

# def _kge(o, s):
#     m = np.isfinite(o) & np.isfinite(s); o, s = o[m], s[m]
#     if len(o) < 3 or o.std() == 0: return np.nan
#     r = np.corrcoef(o, s)[0, 1]
#     return float(1 - np.sqrt((r-1)**2 + (s.std()/o.std()-1)**2 + (s.mean()/o.mean()-1)**2))

# def _nse(o, s):
#     m = np.isfinite(o) & np.isfinite(s); o, s = o[m], s[m]
#     d = ((o - o.mean())**2).sum()
#     return float(1 - ((o-s)**2).sum()/d) if d > 0 and len(o) > 2 else np.nan


# # ── Panel functions (unchanged) ───────────────────────────────────

# def _ts(ax, t, obs, base, bud, da, lo, hi, first_row=False):
#     ax.fill_between(t, lo, hi, color=C["ribbon"], alpha=0.20, zorder=1)
#     ax.plot(t, base, color=C["base"], lw=LW["base"], ls=LS["base"], zorder=2)
#     ax.plot(t, bud,  color=C["bud"],  lw=LW["bud"],  ls=LS["bud"],  zorder=3)
#     ax.plot(t, da,   color=C["da"],   lw=LW["da"],   ls=LS["da"],   zorder=4)
#     ax.plot(t, obs,  color=C["obs"],  lw=LW["obs"],  ls=LS["obs"],  zorder=5)
#     ax.set_xlim(t[0], t[-1]); ax.set_ylim(bottom=0)
#     ax.set_ylabel("Q (mm/month)")
#     ax.xaxis.set_major_locator(mdates.YearLocator(2))
#     ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
#     ax.grid(axis="y")
#     if not first_row:
#         plt.setp(ax.get_xticklabels(), visible=False)


# def _scatter(ax, obs, base, bud, da, kge_b, kge_bud, kge_da):
#     lim = max(np.nanmax([obs, base, bud, da]) * 1.06, 1.0)
#     for sim, color, marker in [
#         (base, C["bud"], "o"),
#         (bud,  C["base"],  "s"),
#         (da,   C["da"],   "^"),
#     ]:
#         m = np.isfinite(obs) & np.isfinite(sim)
#         ax.scatter(obs[m], sim[m], s=30, color=color, marker=marker,
#                    alpha=0.85, linewidths=0, zorder=3)
#     ax.plot([0, lim], [0, lim], color="0.25", lw=1.0, ls="--", zorder=5)
#     ax.set_xlim(0, lim); ax.set_ylim(0, lim)
#     ax.set_aspect("equal"); ax.grid(True)
#     ax.set_xlabel("Obs Q"); ax.set_ylabel("Sim Q")
#     for txt, col, y0 in [
#         (f"KGE={kge_bud:.2f}", C["base"],  0.97),
#         (f"KGE={kge_b:.2f}",   C["bud"], 0.87),
#         (f"KGE={kge_da:.2f}",  C["da"],   0.77),
#     ]:
#         ax.text(0.04, y0, txt, transform=ax.transAxes, va="top", ha="left",
#                 fontsize=9.5, color=col, fontfamily="monospace", fontweight="bold")


# def _fdc(ax, obs, base, bud, da):
#     def _f(x):
#         v = np.sort(x[np.isfinite(x)])[::-1]
#         return np.arange(1, len(v)+1)/(len(v)+1)*100, v
#     for sim, color, ls, lw in [
#         (obs,  C["obs"],  "-.", 2.0),
#         (base, C["bud"], "--", 1.5),
#         (bud,  C["base"],  ":",  1.5),
#         (da,   C["da"],   "-",  2.0),
#     ]:
#         p, v = _f(sim)
#         if len(v): ax.semilogy(p, v, color=color, lw=lw, ls=ls, zorder=3)
#     ax.set_xlabel("Exceedance (%)"); ax.set_ylabel("Q (log)")
#     ax.set_xlim(0, 100); ax.grid(True, which="both", alpha=0.35)


# def _clim(ax, t, obs, base, bud, da):
#     mon  = np.array([d.month for d in t])
#     xpos = np.array([(m - 10) % 12 for m in mon])
#     for sim, color, ls, lw, marker, label in [
#         (obs,  C["obs"],  "-.", 2.0, "o",  "Observed"),
#         (base, C["bud"], "--", 1.5, None, "BASE"),
#         (bud,  C["base"],  ":",  1.5, None, "Budyko"),
#         (da,   C["da"],   "-",  2.0, "^",  "BUDYKO-DA"),
#     ]:
#         cl = np.array([np.nanmean(sim[xpos == x]) for x in range(12)])
#         ax.plot(range(12), cl, color=color, lw=lw, ls=ls,
#                 marker=marker, ms=4, label=label, zorder=3)
#     ax.set_xticks(range(12)); ax.set_xticklabels(MON, fontsize=10)
#     ax.set_xlim(-0.5, 11.5); ax.set_ylim(bottom=0)
#     ax.set_xlabel("Month"); ax.set_ylabel("Mean Q")
#     ax.grid(axis="y")
#     # ax.legend(loc="upper right", fontsize=8, frameon=False, handlelength=1.5)



# # ── Multi-basin figure ────────────────────────────────────────────

# def plot_top6_dashboard(
#     top6_df: pd.DataFrame,
#     result_dir_base:   str,
#     result_dir_budyko: str,
#     result_dir_da:     str,
#     out_dir: str,
#     start_year: int = 2005,
#     end_year:   int = 2014,
#     fname: str = "fig_basin_dashboard_6rows",
# ):
#     """
#     One row per basin, four panels per row.
#     Filters basins where KGE >= 0.5 in ALL three scenarios.
#     top6_df must have columns: gauge_id, KGE_Base, KGE_Budyko, KGE_DA,
#     and optionally: AI_ass/aridity, EI_base/EI_ass, dom_land_cover_short.
#     """
#     n_rows = len(top6_df)

#     fig = plt.figure(figsize=(18.0, n_rows * 2.6 + 0.6))

#     gs = gridspec.GridSpec(
#         n_rows, 4, figure=fig,
#         left=0.06, right=0.99,
#         top=0.97,  bottom=0.05,
#         hspace=0.38, wspace=0.28,
#         width_ratios=[2.2, 1, 1, 1.2],
#     )

#     # Shared legend on the very first row time-series panel
#     legend_drawn = False

#     for row_i, (_, mrow) in enumerate(top6_df.iterrows()):
#         gid = str(mrow["gauge_id"]).zfill(8)

#         # Load data
#         try:
#             db  = _load(os.path.join(result_dir_base,   f"results_BASE_{gid}.feather"))
#             dd  = _load(os.path.join(result_dir_budyko, f"results_BUDYKO_{gid}.feather"))
#             da  = _load(os.path.join(result_dir_da,     f"results_BUDYKO_DA_{gid}.feather"))
#             ens = _load(os.path.join(result_dir_da,     f"enkf_ensemble_BUDYKO_DA_{gid}.feather"))
#         except FileNotFoundError as e:
#             print(f"  ⚠  {gid}: {e}"); continue

#         t_all = ens.loc[f"{start_year}":f"{end_year}-12-31"].index
#         obs   = _align(_get(db,  "Q_obs"),              db.index,  t_all)
#         base  = _align(_get(db,  "Q_base", "Q_ke"),     db.index,  t_all)
#         bud_q = _align(_get(dd,  "Q_budyko", "Q_B"),    dd.index,  t_all)
#         da_q  = _align(_get(da,  "Q_ass"),               da.index,  t_all)
#         q_lo  = _align(_get(ens, "Q_ens_p05"),          ens.index, t_all)
#         q_hi  = _align(_get(ens, "Q_ens_p95"),          ens.index, t_all)

#         kge_b   = float(mrow.get("KGE_Base",   _kge(obs, base)))
#         kge_bud = float(mrow.get("KGE_Budyko", _kge(obs, bud_q)))
#         kge_da  = float(mrow.get("KGE_DA",     _kge(obs, da_q)))
#         nse_b   = _nse(obs, base)
#         nse_da  = _nse(obs, da_q)

#         # Axes
#         ax_ts  = fig.add_subplot(gs[row_i, 0])
#         ax_sc  = fig.add_subplot(gs[row_i, 1])
#         ax_fdc = fig.add_subplot(gs[row_i, 2])
#         ax_cl  = fig.add_subplot(gs[row_i, 3])

#         # Draw panels
#         is_last = (row_i == n_rows - 1)
#         _ts(ax_ts, t_all, obs, base, bud_q, da_q, q_lo, q_hi, first_row=is_last)
#         _scatter(ax_sc, obs, base, bud_q, da_q, kge_b, kge_bud, kge_da)
#         _fdc(ax_fdc, obs, base, bud_q, da_q)
#         _clim(ax_cl, t_all, obs, base, bud_q, da_q)

#         # Row title: basin ID, vegetation, AI — then colour-coded KGE per scenario
#         ai  = mrow.get("AI_ass",  mrow.get("aridity",  np.nan))
#         veg = str(mrow.get("dom_land_cover_short",
#                             mrow.get("dom_land_cover", "—")))
#         ai_s = f"{float(ai):.2f}" if pd.notna(ai) else "—"

#         # Plain text prefix
#         ax_ts.set_title(
#             f"USGS: {gid}  |  Veg: {veg}  |  AI: {ai_s}",
#             loc="left", fontsize=10, fontweight="bold", pad=3, color="0.15",
#         )
#         # Colour-coded KGE values — no subscripts, color identifies scenario
#         for x_off, kge_val, col, txt in [
#             (0.50, kge_bud, C["base"], f"KGE = {kge_bud:.2f}"),
#             (0.65, kge_b,   C["bud"],  f"| KGE = {kge_b:.2f}"),
#             (0.80, kge_da,  C["da"],   f"| KGE = {kge_da:.2f}"),
#         ]:
#             ax_ts.text(
#                 x_off, 1.04, txt,
#                 transform=ax_ts.transAxes,
#                 fontsize=9.5, fontweight="bold",
#                 color=col, va="bottom", ha="left", clip_on=False
#             )
#         # Panel labels (a–d) on first row only
#         if row_i == 0:
#             for ax, lbl in zip([ax_ts, ax_sc, ax_fdc, ax_cl],
#                                 ["(a)", "(b)", "(c)", "(d)"]):
#                 ax.set_title(lbl, loc="right", fontsize=16,
#                              fontweight="bold", pad=3, color="blue")

#         # Shared legend on first row time-series panel
#         if row_i == 1 and not legend_drawn:
#             ax_ts.legend(
#                 handles=[
#                     Line2D([0],[0], color=C["obs"],  lw=2.0, ls="-.",
#                            label="Observed"),
#                     Line2D([0],[0], color=C["bud"],  lw=1.5, ls=":",
#                            label="Budyko"),
#                     Line2D([0],[0], color=C["base"], lw=1.5, ls="--",
#                            label="BASE"),
#                     Line2D([0],[0], color=C["da"],   lw=2.0, ls="-",
#                            label="BUDYKO-DA"),
#                     Patch(fc=C["ribbon"], alpha=0.35, label="DA 5–95%"),
#                 ],
#                 loc="upper center", ncol=2, fontsize=10,
#                 handlelength=1.8, frameon=False, edgecolor="0.65",
#             )
#             legend_drawn = True

#         print(f"  ✓  row {row_i+1}  {gid}")

#     os.makedirs(out_dir, exist_ok=True)
#     path = os.path.join(out_dir, f"{fname}.png")
#     fig.savefig(path, dpi=300)
#     print(f"\n  ✓  {path}")
#     plt.show()
#     return fig


# # ── Entry point ───────────────────────────────────────────────────

# if __name__ == "__main__":

#     ROOT = r"F:\Github_repos\Bayesian_DA_Budyko_modeling"
#     SIM  = os.path.join(ROOT, "Simulation_results",
#                         "Simulation_with_BASIN_CALIB_PARAMS")

#     df = attrs_all_simu_mean_505_basins_BASIN_CAL
#     top6 = (
#         df.dropna(subset=["KGE_Base","KGE_Budyko","KGE_DA"])
#         .loc[
#             (df["KGE_Base"]   >= 0.50) &
#             (df["KGE_Budyko"] >= 0.50) &
#             (df["KGE_DA"]     >= 0.50)
#         ]
#         .sort_values("KGE_DA", ascending=False)
#         .head(6)
#         .reset_index(drop=True)
#     )
#     print(f"Basins with KGE >= 0.50 in all scenarios: {len(top6)}")

#     plot_top6_dashboard(
#         top6_df           = top6,
#         result_dir_base   = os.path.join(SIM, "BASE_MODEL"),
#         result_dir_budyko = os.path.join(SIM, "BUDYKO_MODEL"),
#         result_dir_da     = os.path.join(SIM, "BUDYKO_DA"),
#         out_dir           = os.path.join(SIM, "figures"),
#     )