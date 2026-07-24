"""
Figure_4_mechanism.py  — six panels
Row 1: (a) WB closure  (b) Ridge CV  (c) Budyko KGE vs DA KGE scatter
Row 2: (d)(e)(f) WB flux heatmaps — Base, Budyko, DA  + shared vertical colorbar

Panel (c): x = Budyko KGE, y = DA KGE
           marker shape  = vegetation class (from omega thresholds)
           marker color  = omega value (continuous, YlGn colormap)
           xlim = ylim = (-5, 1)
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
import matplotlib.colors as mcolors

font_prop = FontProperties(fname=r"C:/Windows/Fonts/times.ttf")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"]   = 14

ROOT       = r"F:\Github_repos\Bayesian_DA_Budyko_modeling"
SIM        = os.path.join(ROOT,"Simulation_results","Simulation_with_BASIN_CALIB_PARAMS")
OUT_DIR    = os.path.join(SIM,"figures")
DIR_BASE   = os.path.join(SIM,"BASE_MODEL")
DIR_BUDYKO = os.path.join(SIM,"BUDYKO_MODEL")
DIR_DA     = os.path.join(SIM,"BUDYKO_DA")

SCEN_COLORS = {"Base":"#1528D8","Budyko":"#f30af3","DA":"darkgreen"}
SCEN_ORDER  = ["Base","Budyko","DA"]

FS       = 16
FS_TITLE = 14
FS_TICK  = 14
FS_ANNOT = 14

# ── Vegetation classification from omega ─────────────────────────────────
# Higher omega = more vegetated = more ET-controlled catchment
VEG_BINS   = [0,    2.0,      2.6,       3.0,   3.5,   4.0,   99]
VEG_LABELS = ["GL","WS + SL","CL/NVM",  "MF",  "DBF", "EF"]
VEG_MARKERS= {"GL":"o","WS + SL":"s","CL/NVM":"D","MF":"^","DBF":"P","EF":"*"}
VEG_DESC   = {
    "GL":      "Grassland",
    "WS + SL": "Woody Savanna + Shrubland",
    "CL/NVM":  "Cropland / Non-vegetated",
    "MF":      "Mixed Forest",
    "DBF":     "Deciduous Broadleaf Forest",
    "EF":      "Evergreen Forest",
}

def ls(d, pat):   return sorted(glob.glob(os.path.join(d, pat)))
def bid(fp, pfx): return os.path.basename(fp).replace(".feather","").replace(pfx,"")

def safe_arr(col):
    if isinstance(col, np.ndarray): return col.astype(float)
    return pd.to_numeric(col, errors="coerce").values.astype(float)

def bm(col):
    v = safe_arr(col); v = v[np.isfinite(v)]
    return float(np.nanmean(v)) if v.size > 0 else np.nan

def kge_score(o, s):
    o = safe_arr(o); s = safe_arr(s)
    mask = np.isfinite(o) & np.isfinite(s)
    if mask.sum() < 10: return np.nan
    o, s = o[mask], s[mask]
    r = np.corrcoef(o, s)[0,1]
    a = s.std()/(o.std()+1e-12)
    b = s.mean()/(o.mean()+1e-12)
    return float(1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2))

def cv_of(col):
    v = safe_arr(col); v = v[np.isfinite(v)]
    if v.size == 0: return np.nan
    m = np.nanmean(v); s = np.nanstd(v, ddof=1)
    return np.nan if (not np.isfinite(m) or m==0) else float(s/abs(m))

def omega_to_veg(omega):
    for i in range(len(VEG_BINS)-1):
        if VEG_BINS[i] <= omega < VEG_BINS[i+1]:
            return VEG_LABELS[i]
    return VEG_LABELS[-1]


# ════════════════════════════════════════════════════════════════════════
# PANEL (a) — Water-balance closure
# ════════════════════════════════════════════════════════════════════════
WB_CFG = {
    "Base":   (DIR_BASE,   "results_BASE_",      "ET_ke", "Q_base",   "dS_base"),
    "Budyko": (DIR_BUDYKO, "results_BUDYKO_",    "ET_B",  "Q_budyko", "dS_budyko"),
    "DA":     (DIR_DA,     "results_BUDYKO_DA_", "ET_ass","Q_ass",    "dS_ass"),
}

def compute_wb():
    out = {}
    for sc,(d,pf,et,q,ds) in WB_CFG.items():
        res = []
        for fp in ls(d, f"{pf}*.feather"):
            try:
                df = pd.read_feather(fp)
                if not all(c in df.columns for c in ["P",et,q,ds]): continue
                r = (safe_arr(df["P"]) - safe_arr(df[et])
                     - safe_arr(df[q])  - safe_arr(df[ds]))
                r = r[np.isfinite(r)]
                if r.size: res.append(float(np.mean(np.abs(r))))
            except: pass
        out[sc] = np.array(res)
        print(f"  WB {sc}: {len(res)} basins")
    return out

def plot_wb(ax, wb):
    bp = ax.boxplot([wb[s] for s in SCEN_ORDER],
                    positions=[1,2,3], widths=0.5,
                    patch_artist=True, showfliers=False)
    for patch,s in zip(bp["boxes"],SCEN_ORDER):
        patch.set_facecolor(SCEN_COLORS[s]); patch.set_alpha(0.65)
    for med in bp["medians"]: med.set(color="black", linewidth=2)
    for i,s in enumerate(SCEN_ORDER):
        m = float(np.nanmedian(wb[s]))
        ax.text(i+1, m, f"{m:.2f}", ha="center", va="bottom",
                fontsize=FS_ANNOT+1, fontweight="bold")
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(SCEN_ORDER, fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylabel("|P − ET − Q − ΔS| [mm/month]", fontsize=16)
    ax.set_title("(b)", loc="left",
                 fontweight="bold", fontsize=FS_TITLE)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)


# ════════════════════════════════════════════════════════════════════════
# PANEL (b) — Ridge plot CV
# ════════════════════════════════════════════════════════════════════════
TARGET_COLS    = ["Q_obs","ET_ke","Q_base","ET_B","omega_true","omega_MLR",
                  "Q_budyko","S_ens","G_ens","ET_ens","Q_ens"]
VARIABLE_ORDER = ["Q_obs","omega_true","omega_MLR","ET_ke","ET_B","ET_ens",
                  "Q_base","Q_budyko","Q_ens","S_ens","G_ens"]
VAR_LABEL = {
    "Q_obs":    "Q (Obs)",    "ET_ke":   "ET (Base)",
    "Q_base":   "Q (Base)",   "ET_B":    "ET (Budyko)",
    "Q_budyko": "Q (Budyko)", "ET_ens":  "ET (DA)",
    "Q_ens":    "Q (DA)",     "S_ens":   "S (DA)",
    "G_ens":    "G (DA)",
    "omega_true": r"$\omega$ (True)",
    "omega_MLR":  r"$\omega$ (MLR)",
}

def build_cv_data():
    cv_list = {}
    for fp in ls(DIR_BASE,"results_BASE_*.feather"):
        basin = bid(fp,"results_BASE_")
        try: df = pd.read_feather(fp)
        except: continue
        cvs = cv_list.get(basin,{})
        for c in ("Q_obs","ET_ke","Q_base"):
            if c in df.columns: cvs[c] = cv_of(df[c])
        cv_list[basin] = cvs

    for fp in ls(DIR_BUDYKO,"results_BUDYKO_*.feather"):
        basin = bid(fp,"results_BUDYKO_")
        try: df = pd.read_feather(fp)
        except: continue
        cvs = cv_list.get(basin,{})
        for c in ("omega_true","omega_MLR","ET_B","Q_budyko"):
            if c in df.columns: cvs[c] = cv_of(df[c])
        cv_list[basin] = cvs

    for fp in ls(DIR_DA,"enkf_ensemble_BUDYKO_DA_*.feather"):
        basin = bid(fp,"enkf_ensemble_BUDYKO_DA_")
        try: df = pd.read_feather(fp)
        except: continue
        cvs = cv_list.get(basin,{})
        for out,src in [("ET_ens","ET_ens_mean"),("Q_ens","Q_ens_mean"),
                         ("S_ens","S_ens_mean"),("G_ens","G_ens_mean")]:
            if src in df.columns: cvs[out] = cv_of(df[src])
        cv_list[basin] = cvs

    cv_df = pd.DataFrame.from_dict(cv_list, orient="index")
    cv_df.index.name = "basin"
    for c in TARGET_COLS:
        if c not in cv_df.columns: cv_df[c] = np.nan
    cv_df = cv_df[TARGET_COLS]

    cv_long = cv_df.reset_index().melt(id_vars="basin",
                var_name="variable", value_name="cv")
    cv_long["cv"] = pd.to_numeric(cv_long["cv"], errors="coerce")
    levels = list(reversed(VARIABLE_ORDER))
    cv_long["variable"] = pd.Categorical(cv_long["variable"],
                           categories=levels, ordered=True)
    return cv_long.sort_values("variable"), levels

def plot_ridge(ax, cv_long, levels):
    stats = (cv_long.groupby("variable", observed=True)["cv"]
             .agg(mean_cv="mean",
                  var_cv=lambda x: np.nanvar(x, ddof=1)).reset_index())
    stats["label"] = stats.apply(
        lambda r: "μ=NA" if pd.isna(r["mean_cv"])
        else f"μ={r['mean_cv']:.3f}, var={r['var_cv']:.3f}", axis=1)

    x_grid = np.linspace(-1, 5, 500)
    rc = cm.get_cmap("jet", len(levels))
    vc = {v: rc(i) for i,v in enumerate(levels)}

    for i,var in enumerate(levels):
        vals = cv_long.loc[cv_long["variable"]==var,"cv"].values
        vals = vals[np.isfinite(vals)]
        dens = np.zeros_like(x_grid)
        if vals.size >= 5:
            kde = gaussian_kde(vals)
            dens = kde(x_grid)
            if dens.max() > 0: dens = dens/dens.max()*0.8
        ax.fill_between(x_grid, i, i+dens, color=vc[var], alpha=0.9)
        ax.plot(x_grid, i+dens, color="black", lw=0.5)

    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels([VAR_LABEL[v] for v in levels],
                       fontproperties=font_prop, fontsize=FS_TICK)
    ax.set_xlabel("CV", fontsize=FS)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    ax.set_title("(a)", loc="left",
                 fontweight="bold", fontsize=FS_TITLE)
    for i,var in enumerate(levels):
        lab = stats.loc[stats["variable"]==var,"label"].values
        ax.text(5, i+0.4, lab[0] if lab.size else "μ=NA",
                ha="right", va="bottom", fontsize=10,
                fontproperties=font_prop)


# ════════════════════════════════════════════════════════════════════════
# PANEL (c) — Budyko KGE vs DA KGE
#   marker shape = vegetation class (from omega thresholds)
#   marker color = omega value (YlGn continuous colormap)
#   xlim = ylim = (-5, 1)
# ════════════════════════════════════════════════════════════════════════
def build_kge_scatter():
    base_data, budyko_data, da_data = {}, {}, {}

    for fp in ls(DIR_BASE,"results_BASE_*.feather"):
        basin = bid(fp,"results_BASE_")
        try: df = pd.read_feather(fp)
        except: continue
        if not all(c in df.columns for c in
                   ("time","Q_obs","Q_base","omega_true")): continue
        base_data[basin] = df.set_index("time") if "time" in df.columns else df

    for fp in ls(DIR_BUDYKO,"results_BUDYKO_*.feather"):
        basin = bid(fp,"results_BUDYKO_")
        try: df = pd.read_feather(fp)
        except: continue
        if not all(c in df.columns for c in ("time","Q_obs","Q_budyko")): continue
        budyko_data[basin] = df.set_index("time") if "time" in df.columns else df

    for fp in ls(DIR_DA,"results_BUDYKO_DA_*.feather"):
        basin = bid(fp,"results_BUDYKO_DA_")
        try: df = pd.read_feather(fp)
        except: continue
        if not all(c in df.columns for c in ("time","Q_obs","Q_ass")): continue
        da_data[basin] = df.set_index("time") if "time" in df.columns else df

    print(f"  Base:{len(base_data)}  Budyko:{len(budyko_data)}  DA:{len(da_data)}")

    recs = {}
    for basin, bdf in base_data.items():
        omega = bm(bdf["omega_true"])

        kg_bud = np.nan
        if basin in budyko_data:
            budf = budyko_data[basin]
            common = bdf.index.intersection(budf.index)
            if len(common) >= 10:
                kg_bud = kge_score(bdf.loc[common,"Q_obs"].values,
                                   budf.loc[common,"Q_budyko"].values)

        kg_da = np.nan
        if basin in da_data:
            dadf = da_data[basin]
            common = bdf.index.intersection(dadf.index)
            if len(common) >= 10:
                kg_da = kge_score(bdf.loc[common,"Q_obs"].values,
                                  dadf.loc[common,"Q_ass"].values)

        if np.isfinite(kg_bud) and np.isfinite(kg_da) and np.isfinite(omega):
            recs[basin] = {
                "kge_budyko": kg_bud,
                "kge_da":     kg_da,
                "omega":      omega,
                "veg":        omega_to_veg(omega),
            }

    df_out = pd.DataFrame.from_dict(recs, orient="index")
    print(f"  Scatter basins: {len(df_out)}")
    if len(df_out) > 0:
        print(f"  Veg counts:\n{df_out['veg'].value_counts()}")
    return df_out

def plot_scatter(ax, sdf):
    if len(sdf) == 0:
        ax.text(0.5,0.5,"No overlapping basins",
                ha="center",va="center",
                transform=ax.transAxes,fontsize=FS,color="red")
        ax.set_title("(c)",
                     loc="left",fontweight="bold",fontsize=FS_TITLE)
        return

    cmap  = plt.cm.bwr_r
    omega_vals = sdf["omega"]
    norm  = mcolors.Normalize(vmin=0.5,
                               vmax=3.5)

    sc = None
    for veg in VEG_LABELS:
        grp = sdf[sdf["veg"]==veg]
        if len(grp) == 0: continue
        mk = VEG_MARKERS[veg]
        sc = ax.scatter(grp["kge_budyko"], grp["kge_da"],
                        c=grp["omega"], cmap=cmap, norm=norm,
                        marker=mk, s=55, alpha=0.85,
                        edgecolors="gray", linewidths=0.4, zorder=4)

    # 1:1 line
    ax.plot([-5,1],[-5,1], "k--", lw=1.0, alpha=0.5, zorder=2)

    # # KGE = 0.5 reference lines
    # ax.axhline(0.5, color="gray", lw=0.8, ls=(0,(4,3)), alpha=0.5)
    # ax.axvline(0.5, color="gray", lw=0.8, ls=(0,(4,3)), alpha=0.5)

    # # First quadrant annotation
    # ax.text(0.97, 0.97,
    #         "Both KGE > 0.5",
    #         transform=ax.transAxes, ha="right", va="top",
    #         fontsize=FS_ANNOT, color="#1a6e1a",
    #         bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f9ee",
    #                   edgecolor="#1a6e1a", alpha=0.85))

    # Colorbar for omega
    if sc is not None:
        cb = plt.colorbar(sc,ax=ax,fraction=0.04,pad=0.02,shrink=1.0,aspect=35,extend="max")
        # cb.set_label(r"MLR$_{\omega}$", fontsize=FS_TICK)
        cb.set_label(r"$\omega_{\mathrm{MLR}}$", fontsize=FS_TICK)
        cb.ax.tick_params(labelsize=FS_TICK-1)

    # Marker legend for vegetation classes
    leg_handles = []
    for veg in VEG_LABELS:
        if veg not in sdf["veg"].values: continue
        leg_handles.append(
            Line2D([0],[0], marker=VEG_MARKERS[veg],
                   color="gray", markerfacecolor="gray",
                   markersize=7, linestyle="None",
                   label=f"{veg}"))
    ax.legend(handles=leg_handles, title="Vegetation types",
              frameon=1, fontsize=12,
              title_fontsize=14,
              loc="lower right", ncol=1)

    ax.set_xlim(-3, 1.2); ax.set_ylim(-3, 1.2)
    ax.set_xlabel("KGE (Budyko scenario)", fontsize=FS)
    ax.set_ylabel("KGE (DA scenario)",     fontsize=FS)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_title("(c)",
                 loc="left", fontweight="bold", fontsize=FS_TITLE)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)


# ════════════════════════════════════════════════════════════════════════
# PANELS (d)(e)(f) — WB flux heatmaps
# ════════════════════════════════════════════════════════════════════════
FLUX_CFG = {
    "Base":   {"dir":DIR_BASE,   "pat":"results_BASE_*.feather",
               "pfx":"results_BASE_",
               "cols":{"P":"P","PET":"PET","ET":"ET_ke","Q_obs":"Q_obs",
                        "Q_sim":"Q_base","S":"S_base","G":"G_base","dS":"dS_base"}},
    "Budyko": {"dir":DIR_BUDYKO, "pat":"results_BUDYKO_*.feather",
               "pfx":"results_BUDYKO_",
               "cols":{"P":"P","PET":"PET","ET":"ET_B","Q_obs":"Q_obs",
                        "Q_sim":"Q_budyko","S":"S_budyko",
                        "G":"G_budyko","dS":"dS_budyko"}},
    "DA":     {"dir":DIR_DA,     "pat":"results_BUDYKO_DA_*.feather",
               "pfx":"results_BUDYKO_DA_",
               "cols":{"P":"P","PET":"PET","ET":"ET_ass","Q_obs":"Q_obs",
                        "Q_sim":"Q_ass","S":"S_ass","G":"G_ass","dS":"dS_ass"}},
}

def build_flux_df(cfg):
    recs = {}
    for fp in ls(cfg["dir"], cfg["pat"]):
        basin = bid(fp, cfg["pfx"])
        try: df = pd.read_feather(fp)
        except: continue
        recs[basin] = {lbl: bm(df[src])
                       for lbl,src in cfg["cols"].items()
                       if src in df.columns}
    return pd.DataFrame.from_dict(recs, orient="index")

def plot_heatmap(ax, df, title, border_col, show_ylabel=True):
    corr = df.corr(method="pearson", min_periods=3)
    cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("lightgray")
    im = ax.imshow(np.ma.masked_invalid(corr.values),
                   cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    n = len(corr.columns)
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45,
                       ha="right", fontsize=FS_TICK)
    if show_ylabel:
        ax.set_yticks(range(n))
        ax.set_yticklabels(corr.columns, fontsize=FS_TICK)
    else:
        ax.set_yticks(range(n))
        ax.set_yticklabels([""] * n)
        ax.tick_params(axis="y", left=False)
    ax.set_xticks(np.arange(-0.5,n,1), minor=True)
    ax.set_yticks(np.arange(-0.5,n,1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(n):
        for j in range(n):
            v = corr.values[i,j]
            if np.isfinite(v):
                tc = "white" if abs(v) > 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=tc)
    for sp in ax.spines.values():
        sp.set_edgecolor("black"); sp.set_linewidth(0.6)
    ax.set_title(title, loc="left", fontweight="bold",
                 fontsize=FS_TITLE, color="black")
    return im


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════
def plot_mechanism_figure(out_dir, fname="Figure_4_mechanism"):
    print("Computing WB residuals...")
    wb = compute_wb()

    print("Building CV data for ridge plot...")
    cv_long, levels = build_cv_data()

    print("Building KGE scatter data...")
    sdf = build_kge_scatter()

    print("Building flux DataFrames for heatmaps...")
    flux_dfs = {sc: build_flux_df(FLUX_CFG[sc])
                for sc in ["Base","Budyko","DA"]}

    # ── Layout: 2 rows × 3 cols + colorbar col ───────────────────────
    fig = plt.figure(figsize=(12, 10), dpi=200)
    gs  = gridspec.GridSpec(2, 4,
                             left=0.05, right=0.97,
                             top=0.95, bottom=0.10,
                             hspace=0.21, wspace=0.35,
                             width_ratios=[1,1,1,0.025])

    # ax_a = fig.add_subplot(gs[0,0])
    # ax_b = fig.add_subplot(gs[0,1])
    ax_a = fig.add_subplot(gs[0,1])
    ax_b = fig.add_subplot(gs[0,0])
    ax_c = fig.add_subplot(gs[0,2])
    ax_d = fig.add_subplot(gs[1,0])
    ax_e = fig.add_subplot(gs[1,1])
    ax_f = fig.add_subplot(gs[1,2])
    # cax  = fig.add_subplot(gs[1,3])

    plot_wb(ax_a, wb)
    plot_ridge(ax_b, cv_long, levels)
    plot_scatter(ax_c, sdf)

    last_im = None
    for ax,sc,lbl,bc,sy in [
        (ax_d,"Base",  "(d) Base",   SCEN_COLORS["Base"],   True),
        (ax_e,"Budyko","(e) Budyko", SCEN_COLORS["Budyko"], False),
        (ax_f,"DA",    "(f) DA",     SCEN_COLORS["DA"],     False),
    ]:
        last_im = plot_heatmap(ax, flux_dfs[sc], lbl, bc,
                               show_ylabel=sy)

    # Shared vertical colorbar at right of (f)
    # cb = fig.colorbar(last_im, ax=ax_f, orientation="vertical",
    #                 fraction=0.046, pad=0.02, shrink=1.2, extend="max")
    cb = fig.colorbar(last_im,ax=ax_f,orientation="vertical",fraction=0.046,pad=0.02,shrink=1.0,aspect=35,extend="max")
    cb.set_label("Pearson r", fontsize=FS)
    cb.ax.tick_params(labelsize=FS_TICK)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{fname}.png")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.show(fig)
    return fig


plot_mechanism_figure(OUT_DIR)