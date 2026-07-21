
# ============================================================
# Global style
# ============================================================
plt.rcParams["font.family"] = "Georgia"
plt.rcParams["font.size"] = 12

# ============================================================
# Paths to results folders
# ============================================================
RESULT_DIR_BASE   = r"F:\Github_repos\Bayesian_DA_Budyko_modeling\Simulation_results\Simulation_with_BASIN_CALIB_PARAMS\BASE_MODEL"
RESULT_DIR_BUDYKO = r"F:\Github_repos\Bayesian_DA_Budyko_modeling\Simulation_results\Simulation_with_BASIN_CALIB_PARAMS\BUDYKO_MODEL"
RESULT_DIR_DA     = r"F:\Github_repos\Bayesian_DA_Budyko_modeling\Simulation_results\Simulation_with_BASIN_CALIB_PARAMS\BUDYKO_DA"

METRIC_BASE_CSV   = r"F:\Github_repos\Bayesian_DA_Budyko_modeling\Simulation_results\Simulation_with_BASIN_CALIB_PARAMS\BASE_MODEL\metrics_BASE.csv"
METRIC_BUDYKO_CSV = r"F:\Github_repos\Bayesian_DA_Budyko_modeling\Simulation_results\Simulation_with_BASIN_CALIB_PARAMS\BUDYKO_MODEL\metrics_BUDYKO.csv"
METRIC_DA_CSV     = r"F:\Github_repos\Bayesian_DA_Budyko_modeling\Simulation_results\Simulation_with_BASIN_CALIB_PARAMS\BUDYKO_DA\metrics_BUDYKO_DA.csv"

SFET_PATH         = r"F:\Github_repos\Bayesian_DA_Budyko_modeling\data\processed\SFET.feather"

# ============================================================
# Helper: gauge_id from filename
# ============================================================
def get_gauge_id(fname: str) -> str:
    base = os.path.basename(fname)
    if base.endswith(".feather"):
        base = base[:-8]
    parts = base.split("_")
    return parts[-1].strip()

# ============================================================
# Helper: read "results_*.feather" from a folder
# ============================================================
def read_results_folder(result_dir: str, target_cols: list, desc_name: str) -> pd.DataFrame:
    files = [
        f for f in os.listdir(result_dir)
        if f.startswith("results_") and f.endswith(".feather")
    ]

    records = []

    for fname in tqdm(files, desc=f"Processing results in {desc_name}"):
        basin_id = get_gauge_id(fname)
        fpath = os.path.join(result_dir, fname)

        try:
            df = feather.read_feather(fpath)
            rec = {"gauge_id": basin_id}

            for c in target_cols:
                if c in df.columns:
                    rec[c] = pd.to_numeric(df[c], errors="coerce").mean(skipna=True)
                else:
                    rec[c] = np.nan

            records.append(rec)

        except Exception as e:
            print(f"⚠️ Skipping {basin_id} in {desc_name}: {e}")

    out = pd.DataFrame(records).drop_duplicates(subset=["gauge_id"])
    out = out.sort_values("gauge_id").reset_index(drop=True)
    return out

# ============================================================
# Helper: read "enkf_ensemble_BUDYKO_DA_*.feather"
# ============================================================
def read_ensemble_folder(result_dir: str, desc_name: str) -> pd.DataFrame:
    files = [
        f for f in os.listdir(result_dir)
        if f.startswith("enkf_ensemble_BUDYKO_DA_") and f.endswith(".feather")
    ]

    records = []

    for fname in tqdm(files, desc=f"Processing ensemble in {desc_name}"):
        basin_id = get_gauge_id(fname)
        fpath = os.path.join(result_dir, fname)

        try:
            df = feather.read_feather(fpath)

            rec = {
                "gauge_id": basin_id,
                "ET_ens_mean": pd.to_numeric(df["ET_ens_mean"], errors="coerce").mean(skipna=True) if "ET_ens_mean" in df.columns else np.nan,
                "Q_ens_mean":  pd.to_numeric(df["Q_ens_mean"],  errors="coerce").mean(skipna=True) if "Q_ens_mean"  in df.columns else np.nan,
                "S_ens_mean":  pd.to_numeric(df["S_ens_mean"],  errors="coerce").mean(skipna=True) if "S_ens_mean"  in df.columns else np.nan,
                "G_ens_mean":  pd.to_numeric(df["G_ens_mean"],  errors="coerce").mean(skipna=True) if "G_ens_mean"  in df.columns else np.nan,
            }

            records.append(rec)

        except Exception as e:
            print(f"⚠️ Skipping ensemble {fname}: {e}")

    out = pd.DataFrame(records).drop_duplicates(subset=["gauge_id"])
    out = out.sort_values("gauge_id").reset_index(drop=True)
    return out

# ============================================================
# Helper: safe merge without duplicate non-key columns
# ============================================================
def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: str = "gauge_id", how: str = "left") -> pd.DataFrame:
    overlap = [c for c in right.columns if c in left.columns and c != on]
    if overlap:
        right = right.drop(columns=overlap)
    return left.merge(right, on=on, how=how)

# ============================================================
# Target columns
# ============================================================
base_cols = [
    "P", "PET", "ET_ke", "S_base", "G_base", "Q_base"
]

budyko_cols = [
    "ET_B", "omega_true", "omega_MLR", "S_budyko", "G_budyko", "Q_obs", "Q_budyko"
]

da_results_cols = [
    "ET_ass", "Q_ass", "Q_ens"
]

# ============================================================
# Read simulation summaries
# ============================================================
base_df   = read_results_folder(RESULT_DIR_BASE, base_cols, "BASE_MODEL")
budyko_df = read_results_folder(RESULT_DIR_BUDYKO, budyko_cols, "BUDYKO_MODEL")
da_df_results  = read_results_folder(RESULT_DIR_DA, da_results_cols, "BUDYKO_DA (results)")
da_df_ensemble = read_ensemble_folder(RESULT_DIR_DA, "BUDYKO_DA (ensemble)")

da_df = safe_merge(da_df_results, da_df_ensemble, on="gauge_id", how="left")

all_simu_mean_505_basins = safe_merge(base_df, budyko_df, on="gauge_id", how="outer")
all_simu_mean_505_basins = safe_merge(all_simu_mean_505_basins, da_df, on="gauge_id", how="outer")

# standardize gauge_id
all_simu_mean_505_basins["gauge_id"] = all_simu_mean_505_basins["gauge_id"].astype(str).str.zfill(8)

# ============================================================
# Read metrics CSVs
# Keep ONLY the columns you actually need to avoid _x / _y
# ============================================================
KGE_Base = pd.read_csv(METRIC_BASE_CSV)
KGE_Budyko = pd.read_csv(METRIC_BUDYKO_CSV)
KGE_DA = pd.read_csv(METRIC_DA_CSV)

KGE_Base["gauge_id"] = KGE_Base["gauge_id"].astype(str).str.zfill(8)
KGE_Budyko["gauge_id"] = KGE_Budyko["gauge_id"].astype(str).str.zfill(8)
KGE_DA["gauge_id"] = KGE_DA["gauge_id"].astype(str).str.zfill(8)

KGE_Base = KGE_Base[["gauge_id", "KGE", "NSE"]].rename(
    columns={"KGE": "KGE_Base", "NSE": "NSE_Base"}
)

KGE_Budyko = KGE_Budyko[["gauge_id", "KGE", "NSE"]].rename(
    columns={"KGE": "KGE_Budyko", "NSE": "NSE_Budyko"}
)

KGE_DA = KGE_DA[["gauge_id", "KGE", "NSE"]].rename(
    columns={"KGE": "KGE_DA", "NSE": "NSE_DA"}
)

# ============================================================
# Read CAMELS attributes
# ============================================================
attr_url = "https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/data/contents/camels_attributes_v2.0.feather?download=1"
r = requests.get(attr_url)
r.raise_for_status()

attrs = gpd.read_feather(io.BytesIO(r.content))
attrs = attrs.reset_index(drop=False)

# standardize gauge_id
attrs["gauge_id"] = attrs["gauge_id"].astype(str).str.zfill(8)

# keep only needed columns
attrs_keep = [
    "gauge_id",
    "q_mean",
    "runoff_ratio",
    "slope_fdc",
    "baseflow_index",
    "stream_elas",
    "p_mean",
    "pet_mean",
    "p_seasonality",
    "aridity",
    "frac_forest",
    "lai_max",
    "lai_diff",
    "gvf_max",
    "gvf_diff",
    "dom_land_cover_frac",
    "dom_land_cover",
    "elev_mean",
    "slope_mean",
    "geometry",
]
attrs = attrs[attrs_keep].copy()

# ============================================================
# Land cover mapping
# ============================================================
land_cover_mapping = {
    "Croplands": "CL/NVM",
    "cropland/natural vegetation mosaic": "CL/NVM",
    "Deciduous Broadleaf Forest": "DBF",
    "Evergreen Needleleaf Forest": "EF",
    "Evergreen Broadleaf Forest": "EF",
    "Mixed Forests": "MF",
    "Grasslands": "GL",
    "Savannas": "WS + SL",
    "Woody Savannas": "WS + SL",
    "Closed Shrublands": "WS + SL",
    "Open Shrublands": "WS + SL"
}

# ============================================================
# Merge everything cleanly
# ============================================================
attrs_all_simu_mean_505_basins = safe_merge(all_simu_mean_505_basins, attrs, on="gauge_id", how="left")
attrs_all_simu_mean_505_basins = safe_merge(attrs_all_simu_mean_505_basins, KGE_Base, on="gauge_id", how="left")
attrs_all_simu_mean_505_basins = safe_merge(attrs_all_simu_mean_505_basins, KGE_Budyko, on="gauge_id", how="left")
attrs_all_simu_mean_505_basins = safe_merge(attrs_all_simu_mean_505_basins, KGE_DA, on="gauge_id", how="left")

# ============================================================
# Derived columns
# ============================================================
attrs_all_simu_mean_505_basins["dom_land_cover_short"] = (
    attrs_all_simu_mean_505_basins["dom_land_cover"].map(land_cover_mapping)
)

attrs_all_simu_mean_505_basins["AI_base"] = (
    attrs_all_simu_mean_505_basins["PET"] / attrs_all_simu_mean_505_basins["P"]
)
attrs_all_simu_mean_505_basins["EI_base"] = (
    attrs_all_simu_mean_505_basins["ET_ke"] / attrs_all_simu_mean_505_basins["P"]
)

attrs_all_simu_mean_505_basins["AI_B"] = (
    attrs_all_simu_mean_505_basins["PET"] / attrs_all_simu_mean_505_basins["P"]
)
attrs_all_simu_mean_505_basins["EI_B"] = (
    attrs_all_simu_mean_505_basins["ET_B"] / attrs_all_simu_mean_505_basins["P"]
)

attrs_all_simu_mean_505_basins["AI_ass"] = (
    attrs_all_simu_mean_505_basins["PET"] / attrs_all_simu_mean_505_basins["P"]
)
attrs_all_simu_mean_505_basins["EI_ass"] = (
    attrs_all_simu_mean_505_basins["ET_ass"] / attrs_all_simu_mean_505_basins["P"]
)

# ============================================================
# Load SFET
# ============================================================
SFET = pd.read_feather(SFET_PATH)

SFET["time"] = pd.to_datetime(SFET["time"])
SFET = SFET.set_index("time").sort_index()
SFET.columns = SFET.columns.astype(str).str.strip()
SFET = SFET.loc[:, ~SFET.columns.duplicated()]

SFET_monthly = SFET.resample("MS").sum()
SFET_monthly.index.name = "time"

SFET_longterm = SFET_monthly.mean(axis=0, skipna=True).reset_index()
SFET_longterm.columns = ["gauge_id", "SFET"]
SFET_longterm["gauge_id"] = SFET_longterm["gauge_id"].astype(str).str.zfill(8)

attrs_all_simu_mean_505_basins = safe_merge(
    attrs_all_simu_mean_505_basins,
    SFET_longterm,
    on="gauge_id",
    how="left"
)

# optional final cleanup
# attrs_all_simu_mean_505_basins = attrs_all_simu_mean_505_basins.dropna(axis=0)

# ============================================================
# Final check
# ============================================================
# print(attrs_all_simu_mean_505_basins.columns)

dup_cols = attrs_all_simu_mean_505_basins.columns[
    attrs_all_simu_mean_505_basins.columns.duplicated()
]
print("Duplicated column names:", dup_cols.tolist())

attrs_all_simu_mean_505_basins_BASIN_CAL = attrs_all_simu_mean_505_basins
attrs_all_simu_mean_505_basins_BASIN_CAL