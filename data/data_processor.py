# import os
# import io
# import numpy as np
# import pandas as pd
# import xarray as xr
# import geopandas as gpd
# import rioxarray
# import requests
# from glob import glob
# from tqdm import tqdm
# from shapely.geometry import mapping
# from rioxarray.exceptions import NoDataInBounds
# from typing import Tuple, List

# # ---------------------------------------------------------------------
# # PROJECT SETUP
# # ---------------------------------------------------------------------
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# # Paths
# NLDAS_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'input_data', 'NLDAS_NOAH0125_M_002_1994_2023')
# BASE_FOLDER_STREAMFLOW = os.path.join(PROJECT_ROOT, 'data', 'input_data', 'usgs_streamflow')
# NDVI_PATH = os.path.join(PROJECT_ROOT, 'data', 'input_data', 'NDVI', 'MOD13A2_NDVI_monthly_2000_2018_CAMELS.csv')
# EXTRACTED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')

# # Constants
# TIME_SLICE = slice("2000-01-01", "2014-12-31")
# N_TOP_BASINS = 670
# MISSING_DATA_THRESHOLD = 0.05
# LAMBDA = 2.45e6  # Latent heat of vaporization for water (J/kg)


# # ---------------------------------------------------------------------
# # Streamflow Utilities
# # ---------------------------------------------------------------------
# def read_streamflow_file(file_path):
#     try:
#         df = pd.read_csv(
#             file_path, sep=r'\s+', header=None,
#             names=['gaugeid', 'year', 'month', 'day', 'streamflow', 'flag']
#         )
#         df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
#         df = df[(df['streamflow'] >= 0) & (df['streamflow'] != -999.00)]
#         return df[['gaugeid', 'date', 'streamflow']]
#     except Exception:
#         return None


# def import_streamflow_data(base_folder):
#     if not os.path.exists(base_folder):
#         return pd.DataFrame()

#     all_data = []
#     for folder in range(1, 19):
#         folder_path = os.path.join(base_folder, f"{folder:02}")
#         if not os.path.exists(folder_path):
#             continue

#         file_paths = glob(os.path.join(folder_path, '*_streamflow_qc.txt'))
#         for file_path in file_paths:
#             df = read_streamflow_file(file_path)
#             if df is not None:
#                 gauge_id = os.path.basename(file_path).split('_')[0]
#                 df['gaugeid'] = gauge_id
#                 df['month'] = df['date'].dt.to_period('M')
#                 monthly_df = df.groupby(['gaugeid', 'month']).streamflow.mean().reset_index()
#                 monthly_df = monthly_df.pivot(index='month', columns='gaugeid', values='streamflow')
#                 all_data.append(monthly_df)

#     if not all_data:
#         return pd.DataFrame()

#     streamflow_data = pd.concat(all_data, axis=1)
#     streamflow_data = streamflow_data.loc[:, ~streamflow_data.columns.duplicated()].sort_index()
#     if 'month' in streamflow_data.index.names:
#         streamflow_data.index = streamflow_data.index.to_timestamp()
#     return streamflow_data


# # ---------------------------------------------------------------------
# # NLDAS Extraction with Caching
# # ---------------------------------------------------------------------
# def extract_or_load_nldas_data(
#     attrs_large: gpd.GeoDataFrame, BASIN_IDS: List[str], time_slice: slice
# ) -> Tuple[pd.DataFrame, ...]:
#     VARS_TO_CACHE = ["Rainf", "Evap", "Qsb", "Q_nldas_mm_monthly", "RootMoist", "SoilM_0_200cm", "PotEvap"]

#     # Load cached data if available
#     if os.path.exists(EXTRACTED_DATA_PATH):
#         try:
#             print(f"Loading cached NLDAS data from {EXTRACTED_DATA_PATH}...")
#             cached_df = pd.read_feather(EXTRACTED_DATA_PATH).set_index('time')
#             dfs = {}

#             for var_name in VARS_TO_CACHE:
#                 cols = [c for c in cached_df.columns if c.startswith(var_name + '_')]
#                 if not cols:
#                     raise KeyError(f"Missing {var_name} columns in cached data.")
#                 df_var = cached_df[cols]
#                 df_var.columns = [c.split('_')[-1] for c in cols]
#                 dfs[var_name] = df_var.loc[time_slice.start:time_slice.stop, BASIN_IDS]

#             return (
#                 dfs["Rainf"], dfs["PotEvap"], dfs["Evap"], dfs["Qsb"],
#                 dfs["RootMoist"], dfs["SoilM_0_200cm"], dfs["Q_nldas_mm_monthly"]
#             )
#         except Exception as e:
#             print(f"Error loading cached NLDAS data ({e}). Re-extracting...")
#             try:
#                 os.remove(EXTRACTED_DATA_PATH)
#             except:
#                 pass

#     # Otherwise, extract from NetCDF
#     print(f"Loading NLDAS NetCDF files from {NLDAS_DATA_DIR}...")
#     nc_files = glob(NLDAS_DATA_DIR + '/*.nc')
#     if not nc_files:
#         raise FileNotFoundError(f"No .nc files found in {NLDAS_DATA_DIR}")

#     FluxData_all = xr.open_mfdataset(
#         nc_files, concat_dim='time', combine='nested', coords='minimal',
#         compat='override', parallel=True
#     )
#     FluxData_all = FluxData_all.rio.write_crs("EPSG:4326", inplace=True)

#     vars_to_extract = ["Evap", "Rainf", "PotEvap", "Streamflow", "Qsb", "RootMoist", "SoilM_0_200cm"]
#     dfs = {}

#     for var_name in vars_to_extract:
#         var_da = FluxData_all[var_name].sel(time=time_slice)
#         data_dict = {}
#         for basin_id, row in tqdm(attrs_large.iterrows(), total=len(attrs_large), desc=f"NLDAS-{var_name}"):
#             geom = [mapping(row["geometry"])]
#             try:
#                 clipped = var_da.rio.clip(geom, attrs_large.crs, drop=True)
#                 basin_mean = clipped.mean(dim=["lat", "lon"], skipna=True)
#                 data_dict[basin_id] = basin_mean.to_pandas()
#             except NoDataInBounds:
#                 continue

#         df = pd.DataFrame(data_dict, index=var_da.time.values)
#         dfs[var_name] = df.reindex(BASIN_IDS, axis=1).sort_index()

#     # Convert PotEvap (W/m² → mm/month)
#     PotEvap_df = dfs['PotEvap'].copy()
#     days_in_month = PotEvap_df.index.days_in_month.values
#     seconds_in_month = days_in_month * 86400
#     for col in PotEvap_df.columns:
#         PotEvap_df[col] = PotEvap_df[col].values * seconds_in_month / (LAMBDA * 1000)
#     dfs['PotEvap'] = PotEvap_df

#     # Convert Streamflow (m³/s → mm/month)
#     Q_nldas_mm_monthly = dfs['Streamflow'].copy()
#     for basin in Q_nldas_mm_monthly.columns:
#         area_m2 = attrs_large.loc[basin, 'area_gages2'] * 1e6
#         days_in_month = Q_nldas_mm_monthly.index.days_in_month.values
#         Q_nldas_mm_monthly[basin] = (
#             Q_nldas_mm_monthly[basin].values * 86400 * days_in_month * 1000 / area_m2
#         )
#     dfs['Q_nldas_mm_monthly'] = Q_nldas_mm_monthly

#     # Cache combined results
#     # combined_dfs = {
#     #     var: df.rename(columns=lambda c: f"{var}_{c}")
#     #     for var, df in dfs.items() if var in VARS_TO_CACHE
#     # }
#     # final_df = pd.concat(combined_dfs.values(), axis=1)
#     # os.makedirs(os.path.dirname(EXTRACTED_DATA_PATH), exist_ok=True)
#     # final_df.reset_index(names=['time']).to_feather(EXTRACTED_DATA_PATH)
#     # print(f"✅ Cached to {EXTRACTED_DATA_PATH}")

#     # Save each variable as separate feather
#     CACHE_DIR = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\DA\DA_Github_repo\Bayesian_DA_Budyko_modeling\data\processed"
#     # CACHE_DIR = os.path.join(os.path.dirname(EXTRACTED_DATA_PATH), "extracted_variables_feather")
#     os.makedirs(CACHE_DIR, exist_ok=True)

#     for var_name, df in dfs.items():
#         if var_name not in VARS_TO_CACHE:
#             continue
#         cache_path = os.path.join(CACHE_DIR, f"{var_name}.feather")
#         df.reset_index(names=['time']).to_feather(cache_path)
#         print(f"✅ Saved {var_name} to {cache_path}")


#     return (
#         dfs["Rainf"], dfs["PotEvap"], dfs["Evap"], dfs["Qsb"],
#         dfs["Q_nldas_mm_monthly"], dfs["RootMoist"], dfs["SoilM_0_200cm"]
#     )


# # ---------------------------------------------------------------------
# # Main Loader Function
# # ---------------------------------------------------------------------
# def load_and_prepare_data() -> Tuple[pd.DataFrame, ...]:
#     print("Loading CAMELS attributes and streamflow...")

#     # CAMELS attributes
#     try:
#         r = requests.get(
#             "https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/data/contents/camels_attributes_v2.0.feather"
#         )
#         attrs = gpd.read_feather(io.BytesIO(r.content)).reset_index(drop=False)
#         attrs['geometry_points'] = attrs['geometry'].centroid
#         attrs = attrs.set_geometry('geometry_points')
#     except Exception as e:
#         print(f"Error loading CAMELS attributes: {e}")
#         return (None,) * 11

#     # USGS Streamflow
#     streamflow_data = import_streamflow_data(BASE_FOLDER_STREAMFLOW)
#     if streamflow_data.empty:
#         print("Error loading streamflow data.")
#         return (None,) * 11

#     streamflow_data = streamflow_data.loc['1984-01-01':'2014-12-31']
#     valid_cols = streamflow_data.columns[streamflow_data.isna().mean() <= MISSING_DATA_THRESHOLD]
#     streamflow_data = streamflow_data[valid_cols].interpolate(method='linear').ffill().bfill()

#     attrs['gauge_id'] = attrs['gauge_id'].astype(str).str.strip().str.zfill(8)
#     attrs.set_index('gauge_id', inplace=True)
#     streamflow_data.columns = streamflow_data.columns.astype(str).str.strip().str.zfill(8)

#     area_km2 = attrs['area_gages2'].loc[streamflow_data.columns]

#     # Convert streamflow ft³/s → mm/month
#     Q_USGS = pd.DataFrame(index=streamflow_data.index, columns=streamflow_data.columns)
#     for col in streamflow_data.columns:
#         days = streamflow_data.index.days_in_month.values
#         area_m2 = area_km2.loc[col] * 1e6
#         Q_USGS[col] = streamflow_data[col].values * 0.028316846 * 86400 * days * 1000 / area_m2
#         CACHE_DIR = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\DA\DA_Github_repo\Bayesian_DA_Budyko_modeling\data\processed"
#         os.makedirs(CACHE_DIR, exist_ok=True)
#         q_usgs_path = os.path.join(CACHE_DIR, "Q_USGS.feather")
#         Q_USGS.reset_index(names=['time']).to_feather(q_usgs_path)

#     # NDVI
#     print(f"Loading NDVI data from {NDVI_PATH}...")
#     try:
#         NDVI = pd.read_csv(NDVI_PATH).set_index('month')
#     except FileNotFoundError:
#         print(f"NDVI file not found: {NDVI_PATH}")
#         return (None,) * 11

#     NDVI.replace(-9999, np.nan, inplace=True)
#     NDVI.drop(columns=[c for c in NDVI if NDVI[c].isna().mean() > MISSING_DATA_THRESHOLD], inplace=True)
#     NDVI = NDVI.apply(lambda x: x.fillna(x.mean()), axis=0)
#     NDVI.index = pd.PeriodIndex(NDVI.index, freq='M').to_timestamp()

#     # Filter top basins
#     common_gauges = list(set(attrs.index) & set(Q_USGS.columns))
#     attrs_large = attrs.loc[common_gauges].sort_values('area_gages2', ascending=False).head(N_TOP_BASINS)
#     BASIN_IDS = attrs_large.index.tolist()
#     attrs_large = attrs_large.to_crs("EPSG:4326")

#     print(f"Extracting/Loading NLDAS variables for {len(BASIN_IDS)} basins...")
#     try:
#         Rainf, PotEvap, Evap, Qsb, Q_nldas, RootMoist, SoilM = extract_or_load_nldas_data(
#             attrs_large, BASIN_IDS, TIME_SLICE
#         )
#     except Exception as e:
#         print(f"NLDAS extraction failed: {e}")
#         return (None,) * 11

#     Q_USGS = Q_USGS.loc[TIME_SLICE.start:TIME_SLICE.stop, BASIN_IDS]
#     NDVI = NDVI.loc[TIME_SLICE.start:TIME_SLICE.stop, BASIN_IDS]

#     NDVI_norm = (NDVI - NDVI.min()) / (NDVI.max() - NDVI.min())
#     M_df = (NDVI_norm - NDVI_norm.min().min()) / (NDVI_norm.max().max() - NDVI_norm.min().min())

#     slope = attrs_large['slope_mean']
#     Slope_df = pd.DataFrame({b: slope.loc[b] for b in BASIN_IDS}, index=Rainf.index)

#     S_init_df = RootMoist.iloc[[0]]
#     G_init_df = SoilM.iloc[[0]]

#     return (
#         Rainf, PotEvap, Evap, Qsb, M_df, Slope_df,
#         Q_nldas, Q_USGS, S_init_df, G_init_df, SoilM
#     )

# # =====================================================================
# # 🔥 MAIN EXECUTION BLOCK (ADDED)
# # =====================================================================
# if __name__ == '__main__':
#     print("Starting data loading and preparation...")
    
#     # Call the main function to load, process, and potentially cache the data
#     try:
#         results = load_and_prepare_data()
        
#         # Check if loading was successful
#         if all(df is not None for df in results) and results[7] is not None and not results[7].empty:
#              print(f"✅ Data loading complete for {len(results[7].columns)} basins.")
#              print(f"   Time range: {results[7].index.min().strftime('%Y-%m')} to {results[7].index.max().strftime('%Y-%m')}.")
#              print("   Run `calibration.py` to proceed with model calibration.")
#         else:
#             print("❌ FATAL: Data loading failed. Check error messages above.")
            
#     except Exception as e:
#         print(f"❌ An unexpected error occurred during data processing: {e}")




import os
import io
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray
import requests
from glob import glob
from tqdm import tqdm
from shapely.geometry import mapping
from rioxarray.exceptions import NoDataInBounds
from typing import Tuple, List

# ---------------------------------------------------------------------
# PROJECT SETUP
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Paths
NLDAS_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'input_data', 'NLDAS_NOAH0125_M_002_1994_2023')
BASE_FOLDER_STREAMFLOW = os.path.join(PROJECT_ROOT, 'data', 'input_data', 'usgs_streamflow')
NDVI_PATH = os.path.join(PROJECT_ROOT, 'data', 'input_data', 'NDVI', 'MOD13A2_NDVI_monthly_2000_2018_CAMELS.csv')
EXTRACTED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')


# Paths
NLDAS_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'input_data', 'NLDAS_NOAH0125_M_002_1994_2023')
BASE_FOLDER_STREAMFLOW = os.path.join(PROJECT_ROOT, 'data', 'input_data', 'usgs_streamflow')
NDVI_PATH = os.path.join(PROJECT_ROOT, 'data', 'input_data', 'NDVI', 'MOD13A2_NDVI_monthly_2000_2018_CAMELS.csv')
EXTRACTED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Constants
TIME_SLICE = slice("2000-01-01", "2014-12-31")
N_TOP_BASINS = 670
MISSING_DATA_THRESHOLD = 0.05
# Removed LAMBDA = 2.45e6 as Lv is now calculated dynamically

# ---------------------------------------------------------------------
# Streamflow Utilities
# ---------------------------------------------------------------------
def read_streamflow_file(file_path):
    try:
        df = pd.read_csv(
            file_path, sep=r'\s+', header=None,
            names=['gaugeid', 'year', 'month', 'day', 'streamflow', 'flag']
        )
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df = df[(df['streamflow'] >= 0) & (df['streamflow'] != -999.00)]
        return df[['gaugeid', 'date', 'streamflow']]
    except Exception:
        return None


def import_streamflow_data(base_folder):
    if not os.path.exists(base_folder):
        return pd.DataFrame()

    all_data = []
    for folder in range(1, 19):
        folder_path = os.path.join(base_folder, f"{folder:02}")
        if not os.path.exists(folder_path):
            continue

        file_paths = glob(os.path.join(folder_path, '*_streamflow_qc.txt'))
        for file_path in file_paths:
            df = read_streamflow_file(file_path)
            if df is not None:
                gauge_id = os.path.basename(file_path).split('_')[0]
                df['gaugeid'] = gauge_id
                df['month'] = df['date'].dt.to_period('M')
                monthly_df = df.groupby(['gaugeid', 'month']).streamflow.mean().reset_index()
                monthly_df = monthly_df.pivot(index='month', columns='gaugeid', values='streamflow')
                all_data.append(monthly_df)

    if not all_data:
        return pd.DataFrame()

    streamflow_data = pd.concat(all_data, axis=1)
    streamflow_data = streamflow_data.loc[:, ~streamflow_data.columns.duplicated()].sort_index()
    if 'month' in streamflow_data.index.names:
        streamflow_data.index = streamflow_data.index.to_timestamp()
    return streamflow_data


# ---------------------------------------------------------------------
# NLDAS Extraction with Caching
# ---------------------------------------------------------------------
def extract_or_load_nldas_data(
    attrs_large: gpd.GeoDataFrame, BASIN_IDS: List[str], time_slice: slice
) -> Tuple[pd.DataFrame, ...]:
    # ADDED 'AvgSurfT' to VARS_TO_CACHE for proper loading/extraction
    VARS_TO_CACHE = ["Rainf", "Evap", "Qsb", "Q_nldas_mm_monthly", "RootMoist", "SoilM_0_200cm", "PotEvap", "AvgSurfT"]

    # Load cached data if available
    if os.path.exists(EXTRACTED_DATA_PATH):
        try:
            print(f"Loading cached NLDAS data from {EXTRACTED_DATA_PATH}...")
            cached_df = pd.read_feather(EXTRACTED_DATA_PATH).set_index('time')
            dfs = {}

            for var_name in VARS_TO_CACHE:
                cols = [c for c in cached_df.columns if c.startswith(var_name + '_')]
                if not cols:
                    # Allow missing AvgSurfT if older cache is used, but log it
                    if var_name == "AvgSurfT":
                        continue
                    raise KeyError(f"Missing {var_name} columns in cached data.")
                df_var = cached_df[cols]
                df_var.columns = [c.split('_')[-1] for c in cols]
                dfs[var_name] = df_var.loc[time_slice.start:time_slice.stop, BASIN_IDS]

            # Re-run the PotEvap conversion if AvgSurfT is missing from cache, or assume it's converted
            
            return (
                dfs["Rainf"], dfs["PotEvap"], dfs["Evap"], dfs["Qsb"],
                dfs["Q_nldas_mm_monthly"], dfs["RootMoist"], dfs["SoilM_0_200cm"]
            )
        except Exception as e:
            print(f"Error loading cached NLDAS data ({e}). Re-extracting...")
            try:
                os.remove(EXTRACTED_DATA_PATH)
            except:
                pass

    # Otherwise, extract from NetCDF
    print(f"Loading NLDAS NetCDF files from {NLDAS_DATA_DIR}...")
    nc_files = glob(NLDAS_DATA_DIR + '/*.nc')
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {NLDAS_DATA_DIR}")

    FluxData_all = xr.open_mfdataset(
        nc_files, concat_dim='time', combine='nested', coords='minimal',
        compat='override', parallel=True
    )
    FluxData_all = FluxData_all.rio.write_crs("EPSG:4326", inplace=True)

    # ADDED 'AvgSurfT' to vars_to_extract
    vars_to_extract = ["Evap", "Rainf", "PotEvap", "Streamflow", "Qsb", "RootMoist", "SoilM_0_200cm", "AvgSurfT"]
    dfs = {}

    for var_name in vars_to_extract:
        var_da = FluxData_all[var_name].sel(time=time_slice)
        data_dict = {}
        for basin_id, row in tqdm(attrs_large.iterrows(), total=len(attrs_large), desc=f"NLDAS-{var_name}"):
            geom = [mapping(row["geometry"])]
            try:
                clipped = var_da.rio.clip(geom, attrs_large.crs, drop=True)
                basin_mean = clipped.mean(dim=["lat", "lon"], skipna=True)
                data_dict[basin_id] = basin_mean.to_pandas()
            except NoDataInBounds:
                continue

        df = pd.DataFrame(data_dict, index=var_da.time.values)
        dfs[var_name] = df.reindex(BASIN_IDS, axis=1).sort_index()

    # ---------------------------------------------------------------------------------
    # 🔥 CORRECTED PotEvap (W/m² → mm/month) CONVERSION
    # ---------------------------------------------------------------------------------
    print("Applying temperature-dependent conversion for PotEvap (W/m² → mm/month)...")
    
    # 1. Get the time-related information (required for the conversion)
    time_index = dfs['PotEvap'].index
    # Calculate seconds in each month
    seconds_in_month_series = time_index.to_series().apply(
        lambda x: pd.Period(x, freq='M').days_in_month * 86400
    ).values

    # 2. Calculate Temperature-Dependent Latent Heat of Vaporization (Lv)
    
    # Convert AvgSurfT from Kelvin (K) to Celsius (C)
    T_c_df = dfs['AvgSurfT'] - 273.15 
    
    # Calculate Lv (Latent Heat of Vaporization) in J/kg
    # Lv = (2.501 - 0.00236 * Tc) * 1e6
    Lv_df = (2.501 - 0.00236 * T_c_df) * 1e6
    
    # 3. Calculate PET_mm (PotEvap in mm/month)
    PotEvap_df = dfs['PotEvap'].copy()
    
    # Create a DataFrame of the conversion factor (Seconds in Month / Lv)
    # The 1D array seconds_in_month_series is broadcast across the columns of Lv_df
    conversion_factor_df = pd.DataFrame(
        seconds_in_month_series[:, np.newaxis] / Lv_df.values,
        index=Lv_df.index, 
        columns=Lv_df.columns
    )

    # Apply the conversion: (W/m2) * (s / (J/kg)) = kg/m2 (which is numerically equal to mm)
    PET_mm_df = PotEvap_df * conversion_factor_df

    # 4. Clip at 0.0 (PotEvap cannot be negative)
    PET_mm_df = PET_mm_df.clip(lower=0.0) 
    
    # Replace the old PotEvap DF with the new PET_mm DF
    dfs['PotEvap'] = PET_mm_df
    # ---------------------------------------------------------------------------------

    # Convert Streamflow (m³/s → mm/month)
    Q_nldas_mm_monthly = dfs['Streamflow'].copy()
    for basin in Q_nldas_mm_monthly.columns:
        area_m2 = attrs_large.loc[basin, 'area_gages2'] * 1e6
        days_in_month = Q_nldas_mm_monthly.index.days_in_month.values
        Q_nldas_mm_monthly[basin] = (
            Q_nldas_mm_monthly[basin].values * 86400 * days_in_month * 1000 / area_m2
        )
    dfs['Q_nldas_mm_monthly'] = Q_nldas_mm_monthly

    # Cache combined results
    CACHE_DIR = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\DA\DA_Github_repo\Bayesian_DA_Budyko_modeling\data\processed"
    os.makedirs(CACHE_DIR, exist_ok=True)

    for var_name, df in dfs.items():
        if var_name not in VARS_TO_CACHE:
            continue
        # Skip caching the intermediate AvgSurfT variable
        if var_name == "AvgSurfT":
            continue
            
        cache_path = os.path.join(CACHE_DIR, f"{var_name}.feather")
        df.reset_index(names=['time']).to_feather(cache_path)
        print(f"✅ Saved {var_name} to {cache_path}")


    return (
        dfs["Rainf"], dfs["PotEvap"], dfs["Evap"], dfs["Qsb"],
        dfs["Q_nldas_mm_monthly"], dfs["RootMoist"], dfs["SoilM_0_200cm"]
    )


# ---------------------------------------------------------------------
# Main Loader Function
# ---------------------------------------------------------------------
def load_and_prepare_data() -> Tuple[pd.DataFrame, ...]:
    print("Loading CAMELS attributes and streamflow...")

    # CAMELS attributes
    try:
        r = requests.get(
            "https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/data/contents/camels_attributes_v2.0.feather"
        )
        attrs = gpd.read_feather(io.BytesIO(r.content)).reset_index(drop=False)
        attrs['geometry_points'] = attrs['geometry'].centroid
        attrs = attrs.set_geometry('geometry_points')
    except Exception as e:
        print(f"Error loading CAMELS attributes: {e}")
        return (None,) * 11

    # USGS Streamflow
    streamflow_data = import_streamflow_data(BASE_FOLDER_STREAMFLOW)
    if streamflow_data.empty:
        print("Error loading streamflow data.")
        return (None,) * 11

    streamflow_data = streamflow_data.loc['1984-01-01':'2014-12-31']
    valid_cols = streamflow_data.columns[streamflow_data.isna().mean() <= MISSING_DATA_THRESHOLD]
    streamflow_data = streamflow_data[valid_cols].interpolate(method='linear').ffill().bfill()

    attrs['gauge_id'] = attrs['gauge_id'].astype(str).str.strip().str.zfill(8)
    attrs.set_index('gauge_id', inplace=True)
    streamflow_data.columns = streamflow_data.columns.astype(str).str.strip().str.zfill(8)

    area_km2 = attrs['area_gages2'].loc[streamflow_data.columns]

    # Convert streamflow ft³/s → mm/month
    Q_USGS = pd.DataFrame(index=streamflow_data.index, columns=streamflow_data.columns)
    for col in streamflow_data.columns:
        days = streamflow_data.index.days_in_month.values
        area_m2 = area_km2.loc[col] * 1e6
        Q_USGS[col] = streamflow_data[col].values * 0.028316846 * 86400 * days * 1000 / area_m2
        CACHE_DIR = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\DA\DA_Github_repo\Bayesian_DA_Budyko_modeling\data\processed"
        os.makedirs(CACHE_DIR, exist_ok=True)
        q_usgs_path = os.path.join(CACHE_DIR, "Q_USGS.feather")
        Q_USGS.reset_index(names=['time']).to_feather(q_usgs_path)

    # NDVI
    print(f"Loading NDVI data from {NDVI_PATH}...")
    try:
        NDVI = pd.read_csv(NDVI_PATH).set_index('month')
    except FileNotFoundError:
        print(f"NDVI file not found: {NDVI_PATH}")
        return (None,) * 11

    NDVI.replace(-9999, np.nan, inplace=True)
    NDVI.drop(columns=[c for c in NDVI if NDVI[c].isna().mean() > MISSING_DATA_THRESHOLD], inplace=True)
    NDVI = NDVI.apply(lambda x: x.fillna(x.mean()), axis=0)
    NDVI.index = pd.PeriodIndex(NDVI.index, freq='M').to_timestamp()

    # Filter top basins
    common_gauges = list(set(attrs.index) & set(Q_USGS.columns))
    attrs_large = attrs.loc[common_gauges].sort_values('area_gages2', ascending=False).head(N_TOP_BASINS)
    BASIN_IDS = attrs_large.index.tolist()
    attrs_large = attrs_large.to_crs("EPSG:4326")

    print(f"Extracting/Loading NLDAS variables for {len(BASIN_IDS)} basins...")
    try:
        # Note: The function returns 7 variables, but the original unpacks 7.
        # Ensure the order is matched: Rainf, PotEvap, Evap, Qsb, Q_nldas, RootMoist, SoilM
        Rainf, PotEvap, Evap, Qsb, Q_nldas, RootMoist, SoilM = extract_or_load_nldas_data(
            attrs_large, BASIN_IDS, TIME_SLICE
        )
    except Exception as e:
        print(f"NLDAS extraction failed: {e}")
        return (None,) * 11

    Q_USGS = Q_USGS.loc[TIME_SLICE.start:TIME_SLICE.stop, BASIN_IDS]
    NDVI = NDVI.loc[TIME_SLICE.start:TIME_SLICE.stop, BASIN_IDS]

    NDVI_norm = (NDVI - NDVI.min()) / (NDVI.max() - NDVI.min())
    M_df = (NDVI_norm - NDVI_norm.min().min()) / (NDVI_norm.max().max() - NDVI_norm.min().min())

    slope = attrs_large['slope_mean']
    Slope_df = pd.DataFrame({b: slope.loc[b] for b in BASIN_IDS}, index=Rainf.index)

    S_init_df = RootMoist.iloc[[0]]
    G_init_df = SoilM.iloc[[0]]

    return (
        Rainf, PotEvap, Evap, Qsb, M_df, Slope_df,
        Q_nldas, Q_USGS, S_init_df, G_init_df, SoilM
    )

# =====================================================================
# 🔥 MAIN EXECUTION BLOCK (ADDED)
# =====================================================================
if __name__ == '__main__':
    print("Starting data loading and preparation...")
    
    # Call the main function to load, process, and potentially cache the data
    try:
        results = load_and_prepare_data()
        
        # Check if loading was successful
        if all(df is not None for df in results) and results[7] is not None and not results[7].empty:
             print(f"✅ Data loading complete for {len(results[7].columns)} basins.")
             print(f"   Time range: {results[7].index.min().strftime('%Y-%m')} to {results[7].index.max().strftime('%Y-%m')}.")
             print("   Run `calibration.py` to proceed with model calibration.")
        else:
            print("❌ FATAL: Data loading failed. Check error messages above.")
            
    except Exception as e:
        print(f"❌ An unexpected error occurred during data processing: {e}")



########### --- National Water Model Data ------------------------------------
# import pandas as pd
# import os
# from glob import glob

# data_dir = r"E:\Data\NWM"
# csv_files = glob(os.path.join(data_dir, "*.csv"))

# dfs = []

# for file in csv_files:
#     usgs_id = os.path.splitext(os.path.basename(file))[0]  # e.g. "1333000"
#     df = pd.read_csv(file, sep=',', parse_dates=['time'])
#     df = df.rename(columns={df.columns[1]: usgs_id})
#     dfs.append(df.set_index('time'))
# combined_df = pd.concat(dfs, axis=1)
# combined_df = combined_df.loc['2000-01-01':'2014-12-31']
# monthly_df = combined_df.resample('M').mean()
# monthly_df.to_feather(r'C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\DA\DA_Github_repo\Bayesian_DA_Budyko_modeling\data\processed\NMW.feather')



########### ----------------- NDVI and M ---------------------------
# import pandas as pd

# DATA_DIR = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\DA\DA_Github_repo\Bayesian_DA_Budyko_modeling\data\processed"
# # basin_id = "01013500"

# Evap_df = pd.read_feather(os.path.join(DATA_DIR, "EVap.feather")).set_index("time").dropna(axis=1)
# Qsb_df = pd.read_feather(os.path.join(DATA_DIR, "Qsb.feather")).set_index("time").dropna(axis=1)
# PET_df = pd.read_feather(os.path.join(DATA_DIR, "PotEvap.feather")).set_index("time").dropna(axis=1)
# M_df = pd.read_feather(os.path.join(DATA_DIR, "M.feather")).set_index("time").dropna(axis=1)#.set_index("time")
# Slope_df = pd.read_feather(os.path.join(DATA_DIR, "slope.feather")).dropna(axis=1)#.set_index("time", drop=False)  # or directly no time col if slope static
# basin_id = Evap_df.columns

# # Load and scale NDVI
# NDVI = pd.read_csv(
#     r'C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\DA\DA_Github_repo\Bayesian_DA_Budyko_modeling\data\input_data\NDVI\MOD13A2_NDVI_monthly_2000_2018_CAMELS.csv'
# )
# NDVI = NDVI.replace(-9999, 0)
# NDVI.iloc[:, 1:] = NDVI.iloc[:, 1:] * 0.0001

# NDVI.iloc[:, 0] = pd.to_datetime(NDVI.iloc[:, 0], format='%Y-%m').dt.strftime('%Y-%m-%d')
# NDVI_min = NDVI.iloc[:, 1:].min().min()
# NDVI_max = NDVI.iloc[:, 1:].max().max()
# NDVI_bsin = NDVI.set_index('time')
# NDVI_bsin = NDVI_bsin.loc['2000-01-01':'2014-12-01']
# NDVI_bsin = NDVI_bsin[Evap_df.columns]


# M_basin = (NDVI.iloc[:, 1:] - NDVI_min) / (NDVI_max - NDVI_min)
# M_basin = M_basin.clip(0, 1)

# # Add the same formatted date column
# M_basin.insert(0, NDVI.columns[0], NDVI.iloc[:, 0])
# M_basin = M_basin.set_index('time')
# M_basin = M_basin.loc['2000-01-01':'2014-12-01']
# M_basin = M_basin.reset_index()

# # Save to feather
# M_basin.to_feather(
#     r'C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\DA\DA_Github_repo\Bayesian_DA_Budyko_modeling\data\processed\M.feather'
# )
# NDVI.to_feather(
#     r'C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\DA\DA_Github_repo\Bayesian_DA_Budyko_modeling\data\processed\NDVI.feather'
# )

# print("✅ Saved successfully — dates formatted as 'YYYY-MM-DD' (no hours).")


## ------------------slope -----------------

# DATA_DIR = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\DA\DA_Github_repo\Bayesian_DA_Budyko_modeling\data\processed"

# Evap_df = pd.read_feather(os.path.join(DATA_DIR, "EVap.feather")).set_index("time").dropna(axis=1)
# Qsb_df = pd.read_feather(os.path.join(DATA_DIR, "Qsb.feather")).set_index("time").dropna(axis=1)
# PET_df = pd.read_feather(os.path.join(DATA_DIR, "PotEvap.feather")).set_index("time").dropna(axis=1)
# M_df = pd.read_feather(os.path.join(DATA_DIR, "M.feather")).set_index("time").dropna(axis=1)
# Slope_df = pd.read_feather(os.path.join(DATA_DIR, "slope.feather")).dropna(axis=1)#.set_index("time", drop=False)  # or directly no time col if slope static
# basin_id = Evap_df.columns
# # Subset one basin column and align indexes
# Evap = Evap_df[basin_id]
# Qsb = Qsb_df[basin_id]
# PET = PET_df[basin_id]
# M = M_df[basin_id]
# Slope = Slope_df[basin_id]


# import io
# import requests
# r = requests.get("https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/data/contents/camels_attributes_v2.0.feather")
# attrs = gpd.read_feather(io.BytesIO(r.content))
# r = requests.get("https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/data/contents/camels_attrs_v2_streamflow_v1p2.nc")
# # attrs = attrs.reset_index(drop=False)


# slope_series = attrs.loc[basin_id]['slope_mean']

# slope_df = slope_series.to_frame().T
# slope_df.index.name = None
# slope_df.columns.name = None
# slope_df.index
# slope_df

# Slope_basin_full = pd.DataFrame(np.tile(Slope_df.values, (len(Evap_df.index), 1)), 
#     index=Evap_df.index,
#     columns=Slope_df.columns
# )
# Slope_basin_full

# Slope_basin_full.to_feather(
#     r'C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\DA\DA_Github_repo\Bayesian_DA_Budyko_modeling\data\processed\slope.feather'
# )