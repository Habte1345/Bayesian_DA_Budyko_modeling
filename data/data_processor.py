# import os
# import numpy as np
# import pandas as pd
# import xarray as xr
# import geopandas as gpd
# import rioxarray
# import requests
# import io
# from glob import glob
# from tqdm import tqdm
# from shapely.geometry import mapping
# from rioxarray.exceptions import NoDataInBounds
# from typing import Tuple, List

# BASE_FOLDER_STREAMFLOW = r'C:\Users\hdagne1\Box\Dr.Mesfin Research\Data\CAMELS\basin_timeseries_v1p2_metForcing_obsFlow\basin_dataset_public_v1p2\usgs_streamflow'
# NDVI_PATH = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Data\NDVI\MOD13A2_NDVI_monthly_2000_2018_CAMELS.csv"
# TWSA_PATH = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Data\GRACE\TWSA_Monthly_CAMELS.csv"
# NLDAS_DATA_DIR = r'E:\Data\NLDAS_Data\NLDAS_NOAH0125_M_002_1994_2023'
# TIME_SLICE = slice("2000-01-01", "2014-12-31")
# N_TOP_BASINS = 10
# MISSING_DATA_THRESHOLD = 0.05


# def read_streamflow_file(file_path):
#     try:
#         df = pd.read_csv(file_path, sep='\s+', header=None, names=['gaugeid', 'year', 'month', 'day', 'streamflow', 'flag'])
#         df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
#         df = df[(df['streamflow'] >= 0) & (df['streamflow'] != -999.00)]
#         return df[['gaugeid', 'date', 'streamflow']]
#     except Exception:
#         return None

# def import_streamflow_data(base_folder):
#     all_data = []
#     if not os.path.exists(base_folder):
#         return pd.DataFrame()

#     for folder in range(1, 19):
#         folder_path = os.path.join(base_folder, f"{folder:02}")
#         if not os.path.exists(folder_path): continue
            
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

#     if all_data:
#         streamflow_data_all = pd.concat(all_data, axis=1)
#         streamflow_data_all = streamflow_data_all.loc[:, ~streamflow_data_all.columns.duplicated()].sort_index()
#         if 'month' in streamflow_data_all.index.names:
#             streamflow_data_all.index = streamflow_data_all.index.to_timestamp()
#     else:
#         streamflow_data_all = pd.DataFrame()
#     return streamflow_data_all

# def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, 
#                                      pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
#     print("Loading CAMELS attributes and streamflow...")
#     try:
#         r = requests.get("https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/data/contents/camels_attributes_v2.0.feather")
#         attrs = gpd.read_feather(io.BytesIO(r.content)).reset_index(drop=False)
#         attrs['geometry_points'] = attrs['geometry'].centroid 
#         attrs = attrs.set_geometry('geometry_points')
#     except Exception as e:
#         print(f"FATAL: Error loading CAMELS attributes: {e}")
#         return (None,) * 8

#     streamflow_data_all = import_streamflow_data(BASE_FOLDER_STREAMFLOW)
#     if streamflow_data_all.empty:
#         print("FATAL: USGS streamflow data loading failed or data directory is incorrect.")
#         return (None,) * 8

#     streamflow_data_all = streamflow_data_all.loc['1984-01-01':'2014-12-31']
#     valid_cols = streamflow_data_all.columns[streamflow_data_all.isna().mean() <= MISSING_DATA_THRESHOLD]
#     streamflow_data_all = streamflow_data_all[valid_cols]
#     streamflow_data_all = streamflow_data_all.interpolate(method='linear', axis=0).ffill().bfill()
    
#     attrs['gauge_id'] = attrs['gauge_id'].astype(str).str.strip().str.zfill(8)
#     attrs.set_index('gauge_id', inplace=True)
#     streamflow_data_all.columns = streamflow_data_all.columns.astype(str).str.strip().str.zfill(8)

#     area_km2_series = attrs['area_gages2'].loc[streamflow_data_all.columns]
#     conversion_constant = 0.028316846 * 86400 * 1000 / 1e6
#     Q_USGS_monthly_full = pd.DataFrame({
#         col: streamflow_data_all[col] * conversion_constant / area_km2_series.loc[col]
#         for col in streamflow_data_all.columns
#     }, index=streamflow_data_all.index)

#     print("Loading NDVI data...")
#     NDVI_CAMELS_sites = pd.read_csv(NDVI_PATH).set_index('month')
#     NDVI_CAMELS_sites.replace(-9999, np.nan, inplace=True)
#     cols_to_drop = [col for col in NDVI_CAMELS_sites.columns if NDVI_CAMELS_sites[col].isna().mean() > MISSING_DATA_THRESHOLD]
#     NDVI_CAMELS_sites.drop(columns=cols_to_drop, inplace=True)
#     NDVI_CAMELS_sites = NDVI_CAMELS_sites.apply(lambda x: x.fillna(x.mean()), axis=0)
#     NDVI_CAMELS_sites.index = pd.PeriodIndex(NDVI_CAMELS_sites.index, freq='M').to_timestamp()

#     print(f"Loading NLDAS data from {NLDAS_DATA_DIR}...")
#     # data/data_processor.py (TEMPORARY DEBUGGING BLOCK)
#     try:
#         nc_files = glob(NLDAS_DATA_DIR + '/*.nc')
#         if not nc_files:
#             raise FileNotFoundError("No .nc files found in directory.")

#         # Test loading the files one by one (this will find the corrupt file)
#         for file in tqdm(nc_files, desc="Checking NLDAS files"):
#             xr.open_dataset(file).close() 

#         # If all pass, load them as a multi-file dataset
#         FluxData_all = xr.open_mfdataset(nc_files, concat_dim='time', combine='nested', 
#                                         coords='minimal', compat='override', parallel=True)
#         FluxData_all = FluxData_all.rio.write_crs("EPSG:4326", inplace=True)
#     except Exception as e:
#         print(f"FATAL: Error loading NLDAS netCDF files: {e}")
#         return (None,) * 8
    
#     common_gauges = list(set(attrs.index) & set(Q_USGS_monthly_full.columns))

#     attrs_large = attrs.loc[common_gauges].sort_values(by="area_gages2", ascending=False).head(N_TOP_BASINS)
#     BASIN_IDS = attrs_large.index.tolist()
    
#     print(f"Extracting NLDAS variables for the top {len(BASIN_IDS)} basins...")
#     attrs_large = attrs_large.to_crs("EPSG:4326")
#     vars_to_extract = ["Evap", "Rainf", "Qs", "Qsb", "PotEvap", "Streamflow", "LAI"]
#     dfs = {} 
    
#     for var_name in vars_to_extract:
#         var_da = FluxData_all[var_name].sel(time=TIME_SLICE)
#         data_dict = {}
#         for basin_id, row in tqdm(attrs_large.iterrows(), total=len(attrs_large), desc=f"NLDAS-{var_name}"):
#             geom = [mapping(row["geometry"])]
#             try:
#                 clipped = var_da.rio.clip(geom, attrs_large.crs, drop=True)
#                 basin_mean = clipped.mean(dim=["lat", "lon"], skipna=True)
#                 data_dict[basin_id] = basin_mean.to_pandas()
#             except NoDataInBounds: continue
            
#         df = pd.DataFrame(data_dict, index=var_da.time.values)
#         dfs[var_name] = df.reindex(BASIN_IDS, axis=1).sort_index()

#     Rainf_df = dfs['Rainf']
#     PotEvap_df = dfs['PotEvap']
#     Evap_df = dfs['Evap']
#     Qsb_df = dfs['Qsb']
#     NLDAS_Streamflow_df = dfs['Streamflow']
    
#     Q_USGS_monthly = Q_USGS_monthly_full.loc[TIME_SLICE.start:TIME_SLICE.stop, BASIN_IDS]
#     NDVI_CAMELS_sites_selected = NDVI_CAMELS_sites.loc[TIME_SLICE.start:TIME_SLICE.stop, BASIN_IDS]
    
#     Q_nldas_mm_monthly = NLDAS_Streamflow_df.copy()
#     for basin in NLDAS_Streamflow_df.columns:
#         area_km2 = attrs_large.loc[basin, 'area_gages2']
#         area_m2 = area_km2 * 1e6
#         Q_nldas_mm_monthly[basin] = (NLDAS_Streamflow_df[basin] * 86400 / area_m2) * 1000

#     NDVI_normalized = (NDVI_CAMELS_sites_selected - NDVI_CAMELS_sites_selected.min()) / \
#                       (NDVI_CAMELS_sites_selected.max() - NDVI_CAMELS_sites_selected.min())
#     NDVI_min = np.min(NDVI_normalized.values)
#     NDVI_max = np.max(NDVI_normalized.values)
#     M_df = (NDVI_normalized - NDVI_min) / (NDVI_max - NDVI_min)
    
#     slope_static_series = attrs_large['slope_mean']
#     Slope_df = pd.DataFrame(index=Rainf_df.index, columns=Rainf_df.columns, dtype=float)
#     for basin in Rainf_df.columns:
#         Slope_df[basin] = slope_static_series.loc[basin]

#     return Rainf_df, PotEvap_df, Evap_df, Qsb_df, M_df, Slope_df, Q_nldas_mm_monthly, Q_USGS_monthly


import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray
import requests
import io
from glob import glob
from tqdm import tqdm
from shapely.geometry import mapping
from rioxarray.exceptions import NoDataInBounds
from typing import Tuple, List

BASE_FOLDER_STREAMFLOW = r'C:\Users\hdagne1\Box\Dr.Mesfin Research\Data\CAMELS\basin_timeseries_v1p2_metForcing_obsFlow\basin_dataset_public_v1p2\usgs_streamflow'
NDVI_PATH = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Data\NDVI\MOD13A2_NDVI_monthly_2000_2018_CAMELS.csv"
TWSA_PATH = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Data\GRACE\TWSA_Monthly_CAMELS.csv"
NLDAS_DATA_DIR = r'E:\Data\NLDAS_Data\NLDAS_NOAH0125_M_002_1994_2023'
TIME_SLICE = slice("2000-01-01", "2014-12-31")
N_TOP_BASINS = 10
MISSING_DATA_THRESHOLD = 0.05


def read_streamflow_file(file_path):
    try:
        df = pd.read_csv(file_path, sep='\s+', header=None, names=['gaugeid', 'year', 'month', 'day', 'streamflow', 'flag'])
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df = df[(df['streamflow'] >= 0) & (df['streamflow'] != -999.00)]
        return df[['gaugeid', 'date', 'streamflow']]
    except Exception:
        return None

def import_streamflow_data(base_folder):
    all_data = []
    if not os.path.exists(base_folder):
        return pd.DataFrame()

    for folder in range(1, 19):
        folder_path = os.path.join(base_folder, f"{folder:02}")
        if not os.path.exists(folder_path): continue
            
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

    if all_data:
        streamflow_data_all = pd.concat(all_data, axis=1)
        streamflow_data_all = streamflow_data_all.loc[:, ~streamflow_data_all.columns.duplicated()].sort_index()
        if 'month' in streamflow_data_all.index.names:
            streamflow_data_all.index = streamflow_data_all.index.to_timestamp()
    else:
        streamflow_data_all = pd.DataFrame()
    return streamflow_data_all

def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                     pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                     pd.DataFrame, pd.DataFrame]: # RETURN TUPLE LENGTH INCREASED TO 10
    
    print("Loading CAMELS attributes and streamflow...")
    try:
        r = requests.get("https://www.hydroshare.org/resource/658c359b8c83494aac0f58145b1b04e6/data/contents/camels_attributes_v2.0.feather")
        attrs = gpd.read_feather(io.BytesIO(r.content)).reset_index(drop=False)
        attrs['geometry_points'] = attrs['geometry'].centroid 
        attrs = attrs.set_geometry('geometry_points')
    except Exception as e:
        print(f"FATAL: Error loading CAMELS attributes: {e}")
        return (None,) * 10 # RETURN TUPLE LENGTH INCREASED
        
    streamflow_data_all = import_streamflow_data(BASE_FOLDER_STREAMFLOW)
    if streamflow_data_all.empty:
        print("FATAL: USGS streamflow data loading failed or data directory is incorrect.")
        return (None,) * 10 # RETURN TUPLE LENGTH INCREASED

    streamflow_data_all = streamflow_data_all.loc['1984-01-01':'2014-12-31']
    valid_cols = streamflow_data_all.columns[streamflow_data_all.isna().mean() <= MISSING_DATA_THRESHOLD]
    streamflow_data_all = streamflow_data_all[valid_cols]
    streamflow_data_all = streamflow_data_all.interpolate(method='linear', axis=0).ffill().bfill()
    
    attrs['gauge_id'] = attrs['gauge_id'].astype(str).str.strip().str.zfill(8)
    attrs.set_index('gauge_id', inplace=True)
    streamflow_data_all.columns = streamflow_data_all.columns.astype(str).str.strip().str.zfill(8)

    area_km2_series = attrs['area_gages2'].loc[streamflow_data_all.columns]
    conversion_constant = 0.028316846 * 86400 * 1000 / 1e6
    Q_USGS_monthly_full = pd.DataFrame({
        col: streamflow_data_all[col] * conversion_constant / area_km2_series.loc[col]
        for col in streamflow_data_all.columns
    }, index=streamflow_data_all.index)

    print("Loading NDVI data...")
    NDVI_CAMELS_sites = pd.read_csv(NDVI_PATH).set_index('month')
    NDVI_CAMELS_sites.replace(-9999, np.nan, inplace=True)
    cols_to_drop = [col for col in NDVI_CAMELS_sites.columns if NDVI_CAMELS_sites[col].isna().mean() > MISSING_DATA_THRESHOLD]
    NDVI_CAMELS_sites.drop(columns=cols_to_drop, inplace=True)
    NDVI_CAMELS_sites = NDVI_CAMELS_sites.apply(lambda x: x.fillna(x.mean()), axis=0)
    NDVI_CAMELS_sites.index = pd.PeriodIndex(NDVI_CAMELS_sites.index, freq='M').to_timestamp()

    print(f"Loading NLDAS data from {NLDAS_DATA_DIR}...")
    # data/data_processor.py (TEMPORARY DEBUGGING BLOCK)
    try:
        nc_files = glob(NLDAS_DATA_DIR + '/*.nc')
        if not nc_files:
            raise FileNotFoundError("No .nc files found in directory.")

        # Test loading the files one by one (this will find the corrupt file)
        for file in tqdm(nc_files, desc="Checking NLDAS files"):
            xr.open_dataset(file).close() 

        # If all pass, load them as a multi-file dataset
        FluxData_all = xr.open_mfdataset(nc_files, concat_dim='time', combine='nested', 
                                         coords='minimal', compat='override', parallel=True)
        FluxData_all = FluxData_all.rio.write_crs("EPSG:4326", inplace=True)
    except Exception as e:
        print(f"FATAL: Error loading NLDAS netCDF files: {e}")
        return (None,) * 10 # RETURN TUPLE LENGTH INCREASED
    
    common_gauges = list(set(attrs.index) & set(Q_USGS_monthly_full.columns))

    attrs_large = attrs.loc[common_gauges].sort_values(by="area_gages2", ascending=False).head(N_TOP_BASINS)
    BASIN_IDS = attrs_large.index.tolist()
    
    print(f"Extracting NLDAS variables for the top {len(BASIN_IDS)} basins...")
    attrs_large = attrs_large.to_crs("EPSG:4326")
    
    # 1. ADDED RootMoist and SoilM_0_200cm for initial states
    vars_to_extract = ["Evap", "Rainf", "Qs", "Qsb", "PotEvap", "Streamflow", "LAI", "RootMoist", "SoilM_0_200cm"]
    dfs = {} 
    
    for var_name in vars_to_extract:
        var_da = FluxData_all[var_name].sel(time=TIME_SLICE)
        data_dict = {}
        for basin_id, row in tqdm(attrs_large.iterrows(), total=len(attrs_large), desc=f"NLDAS-{var_name}"):
            geom = [mapping(row["geometry"])]
            try:
                clipped = var_da.rio.clip(geom, attrs_large.crs, drop=True)
                basin_mean = clipped.mean(dim=["lat", "lon"], skipna=True)
                data_dict[basin_id] = basin_mean.to_pandas()
            except NoDataInBounds: continue
            
        df = pd.DataFrame(data_dict, index=var_da.time.values)
        dfs[var_name] = df.reindex(BASIN_IDS, axis=1).sort_index()

    Rainf_df = dfs['Rainf']
    PotEvap_df = dfs['PotEvap']
    Evap_df = dfs['Evap']
    Qsb_df = dfs['Qsb']
    NLDAS_Streamflow_df = dfs['Streamflow']
    
    # 2. Extract NLDAS initial state DataFrames
    RootMoist_df = dfs['RootMoist']
    SoilM_0_200cm_df = dfs['SoilM_0_200cm']
    
    Q_USGS_monthly = Q_USGS_monthly_full.loc[TIME_SLICE.start:TIME_SLICE.stop, BASIN_IDS]
    NDVI_CAMELS_sites_selected = NDVI_CAMELS_sites.loc[TIME_SLICE.start:TIME_SLICE.stop, BASIN_IDS]
    
    Q_nldas_mm_monthly = NLDAS_Streamflow_df.copy()
    for basin in NLDAS_Streamflow_df.columns:
        area_km2 = attrs_large.loc[basin, 'area_gages2']
        area_m2 = area_km2 * 1e6
        Q_nldas_mm_monthly[basin] = (NLDAS_Streamflow_df[basin] * 86400 / area_m2) * 1000

    NDVI_normalized = (NDVI_CAMELS_sites_selected - NDVI_CAMELS_sites_selected.min()) / \
                      (NDVI_CAMELS_sites_selected.max() - NDVI_CAMELS_sites_selected.min())
    NDVI_min = np.min(NDVI_normalized.values)
    NDVI_max = np.max(NDVI_normalized.values)
    M_df = (NDVI_normalized - NDVI_min) / (NDVI_max - NDVI_min)
    
    slope_static_series = attrs_large['slope_mean']
    Slope_df = pd.DataFrame(index=Rainf_df.index, columns=Rainf_df.columns, dtype=float)
    for basin in Rainf_df.columns:
        Slope_df[basin] = slope_static_series.loc[basin]

    # 3. UPDATED RETURN STATEMENT: Total 10 DataFrames returned
    return (Rainf_df, PotEvap_df, Evap_df, Qsb_df, M_df, 
            Slope_df, Q_nldas_mm_monthly, Q_USGS_monthly,
            RootMoist_df, SoilM_0_200cm_df)