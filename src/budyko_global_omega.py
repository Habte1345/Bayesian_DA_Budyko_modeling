import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import ModelParams, two_store_model_step
from src.budyko import BudykoModelEstimator

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SPLIT_FILE = os.path.join(PROJECT_ROOT, "SCE_global_params", "basin_split.json")
GLOBAL_PARAM_FILE = os.path.join(PROJECT_ROOT, "SCE_global_params", "global_calibrated_params.json")
OUT_JSON = os.path.join(PROJECT_ROOT, "SCE_global_params", "omega_global_test.json")

# =========================================================
# LOAD DATA
# =========================================================
def load_feather(name):
    path = os.path.join(DATA_DIR, name)
    df = pd.read_feather(path)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

    return df

print("Loading data...")
PET_df = load_feather("PotEvap.feather")
P_df   = load_feather("Rainf.feather")
SM_df  = load_feather("SoilM_0_200cm.feather")
SLP_df = load_feather("slope.feather")

# =========================================================
# LOAD SPLIT
# =========================================================
with open(SPLIT_FILE, "r") as f:
    split = json.load(f)

train_basins = split["train_basins"]
test_basins  = split["test_basins"]

# =========================================================
# LOAD GLOBAL PARAMS
# =========================================================
with open(GLOBAL_PARAM_FILE, "r") as f:
    gparams = json.load(f)

params = ModelParams(
    Smax=float(gparams["Smax"]),
    Kperc=float(gparams["Kperc"]),
    Kb=float(gparams["Kb"]),
    Ke=float(gparams["Ke"]),
    Cqq=float(gparams["Cqq"]),
    Sfc_frac=0.3,
    beta_et=2.0,
)

S_init = float(gparams["S_init"])
G_init = float(gparams["G_init"])
Gmax   = float(gparams["Smax"]) * float(gparams["Gmax_factor"])

# =========================================================
# STORAGE SIMULATION
# =========================================================
def get_storage_series(P, PET):
    L = len(P)
    S = S_init
    G = G_init

    S_series = np.zeros(L)

    for t in range(L):
        S, G, _, _, *_ = two_store_model_step(
            S, G,
            float(P[t]),
            float(PET[t]),
            params,
            ET_override=float(PET[t] * params.Ke)
        )
        G = np.clip(G, 0.0, Gmax)
        S_series[t] = S

    return S_series

# =========================================================
# STEP 1: COMPUTE omega_true + FEATURES
# =========================================================
records = []

print("\nComputing omega_true for TRAIN basins...")

for basin in tqdm(train_basins, desc="TRAIN basins"):

    if basin not in PET_df.columns:
        continue

    PET = PET_df[basin].values
    P   = P_df[basin].values
    SM  = SM_df[[basin]]
    SLP = SLP_df[[basin]]

    # ---- Correct dS
    S_series = get_storage_series(P, PET)
    dS = np.diff(S_series, prepend=S_series[0])

    # ---- Budyko input
    P_df_b   = pd.DataFrame({basin: P})
    PET_df_b = pd.DataFrame({basin: PET})
    dS_df_b  = pd.DataFrame({basin: dS})

    Ke_df = pd.DataFrame({basin: np.full(len(P), params.Ke)})

    budyko = BudykoModelEstimator(
        P_df=P_df_b,
        dS_df=dS_df_b,
        PotEvap_df=PET_df_b,
        M_basin=SM,
        Slope_basin=SLP,
        calibrated_params=None,
        Ke_df=Ke_df,
    )

    budyko.estimate_budyko_et()
    omega_true = budyko.omega_true[basin]

    # ---- Features
    P_mean   = np.nanmean(P)
    PET_mean = np.nanmean(PET)
    AI       = PET_mean / (P_mean + 1e-6)

    records.append(pd.DataFrame({
        "omega": [np.nanmean(omega_true)],
        "AI": [AI],
        "P": [P_mean],
        "PET": [PET_mean],
        "M": [np.nanmean(SM[basin])],
        "Slope": [float(SLP[basin].values[0])]
    }))

train_df = pd.concat(records, axis=0)

# =========================================================
# STEP 2: TRAIN GLOBAL MLR + METRICS
# =========================================================
print("\nTraining global MLR...")

X = np.column_stack([
    np.ones(len(train_df)),
    train_df["AI"].values,
    train_df["P"].values,
    train_df["PET"].values,
    train_df["M"].values,
    train_df["Slope"].values
])

y = train_df["omega"].values

beta = np.linalg.lstsq(X, y, rcond=None)[0]

y_pred = X @ beta
residuals = y - y_pred

rmse = np.sqrt(np.mean(residuals**2))
mae  = np.mean(np.abs(residuals))
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - ss_res / ss_tot

print("\n================ TRAIN PERFORMANCE ================")
print("R²   =", r2)
print("RMSE =", rmse)
print("MAE  =", mae)
print("==================================================")

# =========================================================
# STEP 3: APPLY MODEL TO TEST BASINS
# =========================================================
omega_test_dict = {}

print("\nApplying trained MLR to TEST basins...")

for basin in tqdm(test_basins, desc="TEST basins"):

    if basin not in PET_df.columns:
        continue

    PET = PET_df[basin]
    P   = P_df[basin]
    SM  = SM_df[basin]
    SLP = SLP_df[basin]

    P_mean   = np.nanmean(P)
    PET_mean = np.nanmean(PET)
    AI       = PET_mean / (P_mean + 1e-6)

    x = np.array([
        1.0,
        AI,
        P_mean,
        PET_mean,
        np.nanmean(SM),
        float(SLP.values[0])
    ])

    omega_mlr = np.dot(beta, x)
    omega_test_dict[basin] = float(omega_mlr)

# =========================================================
# SAVE
# =========================================================
with open(OUT_JSON, "w") as f:
    json.dump({
        "beta": beta.tolist(),
        "omega_test": omega_test_dict
    }, f, indent=4)

print(f"\nSaved global omega to: {OUT_JSON}")



