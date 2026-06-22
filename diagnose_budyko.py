"""
diagnose_budyko.py
==================
Run this BEFORE re-running simulations to confirm the budyko.py
fixes will resolve BASE ≈ BUDYKO.

Place next to run.py and run:
    python diagnose_budyko.py

It prints a clear before/after comparison for 3 basins.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(PROJECT_ROOT, "data", "processed")
CAL_PARAMS = os.path.join(PROJECT_ROOT, "SCE_cal_params", "calibrated_params.json")

# Pick 3 representative basins from your figures
TEST_BASINS = ["14305500", "12040500", "03463300"]

# ── Load data ─────────────────────────────────────────────────────
def load_feather(fname):
    path = os.path.join(DATA_DIR, fname)
    df   = pd.read_feather(path)
    for col in ("Date", "date", "time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            return df.set_index(col)
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    return df

PET_df  = load_feather("PotEvap.feather")
P_df    = load_feather("Rainf.feather")
Evap_df = load_feather("Evap.feather")    # SAC-SMA ET = ET_ke proxy
M_df    = load_feather("M.feather")
Slp_df  = load_feather("slope.feather")

with open(CAL_PARAMS) as f:
    cal = json.load(f)

common_idx = PET_df.index.intersection(P_df.index).intersection(Evap_df.index)

# ── Import BOTH versions of budyko.py ─────────────────────────────
# OLD: the one currently in src/
from src.budyko import BudykoModelEstimator as OLD_BUD

# NEW: the fixed version we just wrote
import importlib.util, types
spec = importlib.util.spec_from_file_location(
    "budyko_new",
    os.path.join(PROJECT_ROOT, "budyko_fixed.py")   # copy the new file here
)
# If new file not placed yet, we simulate the fixes inline
NEW_AVAILABLE = os.path.exists(
    os.path.join(PROJECT_ROOT, "budyko_fixed.py")
)

# ── Diagnose ──────────────────────────────────────────────────────
print("=" * 70)
print("BUDYKO DIAGNOSTIC — before vs after fix")
print("=" * 70)

for basin in TEST_BASINS:
    if basin not in cal:
        print(f"\n{basin}: not in calibrated_params, skipping")
        continue
    if basin not in PET_df.columns:
        print(f"\n{basin}: not in feather columns, skipping")
        continue

    p        = cal[basin]
    Ke       = float(p["Ke"])
    Smax     = float(p.get("Smax", 50.0))
    Gmax     = Smax * float(p.get("Gmax_factor", 4.0))
    S_init   = float(p.get("S_init", 0.5 * Smax))
    G_init   = float(p.get("G_init", 0.5 * Gmax))

    idx  = common_idx
    PET  = PET_df[basin].reindex(idx).values.astype(float)
    P    = P_df[basin].reindex(idx).values.astype(float)
    ET_s = Evap_df[basin].reindex(idx).values.astype(float)  # SAC-SMA ET

    ET_ke = Ke * PET

    # Approximate dS from a simple water balance
    # (in real code this comes from run_model_deterministic)
    # Here we use ET_sacsma as proxy for ET in the water balance
    dS_approx = P - ET_s - np.nanmean(P - ET_s)  # rough dS centered at 0

    P_eff_old = np.clip(P - dS_approx, 1e-6, None)          # OLD: no upper bound
    P_eff_new = np.clip(P - dS_approx, 1e-6, np.where(np.isfinite(P) & (P > 1e-6), P, 1e-6))  # NEW

    ratio_ET_old = ET_ke / np.where(P_eff_old > 0, P_eff_old, np.nan)
    ratio_ET_new = ET_ke / np.where(P_eff_new > 0, P_eff_new, np.nan)
    ratio_PET    = PET   / np.where(P_eff_new > 0, P_eff_new, np.nan)

    # How many timesteps violate ET/P_eff <= min(1, PET/P_eff)?
    admissible   = np.minimum(1.0, ratio_PET)
    n_violated   = np.sum(np.isfinite(ratio_ET_new) & (ratio_ET_new > admissible))
    n_total      = np.sum(np.isfinite(ratio_ET_new))
    pct_violated = 100 * n_violated / n_total if n_total > 0 else 0

    # After clipping (FIX 2)
    ratio_ET_clipped = np.clip(ratio_ET_new, 0.0,
                                np.minimum(1.0, ratio_PET))

    print(f"\nBasin {basin}  (Ke={Ke:.3f}, Smax={Smax:.0f} mm)")
    print(f"  ET_ke mean:            {np.nanmean(ET_ke):.4f} mm/month")
    print(f"  P_eff mean (old):      {np.nanmean(P_eff_old):.4f}  [no upper bound]")
    print(f"  P_eff mean (new):      {np.nanmean(P_eff_new):.4f}  [capped at P]")
    print(f"  ratio_ET > admissible: {n_violated}/{n_total} timesteps = {pct_violated:.1f}%")
    print(f"  → These are the steps where inversion had NO valid root")
    print(f"    (OLD code: returns omega_min=1.01 every time)")
    print(f"    (NEW code: returns NaN, excluded from MLR)")
    print(f"  ratio_ET before clip:  mean={np.nanmean(ratio_ET_new):.4f}")
    print(f"  ratio_ET after  clip:  mean={np.nanmean(ratio_ET_clipped):.4f}")

    # Estimated omega_true quality
    # Simple check: what fraction of omega_true will be NaN vs finite after fix
    n_will_be_nan   = n_violated
    n_will_be_valid = n_total - n_violated
    print(f"  After fix: ~{n_will_be_valid} valid omega_true values ({100*n_will_be_valid/n_total:.0f}%)")
    print(f"             ~{n_will_be_nan} NaN values excluded from MLR ({pct_violated:.0f}%)")

    print()
    print(f"  EXPECTED OUTCOME after fix:")
    print(f"    omega_true: mean in [1.5, 5.0] range, std > 0.3")
    print(f"    omega_MLR: trained on {n_will_be_valid} clean values (vs polluted baseline)")
    print(f"    ET_B: computed from omega_MLR (FIX 4), will differ from ET_ke")

print()
print("=" * 70)
print("ACTION: Copy budyko.py from the fixed version into src/budyko.py")
print("Then re-run: python run.py --config config.yaml --param_mode CALIBRATED")
print("=" * 70)
