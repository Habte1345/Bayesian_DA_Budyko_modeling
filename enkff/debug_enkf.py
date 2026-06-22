"""
debug_enkf.py — Drop this file next to run_simulation.py and run it on ONE basin.

It prints step-by-step diagnostics to reveal exactly why DA has no effect:
  - Ensemble spread (S and G) at each step
  - ET_B (observation) vs model ET (prior mean)
  - Innovation after capping
  - Kalman gain for S and G
  - S before and after analysis

Run:
    python debug_enkf.py

Edit BASIN_ID and the path constants at the top to match your setup.
"""

from __future__ import annotations
import hashlib, json, os, sys
import numpy as np
import pandas as pd

# ── CONFIGURE THESE ──────────────────────────────────────────────────────────
PROJECT_ROOT   = r"F:\Github_repos\Bayesian_DA_Budyko_modeling"
DATA_DIR       = os.path.join(PROJECT_ROOT, "data", "processed")
CAL_PARAMS     = os.path.join(PROJECT_ROOT, "data", "calibrated_params.json")  # adjust path
BASIN_ID       = "01047000"
N_STEPS_PRINT  = 24     # print first N steps in detail
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, PROJECT_ROOT)

from src.model   import ModelParams, two_store_model_step
from src.budyko  import BudykoModelEstimator
from src.enkf    import EnKFConfig, enkf_analysis_step, enkf_forecast_step_states

def _stable_seed(basin_id):
    return int.from_bytes(hashlib.md5(basin_id.encode()).digest()[:4], "little")

def load_f(fname):
    df = pd.read_feather(os.path.join(DATA_DIR, fname)).dropna(axis=1, how="all")
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    return df

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
PET_df   = load_f("PotEvap.feather")
P_df     = load_f("Rainf.feather")
M_df     = load_f("M.feather")
Slp_df   = load_f("slope.feather")

idx = PET_df.index.intersection(P_df.index).intersection(M_df.index).intersection(Slp_df.index)

PET = PET_df[BASIN_ID].reindex(idx).to_numpy(float)
P   = P_df[BASIN_ID].reindex(idx).to_numpy(float)

with open(CAL_PARAMS) as f:
    cal = json.load(f)

p        = cal[BASIN_ID]
Smax     = float(p.get("Smax", 50.0))
Gmax     = Smax * float(p.get("Gmax_factor", 4.0))
S_init   = float(p.get("S_init", 0.5 * Smax))
G_init   = float(p.get("G_init", 0.5 * Gmax))
params   = ModelParams(Smax=Smax, Kperc=float(p["Kperc"]), Kb=float(p["Kb"]),
                       Ke=float(p["Ke"]), Cqq=float(p["Cqq"]),
                       Sfc_frac=0.30, beta_et=2.0)

print(f"\nBasin {BASIN_ID}:  Smax={Smax:.1f}  Gmax={Gmax:.1f}  Ke={params.Ke:.3f}")

# ── Compute ET_B ──────────────────────────────────────────────────────────────
ET_ke = PET * params.Ke

# Base run for dS
S, G = S_init, G_init
dS_arr = np.full(len(P), np.nan)
for t in range(len(P)):
    P_t = float(P[t]) if np.isfinite(P[t]) else 0.0
    PET_t = float(PET[t]) if np.isfinite(PET[t]) else 0.0
    ET_t = float(ET_ke[t]) if np.isfinite(ET_ke[t]) else None
    S, G, _, Q, _, _, _, dS_t = two_store_model_step(S, G, P_t, PET_t, params, ET_override=ET_t)
    G = np.clip(G, 0, Gmax)
    dS_arr[t] = dS_t

# omega MLR (simplified: use constant omega=2.5 for diagnostics)
omega = np.full(len(P), 2.5)

# P_eff capped at P (FIX 1)
P_eff = np.clip(P - dS_arr, 1e-6, P)

# Fu-Budyko ET
ET_B = np.full_like(P, np.nan)
ok = np.isfinite(P_eff) & np.isfinite(PET) & (P_eff > 0) & (PET >= 0)
phi = np.where(ok, PET / P_eff, np.nan)
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    term = np.power(phi, omega)
    e_r  = 1.0 + phi - np.power(1.0 + term, 1.0 / omega)
ET_B[ok] = np.clip(P_eff[ok] * np.where(np.isfinite(e_r[ok]), e_r[ok], np.nan),
                   0, np.minimum(P_eff[ok], PET[ok]))

print(f"\nET summary over {len(P)} steps:")
print(f"  ET_ke : mean={np.nanmean(ET_ke):.2f}  std={np.nanstd(ET_ke):.2f}  "
      f"min={np.nanmin(ET_ke):.2f}  max={np.nanmax(ET_ke):.2f}")
print(f"  ET_B  : mean={np.nanmean(ET_B):.2f}   std={np.nanstd(ET_B):.2f}  "
      f"min={np.nanmin(ET_B):.2f}  max={np.nanmax(ET_B):.2f}")
print(f"  P_eff : mean={np.nanmean(P_eff):.2f}  min={np.nanmin(P_eff):.2f}  "
      f"max={np.nanmax(P_eff):.2f}")
print(f"  P     : mean={np.nanmean(P):.2f}       min={np.nanmin(P):.2f}  "
      f"max={np.nanmax(P):.2f}")

# ── Run EnKF with step-by-step diagnostics ────────────────────────────────────
nens = 100
rng  = np.random.default_rng(_stable_seed(BASIN_ID))

# FIX 2: proc noise as 5% of storage
proc_S_abs = 0.05 * Smax
proc_G_abs = 0.05 * Gmax

cfg = EnKFConfig(
    nens=nens, inflation=1.05,
    R_ET_std=50.0, R_Q_std=100.0,
    R_ET_frac=0.0, R_Q_frac=0.0,
    proc_S_std=proc_S_abs,
    proc_G_std=proc_G_abs,
    P_std_frac=0.10, PET_std_frac=0.03,
)

print(f"\nEnKF config: proc_S_std={proc_S_abs:.2f} mm  proc_G_std={proc_G_abs:.2f} mm  "
      f"R_ET_std={cfg.R_ET_std}")

S0 = np.clip(S_init + rng.normal(0, 0.10*Smax, nens), 0, Smax)
G0 = np.clip(G_init + rng.normal(0, 0.10*Gmax, nens), 0, Gmax)
X  = np.vstack([S0, G0])

prev_et_mean   = float(PET[0] * params.Ke) if np.isfinite(PET[0]) else 30.0
prev_et_spread = 0.3 * prev_et_mean

print(f"\n{'t':>4}  {'P':>6}  {'PET':>6}  {'ET_B':>7}  {'ET_f_mn':>8}  "
      f"{'innov_raw':>9}  {'innov_cap':>9}  {'spread_S':>8}  {'K_S':>7}  "
      f"{'S_prior':>8}  {'S_post':>8}  {'dS_eff':>7}")
print("-" * 110)

ET_ass_all = np.full(len(P), np.nan)

for t in range(len(P)):
    P_t   = float(P[t])   if np.isfinite(P[t])   else 0.0
    PET_t = float(PET[t]) if np.isfinite(PET[t]) else 0.0

    y_ET_raw = float(ET_B[t]) if np.isfinite(ET_B[t]) else None

    # FIX 3: relative innovation cap
    if y_ET_raw is not None:
        cap  = max(3.0*prev_et_spread, 0.30*abs(y_ET_raw), float(cfg.R_ET_std))
        y_ET = float(np.clip(y_ET_raw, prev_et_mean - cap, prev_et_mean + cap))
    else:
        y_ET = None

    # Prior forecast (no analysis yet) to compute Kalman gain manually
    X_f, ET_ens_f, Q_ens_f, diag = enkf_analysis_step(
        X=X, P_t=P_t, PET_t=PET_t, params_cal=params,
        Smax=Smax, Gmax=Gmax, rng=rng, cfg=cfg,
        y_ET=y_ET, y_Q=None, ET_override=None,
    )

    # Manually compute K_S for diagnostics (before analysis was applied)
    # We need the forecast ensemble to do this — re-run a zero-noise forecast
    X_f_diag, ET_f_diag, _ = enkf_forecast_step_states(
        X=X, P_t=P_t, PET_t=PET_t, params_cal=params,
        Smax=Smax, Gmax=Gmax, rng=np.random.default_rng(),
        proc_S_std=0., proc_G_std=0., P_std_frac=0., PET_std_frac=0.,
    )
    dx_S  = X_f_diag[0] - X_f_diag[0].mean()
    dy    = ET_f_diag   - ET_f_diag.mean()
    P_xy_S= np.dot(dx_S, dy) / (nens - 1)
    P_yy  = np.dot(dy, dy)   / (nens - 1)
    K_S   = P_xy_S / (P_yy + cfg.R_ET_std**2) if (P_yy + cfg.R_ET_std**2) > 1e-12 else 0.0

    # Re-forecast from posterior for ET_ass
    _, ET_ens_a, _ = enkf_forecast_step_states(
        X=X_f, P_t=P_t, PET_t=PET_t, params_cal=params,
        Smax=Smax, Gmax=Gmax, rng=np.random.default_rng(),
        proc_S_std=0., proc_G_std=0., P_std_frac=0., PET_std_frac=0.,
    )
    ET_ass_all[t] = float(np.mean(ET_ens_a))

    s_prior = float(X[0].mean())
    s_post  = float(X_f[0].mean())

    prev_et_mean   = float(np.mean(ET_ens_f))
    prev_et_spread = float(np.std(ET_ens_f, ddof=1))

    if t < N_STEPS_PRINT:
        innov_raw = (y_ET_raw - float(np.mean(ET_f_diag))) if y_ET_raw else np.nan
        innov_cap = (y_ET     - float(np.mean(ET_f_diag))) if y_ET     else np.nan
        print(f"{t:>4}  {P_t:>6.1f}  {PET_t:>6.1f}  "
              f"{(y_ET_raw if y_ET_raw else np.nan):>7.2f}  "
              f"{float(np.mean(ET_f_diag)):>8.2f}  "
              f"{innov_raw:>9.2f}  {innov_cap:>9.2f}  "
              f"{diag.spread_S:>8.3f}  {K_S:>7.4f}  "
              f"{s_prior:>8.2f}  {s_post:>8.2f}  "
              f"{s_post - s_prior:>7.2f}")

    X = X_f

print(f"\n{'─'*110}")
print(f"\nFinal ET comparison:")
print(f"  ET_ke  mean = {np.nanmean(ET_ke):.3f}")
print(f"  ET_B   mean = {np.nanmean(ET_B):.3f}")
print(f"  ET_ass mean = {np.nanmean(ET_ass_all):.3f}")
print(f"\n  ET_ke  ≈ ET_ass?  diff = {abs(np.nanmean(ET_ke) - np.nanmean(ET_ass_all)):.4f}")
print(f"  ET_B   ≈ ET_ass?  diff = {abs(np.nanmean(ET_B)  - np.nanmean(ET_ass_all)):.4f}")

print(f"\nIf K_S ≈ 0 every step → ensemble collapsed or ET insensitive to S")
print(f"If innov_cap ≈ 0 every step → cap is too tight")
print(f"If spread_S ≈ 0 → proc_S_std too small")
print(f"\nDone. Share the output above and we will know exactly what to fix.")
