# src/metrics.py

import numpy as np

# =====================================================================
# 5. PERFORMANCE METRICS
# =====================================================================

def calculate_nse(obs, sim):
    """Calculates the Nash-Sutcliffe Efficiency (NSE)."""
    obs = np.asarray(obs).flatten()
    sim = np.asarray(sim).flatten()
    valid = ~(np.isnan(obs) | np.isnan(sim))
    if valid.sum() < 2: return np.nan
    obs_valid = obs[valid]
    sim_valid = sim[valid]
    numerator = np.sum((obs_valid - sim_valid) ** 2)
    denominator = np.sum((obs_valid - np.mean(obs_valid)) ** 2)
    if denominator == 0: return np.nan
    return 1.0 - (numerator / denominator)


def calculate_kge(obs, sim):
    """Calculates the Kling-Gupta Efficiency (KGE)."""
    obs = np.asarray(obs).flatten()
    sim = np.asarray(sim).flatten()
    valid = ~(np.isnan(obs) | np.isnan(sim))
    if valid.sum() < 2: return np.nan
    obs_valid = obs[valid]
    sim_valid = sim[valid]
    
    r = np.corrcoef(obs_valid, sim_valid)[0, 1]
    if np.isnan(r): r = 0.0
    
    beta = np.mean(sim_valid) / (np.mean(obs_valid) + 1e-6)
    alpha = np.std(sim_valid) / (np.std(obs_valid) + 1e-6)
    
    # KGE formula: 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    kge = 1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
    return kge