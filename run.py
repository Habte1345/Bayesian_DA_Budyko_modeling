# run.py
import os
import sys
import argparse
import yaml
import shutil
from datetime import datetime

# ============================================================
# PROJECT ROOT
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.run_simulation import run_simulations_from_config


# ============================================================
# UTILITIES
# ============================================================
def print_centered(text: str, width: int = None):
    if width is None:
        width = shutil.get_terminal_size((120, 20)).columns
    for line in text.split("\n"):
        print(line.center(width))


def banner():
    ascii_header = r"""
==========================================================================================================
                 BAYESIAN DATA ASSIMILATION + BUDYKO-CONSTRAINED HYDROLOGIC SIMULATION

                               Authors:   Habtamu T., Mesfin M

  SCENARIOS IMPLEMENTED:
            [1] BASE MODEL   : Simple scaling factor based ET estimation
            [2] BUDYKO MODEL : Dynamic Budyko-based ET estimation
            [3] BUDYKO DA    : Budyko model with Ensemble Kalman Filter data assimilation
==========================================================================================================
"""
    print_centered(ascii_header)


def load_config(config_path: str) -> dict:
    """
    Load YAML config. Supports relative path from project root.
    """
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found:\n  {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"YAML file is empty/invalid:\n  {config_path}")

    cfg["_config_path"] = config_path
    return cfg

def normalize_scenario_key(s: str) -> str:
    s = str(s).strip().upper()

    mapping = {
        "ALL": "ALL",

        # BASE
        "BASE": "BASE",
        "BASE_MODEL": "BASE",
        "BASE-MODEL": "BASE",

        # BUDYKO
        "BUDYKO": "BUDYKO",
        "BUDYKO_MODEL": "BUDYKO",
        "BUDYKO-MODEL": "BUDYKO",

        # BUDYKO_DA
        "BUDYKO_DA": "BUDYKO_DA",
        "BUDYKO_DA_MODEL": "BUDYKO_DA",
        "BUDYKO+DA": "BUDYKO_DA",
        "DA": "BUDYKO_DA",
        "ENKF": "BUDYKO_DA",
    }

    if s not in mapping:
        raise ValueError(f"Unknown scenario: {s}")

    return mapping[s]



def validate_config(cfg: dict):
    if "scenario" not in cfg:
        raise KeyError("Missing required config key: scenario")

    if "paths" not in cfg:
        raise KeyError("Missing required config key: paths")

    paths = cfg["paths"]
    for k in ["data_dir", "result_dir", "calibrated_params"]:
        if k not in paths:
            raise KeyError(f"Missing required config key: paths.{k}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run Budyko + DA hydrologic simulations"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Force sequential run (debugging mode)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print settings (do not run)",
    )

    args = parser.parse_args()

    # Banner
    banner()

    # Load config
    cfg = load_config(args.config)
    validate_config(cfg)

    # Normalize scenario to canonical key
    cfg["scenario"] = normalize_scenario_key(cfg["scenario"])


    # Optional CLI override
    if args.no_parallel:
        cfg.setdefault("parallel", {})
        cfg["parallel"]["enabled"] = False

    # Show run info
    print("\n" + "-" * 110)
    print(f"✅ Project Root : {PROJECT_ROOT}")
    print(f"✅ Config File  : {cfg['_config_path']}")
    print(f"✅ Scenario     : {cfg['scenario']}")
    print(f"✅ Start Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 110)

    if args.dry_run:
        print("\n🟡 DRY RUN enabled. Configuration loaded successfully. No execution performed.\n")
        return

    # Run simulations
    run_simulations_from_config(cfg)


# ============================================================
# main:
# ============================================================
if __name__ == "__main__":
    main()
