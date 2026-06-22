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

# ✅ IMPORT BOTH SIMULATION ENGINES
from scripts.run_simulation import run_simulations_from_config as run_sim_calibrated
from scripts.run_simulation_uncalib_global import run_simulations_from_config as run_sim_uncalibrated


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

                               Authors:   Habtamu Tamiru, Mesfin Mekonnen, Hamid moradkhani, Parnian Ghaneei

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
    # for k in ["data_dir", "result_dir", "calibrated_params"]:
    #     if k not in paths:
    #         raise KeyError(f"Missing required config key: paths.{k}")
        

    required = ["data_dir", "result_dir"]

    for k in required:
        if k not in paths:
            raise KeyError(f"Missing required config key: paths.{k}")

    # Accept either key
    if "calibrated_params" not in paths and "global_calibrated_params" not in paths:
        raise KeyError("Missing required config key: paths.calibrated_params OR paths.global_calibrated_params")


# ============================================================
# PARAM MODE NORMALIZATION
# ============================================================
def normalize_param_mode(mode: str) -> str:
    """
    Normalize param_mode names into canonical values:
      - CALIBRATED
      - UNCALIBRATED
    """
    s = str(mode).strip().upper()

    mapping = {
        "CALIBRATED": "CALIBRATED",
        "UNCALIBRATED": "UNCALIBRATED",
        "SIMULATION_WITH_BASIN_CALIB_PARAMS": "CALIBRATED",
        "SIMULATION_WITH_GLOBAL_CAL_PARAMS": "UNCALIBRATED",
    }

    if s not in mapping:
        raise ValueError(
            f"Unknown param_mode: {mode}\n"
            f"Allowed: Simulation_with_BASIN_CALIB_PARAMS | Simulation_with_GLOBAL_CAL_PARAMS | CALIBRATED | UNCALIBRATED"
        )

    return mapping[s]


# ============================================================
# APPLY PARAM BLOCK
# ============================================================
def apply_param_block(cfg: dict):
    """
    If cfg['params'] contains CALIBRATED and UNCALIBRATED blocks,
    replace cfg['params'] with the active chosen block.
    """
    if "params" not in cfg:
        return

    params_cfg = cfg["params"]
    if not isinstance(params_cfg, dict):
        return

    # expect canonical blocks
    if "CALIBRATED" not in params_cfg and "UNCALIBRATED" not in params_cfg:
        return

    pmode = cfg.get("param_mode", "CALIBRATED")
    pmode = normalize_param_mode(pmode)

    if pmode not in params_cfg:
        raise ValueError(
            f"params block missing '{pmode}'. Available keys: {list(params_cfg.keys())}"
        )

    chosen = params_cfg[pmode]
    if not isinstance(chosen, dict):
        raise ValueError(f"params.{pmode} must be a dict")

    cfg["params"] = chosen
    cfg["param_mode"] = pmode


# ============================================================
# RESULT DIRECTORY REDIRECTION BY PARAM MODE
# ============================================================
def apply_result_dir_by_param_mode(cfg: dict):
    """
    Save results under different folder depending on param_mode.
    """
    if "paths" not in cfg:
        return
    if "result_dir" not in cfg["paths"]:
        return

    pmode = normalize_param_mode(cfg.get("param_mode", "CALIBRATED"))
    base_result_dir = cfg["paths"]["result_dir"]

    if pmode == "CALIBRATED":
        cfg["paths"]["result_dir"] = os.path.join(base_result_dir, "Simulation_with_BASIN_CALIB_PARAMS")
    elif pmode == "UNCALIBRATED":
        cfg["paths"]["result_dir"] = os.path.join(base_result_dir, "Simulation_with_GLOBAL_CAL_PARAMS")


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
    parser.add_argument(
        "--param-mode", "--param_mode",
        dest="param_mode",
        default=None,
        help="Select parameter/simulation mode: Simulation_with_BASIN_CALIB_PARAMS or Simulation_with_GLOBAL_CAL_PARAMS",
    )

    args = parser.parse_args()

    # Banner
    banner()

    # Load config
    cfg = load_config(args.config)
    validate_config(cfg)

    # Normalize scenario to canonical key
    cfg["scenario"] = normalize_scenario_key(cfg["scenario"])

    # param_mode from CLI overrides YAML
    if args.param_mode is not None:
        cfg["param_mode"] = args.param_mode
    else:
        cfg["param_mode"] = cfg.get("param_mode", "Simulation_with_BASIN_CALIB_PARAMS")

    # Normalize param_mode
    cfg["param_mode"] = normalize_param_mode(cfg["param_mode"])

    # Pick correct params block (very important!)
    apply_param_block(cfg)

    # Redirect result dir based on param_mode
    apply_result_dir_by_param_mode(cfg)

    # Optional CLI override
    if args.no_parallel:
        cfg.setdefault("parallel", {})
        cfg["parallel"]["enabled"] = False

    # Show run info
    print("\n" + "-" * 110)
    print(f"✅ Project Root : {PROJECT_ROOT}")
    print(f"✅ Config File  : {cfg['_config_path']}")
    print(f"✅ Scenario     : {cfg['scenario']}")
    print(f"✅ Param Mode   : {cfg['param_mode']}")
    print(f"✅ Result Dir   : {cfg['paths']['result_dir']}")
    print(f"✅ Start Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 110)

    if args.dry_run:
        print("\n🟡 DRY RUN enabled. Configuration loaded successfully. No execution performed.\n")
        return

    # ============================================================
    # DISPATCH correct engine
    # ============================================================
    if cfg["param_mode"] == "CALIBRATED":
        print("\n✅ Running simulation engine: scripts/run_simulation.py\n")
        run_sim_calibrated(cfg)

    elif cfg["param_mode"] == "UNCALIBRATED":
        print("\n✅ Running simulation engine: scripts/run_simulation_uncalib_global.py\n")
        run_sim_uncalibrated(cfg)

    else:
        raise ValueError(f"Invalid param_mode: {cfg['param_mode']}")


# ============================================================
# main:
# ============================================================
if __name__ == "__main__":
    main()










# # run.py
# import os
# import sys
# import argparse
# import yaml
# import shutil
# from datetime import datetime

# # ============================================================
# # PROJECT ROOT
# # ============================================================
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# # ✅ IMPORT BOTH SIMULATION ENGINES
# from scripts.run_simulation import run_simulations_from_config as run_sim_calibrated
# from scripts.run_simulation_uncalib_params import run_simulations_from_config as run_sim_uncalibrated


# # ============================================================
# # UTILITIES
# # ============================================================
# def print_centered(text: str, width: int = None):
#     if width is None:
#         width = shutil.get_terminal_size((120, 20)).columns
#     for line in text.split("\n"):
#         print(line.center(width))


# def banner():
#     ascii_header = r"""
# ==========================================================================================================
#                  BAYESIAN DATA ASSIMILATION + BUDYKO-CONSTRAINED HYDROLOGIC SIMULATION

#                                Authors:   Habtamu T., Mesfin M

#   SCENARIOS IMPLEMENTED:
#             [1] BASE MODEL   : Simple scaling factor based ET estimation
#             [2] BUDYKO MODEL : Dynamic Budyko-based ET estimation
#             [3] BUDYKO DA    : Budyko model with Ensemble Kalman Filter data assimilation
# ==========================================================================================================
# """
#     print_centered(ascii_header)


# def load_config(config_path: str) -> dict:
#     """
#     Load YAML config. Supports relative path from project root.
#     """
#     if not os.path.isabs(config_path):
#         config_path = os.path.join(PROJECT_ROOT, config_path)

#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"Config file not found:\n  {config_path}")

#     with open(config_path, "r", encoding="utf-8") as f:
#         cfg = yaml.safe_load(f)

#     if cfg is None:
#         raise ValueError(f"YAML file is empty/invalid:\n  {config_path}")

#     cfg["_config_path"] = config_path
#     return cfg


# def normalize_scenario_key(s: str) -> str:
#     s = str(s).strip().upper()

#     mapping = {
#         "ALL": "ALL",

#         # BASE
#         "BASE": "BASE",
#         "BASE_MODEL": "BASE",
#         "BASE-MODEL": "BASE",

#         # BUDYKO
#         "BUDYKO": "BUDYKO",
#         "BUDYKO_MODEL": "BUDYKO",
#         "BUDYKO-MODEL": "BUDYKO",

#         # BUDYKO_DA
#         "BUDYKO_DA": "BUDYKO_DA",
#         "BUDYKO_DA_MODEL": "BUDYKO_DA",
#         "BUDYKO+DA": "BUDYKO_DA",
#         "DA": "BUDYKO_DA",
#         "ENKF": "BUDYKO_DA",
#     }

#     if s not in mapping:
#         raise ValueError(f"Unknown scenario: {s}")

#     return mapping[s]


# def validate_config(cfg: dict):
#     if "scenario" not in cfg:
#         raise KeyError("Missing required config key: scenario")

#     if "paths" not in cfg:
#         raise KeyError("Missing required config key: paths")

#     paths = cfg["paths"]
#     for k in ["data_dir", "result_dir", "calibrated_params"]:
#         if k not in paths:
#             raise KeyError(f"Missing required config key: paths.{k}")


# # ============================================================
# # PARAM MODE NORMALIZATION
# # ============================================================
# def normalize_param_mode(mode: str) -> str:
#     """
#     Normalize param_mode names into canonical values:
#       - CALIBRATED
#       - UNCALIBRATED
#     """
#     s = str(mode).strip().upper()

#     mapping = {
#         "CALIBRATED": "CALIBRATED",
#         "UNCALIBRATED": "UNCALIBRATED",
#         "SIMULATION_WITH_CALIB_PARAMS": "CALIBRATED",
#         "SIMULATION_WITH_UNCALIB_PARAMS": "UNCALIBRATED",
#     }

#     if s not in mapping:
#         raise ValueError(
#             f"Unknown param_mode: {mode}\n"
#             f"Allowed: Simulation_with_CALIB_PARAMS | Simulation_with_UNCALIB_PARAMS | CALIBRATED | UNCALIBRATED"
#         )

#     return mapping[s]


# # ============================================================
# # APPLY PARAM BLOCK
# # ============================================================
# def apply_param_block(cfg: dict):
#     """
#     If cfg['params'] contains CALIBRATED and UNCALIBRATED blocks,
#     replace cfg['params'] with the active chosen block.
#     """
#     if "params" not in cfg:
#         return

#     params_cfg = cfg["params"]
#     if not isinstance(params_cfg, dict):
#         return

#     # expect canonical blocks
#     if "CALIBRATED" not in params_cfg and "UNCALIBRATED" not in params_cfg:
#         return

#     pmode = cfg.get("param_mode", "CALIBRATED")
#     pmode = normalize_param_mode(pmode)

#     if pmode not in params_cfg:
#         raise ValueError(
#             f"params block missing '{pmode}'. Available keys: {list(params_cfg.keys())}"
#         )

#     chosen = params_cfg[pmode]
#     if not isinstance(chosen, dict):
#         raise ValueError(f"params.{pmode} must be a dict")

#     cfg["params"] = chosen
#     cfg["param_mode"] = pmode


# # ============================================================
# # RESULT DIRECTORY REDIRECTION BY PARAM MODE
# # ============================================================
# def apply_result_dir_by_param_mode(cfg: dict):
#     """
#     Save results under different folder depending on param_mode.
#     """
#     if "paths" not in cfg:
#         return
#     if "result_dir" not in cfg["paths"]:
#         return

#     pmode = normalize_param_mode(cfg.get("param_mode", "CALIBRATED"))
#     base_result_dir = cfg["paths"]["result_dir"]

#     if pmode == "CALIBRATED":
#         cfg["paths"]["result_dir"] = os.path.join(base_result_dir, "Simulation_with_CALIB_PARAMS")
#     elif pmode == "UNCALIBRATED":
#         cfg["paths"]["result_dir"] = os.path.join(base_result_dir, "Simulation_with_UNCALIB_PARAMS")


# # ============================================================
# # MAIN
# # ============================================================
# def main():
#     parser = argparse.ArgumentParser(
#         description="Run Budyko + DA hydrologic simulations"
#     )
#     parser.add_argument(
#         "--config",
#         default="config.yaml",
#         help="Path to configuration YAML file",
#     )
#     parser.add_argument(
#         "--no-parallel",
#         action="store_true",
#         help="Force sequential run (debugging mode)",
#     )
#     parser.add_argument(
#         "--dry-run",
#         action="store_true",
#         help="Only print settings (do not run)",
#     )
#     parser.add_argument(
#         "--param-mode", "--param_mode",
#         dest="param_mode",
#         default=None,
#         help="Select parameter/simulation mode: Simulation_with_CALIB_PARAMS or Simulation_with_UNCALIB_PARAMS",
#     )

#     args = parser.parse_args()

#     # Banner
#     banner()

#     # Load config
#     cfg = load_config(args.config)
#     validate_config(cfg)

#     # Normalize scenario to canonical key
#     cfg["scenario"] = normalize_scenario_key(cfg["scenario"])

#     # param_mode from CLI overrides YAML
#     if args.param_mode is not None:
#         cfg["param_mode"] = args.param_mode
#     else:
#         cfg["param_mode"] = cfg.get("param_mode", "Simulation_with_CALIB_PARAMS")

#     # Normalize param_mode
#     cfg["param_mode"] = normalize_param_mode(cfg["param_mode"])

#     # Pick correct params block (very important!)
#     apply_param_block(cfg)

#     # Redirect result dir based on param_mode
#     apply_result_dir_by_param_mode(cfg)

#     # Optional CLI override
#     if args.no_parallel:
#         cfg.setdefault("parallel", {})
#         cfg["parallel"]["enabled"] = False

#     # Show run info
#     print("\n" + "-" * 110)
#     print(f"✅ Project Root : {PROJECT_ROOT}")
#     print(f"✅ Config File  : {cfg['_config_path']}")
#     print(f"✅ Scenario     : {cfg['scenario']}")
#     print(f"✅ Param Mode   : {cfg['param_mode']}")
#     print(f"✅ Result Dir   : {cfg['paths']['result_dir']}")
#     print(f"✅ Start Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print("-" * 110)

#     if args.dry_run:
#         print("\n🟡 DRY RUN enabled. Configuration loaded successfully. No execution performed.\n")
#         return

#     # ============================================================
#     # DISPATCH correct engine
#     # ============================================================
#     if cfg["param_mode"] == "CALIBRATED":
#         print("\n✅ Running simulation engine: scripts/run_simulation.py\n")
#         run_sim_calibrated(cfg)

#     elif cfg["param_mode"] == "UNCALIBRATED":
#         print("\n✅ Running simulation engine: scripts/run_simulation_uncalib_params.py\n")
#         run_sim_uncalibrated(cfg)

#     else:
#         raise ValueError(f"Invalid param_mode: {cfg['param_mode']}")


# # ============================================================
# # main:
# # ============================================================
# if __name__ == "__main__":
#     main()















# # run.py
# import os
# import sys
# import argparse
# import yaml
# import shutil
# from datetime import datetime

# # ============================================================
# # PROJECT ROOT
# # ============================================================
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# from scripts.run_simulation import run_simulations_from_config


# # ============================================================
# # UTILITIES
# # ============================================================
# def print_centered(text: str, width: int = None):
#     if width is None:
#         width = shutil.get_terminal_size((120, 20)).columns
#     for line in text.split("\n"):
#         print(line.center(width))


# def banner():
#     ascii_header = r"""
# ==========================================================================================================
#                  BAYESIAN DATA ASSIMILATION + BUDYKO-CONSTRAINED HYDROLOGIC SIMULATION

#                                Authors:   Habtamu T., Mesfin M

#   SCENARIOS IMPLEMENTED:
#             [1] BASE MODEL   : Simple scaling factor based ET estimation
#             [2] BUDYKO MODEL : Dynamic Budyko-based ET estimation
#             [3] BUDYKO DA    : Budyko model with Ensemble Kalman Filter data assimilation
# ==========================================================================================================
# """
#     print_centered(ascii_header)


# def load_config(config_path: str) -> dict:
#     """
#     Load YAML config. Supports relative path from project root.
#     """
#     if not os.path.isabs(config_path):
#         config_path = os.path.join(PROJECT_ROOT, config_path)

#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"Config file not found:\n  {config_path}")

#     with open(config_path, "r", encoding="utf-8") as f:
#         cfg = yaml.safe_load(f)

#     if cfg is None:
#         raise ValueError(f"YAML file is empty/invalid:\n  {config_path}")

#     cfg["_config_path"] = config_path
#     return cfg


# def normalize_scenario_key(s: str) -> str:
#     s = str(s).strip().upper()

#     mapping = {
#         "ALL": "ALL",

#         # BASE
#         "BASE": "BASE",
#         "BASE_MODEL": "BASE",
#         "BASE-MODEL": "BASE",

#         # BUDYKO
#         "BUDYKO": "BUDYKO",
#         "BUDYKO_MODEL": "BUDYKO",
#         "BUDYKO-MODEL": "BUDYKO",

#         # BUDYKO_DA
#         "BUDYKO_DA": "BUDYKO_DA",
#         "BUDYKO_DA_MODEL": "BUDYKO_DA",
#         "BUDYKO+DA": "BUDYKO_DA",
#         "DA": "BUDYKO_DA",
#         "ENKF": "BUDYKO_DA",
#     }

#     if s not in mapping:
#         raise ValueError(f"Unknown scenario: {s}")

#     return mapping[s]


# def validate_config(cfg: dict):
#     if "scenario" not in cfg:
#         raise KeyError("Missing required config key: scenario")

#     if "paths" not in cfg:
#         raise KeyError("Missing required config key: paths")

#     paths = cfg["paths"]
#     for k in ["data_dir", "result_dir", "calibrated_params"]:
#         if k not in paths:
#             raise KeyError(f"Missing required config key: paths.{k}")


# # ============================================================
# # PARAMETER MODE HANDLING (CALIBRATED vs UNCALIBRATED)
# # ============================================================
# def apply_param_mode(cfg: dict):
#     """
#     If cfg['params'] contains multiple modes (CALIBRATED/UNCALIBRATED),
#     select the correct block based on cfg['param_mode'] and replace cfg['params']
#     with the chosen config.
#     """
#     if "params" not in cfg:
#         return

#     params_cfg = cfg["params"]

#     # If params is already a single dictionary (old style), do nothing.
#     if not isinstance(params_cfg, dict):
#         return

#     # If params does not contain nested mode blocks, do nothing.
#     # (we detect nested mode blocks by checking for dict values with keys like use_calibrated/bounds)
#     if "CALIBRATED" not in params_cfg and "UNCALIBRATED" not in params_cfg:
#         return

#     # Get param_mode
#     param_mode = str(cfg.get("param_mode", "CALIBRATED")).strip().upper()

#     if param_mode not in params_cfg:
#         raise ValueError(
#             f"Invalid param_mode='{param_mode}'. Must be one of: {list(params_cfg.keys())}"
#         )

#     chosen = params_cfg[param_mode]
#     if not isinstance(chosen, dict):
#         raise ValueError(f"cfg['params']['{param_mode}'] must be a dictionary.")

#     # Replace params with the chosen block (so downstream scripts do not change)
#     cfg["params"] = chosen
#     cfg["param_mode"] = param_mode


# # ============================================================
# # MAIN
# # ============================================================
# def main():
#     parser = argparse.ArgumentParser(
#         description="Run Budyko + DA hydrologic simulations"
#     )
#     parser.add_argument(
#         "--config",
#         default="config.yaml",
#         help="Path to configuration YAML file",
#     )
#     parser.add_argument(
#         "--no-parallel",
#         action="store_true",
#         help="Force sequential run (debugging mode)",
#     )
#     parser.add_argument(
#         "--dry-run",
#         action="store_true",
#         help="Only print settings (do not run)",
#     )

#     parser.add_argument(
#         "--param-mode", "--param_mode",
#         dest="param_mode",
#         default=None,
#         help="Parameter mode override: CALIBRATED or UNCALIBRATED",
#     )

#     args = parser.parse_args()

#     # Banner
#     banner()

#     # Load config
#     cfg = load_config(args.config)
#     validate_config(cfg)

#     # Normalize scenario to canonical key
#     cfg["scenario"] = normalize_scenario_key(cfg["scenario"])

#     # ✅ apply CLI override to cfg
#     if args.param_mode is not None:
#         cfg["param_mode"] = args.param_mode

#     # ✅ choose params mode if applicable
#     apply_param_mode(cfg)

#     # Optional CLI override
#     if args.no_parallel:
#         cfg.setdefault("parallel", {})
#         cfg["parallel"]["enabled"] = False

#     # Show run info
#     print("\n" + "-" * 110)
#     print(f"✅ Project Root : {PROJECT_ROOT}")
#     print(f"✅ Config File  : {cfg['_config_path']}")
#     print(f"✅ Scenario     : {cfg['scenario']}")
#     if "param_mode" in cfg:
#         print(f"✅ Param Mode   : {cfg['param_mode']}")
#     print(f"✅ Start Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print("-" * 110)

#     if args.dry_run:
#         print("\n🟡 DRY RUN enabled. Configuration loaded successfully. No execution performed.\n")
#         return

#     # Run simulations
#     run_simulations_from_config(cfg)


# # ============================================================
# # main:
# # ============================================================
# if __name__ == "__main__":
#     main()
