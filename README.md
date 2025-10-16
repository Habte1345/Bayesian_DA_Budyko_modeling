# 🌊 Hydrologic Data Assimilation with Two-Store Model and Budyko Constraint (TwoStore-EnKF-Model)

This repository contains a modular Python implementation of a simple **Two-Store Hydrologic Model** coupled with an **Ensemble Kalman Filter (EnKF)** for state and parameter estimation, including a scenario constrained by the **Budyko hypothesis**.

## Features
* **Modular Design:** Code separated into distinct modules (`model.py`, `enkf.py`, `budyko.py`).
* **EnKF Implementation:** Supports joint state/parameter estimation.
* **Budyko Constraint:** Includes a scenario where Evapotranspiration (ET) is constrained by a dynamically estimated Budyko parameter ($\omega$).

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* The required libraries listed in `requirements.txt`.

### Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/YourUsername/Hydrologic-DA-Budyko.git](https://github.com/YourUsername/Hydrologic-DA-Budyko.git)
    cd Hydrologic-DA-Budyko
    ```
2.  Create and activate a virtual environment (recommended).
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Simulation
1.  Place your input data CSV files (P, PET, Q_NLDAS, Q_USGS, etc.) in the `data/` folder.
2.  Edit the data loading section in `scripts/run_simulation.py` to match your data file names.
3.  Execute the main script:
    ```bash
    python scripts/run_simulation.py
    ```

## 📚 Code Structure

| File | Primary Classes/Functions | Description |
| :--- | :--- | :--- |
| `src/model.py` | `ModelParams`, `run_two_store_model` | Defines the hydrologic model structure and stepping function. |
| `src/enkf.py` | `EnKFConfig`, `enkf_update` | Implements the Ensemble Kalman Filter logic. |
| `src/budyko.py` | `fu_budyko`, `fit_omega_mlr` | Functions related to the Budyko equation and omega regression. |
| `src/metrics.py` | `calculate_nse`, `calculate_kge` | Calculates Nash-Sutcliffe Efficiency (NSE) and Kling-Gupta Efficiency (KGE). |