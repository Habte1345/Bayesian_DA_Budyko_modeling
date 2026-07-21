# ============================================================
# Load libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import json

# ============================================================
# Font settings (Times New Roman)
# ============================================================
LABEL_FONTSIZE = 14

font_path_regular = "C:/Windows/Fonts/times.ttf"
font_path_bold    = "C:/Windows/Fonts/timesbd.ttf"

font_manager.fontManager.addfont(font_path_regular)
font_manager.fontManager.addfont(font_path_bold)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": LABEL_FONTSIZE,
})

# ============================================================
# Load basin calibration CSV
# ============================================================
param_file_path = (
    "F:/Github_repos/Bayesian_DA_Budyko_modeling/"
    "SCE_cal_params/final_calibrated_params_with_KGE.csv"
)

df = pd.read_csv(param_file_path, index_col="Basin")

# ============================================================
# Prepare boxplot data
# ============================================================
params_to_plot = ["Kperc", "Kb", "Ke", "Cqq"]

df_long = (
    df[params_to_plot]
    .reset_index(drop=True)
    .melt(var_name="Parameter", value_name="Value")
)

df_long["Parameter"] = pd.Categorical(
    df_long["Parameter"],
    categories=params_to_plot,
    ordered=True
)

df_stats = df_long.groupby("Parameter")["Value"].median().reset_index()

# ============================================================
# Load global calibration JSONs
# ============================================================
param_json_path = "F:/Github_repos/Bayesian_DA_Budyko_modeling/SCE_global_params/global_calibrated_params.json"
summary_json_path = "F:/Github_repos/Bayesian_DA_Budyko_modeling/SCE_global_params/global_calibration_summary.json"

with open(param_json_path, "r") as f:
    global_params = json.load(f)

with open(summary_json_path, "r") as f:
    summary = json.load(f)

param_names = ["Kperc", "Kb", "Ke", "Cqq"]
param_values = [global_params[p] for p in param_names]

kge_labels = ["Train Mean", "Train Median", "Test Mean", "Test Median"]
kge_values = [
    summary["mean_train_kge"],
    summary["median_train_kge"],
    summary["mean_test_kge"],
    summary["median_test_kge"],
]

# ============================================================
# Create figure (3 panels)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

# Use JET colormap
cmap = plt.cm.coolwarm

# ============================================================
# (a) Basin distributions
# ============================================================
ax = axes[0]

box_data = [
    df_long[df_long["Parameter"] == p]["Value"].dropna().values
    for p in params_to_plot
]

bp = ax.boxplot(
    box_data,
    vert=False,
    patch_artist=True,
    widths=0.4,
    showfliers=False
)

colors_box = [cmap(i) for i in np.linspace(0, 1, len(params_to_plot))]

for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_edgecolor("black")

for element in ["whiskers", "caps", "medians"]:
    for line in bp[element]:
        line.set_color("black")
        line.set_linewidth(1.5)

# median labels
for i, (param, median) in enumerate(zip(df_stats["Parameter"], df_stats["Value"])):
    ax.text(
        median,
        i + 1.3,
        f"{median:.2f}",
        ha="center",
        va="bottom",
        fontsize=LABEL_FONTSIZE + 2,
        fontweight="bold",
        color="blue"
    )

ax.set_yticks(np.arange(1, len(params_to_plot) + 1))
ax.set_yticklabels(params_to_plot)
ax.set_xlabel("Basin-scale [505 Basins]", fontsize=LABEL_FONTSIZE+4, fontweight="bold")
ax.tick_params(axis="both", labelsize=LABEL_FONTSIZE+2)
ax.set_title("(a)", loc="left", fontsize=LABEL_FONTSIZE+8)

# ============================================================
# (b) Global parameters
# ============================================================
ax = axes[1]

colors_global = [cmap(i) for i in np.linspace(0, 1, len(param_names))]

ax.barh(param_names, param_values, color=colors_global)

for i, v in enumerate(param_values):
    ax.text(v, i, f"{v:.2f}", va="center", fontsize=LABEL_FONTSIZE+2)

ax.set_xlabel("Global [Test basins=151]", fontsize=LABEL_FONTSIZE+4, fontweight="bold")
ax.set_xlim(0, 1)
ax.tick_params(axis="both", labelsize=LABEL_FONTSIZE+2)
ax.set_title("(b)", loc="left", fontsize=LABEL_FONTSIZE+8)

# ============================================================
# (c) KGE performance
# ============================================================
ax = axes[2]

colors_kge = 'grey'

ax.barh(kge_labels, kge_values, color=colors_kge)

for i, v in enumerate(kge_values):
    ax.text(v, i, f"{v:.2f}", va="center", fontsize=LABEL_FONTSIZE+2)

ax.set_xlabel("KGE", fontsize=LABEL_FONTSIZE+4, fontweight="bold")
ax.set_xlim(0, 0.7)
ax.tick_params(axis="both", labelsize=LABEL_FONTSIZE+2)
ax.set_title("(c)", loc="left", fontsize=LABEL_FONTSIZE+8)

# ============================================================
# Remove top & right borders
# ============================================================
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ============================================================
# Final layout
# ============================================================
plt.tight_layout()
plt.show()