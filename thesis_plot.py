import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt



## =================================== Ablation Figure =================================================
# file_path = "/home/jeyoung/personal/DeepfakeBench/figures"
# # file_name = "ablation_covariance.csv"
# file_name = "ablation_variance.csv"

# df = pd.read_csv(os.path.join(file_path, file_name))
# weight_col = df.columns[0]
# auc_col = "Average AUC"
# tpr_col = "Average TPR@FPR5%"

# fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# x = df[weight_col]
# a = list(range(len(x)))
# color_auc = "steelblue"
# color_tpr = "darkorange"

# axes[0].plot(a, df[auc_col], marker="o", linewidth=2, color=color_auc)
# axes[0].set_xlabel(weight_col.title())
# axes[0].set_ylabel("Average AUC")
# axes[0].set_title("Average AUC vs " + weight_col.title())
# axes[0].xaxis.set_ticks(a)
# axes[0].xaxis.set_ticklabels(x)
# axes[0].grid(True, linestyle="--", alpha=0.4)

# axes[1].plot(a, df[tpr_col], marker="s", linewidth=2, color=color_tpr)
# axes[1].set_xlabel(weight_col.title())
# axes[1].set_ylabel("Average TPR@FPR5%")
# axes[1].set_title("Average TPR@FPR5% vs " + weight_col.title())
# axes[1].xaxis.set_ticks(a)
# axes[1].xaxis.set_ticklabels(x)
# axes[1].grid(True, linestyle="--", alpha=0.4)

# plt.tight_layout()
# out_name = os.path.splitext(file_name)[0] + "_plot.png"
# plt.savefig(os.path.join(file_path, out_name), dpi=150, bbox_inches="tight")
# plt.show()
# print(f"Saved to {os.path.join(file_path, out_name)}")

## =================================== Introduction Figure ===============================================

file_path = "/home/jeyoung/personal/DeepfakeBench/figures"
file_name = "problem_formulation.csv"

df = pd.read_csv(os.path.join(file_path, file_name))

datasets = ["FF++", "DFD", "Celeb-DF-v2"]
models = df["model_name"].str.upper().tolist()
auc_cols = [f"{d} - AUC" for d in datasets]
tpr_cols = [f"{d} - TPR@FPR=5%" for d in datasets]

x = np.arange(len(datasets))
bar_width = 0.25
offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * bar_width
colors = ["#4C72B0", "#DD8452", "#55A868"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for i, (model, color, offset) in enumerate(zip(models, colors, offsets)):
    auc_vals = df[auc_cols].iloc[i].values
    tpr_vals = df[tpr_cols].iloc[i].values
    axes[0].bar(x + offset, auc_vals, bar_width, label=model, color=color, edgecolor="white", linewidth=0.6)
    axes[1].bar(x + offset, tpr_vals, bar_width, label=model, color=color, edgecolor="white", linewidth=0.6)

for ax, ylabel, ylim in [
    (axes[0], "AUC", (0.0, 1.0)),
    (axes[1], "TPR @ FPR=5%", (0.0, 1.0)),
]:
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=11, framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)

plt.tight_layout()
out_name = os.path.splitext(file_name)[0] + "_plot.png"
plt.savefig(os.path.join(file_path, out_name), dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved to {os.path.join(file_path, out_name)}")