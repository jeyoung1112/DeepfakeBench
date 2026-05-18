import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

file_path = "/home/jeyoung/personal/DeepfakeBench/figures"
file_name = "collapse_evidence.csv"

df = pd.read_csv(os.path.join(file_path, file_name))

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(df["iterations"], df["original"], label="Original", linewidth=1.5)
ax.plot(df["iterations"], df["no varcov"], label="No VarCov", linewidth=1.5)
ax.plot(df["iterations"], df["varcov effect"], label="VarCov Effect", linewidth=1.5)

ax.set_xlabel("Iterations")
ax.set_ylabel("Mean Real feats. Variance")
ax.set_title("Mean Real feats. Variance over Iterations")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(file_path, "collapse_evidence.png"), dpi=150)
plt.show()