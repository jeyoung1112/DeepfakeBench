import pandas as pd
import glob
import json
import re

target_path = "./confusion_matrix_results"
# target_files = glob.glob(target_path + "/*.json")
target_files = glob.glob(target_path + "/*_dfdcp_*.json")

metrics = [
    "tp", "fp", "fn", "tn",
    "sklearn_auc", "eer",
    "tpr_at_fpr1pct", "tpr_at_fpr5pct",
]

rows = []
for path in target_files:
    match = re.match(r".*/(.+)_FaceForensics\+\+_cm_(.+)\.json", path)
    if not match:
        continue

    model_name = match.group(1)
    dataset = match.group(2)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    row = {"model": model_name, "dataset": dataset}
    row.update({m: data.get(m) for m in metrics})
    row["accuracy"] = (data["tp"] + data["tn"]) / (data["tp"] + data["tn"] + data["fp"] + data["fn"])
    rows.append(row)

df = pd.DataFrame(rows).sort_values(["model", "dataset"]).reset_index(drop=True)
print(df.to_string(index=False))
df.to_csv("results_summary.csv", index=False)
print("\nSaved to results_summary.csv")
