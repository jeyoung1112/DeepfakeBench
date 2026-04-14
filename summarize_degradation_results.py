import pandas as pd
import glob
import json
import re

target_path = "./fpr_degradation_results"
target_files = glob.glob(target_path + "/*.json")

metrics = [
    "auc", "eer", "acc",
    "tpr_at_fpr_1", "fpr_at_tpr_99",
]

rows = []
for path in target_files:
    match = re.match(r".*/(.+)_fpr_degradation\.json", path)
    if not match:
        continue

    model_name = match.group(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for dataset, degradations in data.items():
        for degradation_type, params in degradations.items():
            for param, values in params.items():
                row = {
                    "model": model_name,
                    "dataset": dataset,
                    "degradation": degradation_type,
                    "param": param,
                }
                row.update({m: values.get(m) for m in metrics})
                rows.append(row)

df = pd.DataFrame(rows).sort_values(["model", "dataset", "degradation", "param"]).reset_index(drop=True)
print(df.to_string(index=False))
df.to_csv("degradation_results_summary.csv", index=False)
print("\nSaved to results_summary.csv")
