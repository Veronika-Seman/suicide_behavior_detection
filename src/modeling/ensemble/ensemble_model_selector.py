from pathlib import Path
import pandas as pd

# ROOT points to the project root
ROOT = Path(__file__).resolve().parents[3]

# Path where all final model outputs are stored
FINAL_ROOT = ROOT / "src" / "modeling" / "final"

# --- Expected metrics files for each model ---
metrics_files = {
    "BERT_Base" : FINAL_ROOT / "BERT_Base"  / "bert_final_metrics.csv",
    "DistilBERT": FINAL_ROOT / "DistilBERT" / "distilbert_final_metrics.csv",
    "MentalBERT": FINAL_ROOT / "MentalBERT" / "mentalbert_final_metrics.csv",
    "RoBERTa"   : FINAL_ROOT / "RoBERTa"    / "roberta_final_metrics.csv",
    "SVM"       : FINAL_ROOT / "SVM"        / "svm_final_metrics.csv",
    "XLNet"     : FINAL_ROOT / "XLNet"      / "xlnet_final_metrics.csv",
}

# --- Collect results ---
results = []
for model_name, path in metrics_files.items():
    if not path.exists():
        print(f"Missing metrics file: {path}")
        continue
    try:
        # Read CSV into DataFrame
        df = pd.read_csv(path)
        # Add a column to record which model these metrics belong to
        df["model_name"] = model_name
        results.append(df)
    except Exception as e:
        print(f"Problem reading file: {path}\n{e}")

# --- Validation ---
if not results:
    raise RuntimeError(f"No valid metrics files read under:\n{FINAL_ROOT}")

# Merge all metrics into a single DataFrame
all_metrics = pd.concat(results, ignore_index=True)

# --- Sort models by recall (descending order) ---
all_metrics_sorted = all_metrics.sort_values(by="recall", ascending=False)

# Print all metrics sorted by recall
print(all_metrics_sorted)

# --- Select Top 3 models (by recall) for ensemble ---
top3_models = all_metrics_sorted.head(3)
print("\nTop 3 Models for ensemble:")
print(top3_models)
