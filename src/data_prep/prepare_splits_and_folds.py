import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold

# ===== ROOT & PATHS =====
# Resolve project root
ROOT = Path(__file__).resolve().parents[2]

# Source dataset (must contain columns: Text, Lable)
SRC_DATA = ROOT / "data" / "processed" / "final_balanced_dataset.csv"

# Output split files
SPLITS_DIR   = ROOT / "data" / "processed" / "splits"
TRAINVAL_CSV = SPLITS_DIR / "train_val.csv"
TEST_CSV     = SPLITS_DIR / "test.csv"
FOLDS_CSV    = SPLITS_DIR / "fold_assignments.csv"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)

def load_xy_from(path: Path) -> pd.DataFrame:
    """Load dataset with columns [Text, Lable], drop NA, ensure int labels."""
    df = pd.read_csv(path)[["Text", "Lable"]].dropna()
    df["Lable"] = df["Lable"].astype(int)
    return df
# Explanation (load_xy_from):
# Loads only the needed columns, removes missing rows to avoid train-time errors, and casts label dtype to int.

def save_stratified_splits(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> None:
    """Create stratified train_val/test CSVs under SPLITS_DIR."""
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["Lable"], random_state=seed
    )
    train_val_df.to_csv(TRAINVAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
# Explanation (save_stratified_splits):
# Preserves class balance across train_val/test using Stratified split, then writes both CSVs to disk.

def save_folds(train_val_path: Path, out_path: Path, n_splits: int = 5, seed: int = 42) -> None:
    """Create 5-fold validation assignments for train_val.csv and save to out_path."""
    tv = pd.read_csv(train_val_path)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (_, val_idx) in enumerate(skf.split(tv["Text"], tv["Lable"])):
        for i in val_idx:
            rows.append({"index": int(tv.index[i]), "fold": fold})
    pd.DataFrame(rows).to_csv(out_path, index=False)
# Explanation (save_folds):
# Builds stratified folds only on train_val set; stores which row indices belong to each validation fold.

def main():
    df = load_xy_from(SRC_DATA)
    save_stratified_splits(df, test_size=0.2, seed=42)
    save_folds(TRAINVAL_CSV, FOLDS_CSV, n_splits=5, seed=42)
    print(f"Saved: {TRAINVAL_CSV.name}, {TEST_CSV.name}, {FOLDS_CSV.name} → {SPLITS_DIR}")

if __name__ == "__main__":
    main()
