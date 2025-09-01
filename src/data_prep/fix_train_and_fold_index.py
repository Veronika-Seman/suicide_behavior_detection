from pathlib import Path
import pandas as pd

# ===== Project ROOT & default paths =====
ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Defaults (override via function args if needed)
DEFAULT_TRAINVAL = SPLITS_DIR / "train_val.csv"
DEFAULT_TRAINVAL_FIXED = SPLITS_DIR / "train_val_fixed.csv"
DEFAULT_FOLDS = SPLITS_DIR / "fold_assignments.csv"
DEFAULT_FOLDS_FIXED = SPLITS_DIR / "fold_assignments_fixed.csv"


def fix_train_val(input_csv: Path = DEFAULT_TRAINVAL,
                  output_csv: Path = DEFAULT_TRAINVAL_FIXED) -> None:
    """
    Load the prepared train/val CSV, set the 'index' column as the DataFrame index,
    and save a fixed copy. Keeps the index in the file (no index=False) so downstream
    code can read it with `index_col=0`.
    """
    df = pd.read_csv(input_csv)
    if "index" not in df.columns:
        raise ValueError(f"'index' column not found in: {input_csv}")
    df.set_index("index", inplace=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv)
    print(f" Saved fixed train_val → {output_csv}")


def fix_folds(folds_csv: Path = DEFAULT_FOLDS,
              train_val_fixed: Path = DEFAULT_TRAINVAL_FIXED,
              output_csv: Path = DEFAULT_FOLDS_FIXED) -> None:
    """
    Load folds and the fixed train/val; keep only fold rows whose 'index' exists
    in train_val (prevents dangling fold indices). Save to *_fixed.csv without index.
    """
    folds_df = pd.read_csv(folds_csv)
    tv_df = pd.read_csv(train_val_fixed, index_col=0)
    if "index" not in folds_df.columns:
        raise ValueError(f"'index' column not found in: {folds_csv}")
    valid_folds_df = folds_df[folds_df["index"].isin(tv_df.index)]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    valid_folds_df.to_csv(output_csv, index=False)
    print(f" Saved fixed folds → {output_csv} "
          f"(kept {len(valid_folds_df)}/{len(folds_df)} rows)")


def sync_train_val_with_folds(train_val_csv: Path = DEFAULT_TRAINVAL,
                              folds_fixed_csv: Path = DEFAULT_FOLDS_FIXED,
                              output_csv: Path = DEFAULT_TRAINVAL_FIXED) -> None:
    """
    Align train_val rows with the (fixed) folds file:
    keep only rows whose DataFrame index appears in folds 'index', drop
    any stray 'Unnamed: 0' column if present, then save as train_val_fixed.csv.
    """
    folds_df = pd.read_csv(folds_fixed_csv)
    if "index" not in folds_df.columns:
        raise ValueError(f"'index' column not found in: {folds_fixed_csv}")
    tv_df = pd.read_csv(train_val_csv)

    filtered_df = tv_df.loc[tv_df.index.isin(folds_df["index"])].copy()
    if "Unnamed: 0" in filtered_df.columns:
        filtered_df.drop(columns=["Unnamed: 0"], inplace=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_csv, index=False)
    print(f"✔️ Synced train_val with folds → {output_csv} "
          f"(kept {len(filtered_df)}/{len(tv_df)} rows)")


if __name__ == "__main__":
    # Run all steps in a stable order:
    # 1) ensure train_val has 'index' as index
    fix_train_val()
    # 2) make folds consistent with train_val_fixed
    fix_folds()
    # 3) make train_val consistent with the fixed folds (final guard-rail)
    sync_train_val_with_folds()
