from pathlib import Path
import pandas as pd
from .config import DATASET_DIR, FEATURES, TARGET

def find_dataset(directory: Path = DATASET_DIR) -> Path:
    for path in sorted(directory.glob("*.csv")):
        columns = pd.read_csv(path, nrows=0).columns
        if TARGET in columns and all(feature in columns for feature in FEATURES):
            return path
    raise FileNotFoundError(f"No compatible AI4I CSV found in {directory}.")

def validate_dataset(frame: pd.DataFrame) -> None:
    missing = [c for c in FEATURES + [TARGET] if c not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame.empty or frame[TARGET].isna().any():
        raise ValueError("Dataset must be non-empty and have no missing target values.")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Machine failure must contain only 0 and 1.")

def load_dataset(path: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path or find_dataset())
    validate_dataset(frame)
    return frame

def select_features(frame: pd.DataFrame):
    validate_dataset(frame)
    return frame[FEATURES].copy(), frame[TARGET].astype(int).copy()

def dataset_summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicates": int(frame.duplicated().sum()),
        "failure_rate": float(frame[TARGET].mean()),
        "class_distribution": {
            str(k): int(v) for k, v in frame[TARGET].value_counts().sort_index().items()
        },
    }
