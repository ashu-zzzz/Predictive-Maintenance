from pathlib import Path
import json
import joblib
import pandas as pd
from .config import FEATURES, MODELS_DIR

def load_artifacts(model_dir: Path = MODELS_DIR):
    model_path, metadata_path = model_dir / "selected_model.joblib", model_dir / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Run python run_pipeline.py before making predictions.")
    return joblib.load(model_path), json.loads(metadata_path.read_text(encoding="utf-8"))

def validate_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")
    result = frame[FEATURES].copy()
    if not result["Type"].isin(["H", "M", "L"]).all():
        raise ValueError("Type must contain only H, M, or L.")
    for column in FEATURES[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[FEATURES[1:]].isna().any().any():
        raise ValueError("Numeric inputs must contain valid numbers.")
    return result

def predict_frame(frame, model, threshold):
    inputs = validate_inputs(frame)
    scores = model.predict_proba(inputs)[:, 1]
    result = frame.copy()
    result["failure_risk_score"] = scores
    result["predicted_failure"] = (scores >= threshold).astype(int)
    return result
