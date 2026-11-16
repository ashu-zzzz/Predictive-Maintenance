from pathlib import Path
import json
import joblib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from .config import MODELS_DIR, RANDOM_STATE
from .preprocessing import build_preprocessor

def split_data(features, target):
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features, target, test_size=.30, stratify=target, random_state=RANDOM_STATE)
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_holdout, y_holdout, test_size=.50, stratify=y_holdout, random_state=RANDOM_STATE)
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def build_models():
    specs = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=250, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }
    return {name: Pipeline([("preprocessor", build_preprocessor()), ("model", model)])
            for name, model in specs.items()}

def save_model(model, metadata: dict, directory: Path = MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "selected_model.joblib")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
