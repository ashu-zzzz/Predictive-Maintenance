import json
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.config import METRICS_DIR, MODELS_DIR, PLOTS_DIR
from src.data_loader import dataset_summary, load_dataset, select_features
from src.eda import generate_eda_plots
from src.evaluate import choose_threshold, metrics_at_threshold, save_evaluation_artifacts
from src.train import build_models, save_model, split_data

def main():
    print("Loading and validating dataset...")
    frame = load_dataset()
    summary = dataset_summary(frame)
    for directory in (METRICS_DIR, MODELS_DIR, PLOTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Loaded {summary['rows']} rows; failure rate={summary['failure_rate']:.4f}")
    generate_eda_plots(frame, PLOTS_DIR)
    features, target = select_features(frame)
    x_train, x_valid, x_test, y_train, y_valid, y_test = split_data(features, target)
    trained, comparison = {}, []
    print("Training models...")
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        trained[name] = model
        scores = model.predict_proba(x_valid)[:, 1]
        threshold = choose_threshold(y_valid, scores)
        comparison.append({"model": name, "validation_average_precision":
                           average_precision_score(y_valid, scores), "validation_threshold": threshold,
                           **metrics_at_threshold(y_valid, scores, threshold)})
    comparison_frame = pd.DataFrame(comparison).sort_values(
        "validation_average_precision", ascending=False)
    comparison_frame.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    selected_name = str(comparison_frame.iloc[0]["model"])
    selected = trained[selected_name]
    threshold = choose_threshold(y_valid, selected.predict_proba(x_valid)[:, 1])
    results = save_evaluation_artifacts(selected, x_valid, y_valid, x_test, y_test,
                                        METRICS_DIR, threshold)
    metadata = {"selected_model": selected_name, "threshold": threshold,
                "features": list(features.columns), "dataset_summary": summary, **results}
    save_model(selected, metadata)
    print(f"Selected model: {selected_name}; validation threshold={threshold:.6f}")
    print(pd.Series(results["test"]).to_string())

if __name__ == "__main__":
    main()
