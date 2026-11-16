from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

def choose_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = (2 * precision * recall / (precision + recall).clip(min=1e-12))[:-1]
    return float(thresholds[f1.argmax()]) if len(thresholds) else .5

def metrics_at_threshold(y_true, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0,
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }

def save_evaluation_artifacts(model, x_valid, y_valid, x_test, y_test,
                              output_dir: Path, threshold: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_scores = model.predict_proba(x_valid)[:, 1]
    test_scores = model.predict_proba(x_test)[:, 1]
    valid = metrics_at_threshold(y_valid, valid_scores, threshold)
    test = metrics_at_threshold(y_test, test_scores, threshold)
    pd.DataFrame([{"split": "validation", **valid}, {"split": "test", **test}]).to_csv(
        output_dir / "final_metrics.csv", index=False)
    sns.heatmap(confusion_matrix(y_test, test_scores >= threshold), annot=True, fmt="d", cmap="Blues")
    plt.title("Final test confusion matrix"); plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150); plt.close()
    fpr, tpr, _ = roc_curve(y_test, test_scores)
    plt.plot(fpr, tpr, label=f"ROC-AUC={test['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray"); plt.legend()
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.tight_layout(); plt.savefig(output_dir / "roc_curve.png", dpi=150); plt.close()
    precision, recall, _ = precision_recall_curve(y_test, test_scores)
    plt.plot(recall, precision, label=f"AP={test['average_precision']:.3f}"); plt.legend()
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.tight_layout(); plt.savefig(output_dir / "precision_recall_curve.png", dpi=150); plt.close()
    permutation = permutation_importance(model, x_valid, y_valid, scoring="average_precision",
                                          n_repeats=3, random_state=42, n_jobs=-1)
    importance = pd.DataFrame({"feature": x_valid.columns,
                               "importance_mean": permutation.importances_mean})
    importance.sort_values("importance_mean", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False)
    ordered = importance.sort_values("importance_mean")
    plt.barh(ordered["feature"], ordered["importance_mean"], color="#457b9d")
    plt.xlabel("Mean decrease in average precision")
    plt.title("Validation permutation feature importance")
    plt.tight_layout(); plt.savefig(output_dir / "feature_importance.png", dpi=150); plt.close()
    return {"validation": valid, "test": test}
