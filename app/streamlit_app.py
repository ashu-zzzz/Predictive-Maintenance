import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FEATURES, METRICS_DIR
from src.predict import get_range_warnings, load_artifacts, predict_frame

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

st.title("Predictive Maintenance & Machine Failure Detection")
st.caption(
    "An educational prototype using the synthetic UCI AI4I 2020 dataset. "
    "Scores are model risk scores, not guaranteed real-world probabilities."
)

try:
    model, metadata = load_artifacts()
except FileNotFoundError as error:
    st.warning(str(error))
    st.stop()

test_metrics = metadata["test"]
model_name = metadata["selected_model"].replace("_", " ").title()
threshold = metadata["threshold"]

st.sidebar.header("Model")
st.sidebar.write(model_name)
st.sidebar.metric("Alert threshold", f"{threshold:.3f}")

overview, single, batch, performance = st.tabs(
    ["Overview", "Single prediction", "Batch prediction", "Model performance"]
)

with overview:
    st.header("Project overview")
    st.write(
        "This application classifies whether a machine is likely to experience "
        "a failure from six operating measurements. It is not a remaining-useful-life "
        "or exact time-to-failure predictor."
    )
    overview_metrics = [
        ("Selected model", model_name),
        ("Dataset size", f"{metadata['dataset_summary']['rows']:,} rows"),
        ("Failure rate", f"{metadata['dataset_summary']['failure_rate']:.2%}"),
        ("Test accuracy", f"{test_metrics['accuracy']:.3f}"),
        ("Test precision", f"{test_metrics['precision']:.3f}"),
        ("Test recall", f"{test_metrics['recall']:.3f}"),
        ("Test F1-score", f"{test_metrics['f1']:.3f}"),
        ("Test PR-AUC", f"{test_metrics['average_precision']:.3f}"),
    ]
    for start in range(0, len(overview_metrics), 4):
        columns = st.columns(4)
        for column, (label, value) in zip(columns, overview_metrics[start:start + 4]):
            column.metric(label, value)
    distribution = ROOT / "results" / "plots" / "failure_distribution.png"
    if distribution.exists():
        st.image(str(distribution), caption="Dataset failure distribution", width="stretch")

with single:
    st.header("Single machine prediction")
    st.write("Enter the six permitted operating inputs and assess the maintenance risk.")
    with st.form("single_prediction"):
        values = {
            "Type": st.selectbox("Machine type", ["H", "M", "L"]),
            "Air temperature [K]": st.number_input("Air temperature [K]", value=298.1),
            "Process temperature [K]": st.number_input("Process temperature [K]", value=308.6),
            "Rotational speed [rpm]": st.number_input("Rotational speed [rpm]", value=1500.0),
            "Torque [Nm]": st.number_input("Torque [Nm]", value=40.0),
            "Tool wear [min]": st.number_input("Tool wear [min]", value=0.0),
        }
        submitted = st.form_submit_button("Assess failure risk")
    if submitted:
        input_frame = pd.DataFrame([values])
        row = predict_frame(input_frame, model, threshold).iloc[0]
        for warning in get_range_warnings(input_frame, metadata.get("training_ranges")):
            st.warning(warning)
        score = row["failure_risk_score"]
        crossed = bool(row["predicted_failure"])
        st.metric("Failure risk score", f"{score:.2%}")
        st.write(f"Predicted failure status: **{'Failure alert' if crossed else 'No failure alert'}**")
        if crossed:
            st.error(
                "The score crosses the maintenance alert threshold. "
                "Inspect the machine and follow the relevant maintenance process."
            )
        else:
            st.success(
                "The score is below the maintenance alert threshold. "
                "Continue normal monitoring; this is not a guarantee of no failure."
            )

with batch:
    st.header("Batch prediction")
    st.write(
        "Upload a CSV containing only, or at least, the six required input columns. "
        "Identifiers, targets, and failure-mode columns are excluded from the output."
    )
    upload = st.file_uploader("Upload sensor readings", type="csv")
    if upload:
        try:
            uploaded = pd.read_csv(upload)
            result = predict_frame(uploaded, model, threshold)
            for warning in get_range_warnings(result, metadata.get("training_ranges")):
                st.warning(warning)
            st.dataframe(result, use_container_width=True)
            st.download_button(
                "Download predictions",
                result.to_csv(index=False),
                "predictions.csv",
                "text/csv",
            )
        except (ValueError, pd.errors.ParserError) as error:
            st.error(str(error))

with performance:
    st.header("Model performance")
    st.write("All reported values below are generated from the current pipeline artifacts.")
    comparison_path = METRICS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        st.subheader("Validation model comparison")
        st.dataframe(
            comparison[["model", "validation_average_precision", "validation_threshold"]],
            hide_index=True,
            use_container_width=True,
        )
    st.subheader("Selected model: final held-out test metrics")
    metric_names = [
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "average_precision", "balanced_accuracy", "specificity",
        "false_negative_rate", "true_negatives", "false_positives",
        "false_negatives", "true_positives",
    ]
    st.dataframe(
        pd.DataFrame([test_metrics])[metric_names].rename(
            columns={"average_precision": "pr_auc"}
        ),
        hide_index=True,
        use_container_width=True,
    )
    for filename in (
        "confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "feature_importance.png",
    ):
        path = METRICS_DIR / filename
        if path.exists():
            st.image(str(path), caption=filename.replace("_", " ").title(), width="stretch")
