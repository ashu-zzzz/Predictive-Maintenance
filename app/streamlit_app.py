import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import METRICS_DIR
from src.predict import load_artifacts, predict_frame

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")
st.title("Predictive Maintenance & Machine Failure Detection")
st.caption("Educational prototype using the UCI AI4I 2020 synthetic dataset.")
try:
    model, metadata = load_artifacts()
except FileNotFoundError as error:
    st.warning(str(error))
    st.stop()

st.sidebar.metric("Selected model", metadata["selected_model"].replace("_", " ").title())
st.sidebar.metric("Alert threshold", f"{metadata['threshold']:.3f}")
overview, single, batch, performance = st.tabs(
    ["Overview", "Single prediction", "Batch prediction", "Performance"])
with overview:
    st.write("The model estimates failure risk from machine operating conditions. Its score is not a guaranteed real-world probability.")
    cols = st.columns(3)
    cols[0].metric("Dataset rows", f"{metadata['dataset_summary']['rows']:,}")
    cols[1].metric("Failure rate", f"{metadata['dataset_summary']['failure_rate']:.2%}")
    cols[2].metric("Test PR-AUC", f"{metadata['test']['average_precision']:.3f}")
with single:
    with st.form("prediction"):
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
        row = predict_frame(pd.DataFrame([values]), model, metadata["threshold"]).iloc[0]
        st.metric("Model risk score", f"{row['failure_risk_score']:.2%}")
        (st.error if row["predicted_failure"] else st.success)(
            "Alert: threshold crossed." if row["predicted_failure"] else "No alert: below threshold.")
with batch:
    upload = st.file_uploader("Upload CSV with the six required input columns", type="csv")
    if upload:
        try:
            result = predict_frame(pd.read_csv(upload), model, metadata["threshold"])
            st.dataframe(result, use_container_width=True)
            st.download_button("Download predictions", result.to_csv(index=False),
                               "predictions.csv", "text/csv")
        except ValueError as error:
            st.error(str(error))
with performance:
    st.dataframe(pd.DataFrame([metadata["test"]]))
    for name in ("confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png"):
        path = METRICS_DIR / name
        if path.exists():
            st.image(str(path), caption=name.replace("_", " ").title())
