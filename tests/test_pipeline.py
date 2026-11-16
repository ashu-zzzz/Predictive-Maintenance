import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES, TARGET
from src.data_loader import select_features
from src.evaluate import metrics_at_threshold
from src.predict import predict_frame, validate_inputs
from src.preprocessing import build_preprocessor
from src.train import build_models


def make_frame(rows=60):
    frame = pd.DataFrame({
        "Type": (["H", "M", "L"] * ((rows + 2) // 3))[:rows],
        "Air temperature [K]": np.linspace(295, 305, rows),
        "Process temperature [K]": np.linspace(305, 315, rows),
        "Rotational speed [rpm]": np.linspace(1200, 1800, rows),
        "Torque [Nm]": np.linspace(20, 60, rows),
        "Tool wear [min]": np.linspace(0, 200, rows),
        TARGET: ([0, 1] * ((rows + 1) // 2))[:rows],
        "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
        "UDI": range(rows), "Product ID": [f"P{i}" for i in range(rows)],
    })
    return frame


def test_feature_selection_prevents_target_and_identifier_leakage():
    features, target = select_features(make_frame())
    assert list(features.columns) == FEATURES
    assert TARGET not in features
    assert "UDI" not in features
    assert "TWF" not in features
    assert len(target) == len(features)


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="Missing"):
        validate_inputs(pd.DataFrame({"Type": ["M"]}))


def test_preprocessing_encodes_categories_and_scales_numeric_features():
    inputs = make_frame()[FEATURES]
    transformed = build_preprocessor().fit_transform(inputs)
    assert transformed.shape == (60, 8)


@pytest.mark.parametrize("bad_value", ["not-a-number", np.nan, np.inf, -np.inf])
def test_invalid_numeric_inputs_are_rejected(bad_value):
    inputs = make_frame(1)[FEATURES].copy()
    inputs["Torque [Nm]"] = inputs["Torque [Nm]"].astype(object)
    inputs.loc[0, "Torque [Nm]"] = bad_value
    with pytest.raises(ValueError, match="finite"):
        validate_inputs(inputs)


def test_invalid_category_and_input_structure_are_rejected():
    inputs = make_frame(1)[FEATURES].copy()
    inputs.loc[0, "Type"] = "X"
    with pytest.raises(ValueError, match="Type"):
        validate_inputs(inputs)
    with pytest.raises(ValueError, match="DataFrame"):
        validate_inputs({"Type": ["M"]})


def test_all_models_train_on_small_synthetic_data():
    inputs = make_frame()[FEATURES]
    target = make_frame()[TARGET]
    for model in build_models().values():
        model.fit(inputs, target)
        assert model.predict_proba(inputs).shape == (60, 2)


def test_prediction_output_contains_only_inputs_and_generated_columns():
    inputs = make_frame(4)
    model = build_models()["logistic_regression"].fit(inputs[FEATURES], inputs[TARGET])
    output = predict_frame(inputs, model, threshold=0.5)
    assert list(output.columns) == FEATURES + ["failure_risk_score", "predicted_failure"]
    assert output["predicted_failure"].isin([0, 1]).all()


def test_prediction_threshold_is_applied():
    inputs = make_frame(2)[FEATURES]
    model = build_models()["logistic_regression"].fit(inputs, [0, 1])
    output = predict_frame(inputs, model, threshold=1.0)
    assert output["predicted_failure"].eq(0).all()


def test_evaluation_metrics_include_confusion_counts():
    metrics = metrics_at_threshold(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]), threshold=0.5
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["true_negatives"] == 2
    assert metrics["true_positives"] == 2
    assert metrics["false_negative_rate"] == 0.0
