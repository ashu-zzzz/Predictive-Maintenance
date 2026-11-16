import pandas as pd
import pytest
from src.config import FEATURES, TARGET
from src.data_loader import load_dataset, select_features
from src.predict import validate_inputs
from src.preprocessing import build_preprocessor

def test_dataset_and_no_target_leakage():
    features, target = select_features(load_dataset())
    assert list(features.columns) == FEATURES
    assert TARGET not in features.columns
    assert len(features) == len(target)

def test_preprocessor_transforms_data():
    features, _ = select_features(load_dataset().head(10))
    transformed = build_preprocessor().fit_transform(features)
    assert transformed.shape[0] == 10
    assert transformed.shape[1] >= 7

def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="Missing required"):
        validate_inputs(pd.DataFrame({"Type": ["M"]}))
    invalid = {feature: [1] for feature in FEATURES}
    invalid["Type"] = ["X"]
    with pytest.raises(ValueError, match="Type"):
        validate_inputs(pd.DataFrame(invalid))
