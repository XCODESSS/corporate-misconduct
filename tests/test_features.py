"""Tests for text-surface feature engineering."""

import math

import configs.settings as settings
import pandas as pd
from src.features.text_surface_features import (
    add_text_surface_features,
    extract_text_surface_features,
)


def test_extract_text_surface_features_calculates_expected_values() -> None:
    features = extract_text_surface_features(
        "This is a simple sentence. Another sentence contains 123 numbers!"
    )

    assert features["mda_word_count"] == 9.0
    assert features["mda_sentence_count"] == 2.0
    assert features["mda_text_available"] == 1.0
    assert features["mda_digit_ratio"] > 0.0
    assert features["mda_lexical_diversity"] > 0.0


def test_extract_text_surface_features_marks_missing_text() -> None:
    features = extract_text_surface_features(None)

    assert features["mda_text_available"] == 0.0
    assert math.isnan(features["mda_word_count"])


def test_add_text_surface_features_preserves_rows_and_schema() -> None:
    dataframe = pd.DataFrame({"mda": ["A short disclosure.", ""]})

    enriched_dataframe = add_text_surface_features(dataframe)

    assert len(enriched_dataframe) == len(dataframe)
    assert set(settings.TEXT_SURFACE_FEATURE_COLUMNS).issubset(enriched_dataframe)
    assert enriched_dataframe.loc[1, "mda_text_available"] == 0.0
