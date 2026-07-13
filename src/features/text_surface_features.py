"""Text size, readability, and structure features for MDA disclosures."""

from __future__ import annotations

import re
import string
from typing import Any

import configs.settings as settings
import numpy as np
import pandas as pd

TEXT_COLUMN = "mda"
LONG_SENTENCE_WORD_COUNT = 25
COMPLEX_WORD_SYLLABLE_COUNT = 3

WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_PATTERN = re.compile(r"[.!?]+")
VOWEL_GROUP_PATTERN = re.compile(r"[aeiouy]+", re.IGNORECASE)


def add_text_surface_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with text-derived features appended."""
    if TEXT_COLUMN not in dataframe.columns:
        raise ValueError(f"Missing required text column: {TEXT_COLUMN}")

    feature_rows = [
        extract_text_surface_features(text) for text in dataframe[TEXT_COLUMN].tolist()
    ]
    feature_dataframe = pd.DataFrame(
        feature_rows,
        columns=settings.TEXT_SURFACE_FEATURE_COLUMNS,
    )
    return pd.concat([dataframe.reset_index(drop=True), feature_dataframe], axis=1)


def extract_text_surface_features(text: Any) -> dict[str, float]:
    """Calculate interpretable text features for one disclosure."""
    if not _has_text(text):
        return _empty_text_features()

    normalized_text = text.strip()
    words = WORD_PATTERN.findall(normalized_text)
    sentences = _split_sentences(normalized_text)
    if not words or not sentences:
        return _empty_text_features()

    word_count = len(words)
    sentence_count = len(sentences)
    syllable_counts = [_count_syllables(word) for word in words]
    sentence_word_counts = [
        len(WORD_PATTERN.findall(sentence)) for sentence in sentences
    ]
    complex_word_count = sum(
        syllables >= COMPLEX_WORD_SYLLABLE_COUNT for syllables in syllable_counts
    )
    character_count = len(normalized_text)
    average_sentence_words = word_count / sentence_count
    average_syllables_per_word = sum(syllable_counts) / word_count
    complex_word_ratio = complex_word_count / word_count

    return {
        "mda_char_count": float(character_count),
        "mda_word_count": float(word_count),
        "mda_sentence_count": float(sentence_count),
        "mda_avg_sentence_words": average_sentence_words,
        "mda_log_word_count": float(np.log1p(word_count)),
        "mda_flesch_reading_ease": (
            206.835
            - (1.015 * average_sentence_words)
            - (84.6 * average_syllables_per_word)
        ),
        "mda_flesch_kincaid_grade": (
            (0.39 * average_sentence_words)
            + (11.8 * average_syllables_per_word)
            - 15.59
        ),
        "mda_gunning_fog": 0.4 * (average_sentence_words + (100 * complex_word_ratio)),
        "mda_complex_word_ratio": complex_word_ratio,
        "mda_digit_ratio": _character_ratio(normalized_text, str.isdigit),
        "mda_punctuation_ratio": _character_ratio(
            normalized_text,
            lambda character: character in string.punctuation,
        ),
        "mda_uppercase_ratio": _uppercase_ratio(normalized_text),
        "mda_avg_word_length": sum(map(len, words)) / word_count,
        "mda_long_sentence_ratio": (
            sum(
                sentence_word_count >= LONG_SENTENCE_WORD_COUNT
                for sentence_word_count in sentence_word_counts
            )
            / sentence_count
        ),
        "mda_lexical_diversity": len({word.lower() for word in words}) / word_count,
        "mda_text_available": 1.0,
    }


def _has_text(text: Any) -> bool:
    return isinstance(text, str) and bool(text.strip())


def _split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in SENTENCE_PATTERN.split(text)
        if sentence.strip()
    ]


def _count_syllables(word: str) -> int:
    vowel_groups = VOWEL_GROUP_PATTERN.findall(word.lower())
    syllable_count = len(vowel_groups)
    if word.lower().endswith("e") and syllable_count > 1:
        syllable_count -= 1
    return max(syllable_count, 1)


def _character_ratio(text: str, predicate: Any) -> float:
    return sum(predicate(character) for character in text) / len(text)


def _uppercase_ratio(text: str) -> float:
    alphabetic_characters = [character for character in text if character.isalpha()]
    if not alphabetic_characters:
        return 0.0
    return sum(character.isupper() for character in alphabetic_characters) / len(
        alphabetic_characters
    )


def _empty_text_features() -> dict[str, float]:
    features = {
        feature_name: float("nan")
        for feature_name in settings.TEXT_SURFACE_FEATURE_COLUMNS
    }
    features["mda_text_available"] = 0.0
    return features
