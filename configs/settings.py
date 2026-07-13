"""
Global project configuration.

This file centralizes all project paths so that no script contains
hardcoded directories or filenames.
"""

from pathlib import Path

# =============================================================================
# Project Directories
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

REPORTS_DIR = PROJECT_ROOT / "reports"

# =============================================================================
# Raw Dataset
# =============================================================================

FINNLP_DIR = RAW_DIR / "finnlp_dataset"

FIRM_YEARS_FILE = FINNLP_DIR / "firm_years.json"

LABELS_FILE = FINNLP_DIR / "firm_years_labels.json"

AAER_FILE = FINNLP_DIR / "aaer_mark5.csv"

LM_DIR = RAW_DIR / "lm"

LM_SUMMARIES_FILE = LM_DIR / "Loughran-McDonald_10X_Summaries_1993-2025.csv"

# =============================================================================
# Intermediate Outputs
# =============================================================================

INTERIM_CLEANED_DIR = INTERIM_DIR / "cleaned"

INTERIM_VALIDATED_DIR = INTERIM_DIR / "validated"

# =============================================================================
# Processed Outputs
# =============================================================================

FEATURES_DIR = PROCESSED_DIR / "features"

DATASETS_DIR = PROCESSED_DIR / "datasets"

LM_DENSITY_FEATURE_COLUMNS = (
    "negative_density",
    "positive_density",
    "uncertainty_density",
    "litigious_density",
    "weak_modal_density",
    "strong_modal_density",
    "constraining_density",
)

TEXT_SURFACE_FEATURE_COLUMNS = (
    "mda_char_count",
    "mda_word_count",
    "mda_sentence_count",
    "mda_avg_sentence_words",
    "mda_log_word_count",
    "mda_flesch_reading_ease",
    "mda_flesch_kincaid_grade",
    "mda_gunning_fog",
    "mda_complex_word_ratio",
    "mda_digit_ratio",
    "mda_punctuation_ratio",
    "mda_uppercase_ratio",
    "mda_avg_word_length",
    "mda_long_sentence_ratio",
    "mda_lexical_diversity",
    "mda_text_available",
)

MODEL_FEATURE_COLUMNS = LM_DENSITY_FEATURE_COLUMNS + TEXT_SURFACE_FEATURE_COLUMNS

# =============================================================================
# Reports
# =============================================================================

FIGURES_DIR = REPORTS_DIR / "figures"

TABLES_DIR = REPORTS_DIR / "tables"

HYPOTHESES_DIR = REPORTS_DIR / "hypotheses"

# =============================================================================
# Create Required Directories
# =============================================================================

DIRECTORIES = [
    INTERIM_DIR,
    INTERIM_CLEANED_DIR,
    INTERIM_VALIDATED_DIR,
    FEATURES_DIR,
    DATASETS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)
