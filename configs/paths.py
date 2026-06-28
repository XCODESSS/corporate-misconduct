from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
DATASETS_DIR = PROCESSED_DIR / "datasets"
EXTERNAL_DIR = DATA_DIR / "external"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
HYPOTHESES_DIR = REPORTS_DIR / "hypotheses"
