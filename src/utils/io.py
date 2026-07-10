"""
Reusable file I/O utilities.

This module centralizes reading and writing of common file formats
used throughout the project.

Supported formats
-----------------
- JSON
- CSV
- Parquet
- Markdown
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# JSON
# =============================================================================


def load_json(path: Path | str) -> Any:
    """
    Load a JSON file.

    Parameters
    ----------
    path : Path | str

    Returns
    -------
    Any
        Parsed JSON object.
    """
    path = Path(path)

    logger.info(f"Loading JSON: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path | str, indent: int = 4) -> None:
    """
    Save data as JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    logger.info(f"Saved JSON: {path}")


# =============================================================================
# CSV
# =============================================================================


def load_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file.
    """
    path = Path(path)

    logger.info(f"Loading CSV: {path}")

    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: Path | str, **kwargs) -> None:
    """
    Save DataFrame to CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False, **kwargs)

    logger.info(f"Saved CSV: {path}")


# =============================================================================
# Parquet
# =============================================================================


def load_parquet(path: Path | str) -> pd.DataFrame:
    """
    Load a Parquet file.
    """
    path = Path(path)

    logger.info(f"Loading Parquet: {path}")

    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: Path | str) -> None:
    """
    Save DataFrame to Parquet.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(path, index=False)

    logger.info(f"Saved Parquet: {path}")


# =============================================================================
# Markdown
# =============================================================================


def save_markdown(text: str, path: Path | str) -> None:
    """
    Save Markdown text.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    logger.info(f"Saved Markdown: {path}")
