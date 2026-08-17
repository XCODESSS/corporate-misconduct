"""Path resolution and streaming access for raw FinNLP files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import configs.settings as settings
import ijson


@dataclass(frozen=True, slots=True)
class FinNLPDatasetPaths:
    firm_years: Path
    labels: Path
    aaer: Path


class FinNLPDatasetLoader:
    def __init__(self, firm_years=None, labels=None, aaer=None) -> None:
        self._paths = FinNLPDatasetPaths(
            Path(firm_years or settings.FIRM_YEARS_FILE),
            Path(labels or settings.LABELS_FILE),
            Path(aaer or settings.AAER_FILE),
        )

    def get_dataset_paths(self) -> FinNLPDatasetPaths:
        return self._paths

    def stream_firm_years(self) -> Iterator[dict[str, Any]]:
        with self._paths.firm_years.open("rb") as stream:
            yield from ijson.items(stream, "item")
