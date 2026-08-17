"""Atomic one-shot artifact persistence for a newly preregistered holdout."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.utils.fingerprints import sha256_file


class OneShotEvaluationWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.started_path = output_dir / "evaluation_started.json"
        self.complete_path = output_dir / "evaluation_complete.json"

    def begin(self, metadata: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.started_path.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, sort_keys=True, indent=2)

    def commit_text_files(self, payloads: dict[str, str]) -> dict[str, Any]:
        if not self.started_path.exists():
            raise RuntimeError("begin() must be called before committing artifacts")
        temporary = Path(
            tempfile.mkdtemp(prefix="evaluation-", dir=self.output_dir.parent)
        )
        hashes: dict[str, str] = {}
        for name, content in payloads.items():
            temporary_path = temporary / name
            temporary_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.write_text(content, encoding="utf-8")
            hashes[name] = sha256_file(temporary_path)
        for name in payloads:
            destination = self.output_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary / name, destination)
        manifest = {"status": "complete", "sha256": hashes}
        with self.complete_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2)
        return manifest
