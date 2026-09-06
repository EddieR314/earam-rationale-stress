#!/usr/bin/env python3
"""Replace machine-specific paths in an archive and refresh linked hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_record(path: Path) -> dict[str, int | str]:
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def replace_strings(value, source: str, replacement: str):
    if isinstance(value, str):
        return value.replace(source, replacement)
    if isinstance(value, list):
        return [replace_strings(item, source, replacement) for item in value]
    if isinstance(value, dict):
        return {key: replace_strings(item, source, replacement) for key, item in value.items()}
    return value


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--replacement", default="<PROJECT_ROOT>")
    args = parser.parse_args()
    root = args.archive_dir

    for path in root.glob("*.json"):
        write(path, replace_strings(load(path), args.source_root, args.replacement))

    for shuffle_seed in (13, 42, 97):
        cache_path = root / f"cache-shuffle{shuffle_seed}-manifest.json"
        cache_record = file_record(cache_path)
        for model_seed in (13, 42, 97):
            prefix = f"shuffle{shuffle_seed}-model{model_seed}"
            manifest_path = root / f"{prefix}-run-manifest.json"
            manifest = load(manifest_path)
            manifest["inputs"]["feature_manifest"].update(cache_record)
            manifest["artifacts"]["result"].update(
                file_record(root / f"{prefix}-result.json")
            )
            manifest["artifacts"]["test_predictions"].update(
                file_record(root / f"{prefix}-predictions.jsonl")
            )
            write(manifest_path, manifest)

    for model_seed in (13, 42, 97):
        prefix = f"clean-model{model_seed}"
        manifest_path = root / f"{prefix}-run-manifest.json"
        manifest = load(manifest_path)
        manifest["artifacts"]["result"].update(file_record(root / f"{prefix}-result.json"))
        manifest["artifacts"]["test_predictions"].update(
            file_record(root / f"{prefix}-predictions.jsonl")
        )
        write(manifest_path, manifest)

    print(json.dumps({"status": "sanitized", "replacement": args.replacement}, indent=2))


if __name__ == "__main__":
    main()
