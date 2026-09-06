#!/usr/bin/env python3
"""Validate the published within-label result archive from raw artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from earam_stress.io import read_jsonl
from earam_stress.metrics import paired_bootstrap_macro_f1


SHUFFLE_SEEDS = (13, 42, 97)
MODEL_SEEDS = (13, 42, 97)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_artifact(record: dict, path: Path) -> None:
    require(path.stat().st_size == int(record["bytes"]), f"Byte count mismatch: {path}")
    require(sha256(path) == record["sha256"], f"SHA-256 mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.archive_dir
    commit = (root / "git-commit.txt").read_text(encoding="utf-8-sig").strip()

    for shuffle_seed in SHUFFLE_SEEDS:
        rationale = load(root / f"rationale-shuffle{shuffle_seed}-manifest.json")
        mappings = rationale["mappings"]
        require(rationale["mode"] == "within-label", "Unexpected shuffle mode")
        require(rationale["records"] == 2558, "Unexpected rationale record count")
        require(rationale["fixed_points"] == 0, "Rationale manifest has fixed points")
        require(len(mappings) == 2558, "Incomplete donor mapping")
        require(len({row["recipient_id"] for row in mappings}) == 2558, "Duplicate recipient")
        require(len({row["donor_id"] for row in mappings}) == 2558, "Duplicate donor")
        require(
            all(row["recipient_id"] != row["donor_id"] for row in mappings),
            "Self-assignment in donor mapping",
        )

        cache_path = root / f"cache-shuffle{shuffle_seed}-manifest.json"
        cache = load(cache_path)
        require(cache["code_commit"] == commit, "Cache code revision mismatch")
        require(cache["records"] == cache["completed"] == 2558, "Incomplete feature cache")

        for model_seed in MODEL_SEEDS:
            prefix = f"shuffle{shuffle_seed}-model{model_seed}"
            run_manifest = load(root / f"{prefix}-run-manifest.json")
            result_path = root / f"{prefix}-result.json"
            prediction_path = root / f"{prefix}-predictions.jsonl"
            bootstrap_path = root / f"{prefix}-bootstrap.json"

            require(
                run_manifest["code"]["earam_rationale_stress_commit"] == commit,
                "Evaluation code revision mismatch",
            )
            require(
                run_manifest["inputs"]["feature_manifest"]["sha256"] == sha256(cache_path),
                "Feature manifest hash mismatch",
            )
            validate_artifact(run_manifest["artifacts"]["result"], result_path)
            validate_artifact(run_manifest["artifacts"]["test_predictions"], prediction_path)
            require(load(result_path)["inputs"]["missing_features"] == 0, "Missing features")

            clean_predictions = read_jsonl(root / f"clean-model{model_seed}-predictions.jsonl")
            condition_predictions = read_jsonl(prediction_path)
            observed = paired_bootstrap_macro_f1(
                clean_predictions, condition_predictions, samples=2000, seed=42
            )
            require(observed == load(bootstrap_path), f"Bootstrap mismatch: {prefix}")

    for model_seed in MODEL_SEEDS:
        prefix = f"clean-model{model_seed}"
        manifest = load(root / f"{prefix}-run-manifest.json")
        validate_artifact(manifest["artifacts"]["result"], root / f"{prefix}-result.json")
        validate_artifact(
            manifest["artifacts"]["test_predictions"], root / f"{prefix}-predictions.jsonl"
        )

    print(
        json.dumps(
            {
                "status": "valid",
                "code_commit": commit,
                "rationale_manifests": 3,
                "evaluations_recomputed": 9,
                "test_records_per_evaluation": 255,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
