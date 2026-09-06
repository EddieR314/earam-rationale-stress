from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from pathlib import Path

from .io import read_lines, write_lines
from .provenance import sha256_file


SHUFFLE_MODES = ("within-label", "cross-sample")


def _derangement(indices: list[int], rng: random.Random) -> dict[int, int]:
    if len(indices) < 2:
        raise ValueError("Each shuffle group must contain at least two records")
    donors = list(indices)
    while True:
        rng.shuffle(donors)
        if all(recipient != donor for recipient, donor in zip(indices, donors)):
            return dict(zip(indices, donors))


def shuffle_rationale_pairs(records: list[dict], seed: int, mode: str = "within-label") -> list[dict]:
    """Move both rationales together using a deterministic, fixed-point-free permutation."""
    if mode not in SHUFFLE_MODES:
        raise ValueError(f"Unknown shuffle mode: {mode}")
    if len(records) < 2:
        raise ValueError("At least two records are required")

    groups: dict[object, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if "rationale_1" not in record or "rationale_2" not in record:
            raise ValueError("Every record must contain rationale_1 and rationale_2")
        key = record.get("label") if mode == "within-label" else "all"
        if mode == "within-label" and key is None:
            raise ValueError("within-label shuffle requires a label on every record")
        groups[key].append(index)

    rng = random.Random(seed)
    donors: dict[int, int] = {}
    for key in sorted(groups, key=str):
        try:
            donors.update(_derangement(groups[key], rng))
        except ValueError as exc:
            raise ValueError(f"Shuffle group {key!r} has fewer than two records") from exc

    output = copy.deepcopy(records)
    for recipient_index, donor_index in donors.items():
        donor = records[donor_index]
        recipient = output[recipient_index]
        recipient["rationale_1"] = donor["rationale_1"]
        recipient["rationale_2"] = donor["rationale_2"]
        recipient["rationale_shuffle"] = {
            "mode": mode,
            "seed": seed,
            "donor_id": str(donor.get("id", donor_index)),
        }
    return output


def build_rationale_shuffle_condition(
    dataset_json: str | Path,
    analysis_1: str | Path,
    analysis_2: str | Path,
    output_dir: str | Path,
    seed: int,
    mode: str = "within-label",
) -> dict:
    dataset_path = Path(dataset_json)
    first_path = Path(analysis_1)
    second_path = Path(analysis_2)
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = list(data.items())
    first = read_lines(first_path)
    second = read_lines(second_path)
    lengths = {"dataset": len(items), "analysis_1": len(first), "analysis_2": len(second)}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Dataset and rationales are not aligned: {lengths}")

    records = [
        {
            "id": str(source_id),
            "label": int(item["label"]),
            "rationale_1": first[index],
            "rationale_2": second[index],
        }
        for index, (source_id, item) in enumerate(items)
    ]
    shuffled = shuffle_rationale_pairs(records, seed=seed, mode=mode)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    output_first = root / "analysis_1.txt"
    output_second = root / "analysis_2.txt"
    write_lines(output_first, (record["rationale_1"] for record in shuffled))
    write_lines(output_second, (record["rationale_2"] for record in shuffled))

    mappings = [
        {
            "recipient_id": record["id"],
            "donor_id": record["rationale_shuffle"]["donor_id"],
            "label": record["label"],
        }
        for record in shuffled
    ]
    report = {
        "scope": "controlled rationale-pair shuffle condition",
        "mode": mode,
        "seed": seed,
        "records": len(records),
        "fixed_points": sum(row["recipient_id"] == row["donor_id"] for row in mappings),
        "label_counts": {
            str(label): sum(record["label"] == label for record in records)
            for label in sorted({record["label"] for record in records})
        },
        "inputs": {
            "dataset_json": {"path": str(dataset_path.resolve()), "sha256": sha256_file(dataset_path)},
            "analysis_1": {"path": str(first_path.resolve()), "sha256": sha256_file(first_path)},
            "analysis_2": {"path": str(second_path.resolve()), "sha256": sha256_file(second_path)},
        },
        "outputs": {
            "analysis_1": str(output_first.resolve()),
            "analysis_2": str(output_second.resolve()),
        },
        "mappings": mappings,
    }
    (root / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
