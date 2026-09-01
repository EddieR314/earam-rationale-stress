from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .io import read_lines, write_lines


def _aligned_records(dataset_json: str | Path, analysis_1: str | Path, analysis_2: str | Path) -> list[dict]:
    data = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
    first = read_lines(analysis_1)
    second = read_lines(analysis_2)
    items = list(data.items())
    lengths = {"dataset": len(items), "analysis_1": len(first), "analysis_2": len(second)}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Dataset and rationales are not aligned: {lengths}")
    return [
        {
            "source_id": key,
            "item": item,
            "rationale_1": first[index],
            "rationale_2": second[index],
        }
        for index, (key, item) in enumerate(items)
    ]


def stratified_split(records: list[dict], seed: int) -> dict[str, list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        label = int(record["item"]["label"])
        if label not in {0, 1}:
            raise ValueError(f"Expected binary labels 0/1, got {label}")
        groups[label].append(record)

    rng = random.Random(seed)
    output = {"train": [], "validation": [], "test": []}
    for label in sorted(groups):
        group = list(groups[label])
        rng.shuffle(group)
        train_end = round(len(group) * 0.8)
        validation_end = train_end + round(len(group) * 0.1)
        output["train"].extend(group[:train_end])
        output["validation"].extend(group[train_end:validation_end])
        output["test"].extend(group[validation_end:])
    for split in output.values():
        rng.shuffle(split)
    return output


def _write_split(root: Path, name: str, records: list[dict]) -> dict:
    dataset = {str(index): record["item"] for index, record in enumerate(records)}
    (root / "dataset").mkdir(parents=True, exist_ok=True)
    (root / "rationales").mkdir(parents=True, exist_ok=True)
    (root / "dataset" / f"{name}.json").write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_lines(root / "rationales" / f"{name}_analysis_1.txt", (r["rationale_1"] for r in records))
    write_lines(root / "rationales" / f"{name}_analysis_2.txt", (r["rationale_2"] for r in records))
    return {
        "records": len(records),
        "labels": {
            str(label): sum(int(record["item"]["label"]) == label for record in records)
            for label in (0, 1)
        },
        "source_ids": [record["source_id"] for record in records],
    }


def make_splits(
    dataset_json: str | Path,
    analysis_1: str | Path,
    analysis_2: str | Path,
    output_dir: str | Path,
    seeds: list[int],
) -> dict:
    records = _aligned_records(dataset_json, analysis_1, analysis_2)
    report = {
        "scope": "internal EARAM-style split; not the paper's official MR2 test split",
        "source_records": len(records),
        "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "seeds": {},
    }
    for seed in seeds:
        root = Path(output_dir) / f"seed{seed}"
        split = stratified_split(records, seed)
        seed_report = {name: _write_split(root, name, rows) for name, rows in split.items()}
        all_ids = [source_id for value in seed_report.values() for source_id in value["source_ids"]]
        if len(all_ids) != len(set(all_ids)) or len(all_ids) != len(records):
            raise ValueError(f"Seed {seed} produced overlapping or missing source IDs")
        (root / "split_manifest.json").write_text(
            json.dumps(seed_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report["seeds"][str(seed)] = {
            name: {"records": value["records"], "labels": value["labels"]}
            for name, value in seed_report.items()
        }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def validate_split_dir(path: str | Path) -> dict:
    root = Path(path)
    report = {}
    source_sets = []
    manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    for name in ("train", "validation", "test"):
        data = json.loads((root / "dataset" / f"{name}.json").read_text(encoding="utf-8"))
        first = read_lines(root / "rationales" / f"{name}_analysis_1.txt")
        second = read_lines(root / "rationales" / f"{name}_analysis_2.txt")
        expected = int(manifest[name]["records"])
        if not len(data) == len(first) == len(second) == expected:
            raise ValueError(f"Misaligned {name} split")
        source_sets.append(set(manifest[name]["source_ids"]))
        report[name] = {"records": expected, "aligned": True}
    if any(source_sets[i] & source_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Split source IDs overlap")
    report["disjoint"] = True
    return report
