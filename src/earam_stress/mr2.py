from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path


def _load_ordered_items(path: str | Path) -> list[tuple[str, dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return sorted(data.items(), key=lambda pair: int(pair[0]))


def ascii_suffix_start(items: list[tuple[str, dict]]) -> int:
    """Return the first index of the longest all-ASCII caption suffix."""
    start = len(items)
    for index in range(len(items) - 1, -1, -1):
        caption = str(items[index][1].get("caption", ""))
        if not caption.isascii():
            break
        start = index
    if start == len(items):
        raise ValueError("No non-empty all-ASCII suffix was found")
    return start


def select_earam_subset(path: str | Path) -> tuple[list[tuple[str, dict]], dict]:
    """Reconstruct EARAM's English binary MR2 subset from an official split."""
    items = _load_ordered_items(path)
    start = ascii_suffix_start(items)
    suffix = items[start:]
    selected = [(key, item) for key, item in suffix if int(item["label"]) in {0, 1}]
    return selected, {
        "source_records": len(items),
        "ascii_suffix_first_key": items[start][0],
        "ascii_suffix_records": len(suffix),
        "excluded_unverified": len(suffix) - len(selected),
        "selected_binary_records": len(selected),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _extract_selected_images(archive: Path, output_dir: Path, members: set[str]) -> int:
    extracted = 0
    with tarfile.open(archive, mode="r|gz") as bundle:
        for member in bundle:
            normalized = member.name.removeprefix("./")
            if normalized not in members:
                continue
            if not member.isfile():
                raise ValueError(f"Expected a regular image file, got {member.name}")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"Could not read {member.name} from {archive}")
            target = output_dir / normalized
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            extracted += 1
    missing = len(members) - extracted
    if missing:
        raise ValueError(f"Archive is missing {missing} of {len(members)} selected images")
    return extracted


def prepare_mr2(
    train_json: str | Path,
    validation_json: str | Path,
    output_dir: str | Path,
    archive: str | Path | None = None,
) -> dict:
    """Create the 2,558/319 line-aligned subset used by EARAM."""
    output = Path(output_dir)
    train, train_stats = select_earam_subset(train_json)
    test, test_stats = select_earam_subset(validation_json)

    expected = {"train": 2558, "test": 319}
    actual = {"train": len(train), "test": len(test)}
    if actual != expected:
        raise ValueError(f"Unexpected EARAM subset sizes: expected {expected}, got {actual}")

    split_payloads: dict[str, dict] = {}
    manifest: list[dict] = []
    image_members: set[str] = set()
    for output_split, source_split, records in (
        ("train", "train", train),
        ("test", "val", test),
    ):
        payload = {}
        captions = []
        for row_index, (source_key, item) in enumerate(records):
            payload[str(row_index)] = item
            captions.append(str(item["caption"]))
            image_members.add(str(item["image_path"]))
            manifest.append(
                {
                    "output_split": output_split,
                    "row_index": row_index,
                    "source_split": source_split,
                    "source_key": source_key,
                    "image_path": item["image_path"],
                    "label": int(item["label"]),
                }
            )
        split_payloads[output_split] = payload
        _write_json(output / "dataset_merge" / f"en_{output_split}.json", payload)
        (output / f"en_{output_split}_captions.txt").write_text(
            "\n".join(captions) + "\n", encoding="utf-8"
        )

    _write_json(output / "subset_manifest.json", manifest)
    report = {
        "provenance": {
            "rule": "longest all-ASCII caption suffix, excluding MR2 label 2 (unverified)",
            "earam_train_source": "MR2 train",
            "earam_test_source": "MR2 validation",
        },
        "train": train_stats,
        "test": test_stats,
        "labels": {"0": "non-rumor", "1": "rumor", "excluded_2": "unverified"},
        "images_requested": len(image_members),
        "images_extracted": 0,
    }
    if archive:
        report["images_extracted"] = _extract_selected_images(
            Path(archive), output / "dataset_merge", image_members
        )
    _write_json(output / "preparation_report.json", report)
    return report
