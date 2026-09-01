from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_lines(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


def write_lines(path: str | Path, lines: Iterable[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def import_earam(
    analysis_1_path: str | Path,
    analysis_2_path: str | Path,
    captions_path: str | Path | None = None,
) -> list[dict]:
    analysis_1 = read_lines(analysis_1_path)
    analysis_2 = read_lines(analysis_2_path)
    if len(analysis_1) != len(analysis_2):
        raise ValueError(
            f"Analysis files are not aligned: {len(analysis_1)} != {len(analysis_2)}"
        )

    captions = read_lines(captions_path) if captions_path else [""] * len(analysis_1)
    if len(captions) != len(analysis_1):
        raise ValueError(
            f"Caption and analysis files are not aligned: {len(captions)} != {len(analysis_1)}"
        )

    return [
        {
            "id": str(index),
            "caption": captions[index],
            "rationale_1": analysis_1[index],
            "rationale_2": analysis_2[index],
        }
        for index in range(len(analysis_1))
    ]


def export_earam(
    records: list[dict],
    analysis_1_path: str | Path,
    analysis_2_path: str | Path,
) -> None:
    write_lines(analysis_1_path, (record.get("rationale_1", "") for record in records))
    write_lines(analysis_2_path, (record.get("rationale_2", "") for record in records))
