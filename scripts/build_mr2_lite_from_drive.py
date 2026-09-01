#!/usr/bin/env python3
"""Stream an official MR2 archive and retain only EARAM's 2,558 training images."""

from __future__ import annotations

import argparse
import http.cookiejar
import html
import json
import re
import shutil
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


FILE_ID = "14NNqLKSW1FzLGuGkqwlzyIPXnKDzEFX4"


def open_archive():
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    url = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download"
    for _ in range(5):
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = opener.open(request, timeout=120)
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return response
        body = response.read().decode("utf-8", errors="replace")
        response.close()
        if "Quota exceeded" in body or "download quota" in body.lower():
            raise RuntimeError(
                "The official MR2 Google Drive download quota is currently exceeded. "
                "Retry later, or download the archive from the official Baidu AI Studio "
                "dataset page and pass its path with --archive."
            )
        uuid_match = re.search(r'name="uuid" value="([^"]+)"', body)
        if not uuid_match:
            raise RuntimeError(
                "Google Drive returned an HTML error page without a download confirmation token"
            )
        query = urllib.parse.urlencode(
            {
                "id": FILE_ID,
                "export": "download",
                "confirm": "t",
                "uuid": html.unescape(uuid_match.group(1)),
            }
        )
        url = f"https://drive.usercontent.google.com/download?{query}"
    raise RuntimeError("Google Drive confirmation loop did not reach the archive")


def select_train(metadata: dict) -> dict:
    items = sorted(metadata.items(), key=lambda pair: int(pair[0]))
    suffix_start = len(items)
    for index in range(len(items) - 1, -1, -1):
        if not str(items[index][1].get("caption", "")).isascii():
            break
        suffix_start = index
    selected = [(key, item) for key, item in items[suffix_start:] if int(item["label"]) in {0, 1}]
    if len(selected) != 2558:
        raise ValueError(f"Expected 2,558 EARAM training records, got {len(selected)}")
    return {str(index): item for index, (_, item) in enumerate(selected)}


def write_metadata(output: Path, selected: dict) -> set[str]:
    dataset_root = output / "dataset_merge"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "en_train.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "en_train_captions.txt").write_text(
        "\n".join(
            str(item["caption"]).replace("\r", " ").replace("\n", " ")
            for item in selected.values()
        )
        + "\n",
        encoding="utf-8",
    )
    return {str(item["image_path"]).removeprefix("./") for item in selected.values()}


def extract_tar(stream, output: Path) -> tuple[dict, int]:
    dataset_root = output / "dataset_merge"
    selected = None
    targets: set[str] = set()
    extracted = 0
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if name == "dataset_items_train.json":
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError("Could not read dataset_items_train.json")
                selected = select_train(json.load(source))
                targets = write_metadata(output, selected)
                print(f"Selected {len(selected)} English binary records.", flush=True)
                continue
            if name not in targets:
                continue
            if not member.isfile():
                raise RuntimeError(f"Expected regular file for {name}")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read {name}")
            target = dataset_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted += 1
            if extracted % 100 == 0 or extracted == len(targets):
                print(f"extracted {extracted}/{len(targets)}", flush=True)
            if extracted == len(targets):
                break
    if selected is None:
        raise RuntimeError("Training metadata was not found in the archive")
    if extracted != len(targets):
        raise RuntimeError(f"Only extracted {extracted}/{len(targets)} selected images")
    return selected, extracted


def extract_zip(path: Path, output: Path) -> tuple[dict, int]:
    dataset_root = output / "dataset_merge"
    with zipfile.ZipFile(path) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
        metadata_files = [
            item
            for item in files
            if item.filename.removeprefix("./").endswith("dataset_items_train.json")
        ]
        if not metadata_files:
            raise RuntimeError("dataset_items_train.json was not found in the ZIP archive")
        metadata_file = min(metadata_files, key=lambda item: len(item.filename))
        with archive.open(metadata_file) as source:
            selected = select_train(json.load(source))
        targets = write_metadata(output, selected)
        print(f"Selected {len(selected)} English binary records.", flush=True)

        by_name = {item.filename.removeprefix("./"): item for item in files}
        extracted = 0
        for name in sorted(targets):
            member = by_name.get(name)
            if member is None:
                matches = [
                    item
                    for normalized, item in by_name.items()
                    if normalized.endswith("/" + name)
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected one ZIP member for {name}, found {len(matches)}"
                    )
                member = matches[0]
            target = dataset_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted += 1
            if extracted % 100 == 0 or extracted == len(targets):
                print(f"extracted {extracted}/{len(targets)}", flush=True)
    return selected, extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--archive",
        help="Optional local MR2 .tar.gz or data.zip. If omitted, stream Google Drive.",
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    (output / "dataset_merge").mkdir(parents=True, exist_ok=True)
    source_name = (
        str(Path(args.archive).resolve())
        if args.archive
        else "official MR2 Google Drive archive"
    )
    if args.archive and not Path(args.archive).is_file():
        raise FileNotFoundError(f"MR2 archive not found: {args.archive}")
    print(f"Reading {source_name}; only selected images will be stored.", flush=True)
    archive_format = (
        "zip"
        if args.archive and Path(args.archive).suffix.lower() == ".zip"
        else "tar.gz"
    )
    if archive_format == "zip":
        selected, extracted = extract_zip(Path(args.archive), output)
    else:
        stream = Path(args.archive).open("rb") if args.archive else open_archive()
        with stream as response:
            content_type = (
                response.headers.get("Content-Type", "")
                if hasattr(response, "headers")
                else ""
            )
            if "text/html" in content_type:
                raise RuntimeError("Google Drive returned HTML instead of the MR2 archive")
            selected, extracted = extract_tar(response, output)
    report = {
        "source": source_name,
        "records": len(selected),
        "images": extracted,
        "caption_lines": len(selected),
        "archive_format": archive_format,
        "stored_full_archive": False,
    }
    (output / "lite_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
