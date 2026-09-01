#!/usr/bin/env python3
"""Portable GPU runner for the public EARAM VLR model.

This script imports only `model.VLR` from the official repository. It replaces the released
hard-coded data paths, CLIP checkpoint path, CUDA device, and test-every-epoch loop.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--earam-repo", required=True, help="Clone containing official model.py")
    result.add_argument("--split-dir", required=True, help="One seed directory made by make-splits")
    result.add_argument("--image-root", required=True, help="Root joined with each MR2 image_path")
    result.add_argument("--clip", default="openai/clip-vit-large-patch14")
    result.add_argument("--device", default="auto")
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--epochs", type=int, default=35)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--learning-rate", type=float, default=2e-5)
    result.add_argument("--weight-decay", type=float, default=0.01)
    result.add_argument("--num-workers", type=int, default=4)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--dry-run", action="store_true", help="Validate files without importing ML packages")
    return result


def split_paths(root: Path, split: str) -> dict[str, Path]:
    return {
        "dataset": root / "dataset" / f"{split}.json",
        "analysis_1": root / "rationales" / f"{split}_analysis_1.txt",
        "analysis_2": root / "rationales" / f"{split}_analysis_2.txt",
    }


def validate_inputs(args: argparse.Namespace) -> dict:
    repo = Path(args.earam_repo)
    root = Path(args.split_dir)
    if not (repo / "model.py").is_file():
        raise FileNotFoundError(f"Official EARAM model.py not found under {repo}")
    report = {}
    all_images = []
    for split in ("train", "validation", "test"):
        paths = split_paths(root, split)
        data = json.loads(paths["dataset"].read_text(encoding="utf-8"))
        first = paths["analysis_1"].read_text(encoding="utf-8").splitlines()
        second = paths["analysis_2"].read_text(encoding="utf-8").splitlines()
        if not len(data) == len(first) == len(second):
            raise ValueError(f"Unaligned {split}: {len(data)}, {len(first)}, {len(second)}")
        all_images.extend(Path(args.image_root) / item["image_path"] for item in data.values())
        report[split] = len(data)
    missing = [str(path) for path in all_images if not path.is_file()]
    report["images"] = len(all_images)
    report["missing_images"] = len(missing)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first missing path: {missing[0]}")
    return report


def main() -> None:
    args = parser().parse_args()
    input_report = validate_inputs(args)
    if args.dry_run:
        print(json.dumps({"status": "valid", **input_report}, indent=2))
        return

    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from transformers import CLIPModel, CLIPProcessor

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    repo = str(Path(args.earam_repo).resolve())
    sys.path.insert(0, repo)
    VLR = importlib.import_module("model").VLR

    class DatasetView(Dataset):
        def __init__(self, split: str):
            paths = split_paths(Path(args.split_dir), split)
            self.items = list(json.loads(paths["dataset"].read_text(encoding="utf-8")).items())
            self.first = paths["analysis_1"].read_text(encoding="utf-8").splitlines()
            self.second = paths["analysis_2"].read_text(encoding="utf-8").splitlines()

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            row_id, item = self.items[index]
            with Image.open(Path(args.image_root) / item["image_path"]) as image:
                pixels = image.convert("RGB").copy()
            return row_id, item["caption"], pixels, self.first[index], self.second[index], int(item["label"])

    def collate(batch):
        ids, captions, images, first, second, labels = zip(*batch)
        return list(ids), list(captions), list(images), list(first), list(second), torch.tensor(labels)

    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(
            DatasetView(split),
            batch_size=args.batch_size,
            shuffle=split == "train",
            collate_fn=collate,
            num_workers=args.num_workers,
            generator=generator,
            pin_memory=device.startswith("cuda"),
        )
        for split in ("train", "validation", "test")
    }

    clip = CLIPModel.from_pretrained(args.clip).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip)
    for parameter in clip.parameters():
        parameter.requires_grad_(False)

    def encode(captions, images, first, second):
        with torch.no_grad():
            def text_tokens(values):
                inputs = processor(text=values, return_tensors="pt", padding=True, truncation=True, max_length=77)
                inputs = {key: value.to(device) for key, value in inputs.items()}
                hidden = clip.text_model(**inputs).last_hidden_state
                return clip.text_projection(hidden)

            image_inputs = processor(images=images, return_tensors="pt")
            image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
            visual = clip.vision_model(**image_inputs).last_hidden_state
            return text_tokens(captions), clip.visual_projection(visual), text_tokens(first), text_tokens(second)

    def metrics(labels, predictions):
        accuracy = sum(a == b for a, b in zip(labels, predictions)) / len(labels)
        f1s = []
        for target in (0, 1):
            tp = sum(a == target and b == target for a, b in zip(labels, predictions))
            fp = sum(a != target and b == target for a, b in zip(labels, predictions))
            fn = sum(a == target and b != target for a, b in zip(labels, predictions))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return {"accuracy": accuracy, "macro_f1": sum(f1s) / 2}

    model = VLR(dim=768).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    weights = (0.7, 0.15, 0.15)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best.pt"
    history = []

    def evaluate(split: str, save_predictions: bool = False):
        model.eval()
        labels_all, predictions_all, rows = [], [], []
        with torch.no_grad():
            for ids, captions, images, first, second, labels in loaders[split]:
                features = encode(captions, images, first, second)
                logits = model(*features)["label_1"]
                predictions = logits.argmax(dim=1).cpu().tolist()
                truth = labels.tolist()
                labels_all.extend(truth)
                predictions_all.extend(predictions)
                rows.extend({"id": i, "label": y, "prediction": p} for i, y, p in zip(ids, truth, predictions))
        result = metrics(labels_all, predictions_all)
        if save_predictions:
            (output / f"{split}_predictions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
        return result

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        examples = 0
        for _, captions, images, first, second, labels in loaders["train"]:
            labels = labels.to(device)
            features = encode(captions, images, first, second)
            outputs = model(*features)
            loss = sum(weight * criterion(outputs[f"label_{index}"], labels) for index, weight in enumerate(weights, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            examples += labels.size(0)
        validation = evaluate("validation")
        row = {"epoch": epoch, "train_loss": running_loss / examples, "validation": validation}
        history.append(row)
        print(json.dumps(row))
        if validation["macro_f1"] > best:
            best = validation["macro_f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "seed": args.seed}, checkpoint)

    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model"])
    test = evaluate("test", save_predictions=True)
    result = {
        "scope": "internal EARAM-style 80/10/10 experiment; not official-paper reproduction",
        "seed": args.seed,
        "device": device,
        "clip": args.clip,
        "best_epoch": saved["epoch"],
        "best_validation_macro_f1": best,
        "test": test,
        "history": history,
        "inputs": input_report,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
