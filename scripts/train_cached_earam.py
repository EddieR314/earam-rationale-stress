#!/usr/bin/env python3
"""Train the public EARAM VLR from cached CLIP features using batch size one."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--earam-repo", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--checkpoint-in",
        help="Evaluate this clean-model checkpoint on --feature-dir without retraining.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def feature_path(root: Path, source_id: str) -> Path:
    return root / "records" / f"{int(source_id):06d}.pt"


def validate(args: argparse.Namespace) -> dict:
    if not (Path(args.earam_repo) / "model.py").is_file():
        raise FileNotFoundError("Official EARAM model.py was not found")
    if args.checkpoint_in and not Path(args.checkpoint_in).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_in}")
    if args.gradient_accumulation < 1:
        raise ValueError("--gradient-accumulation must be at least 1")
    if not args.checkpoint_in and args.epochs < 1:
        raise ValueError("--epochs must be at least 1 when training")
    manifest = json.loads((Path(args.split_dir) / "split_manifest.json").read_text(encoding="utf-8"))
    report = {}
    missing = []
    for split in ("train", "validation", "test"):
        source_ids = manifest[split]["source_ids"]
        missing.extend(
            str(feature_path(Path(args.feature_dir), value))
            for value in source_ids
            if not feature_path(Path(args.feature_dir), value).is_file()
        )
        report[split] = len(source_ids)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} cached records; first: {missing[0]}")
    report["missing_features"] = 0
    return report


def binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    accuracy = sum(left == right for left, right in zip(labels, predictions)) / len(labels)
    f1s = []
    for target in (0, 1):
        tp = sum(left == target and right == target for left, right in zip(labels, predictions))
        fp = sum(left != target and right == target for left, right in zip(labels, predictions))
        fn = sum(left == target and right != target for left, right in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": accuracy, "macro_f1": sum(f1s) / 2}


def main() -> None:
    args = build_parser().parse_args()
    input_report = validate(args)
    if args.dry_run:
        print(json.dumps({"status": "valid", **input_report}, indent=2))
        return

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    sys.path.insert(0, str(Path(args.earam_repo).resolve()))
    VLR = importlib.import_module("model").VLR
    split_manifest = json.loads((Path(args.split_dir) / "split_manifest.json").read_text(encoding="utf-8"))

    class CachedDataset(Dataset):
        def __init__(self, split: str):
            self.source_ids = split_manifest[split]["source_ids"]

        def __len__(self):
            return len(self.source_ids)

        def __getitem__(self, index):
            source_id = self.source_ids[index]
            record = torch.load(
                feature_path(Path(args.feature_dir), source_id),
                map_location="cpu",
                weights_only=True,
            )
            return record

    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(
            CachedDataset(split),
            batch_size=None,
            shuffle=split == "train",
            num_workers=0,
            generator=generator,
        )
        for split in ("train", "validation", "test")
    }

    model = VLR(dim=768).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    loss_weights = (0.7, 0.15, 0.15)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best.pt"
    history = []

    def tensors(record):
        fields = ("caption", "image", "rationale_1", "rationale_2")
        return tuple(record[name].to(device, non_blocking=True) for name in fields)

    def evaluate(split: str, save: bool = False):
        model.eval()
        labels, predictions, rows = [], [], []
        with torch.inference_mode():
            for record in loaders[split]:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(*tensors(record))["label_1"]
                prediction = int(logits.argmax(dim=1).item())
                label = int(record["label"])
                probabilities = logits.squeeze(0).detach().float().cpu().tolist()
                labels.append(label)
                predictions.append(prediction)
                rows.append(
                    {
                        "id": record["source_id"],
                        "label": label,
                        "prediction": prediction,
                        "probabilities": probabilities,
                    }
                )
        result = binary_metrics(labels, predictions)
        if save:
            (output / f"{split}_predictions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
        return result

    if args.checkpoint_in:
        saved = torch.load(args.checkpoint_in, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        test = evaluate("test", save=True)
        result = {
            "scope": "fixed clean EARAM-style checkpoint evaluated on a cached-feature condition",
            "seed": args.seed,
            "checkpoint": str(Path(args.checkpoint_in).resolve()),
            "checkpoint_epoch": saved.get("epoch"),
            "test": test,
            "inputs": input_report,
        }
        (output / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for index, record in enumerate(loaders["train"], start=1):
            label = torch.tensor([int(record["label"])], device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(*tensors(record))
                raw_loss = sum(
                    weight * criterion(outputs[f"label_{head}"], label)
                    for head, weight in enumerate(loss_weights, start=1)
                )
                loss = raw_loss / args.gradient_accumulation
            scaler.scale(loss).backward()
            if index % args.gradient_accumulation == 0 or index == len(loaders["train"]):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(raw_loss.detach().cpu())
        validation = evaluate("validation")
        row = {
            "epoch": epoch,
            "mean_train_loss": running_loss / len(loaders["train"]),
            "validation": validation,
            "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30 if use_amp else 0.0,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["macro_f1"] > best:
            best = validation["macro_f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "seed": args.seed}, checkpoint)

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    test = evaluate("test", save=True)
    result = {
        "scope": "internal cached-feature EARAM-style experiment; not official reproduction",
        "seed": args.seed,
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
