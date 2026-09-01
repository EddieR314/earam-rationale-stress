#!/usr/bin/env python3
"""Precompute frozen CLIP-Large token features with an 8 GB GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--analysis-1", required=True)
    parser.add_argument("--analysis-2", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clip", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Smoke-test only: encode the first N records")
    return parser


def cache_path(root: Path, source_id: str) -> Path:
    return root / "records" / f"{int(source_id):06d}.pt"


def load_inputs(args: argparse.Namespace) -> list[dict]:
    dataset = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))
    first = Path(args.analysis_1).read_text(encoding="utf-8").splitlines()
    second = Path(args.analysis_2).read_text(encoding="utf-8").splitlines()
    items = list(dataset.items())
    if not len(items) == len(first) == len(second):
        raise ValueError(
            f"Unaligned inputs: dataset={len(items)}, "
            f"analysis_1={len(first)}, analysis_2={len(second)}"
        )
    records = [
        {
            "source_id": source_id,
            "caption": item["caption"],
            "image_path": item["image_path"],
            "label": int(item["label"]),
            "rationale_1": first[index],
            "rationale_2": second[index],
        }
        for index, (source_id, item) in enumerate(items)
    ]
    return records[: args.limit] if args.limit else records


def main() -> None:
    args = build_parser().parse_args()
    records = load_inputs(args)
    output = Path(args.output_dir)
    (output / "records").mkdir(parents=True, exist_ok=True)

    missing_images = [
        str(Path(args.image_root) / record["image_path"])
        for record in records
        if not (Path(args.image_root) / record["image_path"]).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} images; first: {missing_images[0]}")

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install the CUDA PyTorch wheel and check the NVIDIA driver."
        )
    device = torch.device(args.device)
    use_fp16 = device.type == "cuda"
    model_dtype = torch.float16 if use_fp16 else torch.float32
    model = CLIPModel.from_pretrained(args.clip, torch_dtype=model_dtype).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def text_features(text: str):
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        hidden = model.text_model(**inputs).last_hidden_state
        return model.text_projection(hidden)

    completed = 0
    for index, record in enumerate(records, start=1):
        target = cache_path(output, record["source_id"])
        if args.resume and target.is_file():
            completed += 1
            continue
        with Image.open(Path(args.image_root) / record["image_path"]) as opened:
            image = opened.convert("RGB").copy()
        image_inputs = processor(images=[image], return_tensors="pt")
        image_inputs = {
            name: tensor.to(device, dtype=model_dtype)
            if tensor.is_floating_point()
            else tensor.to(device)
            for name, tensor in image_inputs.items()
        }
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_fp16
        ):
            visual = model.vision_model(**image_inputs).last_hidden_state
            payload = {
                "source_id": record["source_id"],
                "label": record["label"],
                "caption": text_features(record["caption"]),
                "image": model.visual_projection(visual),
                "rationale_1": text_features(record["rationale_1"]),
                "rationale_2": text_features(record["rationale_2"]),
            }
        torch.save(
            {
                key: value.detach().cpu().to(torch.float16) if torch.is_tensor(value) else value
                for key, value in payload.items()
            },
            target,
        )
        completed += 1
        if index == 1 or index % 50 == 0 or index == len(records):
            allocated = torch.cuda.max_memory_allocated() / 2**30 if use_fp16 else 0.0
            print(f"encoded {index}/{len(records)} | peak_cuda_gib={allocated:.2f}", flush=True)

    manifest = {
        "scope": "frozen CLIP token cache for low-memory EARAM training",
        "clip": args.clip,
        "dtype": "float16" if use_fp16 else "float32",
        "records": len(records),
        "completed": completed,
        "estimated_full_cache_gib": 1.9,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
