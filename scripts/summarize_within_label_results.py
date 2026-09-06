#!/usr/bin/env python3
"""Aggregate the 3 x 3 within-label shuffle evaluation matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions-root", type=Path, default=Path("runs/conditions"))
    parser.add_argument(
        "--handoff-dir",
        type=Path,
        help="Flat archived results named shuffle<seed>-model<seed>-bootstrap.json",
    )
    parser.add_argument("--shuffle-seeds", type=int, nargs="+", default=[13, 42, 97])
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[13, 42, 97])
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def load_matrix(
    conditions_root: Path,
    shuffle_seeds: list[int],
    model_seeds: list[int],
    handoff_dir: Path | None = None,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    clean_by_model: dict[int, float] = {}
    for shuffle_seed in shuffle_seeds:
        for model_seed in model_seeds:
            if handoff_dir is None:
                source = (
                    conditions_root
                    / f"within-label-shuffle{shuffle_seed}"
                    / f"model-seed{model_seed}"
                    / "paired_bootstrap.json"
                )
            else:
                source = handoff_dir / f"shuffle{shuffle_seed}-model{model_seed}-bootstrap.json"
            payload = json.loads(source.read_text(encoding="utf-8"))
            clean = float(payload["clean_macro_f1"])
            candidate = float(payload["candidate_macro_f1"])
            delta = float(payload["macro_f1_delta"])
            ci95 = payload["macro_f1_delta_ci95"]
            if len(ci95) != 2:
                raise ValueError(f"Expected a two-sided CI in {source}")
            if abs((candidate - clean) - delta) > 1e-12:
                raise ValueError(f"Inconsistent Macro-F1 delta in {source}")
            previous_clean = clean_by_model.setdefault(model_seed, clean)
            if abs(previous_clean - clean) > 1e-12:
                raise ValueError(f"Clean Macro-F1 changed for model seed {model_seed}")
            rows.append(
                {
                    "shuffle_seed": shuffle_seed,
                    "model_seed": model_seed,
                    "records": int(payload["records"]),
                    "clean_macro_f1": clean,
                    "shuffled_macro_f1": candidate,
                    "macro_f1_delta": delta,
                    "ci95_low": float(ci95[0]),
                    "ci95_high": float(ci95[1]),
                }
            )
    return rows


def write_csv(rows: list[dict[str, float | int]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, float | int]], target: Path) -> None:
    clean_mean = mean(float(row["clean_macro_f1"]) for row in rows)
    shuffled_mean = mean(float(row["shuffled_macro_f1"]) for row in rows)
    delta_mean = mean(float(row["macro_f1_delta"]) for row in rows)
    positive_ci = sum(float(row["ci95_low"]) > 0 for row in rows)

    lines = [
        "# Within-label rationale shuffle control",
        "",
        "This follow-up holds each donor rationale's class label constant while breaking its",
        "sample-level correspondence. Both rationale channels move together, with zero fixed points.",
        "The clean checkpoint is frozen for every intervention evaluation.",
        "",
        f"- Clean mean Macro-F1: **{clean_mean:.4f}**",
        f"- Within-label shuffled mean Macro-F1: **{shuffled_mean:.4f}**",
        f"- Mean change: **{delta_mean:+.4f}**",
        f"- Individual paired-bootstrap intervals strictly above zero: **{positive_ci}/{len(rows)}**",
        "",
        "| Shuffle seed | Model seed | Clean | Within-label shuffle | Delta | Paired 95% CI |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {shuffle_seed} | {model_seed} | {clean_macro_f1:.4f} | "
            "{shuffled_macro_f1:.4f} | {macro_f1_delta:+.4f} | "
            "[{ci95_low:+.4f}, {ci95_high:+.4f}] |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Within-label shuffling did not reduce mean Macro-F1 in this internal run. Therefore,",
            "the earlier unrestricted-shuffle drop cannot be attributed to sample-level semantic",
            "misalignment alone. This result does not show that rationales are generally useless:",
            "the experiment uses internal splits, one released architecture path, and generated",
            "rationales whose label-associated signals remain intact under this control.",
            "",
            "These are internal EARAM-style results, not an official-score reproduction.",
        ]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_matrix(
        args.conditions_root,
        args.shuffle_seeds,
        args.model_seeds,
        handoff_dir=args.handoff_dir,
    )
    write_csv(rows, args.output_csv)
    write_markdown(rows, args.output_markdown)
    print(
        json.dumps(
            {
                "evaluations": len(rows),
                "clean_mean_macro_f1": mean(float(row["clean_macro_f1"]) for row in rows),
                "shuffled_mean_macro_f1": mean(
                    float(row["shuffled_macro_f1"]) for row in rows
                ),
                "mean_delta": mean(float(row["macro_f1_delta"]) for row in rows),
                "output_csv": str(args.output_csv),
                "output_markdown": str(args.output_markdown),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
