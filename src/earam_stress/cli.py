from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import export_earam, import_earam, read_jsonl, write_jsonl
from .metrics import classification_metrics, summarize_records
from .mr2 import prepare_mr2
from .perturb import PERTURBERS, perturb_records
from .probe import run_text_probe
from .scoring import filter_records
from .splits import make_splits, validate_split_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="earam-stress")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import-earam", help="Import EARAM line-aligned analyses")
    import_parser.add_argument("--analysis-1", required=True)
    import_parser.add_argument("--analysis-2", required=True)
    import_parser.add_argument("--captions")
    import_parser.add_argument("--output", required=True)

    perturb_parser = subparsers.add_parser("perturb", help="Apply controlled rationale corruption")
    perturb_parser.add_argument("--input", required=True)
    perturb_parser.add_argument("--output", required=True)
    perturb_parser.add_argument("--type", choices=[*PERTURBERS, "irrelevant"], required=True)
    perturb_parser.add_argument("--severity", type=float, default=0.5)
    perturb_parser.add_argument("--target-rate", type=float, default=1.0)
    perturb_parser.add_argument("--seed", type=int, default=42)

    filter_parser = subparsers.add_parser("filter", help="Apply the lightweight reliability filter")
    filter_parser.add_argument("--input", required=True)
    filter_parser.add_argument("--output", required=True)
    filter_parser.add_argument("--threshold", type=float, default=0.35)
    filter_parser.add_argument("--strategy", choices=["peer", "drop"], default="peer")

    matrix_parser = subparsers.add_parser("matrix", help="Generate a complete stress-test matrix")
    matrix_parser.add_argument("--input", required=True)
    matrix_parser.add_argument("--output-dir", required=True)
    matrix_parser.add_argument("--rates", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    matrix_parser.add_argument("--severity", type=float, default=0.7)
    matrix_parser.add_argument("--seed", type=int, default=42)
    matrix_parser.add_argument("--threshold", type=float, default=0.4)
    matrix_parser.add_argument("--strategy", choices=["peer", "drop"], default="peer")

    calibrate_parser = subparsers.add_parser("calibrate", help="Tune the filter threshold on labeled synthetic corruptions")
    calibrate_parser.add_argument("--clean", required=True)
    calibrate_parser.add_argument("--input", required=True)
    calibrate_parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
    )
    calibrate_parser.add_argument("--strategy", choices=["peer", "drop"], default="peer")
    calibrate_parser.add_argument("--output")

    export_parser = subparsers.add_parser("export-earam", help="Export line-aligned EARAM analyses")
    export_parser.add_argument("--input", required=True)
    export_parser.add_argument("--analysis-1", required=True)
    export_parser.add_argument("--analysis-2", required=True)

    summary_parser = subparsers.add_parser("summarize", help="Compare clean and transformed rationales")
    summary_parser.add_argument("--clean", required=True)
    summary_parser.add_argument("--candidate", required=True)
    summary_parser.add_argument("--output")

    predictions_parser = subparsers.add_parser("evaluate-predictions")
    predictions_parser.add_argument("--input", required=True, help="JSONL with label and prediction")
    predictions_parser.add_argument("--output")

    mr2_parser = subparsers.add_parser(
        "prepare-mr2", help="Reconstruct EARAM's English binary MR2 subset"
    )
    mr2_parser.add_argument("--train-json", required=True)
    mr2_parser.add_argument("--validation-json", required=True)
    mr2_parser.add_argument("--output-dir", required=True)
    mr2_parser.add_argument("--archive", help="Optional full MR2 .tar.gz for selective image extraction")

    probe_parser = subparsers.add_parser(
        "run-text-probe", help="Run a low-cost text-only held-out diagnostic"
    )
    probe_parser.add_argument("--dataset-json", required=True)
    probe_parser.add_argument("--analysis-1", required=True)
    probe_parser.add_argument("--analysis-2", required=True)
    probe_parser.add_argument("--rates", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    probe_parser.add_argument("--severity", type=float, default=0.7)
    probe_parser.add_argument("--seed", type=int, default=42)
    probe_parser.add_argument("--threshold", type=float, default=0.65)
    probe_parser.add_argument("--strategy", choices=["peer", "drop"], default="peer")
    probe_parser.add_argument("--output", required=True)

    split_parser = subparsers.add_parser(
        "make-splits", help="Create aligned internal 80/10/10 EARAM-style splits"
    )
    split_parser.add_argument("--dataset-json", required=True)
    split_parser.add_argument("--analysis-1", required=True)
    split_parser.add_argument("--analysis-2", required=True)
    split_parser.add_argument("--output-dir", required=True)
    split_parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 97])

    validate_parser = subparsers.add_parser("validate-split", help="Validate one generated seed split")
    validate_parser.add_argument("--split-dir", required=True)
    return parser


def emit(payload: dict, output: str | None) -> None:
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if output:
        Path(output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "import-earam":
        write_jsonl(args.output, import_earam(args.analysis_1, args.analysis_2, args.captions))
    elif args.command == "perturb":
        records = read_jsonl(args.input)
        write_jsonl(
            args.output,
            perturb_records(records, args.type, args.severity, args.seed, args.target_rate),
        )
    elif args.command == "filter":
        write_jsonl(args.output, filter_records(read_jsonl(args.input), args.threshold, args.strategy))
    elif args.command == "matrix":
        source = read_jsonl(args.input)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for perturbation in [*PERTURBERS, "irrelevant"]:
            for rate in args.rates:
                stem = f"{perturbation}.r{rate:.2f}.s{args.severity:.2f}.seed{args.seed}"
                corrupted = perturb_records(source, perturbation, args.severity, args.seed, rate)
                filtered = filter_records(corrupted, args.threshold, args.strategy)
                corrupted_path = output_dir / f"{stem}.jsonl"
                filtered_path = output_dir / f"{stem}.filtered.jsonl"
                write_jsonl(corrupted_path, corrupted)
                write_jsonl(filtered_path, filtered)
                manifest.append(
                    {
                        "condition": stem,
                        "perturbation": perturbation,
                        "target_rate": rate,
                        "severity": args.severity,
                        "seed": args.seed,
                        "corrupted_path": str(corrupted_path),
                        "filtered_path": str(filtered_path),
                        "corrupted_summary": summarize_records(source, corrupted),
                        "filtered_summary": summarize_records(source, filtered),
                    }
                )
        emit({"conditions": manifest}, str(output_dir / "manifest.json"))
    elif args.command == "calibrate":
        clean = read_jsonl(args.clean)
        corrupted = read_jsonl(args.input)
        rows = []
        for threshold in args.thresholds:
            filtered = filter_records(corrupted, threshold, args.strategy)
            detection = summarize_records(clean, filtered).get("filter_detection", {})
            rows.append({"threshold": threshold, **detection})
        best = max(rows, key=lambda row: row.get("f1", 0.0))
        emit({"best": best, "results": rows}, args.output)
    elif args.command == "export-earam":
        export_earam(read_jsonl(args.input), args.analysis_1, args.analysis_2)
    elif args.command == "summarize":
        emit(summarize_records(read_jsonl(args.clean), read_jsonl(args.candidate)), args.output)
    elif args.command == "evaluate-predictions":
        records = read_jsonl(args.input)
        emit(
            classification_metrics(
                [int(record["label"]) for record in records],
                [int(record["prediction"]) for record in records],
            ),
            args.output,
        )
    elif args.command == "prepare-mr2":
        emit(
            prepare_mr2(
                args.train_json,
                args.validation_json,
                args.output_dir,
                args.archive,
            ),
            None,
        )
    elif args.command == "run-text-probe":
        emit(
            run_text_probe(
                args.dataset_json,
                args.analysis_1,
                args.analysis_2,
                args.rates,
                args.severity,
                args.seed,
                args.threshold,
                args.strategy,
            ),
            args.output,
        )
    elif args.command == "make-splits":
        emit(
            make_splits(
                args.dataset_json,
                args.analysis_1,
                args.analysis_2,
                args.output_dir,
                args.seeds,
            ),
            None,
        )
    elif args.command == "validate-split":
        emit(validate_split_dir(args.split_dir), None)


if __name__ == "__main__":
    main()
