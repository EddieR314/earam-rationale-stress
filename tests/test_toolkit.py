import tempfile
import unittest
import csv
import json
import importlib.util
from pathlib import Path

from earam_stress.conditions import build_rationale_shuffle_condition, shuffle_rationale_pairs
from earam_stress.io import export_earam, import_earam, read_lines
from earam_stress.metrics import classification_metrics, paired_bootstrap_macro_f1, summarize_records
from earam_stress.mr2 import ascii_suffix_start, select_earam_subset
from earam_stress.perturb import perturb_records
from earam_stress.provenance import file_record, runtime_record
from earam_stress.scoring import filter_records, rationale_score
from earam_stress.splits import make_splits, validate_split_dir


RECORDS = [
    {
        "id": "0",
        "label": 0,
        "caption": "A chart shows unemployment falling after the election.",
        "rationale_1": "The chart shows a lower unemployment rate. The image supports the caption.",
        "rationale_2": "The visual evidence is consistent with the text, so the report appears reliable.",
    },
    {
        "id": "1",
        "label": 1,
        "caption": "A child is shown in snow while the caption discusses unemployment.",
        "rationale_1": "The image depicts a child and provides no evidence for the economic claim.",
        "rationale_2": "The photo does not support the caption, so the pairing may be misleading.",
    },
]


class ToolkitTests(unittest.TestCase):
    def test_archive_sanitizer_replaces_nested_paths(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "sanitize_within_label_archive.py"
        spec = importlib.util.spec_from_file_location("archive_sanitizer", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        payload = {"path": "C:\\Users\\person\\project\\file.json", "nested": ["unchanged"]}
        observed = module.replace_strings(payload, "C:\\Users\\person\\project", "<PROJECT_ROOT>")
        self.assertEqual(observed["path"], "<PROJECT_ROOT>\\file.json")
        self.assertEqual(observed["nested"], ["unchanged"])

    def test_within_label_matrix_summary_validates_and_writes_outputs(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_within_label_results.py"
        spec = importlib.util.spec_from_file_location("within_label_summary", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for shuffle_seed in (13, 42):
                for model_seed, clean in ((13, 0.8), (42, 0.9)):
                    target = root / f"within-label-shuffle{shuffle_seed}" / f"model-seed{model_seed}"
                    target.mkdir(parents=True)
                    candidate = clean + shuffle_seed / 10000
                    payload = {
                        "records": 10,
                        "clean_macro_f1": clean,
                        "candidate_macro_f1": candidate,
                        "macro_f1_delta": candidate - clean,
                        "macro_f1_delta_ci95": [-0.01, 0.03],
                    }
                    (target / "paired_bootstrap.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )
            rows = module.load_matrix(root, [13, 42], [13, 42])
            self.assertEqual(len(rows), 4)
            flat = root / "flat"
            flat.mkdir()
            for shuffle_seed in (13, 42):
                for model_seed in (13, 42):
                    source = (
                        root
                        / f"within-label-shuffle{shuffle_seed}"
                        / f"model-seed{model_seed}"
                        / "paired_bootstrap.json"
                    )
                    (flat / f"shuffle{shuffle_seed}-model{model_seed}-bootstrap.json").write_text(
                        source.read_text(encoding="utf-8"), encoding="utf-8"
                    )
            self.assertEqual(module.load_matrix(root, [13, 42], [13, 42], flat), rows)
            csv_target = root / "summary.csv"
            markdown_target = root / "summary.md"
            module.write_csv(rows, csv_target)
            module.write_markdown(rows, markdown_target)
            self.assertIn("shuffle_seed", csv_target.read_text(encoding="utf-8"))
            self.assertIn("Interpretation boundary", markdown_target.read_text(encoding="utf-8"))

    def test_provenance_records_file_hash_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "artifact.txt"
            target.write_text("auditable\n", encoding="utf-8")
            record = file_record(target)
            self.assertEqual(record["bytes"], 10)
            self.assertEqual(len(record["sha256"]), 64)
            runtime = runtime_record()
            self.assertIn("python", runtime)
            self.assertIn("platform", runtime)

    def test_within_label_shuffle_moves_pairs_without_fixed_points(self):
        records = [
            {
                "id": str(index),
                "label": index // 3,
                "rationale_1": f"first-{index}",
                "rationale_2": f"second-{index}",
            }
            for index in range(6)
        ]
        shuffled = shuffle_rationale_pairs(records, seed=19, mode="within-label")
        self.assertEqual(shuffled, shuffle_rationale_pairs(records, seed=19, mode="within-label"))
        source = {record["id"]: record for record in records}
        for recipient in shuffled:
            donor_id = recipient["rationale_shuffle"]["donor_id"]
            donor = source[donor_id]
            self.assertNotEqual(recipient["id"], donor_id)
            self.assertEqual(recipient["label"], donor["label"])
            self.assertEqual(
                (recipient["rationale_1"], recipient["rationale_2"]),
                (donor["rationale_1"], donor["rationale_2"]),
            )

        with self.assertRaisesRegex(ValueError, "fewer than two"):
            shuffle_rationale_pairs(records[:4], seed=19, mode="within-label")

    def test_shuffle_condition_writes_aligned_outputs_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "test.json"
            first = root / "a1.txt"
            second = root / "a2.txt"
            dataset.write_text(
                json.dumps(
                    {
                        str(index): {"caption": f"caption-{index}", "label": index // 2}
                        for index in range(4)
                    }
                ),
                encoding="utf-8",
            )
            first.write_text("a\nb\nc\nd\n", encoding="utf-8")
            second.write_text("A\nB\nC\nD\n", encoding="utf-8")
            report = build_rationale_shuffle_condition(
                dataset, first, second, root / "condition", seed=7
            )
            self.assertEqual(report["mode"], "within-label")
            self.assertEqual(report["fixed_points"], 0)
            self.assertEqual(len(read_lines(root / "condition" / "analysis_1.txt")), 4)
            self.assertEqual(len(read_lines(root / "condition" / "analysis_2.txt")), 4)
            self.assertEqual(
                json.loads((root / "condition" / "manifest.json").read_text(encoding="utf-8")),
                report,
            )
            self.assertEqual(len(report["inputs"]["dataset_json"]["sha256"]), 64)

    def test_pilot_summary_deltas_are_consistent(self):
        summary = Path(__file__).resolve().parents[1] / "docs" / "results_summary.csv"
        with summary.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        clean = float(rows[0]["mean_macro_f1"])
        self.assertEqual(rows[0]["condition"], "Clean rationale")
        for row in rows:
            observed = float(row["mean_macro_f1"]) - clean
            reported = float(row["delta_from_clean"])
            # Displayed means are rounded to four decimals, while deltas were
            # computed from the unrounded run means.
            self.assertLessEqual(abs(observed - reported), 0.00011)

    def test_all_perturbations_are_deterministic_and_change_text(self):
        names = ("evidence_deletion", "conclusion_flip", "unsupported_claim", "contradiction", "irrelevant")
        for name in names:
            with self.subTest(name=name):
                first = perturb_records(RECORDS, name, 0.6, seed=7)
                second = perturb_records(RECORDS, name, 0.6, seed=7)
                self.assertEqual(first, second)
                self.assertTrue(any(record["corrupted_fields"] for record in first))

    def test_filter_attaches_scores_and_can_remove_bad_rationale(self):
        corrupted = perturb_records(RECORDS, "unsupported_claim", 1.0, seed=3)
        filtered = filter_records(corrupted, threshold=0.55, strategy="peer")
        self.assertIn("reliability_1", filtered[0])
        self.assertTrue(any(record["filtered_fields"] for record in filtered))
        self.assertTrue(any(record["filter_actions"] for record in filtered))

    def test_summary_and_classification_metrics(self):
        corrupted = perturb_records(RECORDS, "irrelevant", 1.0, seed=2)
        filtered = filter_records(corrupted, threshold=0.55, strategy="peer")
        summary = summarize_records(RECORDS, filtered)
        self.assertEqual(summary["records"], 2)
        self.assertGreater(summary["mean_text_change"], 0)
        self.assertIn("filter_detection", summary)
        metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertGreater(metrics["macro_f1"], 0)
        self.assertLess(metrics["macro_f1"], 1)

    def test_paired_bootstrap_aligns_ids_and_is_deterministic(self):
        clean = [
            {"id": str(index), "label": label, "prediction": prediction}
            for index, (label, prediction) in enumerate([(0, 0), (0, 0), (1, 1), (1, 1)])
        ]
        candidate = [
            {"id": "2", "label": 1, "prediction": 0},
            {"id": "0", "label": 0, "prediction": 0},
            {"id": "3", "label": 1, "prediction": 1},
            {"id": "1", "label": 0, "prediction": 1},
        ]
        first = paired_bootstrap_macro_f1(clean, candidate, samples=100, seed=5)
        second = paired_bootstrap_macro_f1(clean, candidate, samples=100, seed=5)
        self.assertEqual(first, second)
        self.assertLess(first["macro_f1_delta"], 0)
        self.assertEqual(first["records"], 4)

    def test_line_aligned_adapter_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a1.txt"
            second = root / "a2.txt"
            first.write_text("one\ntwo\n", encoding="utf-8")
            second.write_text("three\nfour\n", encoding="utf-8")
            records = import_earam(first, second)
            out_first = root / "out1.txt"
            out_second = root / "out2.txt"
            export_earam(records, out_first, out_second)
            self.assertEqual(read_lines(out_first), ["one", "two"])
            self.assertEqual(read_lines(out_second), ["three", "four"])

    def test_reliability_score_is_bounded(self):
        result = rationale_score(
            RECORDS[0]["caption"], RECORDS[0]["rationale_1"], RECORDS[0]["rationale_2"]
        )
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 1)

    def test_mr2_ascii_suffix_and_binary_filter(self):
        items = [
            ("0", {"caption": "中文", "label": 0}),
            ("1", {"caption": "English rumor", "label": 1}),
            ("2", {"caption": "English unverified", "label": 2}),
        ]
        self.assertEqual(ascii_suffix_start(items), 1)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "items.json"
            source.write_text(
                __import__("json").dumps(dict(items), ensure_ascii=False), encoding="utf-8"
            )
            selected, stats = select_earam_subset(source)
            self.assertEqual([key for key, _ in selected], ["1"])
            self.assertEqual(stats["excluded_unverified"], 1)

    def test_internal_splits_are_aligned_stratified_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {
                str(index): {
                    "caption": f"caption {index}",
                    "image_path": f"train/img/{index}.jpg",
                    "label": index % 2,
                }
                for index in range(40)
            }
            dataset = root / "data.json"
            dataset.write_text(__import__("json").dumps(data), encoding="utf-8")
            first = root / "a1.txt"
            second = root / "a2.txt"
            first.write_text("\n".join(f"first {i}" for i in range(40)) + "\n", encoding="utf-8")
            second.write_text("\n".join(f"second {i}" for i in range(40)) + "\n", encoding="utf-8")
            output = root / "splits"
            report = make_splits(dataset, first, second, output, [13])
            self.assertEqual(report["seeds"]["13"]["train"]["records"], 32)
            validation = validate_split_dir(output / "seed13")
            self.assertTrue(validation["disjoint"])


if __name__ == "__main__":
    unittest.main()
