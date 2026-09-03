import tempfile
import unittest
import csv
from pathlib import Path

from earam_stress.io import export_earam, import_earam, read_lines
from earam_stress.metrics import classification_metrics, summarize_records
from earam_stress.mr2 import ascii_suffix_start, select_earam_subset
from earam_stress.perturb import perturb_records
from earam_stress.scoring import filter_records, rationale_score
from earam_stress.splits import make_splits, validate_split_dir


RECORDS = [
    {
        "id": "0",
        "caption": "A chart shows unemployment falling after the election.",
        "rationale_1": "The chart shows a lower unemployment rate. The image supports the caption.",
        "rationale_2": "The visual evidence is consistent with the text, so the report appears reliable.",
    },
    {
        "id": "1",
        "caption": "A child is shown in snow while the caption discusses unemployment.",
        "rationale_1": "The image depicts a child and provides no evidence for the economic claim.",
        "rationale_2": "The photo does not support the caption, so the pairing may be misleading.",
    },
]


class ToolkitTests(unittest.TestCase):
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
