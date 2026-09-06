from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import random

from .scoring import rationale_score


def text_change_ratio(original: str, candidate: str) -> float:
    return 1.0 - SequenceMatcher(None, original, candidate).ratio()


def summarize_records(clean: list[dict], candidate: list[dict]) -> dict:
    if len(clean) != len(candidate):
        raise ValueError("Clean and candidate record counts differ")
    changes: list[float] = []
    clean_scores: list[float] = []
    candidate_scores: list[float] = []
    filtered = Counter()
    tp = fp = fn = tn = 0

    for left, right in zip(clean, candidate):
        if str(left.get("id")) != str(right.get("id")):
            raise ValueError("Record IDs are not aligned")
        for field, peer in (("rationale_1", "rationale_2"), ("rationale_2", "rationale_1")):
            left_text = left.get(field, "")
            right_text = right.get(field, "")
            changes.append(text_change_ratio(left_text, right_text))
            clean_scores.append(
                rationale_score(left.get("caption", ""), left_text, left.get(peer, ""))["score"]
            )
            candidate_scores.append(
                rationale_score(right.get("caption", ""), right_text, right.get(peer, ""))["score"]
            )
        filtered.update(right.get("filtered_fields", []))
        corrupted_fields = set(right.get("corrupted_fields", []))
        filtered_fields = set(right.get("filtered_fields", []))
        for field in ("rationale_1", "rationale_2"):
            corrupted = field in corrupted_fields
            rejected = field in filtered_fields
            tp += int(corrupted and rejected)
            fp += int(not corrupted and rejected)
            fn += int(corrupted and not rejected)
            tn += int(not corrupted and not rejected)

    average = lambda values: sum(values) / len(values) if values else 0.0
    result = {
        "records": len(clean),
        "rationales": len(changes),
        "mean_text_change": average(changes),
        "mean_clean_reliability": average(clean_scores),
        "mean_candidate_reliability": average(candidate_scores),
        "reliability_delta": average(candidate_scores) - average(clean_scores),
        "filtered_rationales": sum(filtered.values()),
    }
    if tp + fp + fn + tn:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result["filter_detection"] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        }
    return result


def _classification_metrics(
    labels: list[int], predictions: list[int], classes: list[int]
) -> dict[str, float]:
    f1_values: list[float] = []
    correct = sum(label == prediction for label, prediction in zip(labels, predictions))
    for class_id in classes:
        tp = sum(l == class_id and p == class_id for l, p in zip(labels, predictions))
        fp = sum(l != class_id and p == class_id for l, p in zip(labels, predictions))
        fn = sum(l == class_id and p != class_id for l, p in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": correct / len(labels), "macro_f1": sum(f1_values) / len(f1_values)}


def classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("labels and predictions must be non-empty and aligned")
    classes = sorted(set(labels) | set(predictions))
    return _classification_metrics(labels, predictions, classes)


def paired_bootstrap_macro_f1(
    clean: list[dict], candidate: list[dict], samples: int = 2_000, seed: int = 42
) -> dict:
    if samples < 1:
        raise ValueError("samples must be at least one")
    if len({str(row["id"]) for row in clean}) != len(clean):
        raise ValueError("Clean prediction IDs must be unique")
    candidate_by_id = {str(row["id"]): row for row in candidate}
    if len(candidate_by_id) != len(candidate) or set(candidate_by_id) != {
        str(row["id"]) for row in clean
    }:
        raise ValueError("Clean and candidate prediction IDs must match uniquely")

    aligned = []
    for left in clean:
        right = candidate_by_id[str(left["id"])]
        if int(left["label"]) != int(right["label"]):
            raise ValueError(f"Label mismatch for record {left['id']}")
        aligned.append((int(left["label"]), int(left["prediction"]), int(right["prediction"])))
    if not aligned:
        raise ValueError("Prediction files must not be empty")

    labels = [row[0] for row in aligned]
    clean_predictions = [row[1] for row in aligned]
    candidate_predictions = [row[2] for row in aligned]
    classes = sorted(set(labels) | set(clean_predictions) | set(candidate_predictions))
    clean_score = _classification_metrics(labels, clean_predictions, classes)["macro_f1"]
    candidate_score = _classification_metrics(labels, candidate_predictions, classes)["macro_f1"]

    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        indices = [rng.randrange(len(aligned)) for _ in aligned]
        sampled_labels = [labels[index] for index in indices]
        sampled_clean = [clean_predictions[index] for index in indices]
        sampled_candidate = [candidate_predictions[index] for index in indices]
        deltas.append(
            _classification_metrics(sampled_labels, sampled_candidate, classes)["macro_f1"]
            - _classification_metrics(sampled_labels, sampled_clean, classes)["macro_f1"]
        )

    deltas.sort()

    def percentile(probability: float) -> float:
        position = (len(deltas) - 1) * probability
        lower = int(position)
        upper = min(lower + 1, len(deltas) - 1)
        weight = position - lower
        return deltas[lower] * (1.0 - weight) + deltas[upper] * weight

    return {
        "records": len(aligned),
        "bootstrap_samples": samples,
        "seed": seed,
        "clean_macro_f1": clean_score,
        "candidate_macro_f1": candidate_score,
        "macro_f1_delta": candidate_score - clean_score,
        "macro_f1_delta_ci95": [percentile(0.025), percentile(0.975)],
    }
