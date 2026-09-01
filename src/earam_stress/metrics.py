from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher

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


def classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("labels and predictions must be non-empty and aligned")
    classes = sorted(set(labels) | set(predictions))
    f1_values: list[float] = []
    correct = 0
    for label, prediction in zip(labels, predictions):
        correct += int(label == prediction)
    for class_id in classes:
        tp = sum(l == class_id and p == class_id for l, p in zip(labels, predictions))
        fp = sum(l != class_id and p == class_id for l, p in zip(labels, predictions))
        fn = sum(l == class_id and p != class_id for l, p in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": correct / len(labels), "macro_f1": sum(f1_values) / len(f1_values)}
