from __future__ import annotations

import json
from pathlib import Path

from .io import read_lines
from .metrics import classification_metrics
from .perturb import PERTURBERS, perturb_records
from .scoring import filter_records


def _probe_text(record: dict) -> str:
    return " [CAPTION] ".join(
        [
            str(record.get("caption", "")),
            str(record.get("rationale_1", "")),
            str(record.get("rationale_2", "")),
        ]
    )


def _detector_text(caption: str, rationale: str) -> str:
    return f"[CAPTION] {caption} [RATIONALE] {rationale}"


def _train_corruption_detector(train_records: list[dict], seed: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts: list[str] = []
    labels: list[int] = []
    for record in train_records:
        for field in ("rationale_1", "rationale_2"):
            texts.append(_detector_text(record["caption"], record[field]))
            labels.append(0)
    for offset, perturbation in enumerate([*PERTURBERS, "irrelevant"]):
        corrupted = perturb_records(train_records, perturbation, 0.7, seed + offset, 1.0)
        for record in corrupted:
            for field in ("rationale_1", "rationale_2"):
                texts.append(_detector_text(record["caption"], record[field]))
                labels.append(1)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=50_000, sublinear_tf=True
    )
    features = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=seed)
    classifier.fit(features, labels)
    return vectorizer, classifier


def _learned_filter(records: list[dict], vectorizer, classifier) -> list[dict]:
    output = []
    for source in records:
        record = dict(source)
        fields = ("rationale_1", "rationale_2")
        texts = [_detector_text(record["caption"], record[field]) for field in fields]
        probabilities = classifier.predict_proba(vectorizer.transform(texts))[:, 1]
        rejected = [bool(value >= 0.5) for value in probabilities]
        record["learned_corruption_probability"] = {
            fields[index]: float(probabilities[index]) for index in range(2)
        }
        record["learned_filtered_fields"] = [fields[index] for index in range(2) if rejected[index]]
        first, second = record[fields[0]], record[fields[1]]
        if rejected == [True, False]:
            record[fields[0]] = second
        elif rejected == [False, True]:
            record[fields[1]] = first
        elif rejected == [True, True]:
            safer = first if probabilities[0] <= probabilities[1] else second
            record[fields[0]] = safer
            record[fields[1]] = safer
        output.append(record)
    return output


def load_labeled_earam_records(
    dataset_json: str | Path,
    analysis_1_path: str | Path,
    analysis_2_path: str | Path,
) -> list[dict]:
    data = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
    items = list(data.items())
    analysis_1 = read_lines(analysis_1_path)
    analysis_2 = read_lines(analysis_2_path)
    lengths = {"dataset": len(items), "analysis_1": len(analysis_1), "analysis_2": len(analysis_2)}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Dataset and rationales are not aligned: {lengths}")
    return [
        {
            "id": str(key),
            "caption": item["caption"],
            "label": int(item["label"]),
            "rationale_1": analysis_1[index],
            "rationale_2": analysis_2[index],
        }
        for index, (key, item) in enumerate(items)
    ]


def run_text_probe(
    dataset_json: str | Path,
    analysis_1_path: str | Path,
    analysis_2_path: str | Path,
    rates: list[float],
    severity: float,
    seed: int,
    threshold: float,
    strategy: str,
) -> dict:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError("run-text-probe requires scikit-learn") from exc

    records = load_labeled_earam_records(dataset_json, analysis_1_path, analysis_2_path)
    indices = list(range(len(records)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=[record["label"] for record in records],
    )
    train_records = [records[index] for index in train_indices]
    test_records = [records[index] for index in test_indices]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True, max_features=50_000
    )
    train_features = vectorizer.fit_transform(_probe_text(record) for record in train_records)
    classifier = LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=seed)
    classifier.fit(train_features, [record["label"] for record in train_records])
    detector_vectorizer, detector = _train_corruption_detector(train_records, seed)

    def evaluate(candidate: list[dict]) -> dict:
        features = vectorizer.transform(_probe_text(record) for record in candidate)
        predictions = classifier.predict(features)
        return classification_metrics(
            [record["label"] for record in candidate], [int(value) for value in predictions]
        )

    clean_metrics = evaluate(test_records)
    learned_clean = _learned_filter(test_records, detector_vectorizer, detector)
    clean_rejected = sum(len(record["learned_filtered_fields"]) for record in learned_clean)
    conditions = []
    for perturbation in [*PERTURBERS, "irrelevant"]:
        for rate in rates:
            corrupted = perturb_records(test_records, perturbation, severity, seed, rate)
            filtered = filter_records(corrupted, threshold, strategy)
            corrupted_metrics = evaluate(corrupted)
            filtered_metrics = evaluate(filtered)
            learned_filtered = _learned_filter(corrupted, detector_vectorizer, detector)
            learned_metrics = evaluate(learned_filtered)
            learned_rejected = sum(
                len(record["learned_filtered_fields"]) for record in learned_filtered
            )
            conditions.append(
                {
                    "perturbation": perturbation,
                    "target_rate": rate,
                    "corrupted": corrupted_metrics,
                    "filtered": filtered_metrics,
                    "learned_filtered": learned_metrics,
                    "learned_rejected_rationale_rate": learned_rejected / (2 * len(corrupted)),
                    "macro_f1_delta_corrupted": corrupted_metrics["macro_f1"] - clean_metrics["macro_f1"],
                    "macro_f1_delta_filtered": filtered_metrics["macro_f1"] - clean_metrics["macro_f1"],
                    "macro_f1_delta_learned": learned_metrics["macro_f1"] - clean_metrics["macro_f1"],
                    "macro_f1_gain_learned_over_corrupted": (
                        learned_metrics["macro_f1"] - corrupted_metrics["macro_f1"]
                    ),
                }
            )
    return {
        "scope": "text-only diagnostic probe; not an EARAM reproduction",
        "split": {
            "source_records": len(records),
            "train_records": len(train_records),
            "held_out_records": len(test_records),
            "stratified_test_fraction": 0.2,
            "seed": seed,
        },
        "model": "TF-IDF (word 1-2 grams) + class-balanced logistic regression",
        "severity": severity,
        "filter_threshold": threshold,
        "filter_strategy": strategy,
        "clean": clean_metrics,
        "learned_filter_clean_false_positive_rate": clean_rejected / (2 * len(test_records)),
        "learned_filter_limit": (
            "trained and tested on perturbations from the same synthetic generators; "
            "it detects generator artifacts, not factual correctness"
        ),
        "conditions": conditions,
    }
