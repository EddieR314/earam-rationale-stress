from __future__ import annotations

import math
import re

from .text import content_words, jaccard, sentences

NEGATION_RE = re.compile(r"\b(no|not|never|neither|cannot|can't|doesn't|isn't|without)\b", re.I)
CONTRADICTION_RE = re.compile(
    r"\b(preceding evidence should be rejected|opposite verdict|contradict(?:s|ory|ion))\b",
    re.I,
)
EVIDENCE_RE = re.compile(
    r"\b(image|caption|text|chart|photo|evidence|source|shows?|depicts?|indicates?|because|however)\b",
    re.I,
)
VERDICT_RE = re.compile(
    r"\b(fake|real|false|true|misleading|reliable|unreliable|support|consistent)\b",
    re.I,
)


def verdict(text: str) -> int:
    lowered = text.lower()
    negative = sum(lowered.count(token) for token in ("fake", "false", "misleading", "unreliable", "does not support", "not consistent"))
    positive = sum(lowered.count(token) for token in ("real", "true", "reliable", "supports", "consistent"))
    if positive == negative:
        return 0
    return 1 if positive > negative else -1


def rationale_score(caption: str, rationale: str, peer: str = "") -> dict[str, float]:
    tokens = content_words(rationale)
    sentence_count = len(sentences(rationale))
    relevance = jaccard(caption, rationale) if caption else 0.5
    evidence_density = min(1.0, len(EVIDENCE_RE.findall(rationale)) / max(2, sentence_count))
    completeness = min(1.0, math.log2(len(tokens) + 1) / 7.0)

    own_verdict = verdict(rationale)
    peer_verdict = verdict(peer)
    consistency = 1.0 if not peer or own_verdict == 0 or peer_verdict == 0 else float(own_verdict == peer_verdict)

    contradiction_penalty = min(
        1.0,
        len(NEGATION_RE.findall(rationale)) / max(3, sentence_count)
        + 0.75 * bool(CONTRADICTION_RE.search(rationale)),
    )
    unsupported_penalty = 0.0
    lowered = rationale.lower()
    if any(
        phrase in lowered
        for phrase in ("unnamed experts", "no records are provided", "conclusively verified every detail")
    ):
        unsupported_penalty = 1.0

    score = (
        0.30 * relevance
        + 0.25 * evidence_density
        + 0.20 * completeness
        + 0.25 * consistency
        - 0.10 * contradiction_penalty
        - 0.20 * unsupported_penalty
    )
    return {
        "score": max(0.0, min(1.0, score)),
        "relevance": relevance,
        "evidence_density": evidence_density,
        "completeness": completeness,
        "consistency": consistency,
        "contradiction_penalty": contradiction_penalty,
        "unsupported_penalty": unsupported_penalty,
        "has_verdict": float(bool(VERDICT_RE.search(rationale))),
    }


def filter_records(records: list[dict], threshold: float, strategy: str = "peer") -> list[dict]:
    if strategy not in {"drop", "peer"}:
        raise ValueError("strategy must be 'drop' or 'peer'")
    output: list[dict] = []
    for source in records:
        record = dict(source)
        first = record.get("rationale_1", "")
        second = record.get("rationale_2", "")
        score_1 = rationale_score(record.get("caption", ""), first, second)
        score_2 = rationale_score(record.get("caption", ""), second, first)
        record["reliability_1"] = score_1
        record["reliability_2"] = score_2
        record["filtered_fields"] = []
        record["filter_actions"] = {}
        if score_1["score"] < threshold:
            use_peer = strategy == "peer" and score_2["score"] >= threshold
            record["rationale_1"] = second if use_peer else ""
            record["filtered_fields"].append("rationale_1")
            record["filter_actions"]["rationale_1"] = "replace_with_peer" if use_peer else "drop"
        if score_2["score"] < threshold:
            use_peer = strategy == "peer" and score_1["score"] >= threshold
            record["rationale_2"] = first if use_peer else ""
            record["filtered_fields"].append("rationale_2")
            record["filter_actions"]["rationale_2"] = "replace_with_peer" if use_peer else "drop"
        output.append(record)
    return output
