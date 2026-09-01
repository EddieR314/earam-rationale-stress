from __future__ import annotations

import copy
import random
import re
from collections.abc import Callable

from .text import sentences


UNSUPPORTED_CLAIMS = (
    "Independent authorities have conclusively verified every detail of this report.",
    "Multiple unnamed experts confirm that the claim is unquestionably accurate.",
    "Official records prove the conclusion, although no records are provided here.",
)


VERDICT_SWAPS = (
    (re.compile(r"\bdoes not support\b", re.I), "supports"),
    (re.compile(r"\bnot consistent with\b", re.I), "consistent with"),
    (re.compile(r"\bfake news\b", re.I), "real news"),
    (re.compile(r"\bfalse claim\b", re.I), "true claim"),
    (re.compile(r"\bmisleading\b", re.I), "reliable"),
    (re.compile(r"\bunreliable\b", re.I), "reliable"),
    (re.compile(r"\bsupports\b", re.I), "does not support"),
    (re.compile(r"\bconsistent with\b", re.I), "not consistent with"),
    (re.compile(r"\breal news\b", re.I), "fake news"),
    (re.compile(r"\btrue claim\b", re.I), "false claim"),
    (re.compile(r"\breliable\b", re.I), "unreliable"),
)


def evidence_deletion(text: str, rng: random.Random, severity: float) -> str:
    parts = sentences(text)
    if len(parts) <= 1:
        words = text.split()
        keep = max(1, round(len(words) * (1.0 - severity)))
        return " ".join(words[:keep])
    delete_count = min(len(parts) - 1, max(1, round(len(parts) * severity)))
    deleted = set(rng.sample(range(len(parts)), delete_count))
    return " ".join(part for index, part in enumerate(parts) if index not in deleted)


def conclusion_flip(text: str, rng: random.Random, severity: float) -> str:
    del rng, severity
    for pattern, replacement in VERDICT_SWAPS:
        changed, count = pattern.subn(replacement, text, count=1)
        if count:
            return changed
    return text.rstrip() + " Therefore, the opposite verdict should be accepted."


def unsupported_claim(text: str, rng: random.Random, severity: float) -> str:
    count = max(1, round(3 * severity))
    additions = rng.sample(UNSUPPORTED_CLAIMS, min(count, len(UNSUPPORTED_CLAIMS)))
    return text.rstrip() + " " + " ".join(additions)


def contradiction(text: str, rng: random.Random, severity: float) -> str:
    del severity
    parts = sentences(text)
    anchor = rng.choice(parts) if parts else text
    return text.rstrip() + f" However, the preceding evidence should be rejected: {anchor}"


Perturber = Callable[[str, random.Random, float], str]
PERTURBERS: dict[str, Perturber] = {
    "evidence_deletion": evidence_deletion,
    "conclusion_flip": conclusion_flip,
    "unsupported_claim": unsupported_claim,
    "contradiction": contradiction,
}


def perturb_records(
    records: list[dict],
    perturbation: str,
    severity: float,
    seed: int,
    target_rate: float = 1.0,
) -> list[dict]:
    if perturbation not in {*PERTURBERS, "irrelevant"}:
        raise ValueError(f"Unknown perturbation: {perturbation}")
    if not 0.0 <= severity <= 1.0 or not 0.0 <= target_rate <= 1.0:
        raise ValueError("severity and target_rate must be between 0 and 1")

    rng = random.Random(seed)
    output = copy.deepcopy(records)
    rationales = [
        record[field]
        for record in records
        for field in ("rationale_1", "rationale_2")
        if record.get(field)
    ]

    for record in output:
        record["perturbation"] = perturbation
        record["severity"] = severity
        record["corrupted_fields"] = []
        for field in ("rationale_1", "rationale_2"):
            original = record.get(field, "")
            if not original or rng.random() > target_rate:
                continue
            if perturbation == "irrelevant":
                candidates = [candidate for candidate in rationales if candidate != original]
                changed = rng.choice(candidates) if candidates else original
            else:
                changed = PERTURBERS[perturbation](original, rng, severity)
            record[field] = changed
            if changed != original:
                record["corrupted_fields"].append(field)
    return output
