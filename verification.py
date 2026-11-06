"""Conservative guards. A passing guard is not a factuality guarantee."""

import re
from dataclasses import dataclass, field, asdict
from itertools import combinations
from metrics import normalize_text


@dataclass
class Answer:
    answer: str = ""
    status: str = "abstained"
    reason: str = "insufficient_evidence"
    confidence: float | None = None
    evidence: list = field(default_factory=list)
    fallback_used: bool = False
    candidate: str = ""

    def to_dict(self):
        return asdict(self)


def contains_span(answer, context):
    a, c = normalize_text(answer).split(), normalize_text(context).split()
    return bool(a) and any(c[i : i + len(a)] == a for i in range(len(c) - len(a) + 1))


def is_boolean_question(question):
    # The extractive QA model has no boolean classification head. Never use gold labels here.
    return bool(
        re.match(
            r"^\s*(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would|should|must)\b",
            question,
            re.I,
        )
    )


def build_candidate_contexts(question, top_paragraphs, backend):
    contexts = list(top_paragraphs)
    contexts.extend(a + "\n" + b for a, b in combinations(top_paragraphs, 2))
    sentences = [
        s.strip() for p in top_paragraphs for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()
    ]
    if sentences:
        scores = backend.similarities(question, sentences)
        if len(scores) != len(sentences):
            raise RuntimeError("Sentence score count mismatch")
        ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
        for count in (4, 6):
            if len(ranked) >= count:
                contexts.append(" ".join(s for s, _ in ranked[:count]))
    return list(dict.fromkeys(c.strip() for c in contexts if c.strip()))


def select_best_span(predictions):
    valid = [
        p
        for p in predictions
        if p["answer"].strip() and 0 <= p["score"] <= 1 and contains_span(p["answer"], p["context"])
    ]
    # A shorter answer only wins an actual tie. The stored score always belongs to its span.
    return max(valid, key=lambda p: (p["score"], -len(p["answer"].split()))) if valid else None


def validate_anchor(question, span, conflicts, threshold):
    if is_boolean_question(question):
        return Answer(reason="boolean_question_requires_non_extractive_reasoning")
    if not span or span["score"] < threshold:
        return Answer(reason="low_qa_confidence", confidence=span["score"] if span else None)
    if conflicts:
        return Answer(reason="evidence_conflict", confidence=span["score"])
    if not contains_span(span["answer"], span["context"]):
        return Answer(reason="anchor_not_in_evidence", confidence=span["score"])
    return Answer(
        answer=span["answer"],
        status="answered",
        reason="extractive_anchor",
        confidence=span["score"],
        evidence=[span["context"]],
    )


def verify_candidate(candidate, anchor, fallback="abstain"):
    if anchor.status != "answered":
        return Answer(reason=anchor.reason, confidence=anchor.confidence, candidate=candidate)
    # Similarity is insufficient: the final answer must agree with the supported anchor.
    if normalize_text(candidate) and normalize_text(candidate) == normalize_text(anchor.answer):
        return Answer(
            answer=candidate.strip(),
            status="answered",
            reason="candidate_matches_anchor",
            confidence=anchor.confidence,
            evidence=anchor.evidence,
            candidate=candidate,
        )
    if fallback == "anchor":
        return Answer(
            answer=anchor.answer,
            status="answered",
            reason="validated_anchor_fallback",
            confidence=anchor.confidence,
            evidence=anchor.evidence,
            fallback_used=True,
            candidate=candidate,
        )
    return Answer(
        reason="candidate_not_supported_by_anchor",
        confidence=anchor.confidence,
        candidate=candidate,
    )
