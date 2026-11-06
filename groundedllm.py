"""Extractive-only convenience API; this path never loads GPT-2."""

from config import Config
from nli_checker import find_contradictions
from similarity import rank_paragraphs
from verification import (
    Answer,
    build_candidate_contexts,
    select_best_span,
    validate_anchor,
    is_boolean_question,
)


def hallucination_aware_answer(question, paragraphs, backend=None, config=None):
    if backend is None:
        from runtime import default_backend

        backend = default_backend()
    config = config or getattr(backend, "config", Config())
    if is_boolean_question(question):
        return Answer(reason="boolean_question_requires_non_extractive_reasoning").to_dict()
    ranked = rank_paragraphs(question, paragraphs, backend, config.top_k)
    texts = [p["text"] for p in ranked]
    if not texts:
        return Answer(reason="empty_context").to_dict()
    contexts = build_candidate_contexts(question, texts, backend)
    span = select_best_span(backend.extract(question, contexts))
    conflicts = find_contradictions(texts, backend, config.contradiction_threshold)
    return validate_anchor(question, span, conflicts, config.qa_threshold).to_dict()
