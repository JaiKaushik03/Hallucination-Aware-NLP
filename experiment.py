"""One shared candidate and one shared QA pass per question."""

from data_loader import get_paragraphs
from metrics import score
from nli_checker import find_contradictions
from similarity import rank_paragraphs
from verification import (
    Answer,
    build_candidate_contexts,
    select_best_span,
    validate_anchor,
    verify_candidate,
    is_boolean_question,
)

SYSTEMS = ("baseline_gpt2", "verified_gpt2", "extractive_qa")


def error_answer(error):
    return Answer(status="error", reason=f"{type(error).__name__}: {error}")


def evaluate_sample(sample, backend, config, *, fail_fast=False):
    record = {
        "sample_id": sample["id"],
        "question": sample["question"],
        "gold_answer": sample["answer"],
        "question_type": sample.get("type", "unknown"),
        "gold_answer_type": "boolean" if sample["answer"].lower() in {"yes", "no"} else "span",
        "supporting_facts": sample.get("supporting_facts", {}),
        "retrieved_evidence": [],
        "conflicts": [],
        "systems": {},
    }
    outputs = {}
    try:
        ranked = rank_paragraphs(sample["question"], get_paragraphs(sample), backend, config.top_k)
        record["retrieved_evidence"] = [
            {**p, "title": sample["context"]["title"][p["id"]]} for p in ranked
        ]
        texts = [p["text"] for p in ranked]
        if not texts:
            outputs = {name: Answer(reason="empty_context") for name in SYSTEMS}
        else:
            try:
                candidate = backend.generate(sample["question"], texts)
                outputs["baseline_gpt2"] = Answer(
                    answer=candidate,
                    status="answered" if candidate.strip() else "abstained",
                    reason="unverified_generation" if candidate.strip() else "empty_generation",
                    candidate=candidate,
                )
                record["generation_trace"] = getattr(backend, "generation_trace", {})
            except Exception as error:
                if fail_fast:
                    raise
                candidate = ""
                outputs["baseline_gpt2"] = error_answer(error)
            try:
                if is_boolean_question(sample["question"]):
                    anchor = Answer(reason="boolean_question_requires_non_extractive_reasoning")
                else:
                    contexts = build_candidate_contexts(sample["question"], texts, backend)
                    span = select_best_span(backend.extract(sample["question"], contexts))
                    conflicts = find_contradictions(texts, backend, config.contradiction_threshold)
                    record["conflicts"] = conflicts
                    record["selected_span"] = span
                    anchor = validate_anchor(
                        sample["question"], span, conflicts, config.qa_threshold
                    )
                outputs["extractive_qa"] = anchor
                outputs["verified_gpt2"] = (
                    verify_candidate(candidate, anchor, config.fallback)
                    if outputs["baseline_gpt2"].status != "error"
                    else error_answer(RuntimeError("baseline candidate generation failed"))
                )
            except Exception as error:
                if fail_fast:
                    raise
                outputs["verified_gpt2"] = error_answer(error)
                outputs["extractive_qa"] = error_answer(error)
    except Exception as error:
        if fail_fast:
            raise
        outputs = {name: error_answer(error) for name in SYSTEMS}
    for system in SYSTEMS:
        result = outputs[system].to_dict()
        result["metrics"] = score(sample["answer"], result["answer"], result["status"])
        record["systems"][system] = result
    return record
