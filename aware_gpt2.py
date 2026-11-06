"""Structured verified answers. Rejections are statuses, never pseudo-answer strings."""

from config import Config
from experiment import evaluate_sample


def hallucination_aware_gpt2_answer(question, paragraphs, backend=None, config=None):
    if backend is None:
        from runtime import default_backend

        backend = default_backend()
    config = config or getattr(backend, "config", Config())
    sample = {
        "id": "interactive",
        "question": question,
        "answer": "unused",
        "context": {
            "title": [str(i) for i in range(len(paragraphs))],
            "sentences": [[p] for p in paragraphs],
        },
    }
    result = evaluate_sample(sample, backend, config)["systems"]["verified_gpt2"]
    result.pop("metrics", None)  # Interactive callers did not supply a reference answer.
    return result
