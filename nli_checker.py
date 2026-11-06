"""Contradiction checks cover every unique retrieved pair."""

from itertools import combinations


def find_contradictions(paragraphs, backend, threshold=0.90):
    conflicts = []
    for i, j in combinations(range(len(paragraphs)), 2):
        score = backend.contradiction(paragraphs[i], paragraphs[j])
        if score >= threshold:
            conflicts.append({"first": i, "second": j, "score": float(score)})
    return conflicts


def count_contradictions(paragraphs, backend=None, threshold=0.90):
    if backend is None:
        from runtime import default_backend

        backend = default_backend()
    return len(find_contradictions(paragraphs, backend, threshold))
