"""Batch retrieval. Similarity is not used as proof of factual correctness."""


def rank_paragraphs(question, paragraphs, backend, k=4):
    indexed = [(i, p) for i, p in enumerate(paragraphs) if p.strip()]
    if not indexed:
        return []
    scores = backend.similarities(question, [p for _, p in indexed])
    if len(scores) != len(indexed):
        raise RuntimeError("Embedding score count does not match evidence")
    ranked = [{"id": i, "text": p, "similarity": float(s)} for (i, p), s in zip(indexed, scores)]
    return sorted(ranked, key=lambda r: (-r["similarity"], r["id"]))[:k]


def compute_similarity(question, paragraphs, backend=None):
    if not paragraphs:
        return 0.0
    if backend is None:
        from runtime import default_backend

        backend = default_backend()
    return max(backend.similarities(question, paragraphs))
