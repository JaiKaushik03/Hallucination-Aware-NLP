"""Convenience API. The evaluation harness shares this candidate with its verifier."""


def generate_answer(question, paragraphs, backend=None):
    if backend is None:
        from runtime import default_backend

        backend = default_backend()
    if not any(p.strip() for p in paragraphs):
        return ""
    return backend.generate(question, paragraphs)
