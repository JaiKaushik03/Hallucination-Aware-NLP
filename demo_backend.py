"""A deterministic test double, explicitly selected with --demo. Not a model."""

import re


class DemoBackend:
    def similarities(self, question, texts):
        q = set(re.findall(r"\w+", question.lower()))
        return [len(q & set(re.findall(r"\w+", p.lower()))) / max(len(q), 1) for p in texts]

    def generate(self, question, paragraphs):
        # Predicts only synthetic fixture markers. Never silently substitutes for real models.
        match = re.search(r"\[ANSWER: ([^\]]+)\]", " ".join(paragraphs))
        return match.group(1) if match else "unverified demo guess"

    def extract(self, question, contexts):
        results = []
        for context in contexts:
            match = re.search(r"\[ANSWER: ([^\]]+)\]", context)
            results.append(
                {
                    "answer": match.group(1) if match else "",
                    "score": 0.95 if match else 0.0,
                    "context": context,
                }
            )
        return results

    def contradiction(self, first, second):
        return 0.99 if "[CONFLICT]" in first + second else 0.01
