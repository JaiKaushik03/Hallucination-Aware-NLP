"""HotpotQA answer EM/F1 conventions. No substring-based correctness."""

import re
import string
from collections import Counter
from math import sqrt


def normalize_text(text):
    text = "" if text is None else str(text)
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(gold, prediction):
    return bool(str(prediction or "").strip()) and normalize_text(gold) == normalize_text(
        prediction
    )


def token_f1(gold, prediction):
    gold, prediction = normalize_text(gold), normalize_text(prediction)
    if (
        gold in {"yes", "no", "noanswer"} or prediction in {"yes", "no", "noanswer"}
    ) and gold != prediction:
        return 0.0
    g, p = gold.split(), prediction.split()
    common = sum((Counter(g) & Counter(p)).values())
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def rouge_l(gold, prediction):
    """Token LCS F1; supplemental, not an official HotpotQA metric."""
    g, p = normalize_text(gold).split(), normalize_text(prediction).split()
    if not g or not p:
        return 0.0
    previous = [0] * (len(p) + 1)
    for gt in g:
        current = [0]
        for j, pt in enumerate(p, 1):
            current.append(previous[j - 1] + 1 if gt == pt else max(previous[j], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(g) + len(p))


def score(gold, prediction, status="answered"):
    if status != "answered":
        return {"exact_match": 0.0, "token_f1": 0.0, "rouge_l": 0.0}
    return {
        "exact_match": float(exact_match(gold, prediction)),
        "token_f1": token_f1(gold, prediction),
        "rouge_l": rouge_l(gold, prediction),
    }


def wilson_interval(correct, total):
    if not total:
        return None
    z = 1.959963984540054
    p = correct / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    delta = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [max(0.0, center - delta), min(1.0, center + delta)]


def summarize(records):
    if not records:
        raise ValueError("Cannot summarize an empty evaluation")
    summary = {}
    for system in records[0]["systems"]:
        items = [r["systems"][system] for r in records]
        n = len(items)
        answered = [i for i in items if i["status"] == "answered"]
        correct = sum(i["metrics"]["exact_match"] for i in items)
        summary[system] = {
            "questions": n,
            "answered": len(answered),
            "abstentions": sum(i["status"] == "abstained" for i in items),
            "errors": sum(i["status"] == "error" for i in items),
            "coverage": len(answered) / n,
            "exact_match": correct / n,
            "token_f1": sum(i["metrics"]["token_f1"] for i in items) / n,
            "rouge_l": sum(i["metrics"]["rouge_l"] for i in items) / n,
            "answered_exact_match": correct / len(answered) if answered else None,
            "exact_match_95_wilson": wilson_interval(correct, n),
            "fallbacks": sum(bool(i.get("fallback_used")) for i in items),
        }
    return summary
