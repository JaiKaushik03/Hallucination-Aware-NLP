"""Re-score archived predictions without presenting them as a fresh experiment."""

import argparse
import hashlib
from pathlib import Path
from metrics import score
from reporting import atomic_json


def audit(path):
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True, data_only=True)
    try:
        iterator = iter(book["Results"].values)
        headers = next(iterator)
        rows = [dict(zip(headers, row)) for row in iterator]
        if not rows:
            raise ValueError("Legacy workbook contains no results")
        systems = {}
        for name, column, rejected in [
            ("baseline_gpt2", "Normal LLM Answer", None),
            ("verified_gpt2", "Aware GPT2 Answer", "Aware GPT2 Rejected"),
            ("extractive_qa", "Aware LLM Answer", "Aware Rejected"),
        ]:
            values = [
                score(
                    row["HotpotQA Correct Answer"],
                    row[column] or "",
                    "abstained" if rejected and row[rejected] else "answered",
                )
                for row in rows
            ]
            systems[name] = {
                metric: sum(value[metric] for value in values) / len(values) for metric in values[0]
            }
            systems[name]["stored_rejections"] = (
                sum(bool(row[rejected]) for row in rows) if rejected else None
            )
        original_summary = list(book["Summary"].values)
        return {
            "input_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "questions": len(rows),
            "kind": "audit_of_original_predictions_not_a_new_run",
            "metrics_are_fractions": True,
            "recomputed": systems,
            "original_summary": original_summary,
            "note": "These predictions came from an unknown earlier implementation. Their accuracy cannot be interpreted as hallucination rate or evidence for the corrected pipeline. The original README confused inclusive matching with exact accuracy and reported rejections not present in this workbook.",
        }
    finally:
        book.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "legacy/LLM_Evaluation_Output_Final.xlsx",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "legacy/audit.json")
    args = parser.parse_args()
    atomic_json(args.output, audit(args.input))
    print(f"Legacy audit saved to {args.output}")


if __name__ == "__main__":
    main()
