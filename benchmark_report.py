"""Summarize an existing run without executing models or changing predictions."""
import argparse
import json
import math
from pathlib import Path
from reporting import read_checkpoint
from metrics import summarize


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def build_report(folder):
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    records = read_checkpoint(folder / "results.jsonl", set(metadata["sample_ids"]))
    if not records:
        raise ValueError("No completed questions; no benchmark result exists yet")
    # Keep processing/checkpoint order: the first completed question may include
    # model initialization. Shared pipeline timing is not per-system latency.
    durations = [r.get("duration_seconds") for r in records]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in durations):
        raise ValueError("Every completed question needs a finite nonnegative duration")
    return {
        "mode": metadata["mode"],
        "benchmark_is_real_models": metadata["mode"] == "real_models",
        "selected_questions": len(metadata["sample_ids"]),
        "completed_questions": len(records),
        "complete": len(records) == len(metadata["sample_ids"]),
        "source_hash": metadata["source_hash"],
        "sample_fingerprint": metadata["sample_fingerprint"],
        "seed": metadata["config"]["seed"],
        "metrics": summarize(records),
        "timing": {
            "scope": "shared retrieval/generation/verification pipeline per question",
            "first_completed_question_s": durations[0],
            "all_question_median_s": percentile(durations, .5),
            "all_question_p95_s": percentile(durations, .95),
            "excluding_first_median_s": percentile(durations[1:], .5),
            "excluding_first_p95_s": percentile(durations[1:], .95),
            "summed_question_runtime_s": sum(durations),
            "excludes": "dataset preparation and any idle time between resumed processes",
        },
        "limitations": [
            "Synthetic demo mode is a software fixture, not a model benchmark.",
            "Scores include all completed questions, including errors and abstentions.",
            "Incomplete runs are partial evidence and must not be presented as complete.",
            "Latency is shared across systems and cannot be attributed to an individual model.",
            "Accuracy and abstention do not establish a hallucination reduction rate.",
            "A small sample cannot establish broad model quality.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = build_report(args.run)
    output = args.run / "benchmark.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
