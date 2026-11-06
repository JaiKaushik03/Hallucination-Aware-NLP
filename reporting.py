"""Durable JSONL checkpoints and safe, secondary tabular exports."""

import csv
import json
import os
import re
from pathlib import Path
from metrics import summarize


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_checkpoint(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_checkpoint(path, allowed_ids):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    records, seen, offset = [], set(), 0
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i == len(lines) - 1 and not line.endswith(b"\n"):
                # A killed process can leave just its final write incomplete.
                with path.open("r+b") as handle:
                    handle.truncate(offset)
                break
            raise ValueError(
                "Invalid checkpoint data; refusing to discard complete lines"
            ) from None
        identifier = record.get("sample_id")
        if identifier not in allowed_ids or identifier in seen:
            raise ValueError("Checkpoint contains an unexpected or duplicate sample ID")
        if set(record.get("systems", {})) != {"baseline_gpt2", "verified_gpt2", "extractive_qa"}:
            raise ValueError("Checkpoint system schema does not match this version")
        seen.add(identifier)
        records.append(record)
        offset += len(line)
    if records and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return records


def safe_cell(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def excel_cell(value):
    if isinstance(value, str):
        value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "\ufffd", value)[:32000]
    return safe_cell(value)


def flat_rows(records):
    for record in records:
        for system, result in record["systems"].items():
            yield {
                "sample_id": record["sample_id"],
                "question": record["question"],
                "reference_answer": record["gold_answer"],
                "system": system,
                "prediction": result["answer"],
                "status": result["status"],
                "reason": result["reason"],
                "confidence": result["confidence"],
                "fallback_used": result["fallback_used"],
                **result["metrics"],
            }


def write_reports(output_dir, records, metadata, *, xlsx=False):
    output_dir = Path(output_dir)
    overall = summarize(records)
    groups = {}
    for answer_type in sorted({r["gold_answer_type"] for r in records}):
        groups[answer_type] = summarize(
            [r for r in records if r["gold_answer_type"] == answer_type]
        )
    selected = len(metadata.get("sample_ids", records))
    result = {
        "selected_questions": selected,
        "completed_questions": len(records),
        "complete": len(records) == selected,
        "schema_version": 1,
        "mode": metadata["mode"],
        "metrics_are_fractions": True,
        "hallucination_rate": None,
        "note": "Answer accuracy and verifier rejection are not hallucination measurements. Demo outputs are synthetic tests, not benchmark results.",
        "overall": overall,
        "by_reference_answer_type": groups,
    }
    atomic_json(output_dir / "summary.json", result)
    rows = list(flat_rows(records))
    with (output_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: safe_cell(v) for k, v in row.items()} for row in rows)
    if xlsx:
        # Optional application export. JSONL remains the authoritative, lossless evidence log.
        from openpyxl import Workbook

        book = Workbook()
        sheet = book.active
        sheet.title = "Results"
        sheet.append(list(rows[0]))
        for row in rows:
            sheet.append([excel_cell(v) for v in row.values()])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        summary = book.create_sheet("Summary")
        summary.append(["Mode", metadata["mode"]])
        summary.append(["Note", result["note"]])
        summary.append(
            [
                "System",
                "Questions",
                "EM (fraction)",
                "F1 (fraction)",
                "Coverage",
                "Abstentions",
                "Errors",
            ]
        )
        for system, metrics in overall.items():
            summary.append(
                [
                    system,
                    metrics["questions"],
                    metrics["exact_match"],
                    metrics["token_f1"],
                    metrics["coverage"],
                    metrics["abstentions"],
                    metrics["errors"],
                ]
            )
        provenance = book.create_sheet("Provenance")
        for key, value in metadata.items():
            provenance.append([excel_cell(key), excel_cell(json.dumps(value, ensure_ascii=False))])
        temporary = output_dir / "evaluation.partial.xlsx"
        book.save(temporary)
        os.replace(temporary, output_dir / "evaluation.xlsx")
    return result
