"""Reproducible evaluation CLI. Importing this module does not start a run."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import time
from config import Config
from data_loader import load_data
from experiment import evaluate_sample
from reporting import atomic_json, append_checkpoint, read_checkpoint, write_reports

ROOT = Path(__file__).resolve().parent


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(ROOT.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_metadata(samples, config, *, demo, split, input_file):
    versions = {}
    for name in ("torch", "transformers", "datasets", "openpyxl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    fingerprint = hashlib.sha256(
        json.dumps(samples, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "mode": "synthetic_demo" if demo else "real_models",
        "config": config.to_dict(),
        "source_hash": source_hash(),
        "sample_ids": [s["id"] for s in samples],
        "sample_fingerprint": fingerprint,
        "split": "local_file" if input_file else split,
        "python": platform.python_version(),
        "packages": versions,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def prepare_run(output, metadata, *, resume):
    output = Path(output)
    marker = output / "metadata.json"
    if resume:
        if not marker.exists():
            raise ValueError("--resume needs an existing run with metadata.json")
        previous = json.loads(marker.read_text(encoding="utf-8"))
        for key in (
            "schema_version",
            "mode",
            "config",
            "source_hash",
            "sample_ids",
            "sample_fingerprint",
            "split",
            "python",
            "packages",
        ):
            if previous.get(key) != metadata.get(key):
                raise ValueError(f"Cannot resume: {key} changed. Use a new output directory.")
        return previous, read_checkpoint(output / "results.jsonl", set(metadata["sample_ids"]))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty. Use --resume or choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, metadata)
    return metadata, []


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate answer accuracy, abstention and errors; not an automatic hallucination benchmark."
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--qa-threshold", type=float, default=0.15)
    parser.add_argument("--contradiction-threshold", type=float, default=0.90)
    parser.add_argument("--fallback", choices=["abstain", "anchor"], default="abstain")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--input", type=Path, help="HotpotQA-shaped local JSON/JSONL")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--xlsx", action="store_true", help="Also export a supplementary Excel workbook"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Explicit synthetic test double, not a real-model evaluation",
    )
    return parser.parse_args(argv)


def run(argv=None):
    args = parse_args(argv)
    config = Config(
        seed=args.seed,
        device=args.device,
        top_k=args.top_k,
        qa_threshold=args.qa_threshold,
        contradiction_threshold=args.contradiction_threshold,
        fallback=args.fallback,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        threads=args.threads,
    )
    if args.resume and not args.output:
        raise ValueError("--resume requires --output pointing to the original run")
    input_file = args.input
    if args.demo and input_file is None:
        input_file = ROOT / "examples" / "demo.json"
        if args.sample_size == 200:
            args.sample_size = 4
    samples = load_data(
        args.sample_size,
        split=args.split,
        seed=args.seed,
        input_file=input_file,
        revision=config.dataset_revision,
    )
    output = args.output or ROOT / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    metadata = make_metadata(
        samples, config, demo=args.demo, split=args.split, input_file=input_file
    )
    metadata, records = prepare_run(output, metadata, resume=args.resume)
    done = {r["sample_id"] for r in records}
    if args.demo:
        from demo_backend import DemoBackend

        backend = DemoBackend()
        print("SYNTHETIC DEMO: no model inference; these scores are not research results.")
    else:
        from models import ModelBackend

        backend = ModelBackend(config)
        # Fail clearly before running if required libraries/device are unavailable.
        import transformers  # noqa: F401

        print(
            f"Real-model evaluation on {backend.device}. Model downloads may take several minutes."
        )
    interrupted = False
    try:
        for index, sample in enumerate(samples, 1):
            if sample["id"] in done:
                continue
            started = time.monotonic()
            record = evaluate_sample(sample, backend, config, fail_fast=args.fail_fast)
            record["duration_seconds"] = round(time.monotonic() - started, 4)
            append_checkpoint(output / "results.jsonl", record)
            records.append(record)
            print(
                f"[{index}/{len(samples)}] {sample['id']} "
                + " ".join(
                    f"{name}={result['status']}" for name, result in record["systems"].items()
                ),
                flush=True,
            )
    except KeyboardInterrupt:
        interrupted = True
        print(
            "Interrupted. Completed questions are saved; resume using the same arguments and --resume.",
            file=sys.stderr,
        )
    if not records:
        return 130 if interrupted else 1
    # Restore selected dataset order when resuming.
    order = {s["id"]: i for i, s in enumerate(samples)}
    records.sort(key=lambda r: order[r["sample_id"]])
    summary = write_reports(output, records, metadata, xlsx=args.xlsx)
    print(f"\nSaved {len(records)}/{len(samples)} questions to {output.resolve()}")
    for name, metrics in summary["overall"].items():
        print(
            f"{name:20} EM={metrics['exact_match']:.2%} F1={metrics['token_f1']:.2%} coverage={metrics['coverage']:.2%} abstain={metrics['abstentions']} errors={metrics['errors']}"
        )
    if interrupted:
        return 130
    return 2 if any(m["errors"] for m in summary["overall"].values()) else 0


def main():
    try:
        return run()
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(
            f"Error: {error}\nSee README.md for setup, cache and resume instructions.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
