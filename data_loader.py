"""Lazy data loading: importing this module never downloads a dataset."""

import hashlib
import json
import random
from pathlib import Path


def validate_sample(sample):
    if (
        not isinstance(sample, dict)
        or not isinstance(sample.get("question"), str)
        or not sample["question"].strip()
    ):
        raise ValueError("Every sample needs a nonempty question")
    if not isinstance(sample.get("answer"), str) or not sample["answer"].strip():
        raise ValueError("Every sample needs a nonempty reference answer")
    context = sample.get("context", {})
    if not isinstance(context, dict):
        raise ValueError("context must contain aligned title and sentences arrays")
    sentences = context.get("sentences", [])
    titles = context.get("title", [])
    if (
        not isinstance(sentences, list)
        or not isinstance(titles, list)
        or len(sentences) != len(titles)
    ):
        raise ValueError("Context titles and sentence lists must be aligned")
    if any(
        not isinstance(s, list) or any(not isinstance(t, str) for t in s) for s in sentences
    ) or any(not isinstance(t, str) for t in titles):
        raise ValueError("Context must contain text titles and lists of text sentences")
    sample = dict(sample)
    sample["id"] = str(
        sample.get("id")
        or hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest()[:24]
    )
    return sample


def get_paragraphs(sample):
    """Preserve paragraph order; evidence IDs refer back to this order."""
    return [" ".join(s).strip() for s in sample["context"]["sentences"]]


def load_data(sample_size=200, *, split="validation", seed=42, input_file=None, revision=None):
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if input_file:
        path = Path(input_file)
        with path.open(encoding="utf-8-sig") as handle:
            raw = (
                json.load(handle)
                if path.suffix.lower() == ".json"
                else [json.loads(line) for line in handle if line.strip()]
            )
        if not isinstance(raw, list):
            raise ValueError("Input must be a JSON list or JSONL records")
        raw = [validate_sample(s) for s in raw]
        if len(raw) < sample_size:
            raise ValueError(f"Requested {sample_size} questions; input has {len(raw)}")
        random.Random(seed).shuffle(raw)
        selected = raw[:sample_size]
    else:
        if split not in {"train", "validation"}:
            raise ValueError("HotpotQA distractor supports train or validation")
        from datasets import load_dataset
        from config import Config

        revision = revision or Config().dataset_revision

        dataset = load_dataset(
            "hotpotqa/hotpot_qa",
            "distractor",
            split=split,
            revision=revision,
            cache_dir=str(Path(__file__).resolve().parent / ".cache" / "datasets"),
        )
        # The datasets library can fall back to a different cached revision offline.
        # Refuse that fallback instead of labeling it as the configured experiment.
        if len(revision) == 40 and any(
            revision not in Path(item["filename"]).parts for item in dataset.cache_files
        ):
            raise RuntimeError("Cached dataset revision does not match the pinned revision")
        if not dataset.cache_files:
            raise RuntimeError("Cannot verify the provenance of the downloaded dataset cache")
        if len(dataset) < sample_size:
            raise ValueError("Requested more samples than the dataset contains")
        selected = [
            validate_sample(s) for s in dataset.shuffle(seed=seed).select(range(sample_size))
        ]
    ids = [s["id"] for s in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs are not allowed")
    return selected
