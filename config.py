"""Configuration is explicit and model weights are pinned to immutable revisions."""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Config:
    seed: int = 42
    device: str = "auto"
    top_k: int = 4
    qa_threshold: float = 0.15
    contradiction_threshold: float = 0.90
    fallback: str = "abstain"
    max_new_tokens: int = 32
    batch_size: int = 4
    threads: int = 4
    generator_model: str = "openai-community/gpt2"
    generator_revision: str = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    qa_model: str = "deepset/roberta-base-squad2"
    qa_revision: str = "adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
    nli_model: str = "facebook/bart-large-mnli"
    nli_revision: str = "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    dataset_name: str = "hotpotqa/hotpot_qa"
    dataset_revision: str = "1908d6afbbead072334abe2965f91bd2709910ab"

    def __post_init__(self):
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if not 1 <= self.top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        if not 0 <= self.qa_threshold <= 1 or not 0 <= self.contradiction_threshold <= 1:
            raise ValueError("confidence thresholds must be between 0 and 1")
        if self.fallback not in {"abstain", "anchor"}:
            raise ValueError("fallback must be abstain or anchor")
        if not 1 <= self.max_new_tokens <= 128 or self.batch_size < 1 or self.threads < 1:
            raise ValueError("invalid token budget or batch size")

    def to_dict(self):
        return asdict(self)
