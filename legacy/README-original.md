# Exploring Hallucination Detection in Large Language Models

**Florida International University — NLP Research Project, 2025**

**Authors:** Sanskar Lohani, Aniruddha Tayade, Jai

---

## Overview

This project investigates hallucination in autoregressive language models and evaluates whether post-generation verification pipelines can reduce it — without any model retraining. Three systems are benchmarked on 200 multi-hop questions drawn from the HotpotQA distractor setting:

| System | Description |
|--------|-------------|
| **System 1 — Baseline GPT-2** | Raw GPT-2 generation; no context grounding or verification |
| **System 2 — Hallucination-Aware LLM** | Same GPT-2 generator, constrained by a RoBERTa QA anchor + NLI + semantic similarity pipeline |
| **System 3 — Grounded LLM** | RoBERTa extractive QA as the sole answer source; pipeline guards filter low-confidence and contradictory outputs |

The central finding is that **the verification pipeline alone reduces hallucination by 28 percentage points** (89.5% → 61.5%) without touching GPT-2's weights, and a fully extractive system drives it down to 49.0%.

---

## Key Results

| Metric | Baseline GPT-2 | Hallucination-Aware LLM | Grounded LLM |
|--------|:--------------:|:-----------------------:|:------------:|
| Hallucination Rate | 89.5% | 61.5% | **49.0%** |
| Token F1 | 0.55% | 34.63% | **48.08%** |
| Exact Match | ~10.5% | ~38.5% | ~51.0% |
| ROUGE-L | low | moderate | highest |
| Rejection Rate | N/A | ~28% | ~10% |

> Full per-question results are saved to `LLM_Evaluation_Output_Final.xlsx` after running `main.py`.

---

## Architecture

### Pipeline Overview

```
Question + Context Paragraphs
          |
          v
 [Similarity Retrieval]          <-- all-MiniLM-L6-v2 (sentence-transformers)
 Top-k paragraphs ranked by
 cosine similarity to question
          |
          v
 [Candidate Context Builder]
 Single paragraphs, paragraph
 pairs, top-sentence bundles
          |
          v
 [RoBERTa QA Anchor]             <-- deepset/roberta-base-squad2
 Extractive span with confidence
          |
   +------+------+
   |             |
   v             v
[NLI Check]  [Grounding Check]   <-- facebook/bart-large-mnli
Contradiction  Answer span must
detection on   appear in context
evidence pairs
   |             |
   +------+------+
          |
          v
     System 2 only:
 [GPT-2 Constrained Generation]  <-- gpt2 (huggingface)
 Anchor-seeded prompt, 8 new
 tokens, semantic similarity guard
          |
          v
      Final Answer
```

### System Descriptions

**System 1 — Baseline GPT-2 (`normal_llm.py`)**
- Concatenates the first two retrieved paragraphs into a prompt
- GPT-2 generates freely up to 200 tokens with `temperature=0.7`
- No grounding, no verification — answer is whatever GPT-2 produces

**System 2 — Hallucination-Aware LLM (`aware_gpt2.py`)**
- Retrieves top-4 paragraphs by semantic similarity (`all-MiniLM-L6-v2`)
- Builds enriched candidate contexts (singles, pairs, sentence bundles)
- RoBERTa QA model extracts a high-confidence anchor span
- Grounding check: anchor must appear verbatim in its source context
- NLI contradiction check: high-confidence contradictions between paragraphs trigger a rejection
- GPT-2 generates a short constrained answer seeded with the anchor
- Final semantic similarity guard: falls back to anchor if GPT-2 drifts

**System 3 — Grounded LLM (`groundedllm.py`)**
- Same retrieval and context-building pipeline as System 2
- RoBERTa QA span is the direct answer — GPT-2 is not used
- Softer fallback logic: low-confidence answers pull from the first sentence of the best paragraph rather than hard-rejecting
- Only strong contradictions (BART-MNLI score ≥ 0.90) trigger `"Evidence conflict detected."`

**Supporting Modules**

| File | Role |
|------|------|
| `nli_checker.py` | BART-MNLI contradiction detection between paragraph pairs |
| `similarity.py` | Cosine similarity via `all-MiniLM-L6-v2` sentence embeddings |
| `data_loader.py` | HotpotQA dataset loader; samples distractor-setting questions |
| `main.py` | Orchestrates all three systems, computes metrics, writes Excel output |

---

## File Structure

```
.
├── main.py               # Evaluation harness — runs all three systems on 200 questions
├── normal_llm.py         # System 1: baseline GPT-2 generation
├── aware_gpt2.py         # System 2: GPT-2 constrained by verification pipeline
├── groundedllm.py        # System 3: RoBERTa extractive QA with pipeline guards
├── nli_checker.py        # NLI contradiction detection (BART-MNLI)
├── similarity.py         # Semantic similarity scoring (MiniLM)
├── data_loader.py        # HotpotQA loader and paragraph extractor
├── load_data.py          # Utility for dataset loading
└── LLM_Evaluation_Output_Final.xlsx   # Generated output (not tracked in git)
```

> The HotpotQA dataset is **not included**. It is downloaded automatically at runtime via the Hugging Face `datasets` library.

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- pip
- (Recommended) a CUDA-capable GPU — all models fall back to CPU automatically

### Install Dependencies

```bash
pip install torch transformers sentence-transformers datasets scikit-learn pandas openpyxl
```

### Model Downloads (automatic on first run)

The following models are downloaded from Hugging Face on the first execution:

| Model | Purpose |
|-------|---------|
| `gpt2` | Baseline and constrained text generation |
| `deepset/roberta-base-squad2` | Extractive QA anchor |
| `facebook/bart-large-mnli` | NLI contradiction detection |
| `all-MiniLM-L6-v2` | Semantic similarity / retrieval |

---

## How to Run

```bash
python main.py
```

This will:
1. Download and sample 200 questions from HotpotQA (distractor setting)
2. Run all three systems on each question
3. Print a live progress log with per-question confidence and similarity scores
4. Write full results to `LLM_Evaluation_Output_Final.xlsx`
5. Print a summary table to the console:

```
System                                        Exact Acc   Inclusive Acc   Rejection Rate
------------------------------------------------------------------------------------------
Normal LLM (GPT-2)                              10.50%           10.50%              N/A
Hallucination-Aware LLM (GPT-2 + Pipeline)      38.50%           38.50%           28.00%
Grounded LLM (RoBERTa + Pipeline)               51.00%           51.00%           10.50%
```

> Runtime varies by hardware. On CPU, expect ~5–10 minutes for 200 questions due to four concurrent model inferences per sample.

---

## Requirements

```
torch>=1.13
transformers>=4.30
sentence-transformers>=2.2
datasets>=2.0
scikit-learn>=1.0
pandas>=1.5
openpyxl>=3.0
```

---

## Key Findings

1. **Verification pipelines are powerful without retraining.** Wrapping GPT-2 with a retrieval + NLI + grounding pipeline cut hallucination from 89.5% to 61.5% — a 28-point improvement using only post-generation checks.

2. **Extractive QA beats generative QA on factoid tasks.** Replacing GPT-2 generation with direct RoBERTa span extraction improved Token F1 from 0.55% to 48.08%, confirming that for multi-hop factoid questions, generation capacity is less important than evidence grounding.

3. **Rejection vs. fallback trade-off matters.** System 2's hard rejection strategy (28% rejection rate) leaves many questions unanswered. System 3's softer fallback approach achieved higher coverage and lower hallucination simultaneously.

4. **Multi-hop reasoning remains challenging.** Even the best system (49.0% hallucination) struggles with questions requiring implicit reasoning chains across two or more paragraphs — a direction for future work with chain-of-thought or iterative retrieval.

---

## Citation

If you use this code or build on this work, please cite:

```
Lohani, S., Tayade, A., & Jai. (2025). Exploring Hallucination Detection in
Large Language Models. Florida International University.
```
