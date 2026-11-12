# Hallucination-Aware NLP — corrected project

## Verified demo output

This project runs in the terminal, not as a web app. On November 7, 2026, the offline CLI demo completed all four synthetic examples without execution errors. The regression suite ran 56 tests successfully, with three optional dependency tests skipped.

- [Actual generated summary](docs/screenshots/demo-summary.json)
- [Readable HTML report](docs/screenshots/demo-results.html) — download and open locally; generated directly from that summary.

**Synthetic software demo only: no model inference, benchmark accuracy, or hallucination-reduction claim.** Browser policy prevented capturing the local HTML report, so no screenshot is presented as if a real model run occurred.


An auditable experiment comparing GPT-2 answers, conservative verification, and extractive QA on HotpotQA. Based on the original project credited to Sanskar Lohani, Aniruddha Tayade, and Jai (Florida International University, 2025).

**The original performance claims are withdrawn.** The bundled workbook did not match the original README, and its permissive substring scoring could mark wrong answers as correct. Accuracy, rejection, and hallucination are different measurements. This project reports answer accuracy, coverage, abstentions, and execution errors; it does not invent a hallucination rate.

## Start here

Use **64-bit Python 3.11–3.13**. The complete dependency environment was checked on Windows with Python 3.13.7. Python 3.8 from the original README is no longer supported.

### Get and open the project

The commands below use Python 3.13 on Windows. If you installed Python 3.11 or 3.12, replace `py -3.13` with your installed version. Install Git if using the clone command.

Open PowerShell in the folder where you want to keep the project:

```powershell
git clone https://github.com/JaiKaushik03/Hallucination-Aware-NLP.git
cd Hallucination-Aware-NLP
```

Alternatively, choose **Code → Download ZIP** on GitHub, extract it, and open PowerShell in the extracted folder containing `main.py`. You can open that same folder in your editor. Run all commands below from this project folder.

This project runs in the **terminal**, rather than opening a website. Each run prints its results and the location of its output files.

### 1. Try the offline demo

No dependencies, model downloads, GPU, or API key are needed:

```powershell
py -3.13 main.py --demo
```

`--demo` explicitly selects a synthetic test double and four artificial examples. **Its scores are software checks, not research results.** The demo never silently replaces real model inference.

### 2. Install for real model evaluation

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` records the resolved Windows/Python 3.13 environment. `requirements.txt` pins the four direct runtime dependencies and can be used on other supported Python versions/platforms. Development lint/format tooling is in `requirements-dev.txt`.

### 3. Check real models on two small examples

```powershell
.\.venv\Scripts\python.exe main.py --input examples/smoke.json --sample-size 2 --device cpu
```

This downloads public model weights on first use (several GB, not included in the ZIP). The CPU path works without CUDA. `--device auto` chooses CUDA if available; explicitly requesting unavailable CUDA fails with an explanation. Models are shared and loaded only when needed. `--threads 4` and `--batch-size 4` control CPU threading and inference batches.

### 4. Run a real HotpotQA evaluation

```powershell
.\.venv\Scripts\python.exe main.py --sample-size 200 --split validation --seed 42 --device auto
```

The loader uses `hotpotqa/hotpot_qa`, the `distractor` configuration, and a pinned dataset revision. It selects a shuffled, seeded validation sample instead of the first 200 training examples. The dataset library may download both train and validation shards while preparing this configuration. Model files are cached under `.cache/models`; dataset caches include `.cache/datasets` and the Hugging Face hub cache. Set `HF_HOME` before starting Python if you want to control the hub-cache location.

Runtime depends heavily on hardware, paragraph lengths, QA candidates, and NLI windows. There is no unsupported 5–10 minute promise. Start with a small sample. Local input files must use the HotpotQA-shaped JSON/JSONL schema shown in `examples/smoke.json`; references are used only by the scorer, never by the inference code.

## What changed

- Removed import-time dataset downloads, model allocations, and benchmark execution.
- Replaced hard-coded GPU 0 with explicit CPU/auto/CUDA handling.
- Fixed generation budgets: reserve new-token space, budget context to the actual GPT-2 window, preserve the question, and decode only newly generated tokens. Greedy generation uses no ineffective temperature setting.
- Pinned Transformers 4.57.6, avoiding the removed v5 question-answering pipeline API. Pinned all four model revisions and the dataset revision to immutable commits. Added the fully resolved dependency lock.
- Batched MiniLM retrieval; ranked evidence is shared across systems. Long evidence is chunked instead of silently scoring only its prefix. No duplicated GPT-2 or QA model instances.
- Kept QA confidence attached to its selected span. Removed capitalization bonuses, arbitrary word chopping, and stale scores from the shorter-answer tie-break.
- Check all retrieved evidence pairs. The model adapter checks windows in both directions; a single above-threshold conflict reaches the guard. The old `>= 2` condition could never trigger because only one pair was evaluated.
- Removed the first-sentence fallback for low-confidence QA. These cases now abstain with a structured reason.
- Removed substring/inclusive correctness. EM and token F1 follow [official HotpotQA answer scoring](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py), including strict yes/no/noanswer handling. Supplemental ROUGE-L is explicitly labeled.
- Removed semantic similarity as proof of a generated answer. A verified answer must agree with the validated extractive anchor; changed or unsupported candidates abstain by default.
- Added a controlled candidate comparison: baseline and verifier share the **same** GPT-2 generation, not separately sampled outputs or different generation prompts. QA extraction is also shared with the extractive system.
- Added explicit statuses, error counts, coverage, conditional accuracy, answer-type breakdowns, an EM confidence interval, lossless evidence logs, checkpoint/resume, and safe CSV/optional Excel export.
- Fixed the undefined `hotpotqa` variable in `load_data.py`. Added regression and adapter tests.

## Systems and interpretation

| System | Behavior |
|---|---|
| `baseline_gpt2` | One greedy, context-grounded GPT-2 candidate with no verification |
| `verified_gpt2` | The exact same candidate, accepted only when it agrees with a sufficiently confident extractive anchor and the evidence conflict guard passes |
| `extractive_qa` | The same validated QA anchor, without a GPT-2 answer |

The default verifier is **abstention-only**. To study anchor substitution separately:

```powershell
.\.venv\Scripts\python.exe main.py --sample-size 20 --fallback anchor
```

Every substituted answer is marked `fallback_used=true`; the summary counts it. Anchor substitution is a different treatment and must not be described as verification-only improvement. Auxiliary QA, evidence selection, thresholds, and abstention all influence outcomes; this experiment does not establish that NLI alone causes an improvement.

The extractive model has no boolean classification/reasoning head. Questions detected as boolean by their wording explicitly abstain in the two guarded systems, instead of returning an unrelated name or sentence. This is a **documented task limitation**, not a solved yes/no reasoning task. Boolean and span-reference results are reported separately, while overall scores retain all questions. Wording-based detection is conservative and imperfect.

NLI and span containment are heuristic checks, not proofs of factuality, question relevance, or multi-hop reasoning. The thresholds (QA 0.15, contradiction 0.90) are configurable experimental settings, not calibrated probabilities. NLI can miss real conflicts or flag unrelated facts. GPT-2 is not instruction-tuned. Similarity retrieval can miss supporting facts. No new 200-question accuracy or hallucination-reduction claim is made until you run and review an appropriate evaluation.

## Results and resume

Each run gets a new timestamped directory under `runs/` unless you specify `--output`:

| File | Purpose |
|---|---|
| `metadata.json` | Configuration, immutable revisions, package/Python versions, code hash, sample IDs and fingerprint |
| `results.jsonl` | Authoritative per-question predictions, evidence, QA span, conflicts, generation context budget, statuses and metrics; flushed after each completed question |
| `summary.json` | All-question EM/F1/ROUGE-L, coverage, answered-only EM, abstentions, errors, fallback counts and answer-type cuts; metrics are fractions in [0,1] |
| `results.csv` | Flattened predictions for inspection; spreadsheet formula prefixes are escaped |
| `evaluation.xlsx` | Optional supplementary workbook when `--xlsx` is supplied |

Excel text cells are limited in size; full evidence is always retained in JSONL. `hallucination_rate` is deliberately null because no independent hallucination annotations are collected. Answer errors and abstentions are not relabeled as hallucinations. Execution errors remain visible and receive zero answer metrics; they are not dropped from the denominator. The program exits with code 2 if any system has execution errors.

Example:

```powershell
.\.venv\Scripts\python.exe main.py --sample-size 200 --seed 42 --output runs/my-evaluation --xlsx
# After an interruption, use identical inference/data arguments:
.\.venv\Scripts\python.exe main.py --sample-size 200 --seed 42 --output runs/my-evaluation --xlsx --resume
```

Resume checks code, configuration, environment versions, selected IDs and the full selected-data fingerprint. It refuses incompatible resumes and duplicate IDs. Only an incomplete final checkpoint write can be repaired automatically; malformed complete records are not silently discarded. Existing output directories are not overwritten. **Use one process per output directory.** Interrupted evaluations summarize completed questions only; the console prints completed/selected counts. A resumed run keeps already recorded errors; use a new directory for a fresh attempt after fixing their cause.

`--fail-fast` is useful for debugging: it raises on the first inference failure while preserving prior completed questions. `Ctrl+C` saves the completed prefix. Outputs are incremental at question granularity; an interrupted question is recomputed.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\python.exe -m pip check
```

The standard-library regression suite does not download models. Two tensor-budget adapter tests additionally use installed PyTorch (and are skipped when it is absent). Checks cover the original false-positive scoring examples, boolean scoring, all evidence pairs, confidence attachment, conservative final answers, shared generation, empty inputs, stage errors, safe exports, import side effects, and checkpoint/resume integrity. See `VALIDATION.md` for the exact checks performed on the delivered source.

## Original artifacts

`legacy/README-original.md` and `legacy/LLM_Evaluation_Output_Final.xlsx` preserve the original material for audit only. They do **not** describe this implementation's results. `legacy/audit.json` re-scores the archived predictions using corrected metrics; it is not a fresh model run.

```powershell
.\.venv\Scripts\python.exe audit_legacy.py
```

## Source layout

`config.py` — experiment settings and revisions; `models.py` — lazy model adapters; `data_loader.py` — validated seeded inputs; `similarity.py` — retrieval; `nli_checker.py` — pair checks; `verification.py` — structured guards; `experiment.py` — shared inference; `metrics.py` — reference-answer scoring; `reporting.py` — durable outputs; `main.py` — CLI.

The original convenience module names remain. Aware/grounded convenience functions now return structured result dictionaries (`answer`, `status`, `reason`, etc.) instead of encoding rejections as answer strings. The standalone extractive API never loads GPT-2. Avoid relying on the legacy string return type.

## Sources and licenses

- [HotpotQA dataset card and CC BY-SA 4.0 license](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- [HotpotQA official evaluation script](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)
- [GPT-2 model card](https://huggingface.co/openai-community/gpt2)
- [RoBERTa SQuAD2 model card](https://huggingface.co/deepset/roberta-base-squad2)
- [BART-MNLI model card](https://huggingface.co/facebook/bart-large-mnli)
- [MiniLM model card and mean-pooling implementation](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Transformers generation parameter documentation](https://huggingface.co/docs/transformers/v4.52.2/main_classes/text_generation)

Model/data licenses remain with their respective owners. Original author attribution is preserved; no license for the original project is invented here. Model weights, caches and the local virtual environment are excluded from the distribution ZIP.


## Reproducible benchmark report

```powershell
python main.py --sample-size 20 --split validation --seed 42 --device cpu --output runs/benchmark-20
python benchmark_report.py runs/benchmark-20
```

The first command runs all three comparison systems with the pinned model and dataset revisions. The second reads existing checkpoints and writes `benchmark.json`; it does not generate answers or alter predictions. It reports exact match, token F1, coverage, answered-only accuracy, abstentions, execution errors and shared per-question pipeline latency (median and p95). The first completed question and timings excluding it are shown separately because initialization can dominate the first question. These durations cannot be attributed to individual models and omit dataset preparation and idle time between resumed processes.

Missing timing data is rejected, and synthetic and partial runs are labelled explicitly. Three additional regression tests cover percentile behavior, empty runs and partial synthetic runs containing errors. The dependency-free suite executes 59 tests, skipping optional tensor tests when PyTorch is absent; skipped tests are not passes.

The GitHub Actions workflow runs regressions and an explicitly synthetic demonstration. An optional manual real-model job evaluates 20 seeded validation questions and retains its outputs. Regression success is not real-model benchmark success. Inspect mode, completed/selected counts and error counts before using any result in a resume. `requirements-benchmark-win-py312.txt` records the separate Windows/Python 3.12 benchmark environment; the older Python 3.13 lock remains available.

### Completed real-model run — November 12, 2026

The [recorded run](docs/benchmarks/seed42-20/benchmark.json) completed all 20 selected HotpotQA validation questions with seed 42 on CPU (4 threads), using the pinned revisions in [metadata](docs/benchmarks/seed42-20/metadata.json). These are fresh model outputs, not demonstration fixtures. [Per-question predictions and evidence](docs/benchmarks/seed42-20/results.jsonl) and the [full summary](docs/benchmarks/seed42-20/summary.json) are retained for inspection.

| System | All-question exact match | Token F1 | Coverage | Abstentions | Execution errors |
|---|---:|---:|---:|---:|---:|
| GPT-2 baseline | 0% | 3.00% | 100% | 0/20 | 0 |
| Verified GPT-2 | 0% | 0% | 0% | 20/20 | 0 |
| Extractive QA | 10% | 10% | 20% | 16/20 | 0 |

The verified system's zero coverage is a failure to provide useful answers on this sample, not evidence of successful hallucination prevention. Extractive QA answered 4 questions and matched 2; its 50% answered-only accuracy must be read alongside 20% coverage and 10% all-question accuracy. Twenty questions cannot establish broad model quality, and no independent hallucination annotations were collected.

The shared retrieval/generation/verification pipeline took 638.859 seconds summed across questions: median 23.031 seconds and nearest-rank p95 74.015 seconds. The first question took 160.953 seconds, including model initialization/downloads; excluding it, median and p95 remained 23.031 and 74.015 seconds. Timings omit dataset preparation and are not individual-model latencies. Python and inference-package versions are recorded in metadata; this is a local Windows development measurement, not a controlled cross-machine comparison.

The next model-quality experiment should investigate retrieval misses and verifier rejection reasons on a separate development split, then lock any changed thresholds before evaluating a larger held-out sample. This run does not establish an accuracy improvement, production readiness, or a hallucination-reduction percentage.
