# Validation of the corrected project

Validated on November 6, 2026, on Windows with Python 3.13.7, CPU PyTorch 2.9.1, Transformers 4.57.6, datasets 4.4.1 and openpyxl 3.1.5. The exact resolved package set is in `requirements-lock.txt`.

- **56 regression/adapter tests passed**, including scoring false positives, strict yes/no F1, confidence/span integrity, all six evidence pairs, final-answer checks, GPU fallback configuration, oversized prompts, model-free imports, stage failure handling, checkpoint integrity, CLI resume, partial-run labeling and safe Excel export.
- Ruff lint passed; Ruff formatting checks passed; all source files compiled; `pip check` reported no incompatible installed requirements. Dependency compatibility is not a claim of a security audit.
- All four real pinned model adapters loaded and ran on CPU. Two small local QA examples completed with zero execution errors. Records are in `validation/verified-cpu/`.
- The real HotpotQA distractor dataset loaded at the pinned revision. A validation-set example completed the real end-to-end pipeline with zero execution errors. Records are in `validation/verified-hotpot/`. The guarded systems abstained on that example; this is recorded rather than hidden or counted as a correct answer.
- A separate real GPT-2 call used an oversized context; token counts reserved the generation budget inside the 1,024-token model window. See `validation/long-context-check.json`.
- The four-question **synthetic** demo and its resume check completed without duplicate result rows. Records are in `validation/verified-demo/`; these are software fixtures, not research results.
- The archived workbook was re-scored using corrected answer metrics. `legacy/audit.json` identifies its original file hash and clearly distinguishes a legacy prediction audit from a fresh experiment.

The packaged real-model run metadata includes a hash of the delivered root-level Python source. Smoke checks and tiny samples establish integration behavior, not benchmark quality or a hallucination reduction. **A new full 200-question benchmark was not run. CUDA execution was not tested on this CPU environment.**

The two guarded systems explicitly abstain on questions detected as boolean because the extractive model does not implement a yes/no reasoning head. This and the conservative verifier's coverage limits remain documented experimental limitations. Passing these checks does not guarantee factual correctness.

Model weights, virtual environments, and caches are intentionally omitted from the ZIP. Public weights download automatically when real inference is first requested on another machine; the synthetic demo and most tests can run without them.
