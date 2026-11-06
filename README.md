# Hallucination-Aware NLP

The corrected project is provided as [Hallucination-Aware-NLP-fixed.zip](Hallucination-Aware-NLP-fixed.zip).

Download and extract the ZIP, then read the included `README.md` and `VALIDATION.md`. The archive includes corrected source, pinned dependencies and model revisions, regression tests, verification records, and clearly archived original results.

## Offline demo

Requires Python 3.11–3.13. From the extracted `Hallucination-Aware-NLP-fixed` folder:

```sh
python main.py --demo
```

The demo uses synthetic fixtures and needs no model downloads. For real inference, follow the included dependency installation and evaluation instructions.

Corrections cover answer scoring, evidence checks, confidence handling, CPU support, generation budgets, reproducible sampling, and resumable results. Verification included 56 tests and real CPU model checks. The original performance claims are withdrawn; a new full 200-question benchmark was not run. Guarded systems explicitly abstain on detected boolean questions.

Original project attribution is preserved in the included documentation.

