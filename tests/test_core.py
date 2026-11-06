import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from config import Config
from data_loader import load_data, validate_sample, get_paragraphs
from demo_backend import DemoBackend
from experiment import evaluate_sample
from groundedllm import hallucination_aware_answer
from metrics import exact_match, normalize_text, token_f1, rouge_l, score, summarize
from nli_checker import count_contradictions, find_contradictions
from similarity import compute_similarity, rank_paragraphs
from verification import Answer, contains_span, select_best_span, validate_anchor, verify_candidate
from reporting import append_checkpoint, read_checkpoint, safe_cell, write_reports
from main import make_metadata, prepare_run

ROOT = Path(__file__).resolve().parents[1]


def sample():
    return {
        "id": "test-one",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "context": {
            "title": ["France", "Europe"],
            "sentences": [["The capital is [ANSWER: Paris]."], ["France is in Europe."]],
        },
    }


class MetricsTests(unittest.TestCase):
    def test_normalization_matches_hotpot_conventions(self):
        self.assertEqual(normalize_text("The U.S., a place!"), "us place")

    def test_no_substring_credit(self):
        for gold, pred in [
            ("no", "Not enough evidence."),
            ("yes", "yesterday"),
            ("France", "France is not the answer"),
        ]:
            with self.subTest(gold=gold):
                self.assertFalse(exact_match(gold, pred))

    def test_boolean_f1_requires_exact_match(self):
        self.assertEqual(token_f1("yes", "yes no"), 0)
        self.assertEqual(token_f1("no", "The New Pornographers is a Canadian"), 0)
        self.assertEqual(token_f1("no", "No."), 1)

    def test_f1_counts_repeated_tokens(self):
        self.assertAlmostEqual(token_f1("cat cat dog", "cat dog"), 0.8)

    def test_empty_metrics(self):
        self.assertEqual(token_f1("Paris", ""), 0)
        self.assertFalse(exact_match("", ""))

    def test_nonempty_article_answer_uses_official_em_normalization(self):
        self.assertTrue(exact_match("A", "A"))
        self.assertFalse(exact_match("A", ""))

    def test_rouge_uses_token_order(self):
        self.assertEqual(rouge_l("red blue", "red blue"), 1)
        self.assertEqual(rouge_l("red blue", "blue red"), 0.5)

    def test_abstention_does_not_match_gold_no(self):
        self.assertEqual(score("no", "no", "abstained")["exact_match"], 0)

    def test_summarize_coverage_denominators_and_errors(self):
        a = evaluate_sample(sample(), DemoBackend(), Config())
        b = copy.deepcopy(a)
        b["systems"]["baseline_gpt2"].update(status="error", metrics=score("Paris", "", "error"))
        result = summarize([a, b])["baseline_gpt2"]
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["exact_match"], 0.5)
        self.assertEqual(result["answered_exact_match"], 1)
        self.assertEqual(result["errors"], 1)

    def test_empty_summary_fails(self):
        with self.assertRaises(ValueError):
            summarize([])


class DataTests(unittest.TestCase):
    def test_seeded_local_selection_is_stable(self):
        path = ROOT / "examples/demo.json"
        self.assertEqual(
            load_data(3, input_file=path, seed=9), load_data(3, input_file=path, seed=9)
        )

    def test_context_alignment_is_validated(self):
        s = sample()
        s["context"]["title"] = []
        with self.assertRaises(ValueError):
            validate_sample(s)

    def test_empty_context_is_allowed_and_preserved(self):
        s = sample()
        s["context"] = {"title": [], "sentences": []}
        self.assertEqual(get_paragraphs(validate_sample(s)), [])

    def test_bad_question_and_reference_are_rejected(self):
        for key in ["question", "answer"]:
            s = sample()
            s[key] = ""
            with self.assertRaises(ValueError):
                validate_sample(s)

    def test_sample_size_limits(self):
        for n in [0, 99]:
            with self.assertRaises(ValueError):
                load_data(n, input_file=ROOT / "examples/demo.json")

    def test_duplicate_ids_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.json"
            path.write_text(json.dumps([sample(), sample()]))
            with self.assertRaises(ValueError):
                load_data(2, input_file=path)

    def test_wrong_offline_cached_revision_is_rejected(self):
        dataset = SimpleNamespace(cache_files=[{"filename": "/cache/wrong-revision/data.arrow"}])
        module = SimpleNamespace(load_dataset=lambda *a, **k: dataset)
        with patch.dict(sys.modules, {"datasets": module}):
            with self.assertRaisesRegex(RuntimeError, "revision"):
                load_data(1, revision="f" * 40)


class GuardTests(unittest.TestCase):
    def test_substring_is_not_span_grounding(self):
        self.assertFalse(contains_span("us", "Russia is large"))
        self.assertFalse(contains_span("", "France"))
        self.assertTrue(contains_span("Paris", "Paris is in France."))

    def test_selected_score_belongs_to_selected_span(self):
        predictions = [
            {"answer": "alpha beta gamma", "score": 0.16, "context": "alpha beta gamma"},
            {"answer": "delta", "score": 0.14, "context": "delta"},
        ]
        chosen = select_best_span(predictions)
        self.assertEqual(chosen["answer"], "alpha beta gamma")
        self.assertEqual(chosen["score"], 0.16)

    def test_equal_score_shorter_span_uses_own_score(self):
        predictions = [
            {"answer": "alpha beta", "score": 0.5, "context": "alpha beta"},
            {"answer": "delta", "score": 0.5, "context": "delta"},
        ]
        self.assertEqual(select_best_span(predictions)["answer"], "delta")

    def test_capitalization_does_not_inflate_confidence(self):
        span = select_best_span([{"answer": "Paris", "score": 0.149, "context": "Paris"}])
        self.assertEqual(validate_anchor("What city?", span, [], 0.15).status, "abstained")

    def test_low_confidence_abstains_instead_of_first_sentence(self):
        span = {"answer": "Paris", "score": 0.01, "context": "Paris"}
        self.assertEqual(validate_anchor("What city?", span, [], 0.15).answer, "")

    def test_one_conflict_is_enough(self):
        anchor = validate_anchor(
            "What city?",
            {"answer": "Paris", "score": 0.9, "context": "Paris"},
            [{"score": 0.99}],
            0.15,
        )
        self.assertEqual(anchor.reason, "evidence_conflict")

    def test_all_six_paragraph_pairs_are_checked(self):
        backend = Mock()
        backend.contradiction.return_value = 0.99
        self.assertEqual(count_contradictions(["a", "b", "c", "d"], backend), 6)
        self.assertEqual(backend.contradiction.call_count, 6)

    def test_conflicts_beyond_first_two_are_found(self):
        backend = Mock()
        backend.contradiction.side_effect = lambda a, b: 0.99 if (a, b) == ("c", "d") else 0
        self.assertEqual(find_contradictions(["a", "b", "c", "d"], backend)[0]["first"], 2)

    def test_unsupported_generation_is_rejected(self):
        anchor = Answer(
            "Paris", "answered", "extractive_anchor", 0.9, ["Paris is the capital of France."]
        )
        self.assertEqual(verify_candidate("Berlin", anchor).status, "abstained")
        self.assertEqual(verify_candidate("Paris is not the answer", anchor).status, "abstained")

    def test_anchor_fallback_is_explicit(self):
        anchor = Answer(
            "Paris", "answered", "extractive_anchor", 0.9, ["Paris is the capital of France."]
        )
        result = verify_candidate("Berlin", anchor, "anchor")
        self.assertEqual(result.answer, "Paris")
        self.assertTrue(result.fallback_used)

    def test_generated_correct_equivalent_is_accepted(self):
        anchor = Answer(
            "Paris", "answered", "extractive_anchor", 0.9, ["Paris is the capital of France."]
        )
        self.assertEqual(verify_candidate("Paris.", anchor).status, "answered")

    def test_boolean_questions_abstain_explicitly(self):
        result = hallucination_aware_answer(
            "Are both bands American?", ["A band is Canadian."], DemoBackend()
        )
        self.assertEqual(result["status"], "abstained")
        self.assertIn("boolean", result["reason"])

    def test_empty_similarity_is_safe(self):
        self.assertEqual(compute_similarity("question", [], Mock()), 0)


class PipelineTests(unittest.TestCase):
    def test_generation_and_qa_are_shared_once(self):
        backend = Mock(wraps=DemoBackend())
        result = evaluate_sample(sample(), backend, Config())
        self.assertEqual(backend.generate.call_count, 1)
        self.assertEqual(backend.extract.call_count, 1)
        self.assertEqual(
            result["systems"]["baseline_gpt2"]["candidate"],
            result["systems"]["verified_gpt2"]["candidate"],
        )

    def test_gold_answer_cannot_change_predictions(self):
        first = sample()
        second = sample()
        second["answer"] = "Berlin"
        a = evaluate_sample(first, DemoBackend(), Config())
        b = evaluate_sample(second, DemoBackend(), Config())
        self.assertEqual(
            [v["answer"] for v in a["systems"].values()],
            [v["answer"] for v in b["systems"].values()],
        )

    def test_no_evidence_does_not_initialize_models(self):
        backend = Mock()
        s = sample()
        s["context"] = {"title": [], "sentences": []}
        result = evaluate_sample(s, backend, Config())
        self.assertTrue(all(r["status"] == "abstained" for r in result["systems"].values()))
        backend.generate.assert_not_called()

    def test_qa_error_preserves_baseline_and_records_errors(self):
        backend = Mock(wraps=DemoBackend())
        backend.extract.side_effect = RuntimeError("QA failure")
        result = evaluate_sample(sample(), backend, Config())
        self.assertEqual(result["systems"]["baseline_gpt2"]["status"], "answered")
        self.assertEqual(result["systems"]["extractive_qa"]["status"], "error")

    def test_generation_error_does_not_prevent_qa(self):
        backend = Mock(wraps=DemoBackend())
        backend.generate.side_effect = RuntimeError("generation failure")
        result = evaluate_sample(sample(), backend, Config())
        self.assertEqual(result["systems"]["baseline_gpt2"]["status"], "error")
        self.assertEqual(result["systems"]["extractive_qa"]["status"], "answered")

    def test_fail_fast_propagates_model_error(self):
        backend = Mock()
        backend.similarities.side_effect = RuntimeError("broken")
        with self.assertRaises(RuntimeError):
            evaluate_sample(sample(), backend, Config(), fail_fast=True)

    def test_extractive_api_does_not_generate(self):
        backend = Mock(wraps=DemoBackend())
        self.assertEqual(
            hallucination_aware_answer("What is the capital?", ["[ANSWER: Paris]"], backend)[
                "answer"
            ],
            "Paris",
        )
        backend.generate.assert_not_called()

    def test_rank_is_stable_and_uses_original_ids(self):
        backend = Mock()
        backend.similarities.return_value = [0.5, 0.5]
        result = rank_paragraphs("question", ["", "alpha", "beta"], backend)
        self.assertEqual([r["id"] for r in result], [1, 2])


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.record = evaluate_sample(sample(), DemoBackend(), Config())

    def test_checkpoint_round_trip(self):
        path = self.root / "results.jsonl"
        append_checkpoint(path, self.record)
        self.assertEqual(read_checkpoint(path, {"test-one"}), [self.record])

    def test_incomplete_final_line_is_recovered(self):
        path = self.root / "results.jsonl"
        append_checkpoint(path, self.record)
        with path.open("ab") as handle:
            handle.write(b'{"sample_id":')
        self.assertEqual(read_checkpoint(path, {"test-one"}), [self.record])
        self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_invalid_complete_line_is_not_deleted(self):
        path = self.root / "results.jsonl"
        path.write_bytes(b"invalid\n")
        with self.assertRaises(ValueError):
            read_checkpoint(path, {"test-one"})
        self.assertEqual(path.read_bytes(), b"invalid\n")

    def test_duplicate_checkpoint_id_is_rejected(self):
        path = self.root / "results.jsonl"
        append_checkpoint(path, self.record)
        append_checkpoint(path, self.record)
        with self.assertRaises(ValueError):
            read_checkpoint(path, {"test-one"})

    def test_resume_rejects_different_config(self):
        metadata = make_metadata(
            [sample()], Config(), demo=True, split="validation", input_file=True
        )
        prepare_run(self.root, metadata, resume=False)
        changed = {**metadata, "config": Config(seed=7).to_dict()}
        with self.assertRaises(ValueError):
            prepare_run(self.root, changed, resume=True)

    def test_existing_run_is_not_overwritten(self):
        metadata = make_metadata(
            [sample()], Config(), demo=True, split="validation", input_file=True
        )
        prepare_run(self.root, metadata, resume=False)
        with self.assertRaises(ValueError):
            prepare_run(self.root, metadata, resume=False)

    def test_csv_escapes_formulas(self):
        for value in ["=1+1", " +1", "@SUM(A1)", "-2"]:
            self.assertTrue(safe_cell(value).startswith("'"))
        self.assertEqual(safe_cell(2.5), 2.5)

    def test_reports_have_no_invented_hallucination_rate(self):
        metadata = {"mode": "synthetic_demo"}
        result = write_reports(self.root, [self.record], metadata)
        self.assertIsNone(result["hallucination_rate"])
        self.assertTrue((self.root / "results.csv").exists())
        self.assertIn("span", result["by_reference_answer_type"])

    def test_partial_run_is_explicit(self):
        result = write_reports(
            self.root,
            [self.record],
            {"mode": "synthetic_demo", "sample_ids": ["test-one", "test-two"]},
        )
        self.assertFalse(result["complete"])
        self.assertEqual(result["completed_questions"], 1)
        self.assertEqual(result["selected_questions"], 2)

    def test_optional_excel_export_is_safe_and_preserves_status(self):
        try:
            from openpyxl import load_workbook
        except ImportError:
            self.skipTest("Install pinned dependencies for Excel export check")
        self.record["question"] = "=SUM(A1)\x00"
        write_reports(self.root, [self.record], {"mode": "synthetic_demo"}, xlsx=True)
        book = load_workbook(self.root / "evaluation.xlsx", read_only=True)
        try:
            self.assertEqual(book["Results"]["B2"].value, "'=SUM(A1)\ufffd")
            self.assertEqual(book["Results"]["F2"].value, "answered")
            self.assertEqual(book["Results"]["B2"].data_type, "s")
        finally:
            book.close()


class ImportAndCliTests(unittest.TestCase):
    def test_imports_do_not_load_models_or_start_evaluation(self):
        script = "import main, aware_gpt2, groundedllm, normal_llm, load_data, similarity, nli_checker, models; import sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")

    def test_cli_demo_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            args = [sys.executable, "main.py", "--demo", "--output", str(Path(folder) / "run")]
            first = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = subprocess.run(args + ["--resume"], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            rows = (Path(folder) / "run/results.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 4)

    def test_invalid_configuration_fails(self):
        for kwargs in [
            {"top_k": 0},
            {"qa_threshold": 2},
            {"fallback": "invent"},
            {"max_new_tokens": 0},
        ]:
            with self.assertRaises(ValueError):
                Config(**kwargs)


if __name__ == "__main__":
    unittest.main()
