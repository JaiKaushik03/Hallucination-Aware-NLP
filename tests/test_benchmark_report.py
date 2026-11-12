import json
import tempfile
import unittest
from pathlib import Path
from benchmark_report import build_report, percentile


class BenchmarkReportTests(unittest.TestCase):
    def test_nearest_rank_percentiles(self):
        self.assertEqual(percentile([4, 1, 3, 2], .5), 2)
        self.assertEqual(percentile([4, 1, 3, 2], .95), 4)
        self.assertIsNone(percentile([], .95))

    def test_refuses_empty_run_and_invents_no_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "metadata.json").write_text(json.dumps({"sample_ids": ["a"]}))
            with self.assertRaisesRegex(ValueError, "No completed"):
                build_report(path)

    def test_synthetic_partial_is_labelled_and_errors_stay_in_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            metadata = {"mode": "synthetic_demo", "sample_ids": ["a", "b"],
                        "source_hash": "fixture", "sample_fingerprint": "fixture",
                        "config": {"seed": 42}}
            systems = {name: {"answer": "", "status": "error", "fallback_used": False,
                             "metrics": {"exact_match": 0., "token_f1": 0., "rouge_l": 0.}}
                       for name in ["baseline_gpt2", "verified_gpt2", "extractive_qa"]}
            record = {"sample_id": "a", "systems": systems, "duration_seconds": 2.5}
            (path / "metadata.json").write_text(json.dumps(metadata))
            (path / "results.jsonl").write_text(json.dumps(record) + "\n")
            result = build_report(path)
            self.assertFalse(result["complete"])
            self.assertFalse(result["benchmark_is_real_models"])
            self.assertEqual(result["metrics"]["baseline_gpt2"]["errors"], 1)
            self.assertEqual(result["timing"]["first_completed_question_s"], 2.5)
            self.assertIsNone(result["timing"]["excluding_first_p95_s"])

if __name__ == "__main__":
    unittest.main()
