import importlib.util
from types import SimpleNamespace
import unittest
from config import Config
from models import ModelBackend


class DeviceTests(unittest.TestCase):
    def test_auto_device_falls_back_to_cpu(self):
        backend = ModelBackend(Config())
        backend.__dict__["torch"] = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        )
        self.assertEqual(backend.device, "cpu")

    def test_explicit_unavailable_cuda_is_an_error(self):
        backend = ModelBackend(Config(device="cuda"))
        backend.__dict__["torch"] = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        )
        with self.assertRaises(RuntimeError):
            _ = backend.device

    def test_explicit_cpu_is_respected_with_cuda_available(self):
        backend = ModelBackend(Config(device="cpu"))
        backend.__dict__["torch"] = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        self.assertEqual(backend.device, "cpu")


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "Install pinned dependencies for tensor adapter tests"
)
class GenerationBudgetTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch

        class Tokenizer:
            eos_token_id = 0

            def encode(self, text, **kwargs):
                return list(text.encode("ascii"))

            def decode(self, ids, **kwargs):
                return bytes(int(i) for i in ids).decode("ascii")

        class Model:
            config = SimpleNamespace(max_position_embeddings=256)

            def generate(inner, **kwargs):
                inner.kwargs = kwargs
                return torch.cat([kwargs["input_ids"], torch.tensor([list(b"Paris")])], dim=1)

        self.backend = ModelBackend(Config(device="cpu", max_new_tokens=16))
        self.backend.__dict__["torch"] = torch
        self.backend.__dict__["generator"] = (Tokenizer(), Model())

    def test_long_context_reserves_generation_space_and_preserves_question(self):
        result = self.backend.generate("What city?", ["Evidence. " * 1000])
        self.assertEqual(result, "Paris")
        trace = self.backend.generation_trace
        self.assertLessEqual(
            trace["input_tokens"] + trace["max_new_tokens"], trace["context_window"]
        )
        kwargs = self.backend.generator[1].kwargs
        self.assertNotIn("max_length", kwargs)
        self.assertNotIn("temperature", kwargs)
        self.assertTrue(
            self.backend.generator[0]
            .decode(kwargs["input_ids"][0])
            .endswith("Question: What city?\nAnswer:")
        )

    def test_oversize_question_fails_instead_of_disappearing(self):
        with self.assertRaises(ValueError):
            self.backend.generate("q" * 1000, ["Evidence"])
