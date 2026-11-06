"""Lazy, shared Hugging Face models; no model initialization during imports."""

from functools import cached_property
from pathlib import Path


class ModelBackend:
    def __init__(self, config):
        self.config = config
        self.cache_dir = str(Path(__file__).resolve().parent / ".cache" / "models")

    @cached_property
    def torch(self):
        import torch

        torch.set_num_threads(self.config.threads)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        return torch

    @cached_property
    def device(self):
        available = self.torch.cuda.is_available()
        if self.config.device == "cuda" and not available:
            raise RuntimeError("CUDA was requested but is not available. Use --device cpu or auto.")
        return "cuda:0" if available and self.config.device != "cpu" else "cpu"

    @cached_property
    def generator(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.generator_model,
            revision=self.config.generator_revision,
            cache_dir=self.cache_dir,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.generator_model,
            revision=self.config.generator_revision,
            use_safetensors=True,
            cache_dir=self.cache_dir,
        )
        model.to(self.device).eval()
        return tokenizer, model

    @cached_property
    def qa(self):
        from transformers import pipeline

        return self._load_qa(pipeline)

    def _load_qa(self, pipeline):
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        model = AutoModelForQuestionAnswering.from_pretrained(
            self.config.qa_model,
            revision=self.config.qa_revision,
            use_safetensors=True,
            cache_dir=self.cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.qa_model, revision=self.config.qa_revision, cache_dir=self.cache_dir
        )
        return pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.device.startswith("cuda") else -1,
        )

    @cached_property
    def embedding(self):
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.embedding_model,
            revision=self.config.embedding_revision,
            cache_dir=self.cache_dir,
        )
        model = (
            AutoModel.from_pretrained(
                self.config.embedding_model,
                revision=self.config.embedding_revision,
                use_safetensors=True,
                cache_dir=self.cache_dir,
            )
            .to(self.device)
            .eval()
        )
        return tokenizer, model

    @cached_property
    def nli_model(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.nli_model, revision=self.config.nli_revision, cache_dir=self.cache_dir
        )
        model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.config.nli_model,
                revision=self.config.nli_revision,
                use_safetensors=True,
                cache_dir=self.cache_dir,
            )
            .to(self.device)
            .eval()
        )
        return tokenizer, model

    def generate(self, question, paragraphs):
        tokenizer, model = self.generator
        suffix = f"\nQuestion: {question}\nAnswer:"
        prefix = "Answer the question briefly using only the context.\nContext:\n"
        before = tokenizer.encode(prefix, add_special_tokens=False)
        after = tokenizer.encode(suffix, add_special_tokens=False)
        limit = getattr(
            model.config, "max_position_embeddings", getattr(model.config, "n_positions", 1024)
        )
        budget = limit - self.config.max_new_tokens - len(before) - len(after)
        if budget < 1:
            raise ValueError("Question exceeds the model's context budget")
        # Both baseline and verifier use this exact candidate and exact prompt evidence.
        evidence = tokenizer.encode("\n\n".join(paragraphs), add_special_tokens=False)[:budget]
        self.generation_trace = {
            "context_used": tokenizer.decode(evidence),
            "input_tokens": len(before) + len(evidence) + len(after),
            "context_window": limit,
            "max_new_tokens": self.config.max_new_tokens,
        }
        ids = self.torch.tensor([before + evidence + after], device=self.device)
        with self.torch.inference_mode():
            generated = model.generate(
                input_ids=ids,
                attention_mask=self.torch.ones_like(ids),
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0, ids.shape[1] :], skip_special_tokens=True).strip()
        return text.splitlines()[0].strip() if text else ""

    def extract(self, question, contexts):
        if not contexts:
            return []
        predictions = self.qa(
            question=[question] * len(contexts),
            context=contexts,
            handle_impossible_answer=True,
            max_seq_len=384,
            doc_stride=128,
            max_answer_len=30,
            batch_size=self.config.batch_size,
        )
        if isinstance(predictions, dict):
            predictions = [predictions]
        if len(predictions) != len(contexts):
            raise RuntimeError("QA returned an unexpected number of spans")
        return [
            {"answer": p["answer"].strip(), "score": float(p["score"]), "context": c}
            for p, c in zip(predictions, contexts)
        ]

    @staticmethod
    def _chunks(tokenizer, text, size):
        ids = tokenizer.encode(text, add_special_tokens=False)
        return [
            tokenizer.decode(ids[i : i + size], skip_special_tokens=True)
            for i in range(0, len(ids), size)
        ] or [""]

    def similarities(self, question, texts):
        if not texts:
            return []
        tokenizer, model = self.embedding
        chunks, groups = [], []
        for text in texts:
            start = len(chunks)
            chunks.extend(self._chunks(tokenizer, text, 224))
            groups.append((start, len(chunks)))
        encoded = []
        all_texts = [question] + chunks
        for start in range(0, len(all_texts), self.config.batch_size):
            inputs = tokenizer(
                all_texts[start : start + self.config.batch_size],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.inference_mode():
                output = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
                encoded.extend(self.torch.nn.functional.normalize(pooled, p=2, dim=1).cpu())
        query, vectors = encoded[0], encoded[1:]
        return [
            max(float(self.torch.dot(query, vectors[i])) for i in range(a, b)) for a, b in groups
        ]

    def contradiction(self, first, second):
        tokenizer, model = self.nli_model
        # Score all windows in both directions: tail evidence is not silently discarded.
        a, b = self._chunks(tokenizer, first, 220), self._chunks(tokenizer, second, 220)
        best = 0.0
        labels = {str(label).lower(): int(index) for index, label in model.config.id2label.items()}
        if "contradiction" not in labels:
            raise RuntimeError("NLI checkpoint does not expose a contradiction label")
        for x in a:
            for y in b:
                inputs = tokenizer(
                    [x, y], [y, x], padding=True, truncation=False, return_tensors="pt"
                ).to(self.device)
                with self.torch.inference_mode():
                    probabilities = self.torch.softmax(model(**inputs).logits, dim=-1)
                best = max(best, float(probabilities[:, labels["contradiction"]].max()))
        return best
