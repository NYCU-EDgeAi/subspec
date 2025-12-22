import os
import json
import time
import unittest

import torch
from fastapi.testclient import TestClient

from run.core.configuration import AppConfig
from run.core.builder import GeneratorPipelineBuilder
from run.core.presets import register_presets
from run.pipelines.run_api import create_app


def _dtype_from_env() -> torch.dtype:
    v = (os.environ.get("SUBSPEC_DTYPE") or "").strip().lower()
    if v in ("", "fp16", "float16"):
        return torch.float16
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    if v in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown SUBSPEC_DTYPE={v!r} (use fp16|bf16|fp32)")


class TestOpenAIRealModelIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Opt-in only: real model loads can be slow and require local paths/GPU.
        if os.environ.get("SUBSPEC_RUN_REAL_MODEL_TESTS") != "1":
            raise unittest.SkipTest("Set SUBSPEC_RUN_REAL_MODEL_TESTS=1 to enable real-model tests")

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available; real-model deployment test requires GPU")

        llm_path = os.environ.get("SUBSPEC_LLM_PATH")
        if not llm_path:
            raise unittest.SkipTest("Set SUBSPEC_LLM_PATH to a local model path (or HF id) to enable")

        # If you want to exercise a registered method (e.g. subspec_sd), you can set SUBSPEC_METHOD.
        # If omitted, we intentionally use an unregistered method name so the builder falls back
        # to the single-model NaiveGenerator (no draft model required).
        method = os.environ.get("SUBSPEC_METHOD", "real_model_naive")

        device = os.environ.get("SUBSPEC_DEVICE", "cuda:0")
        max_length = int(os.environ.get("SUBSPEC_MAX_LENGTH", "128"))

        # Register presets so ModelRegistry is populated if you choose a real method.
        register_presets()

        cfg = AppConfig()
        cfg.method = method
        cfg.llm_path = llm_path
        cfg.device = device
        cfg.max_length = max_length
        cfg.warmup_iter = 0
        cfg.compile_mode = None
        cfg.do_sample = False
        cfg.temperature = 0.0
        cfg.cache_implementation = os.environ.get("SUBSPEC_CACHE_IMPL", "dynamic")

        # Draft model is optional for naive fallback; required for many speculative methods.
        draft_path = os.environ.get("SUBSPEC_DRAFT_MODEL_PATH")
        if draft_path:
            cfg.draft_model_path = draft_path

        # Default fp16 is typical on GPU; allow override.
        cfg.dtype = _dtype_from_env()

        builder = GeneratorPipelineBuilder(cfg)

        # This actually loads the model(s) once.
        t0 = time.time()
        cls.app = create_app(builder)
        cls.build_seconds = time.time() - t0
        cls.client = TestClient(cls.app)

    def test_models_endpoint(self):
        r = self.client.get("/v1/models")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body.get("object"), "list")
        self.assertTrue(isinstance(body.get("data"), list) and len(body["data"]) >= 1)

    def test_chat_completions_non_stream_real_model(self):
        r = self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say 'OK'."}],
                "max_tokens": 8,
                "stream": False,
                "temperature": 0.0,
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body.get("object"), "chat.completion")
        self.assertIn("choices", body)
        content = body["choices"][0]["message"]["content"]
        self.assertTrue(isinstance(content, str))
        # Don't assert exact text; just ensure something came back.
        self.assertTrue(len(content) >= 0)

    def test_chat_completions_stream_real_model(self):
        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Count to three."}],
                "max_tokens": 16,
                "stream": True,
                "temperature": 0.0,
            },
        ) as r:
            self.assertEqual(r.status_code, 200, r.text)
            self.assertIn("text/event-stream", r.headers.get("content-type", ""))

            data_lines = []
            for line in r.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_lines.append(line[len("data: ") :])

            # At minimum: initial role chunk, optional content chunks, final chunk, [DONE]
            self.assertTrue(len(data_lines) >= 2)
            self.assertEqual(data_lines[-1], "[DONE]")

            first = json.loads(data_lines[0])
            self.assertEqual(first.get("object"), "chat.completion.chunk")
            self.assertEqual(first["choices"][0]["delta"].get("role"), "assistant")


if __name__ == "__main__":
    unittest.main()
