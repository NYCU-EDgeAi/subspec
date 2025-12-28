import json
import unittest
from types import SimpleNamespace

import torch
from fastapi.testclient import TestClient

from run.pipelines.run_api import create_app


class _DummyKV:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class _DummyTokenizer:
    eos_token_id = 2

    def __init__(self):
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"):
        # Return a deterministic "prompt" token sequence.
        self.last_messages = messages
        return torch.tensor([[101, 102, 103]], dtype=torch.long)

    def decode(self, token_ids, skip_special_tokens=True):
        # token_ids can be a list[int] or a tensor slice.
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        mapping = {
            10: "Hello",
            11: " world",
            12: "!",
            self.eos_token_id: "",
        }
        return "".join(mapping.get(int(t), f"<{int(t)}>") for t in token_ids)

    def encode(self, text: str):
        # Best-effort token counter for usage.
        if not text:
            return []
        return text.split()


class _DummyGenerator:
    def generate(
        self,
        input_ids,
        temperature=None,
        max_length=None,
        do_sample=False,
        stop_strings=None,
        stream_callback=None,
        **kwargs,
    ):
        # Emit two tokens unless stop_strings asks us to stop early.
        emit = [10, 11]
        if stop_strings and " world" in stop_strings:
            emit = [10]

        if stream_callback is not None:
            for tid in emit:
                stream_callback(torch.tensor([[tid]], dtype=torch.long))
            # optionally end with EOS (dropped by server)
            stream_callback(torch.tensor([[2]], dtype=torch.long))
            return None

        out = torch.tensor([emit + [2]], dtype=torch.long)
        return torch.cat([input_ids, out], dim=1)


class _DummyBuilder:
    def __init__(self):
        self._args = SimpleNamespace(
            device="cpu",
            do_sample=False,
            temperature=0.0,
            max_length=64,
            llm_path="dummy-model",
            method="dummy",
        )

    @property
    def args(self):
        return self._args

    def build(self):
        return _DummyGenerator(), _DummyTokenizer(), _DummyKV(), None


class TestOpenAICompatibleAPI(unittest.TestCase):
    def setUp(self):
        app = create_app(_DummyBuilder())
        self.client = TestClient(app)

    def test_list_models(self):
        r = self.client.get("/v1/models")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["object"], "list")
        self.assertTrue(len(data["data"]) >= 1)
        self.assertEqual(data["data"][0]["id"], "dummy-model")

    def test_chat_completions_non_stream(self):
        r = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(body["choices"][0]["message"]["content"], "Hello world")

    def test_chat_completions_stop_non_stream(self):
        r = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stop": " world",
                "stream": False,
            },
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["choices"][0]["message"]["content"], "Hello")

    def test_chat_completions_stream(self):
        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as r:
            self.assertEqual(r.status_code, 200)
            self.assertIn("text/event-stream", r.headers.get("content-type", ""))

            data_lines = []
            for line in r.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_lines.append(line[len("data: ") :])

            self.assertTrue(data_lines[0].startswith("{"))
            first = json.loads(data_lines[0])
            self.assertEqual(first["object"], "chat.completion.chunk")
            self.assertEqual(first["choices"][0]["delta"].get("role"), "assistant")

            # Ensure at least one content chunk arrived.
            content_chunks = [json.loads(x) for x in data_lines[1:] if x.startswith("{")]
            content_text = "".join(
                c["choices"][0]["delta"].get("content", "") for c in content_chunks
            )
            self.assertIn("Hello", content_text)

            self.assertEqual(data_lines[-1], "[DONE]")

    def test_completions_stream(self):
        with self.client.stream(
            "POST",
            "/v1/completions",
            json={
                "model": "dummy-model",
                "prompt": "hi",
                "stream": True,
            },
        ) as r:
            self.assertEqual(r.status_code, 200)
            data_lines = []
            for line in r.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_lines.append(line[len("data: ") :])

            # Expect at least one chunk and a terminal [DONE].
            self.assertTrue(any(x.startswith("{") for x in data_lines))
            self.assertEqual(data_lines[-1], "[DONE]")


if __name__ == "__main__":
    unittest.main()
