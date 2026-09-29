from __future__ import annotations

import csv
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vllm_predictor import chat_benchmark as bench


class FakeClient:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.request_index = 0

    def tokenize_prompt(self, model, prompt):
        tokens = tuple(range(len(prompt)))
        return len(tokens), tokens

    def tokenize_chat(self, model, messages, thinking):
        length = sum(len(message["content"]) for message in messages) + 10
        tokens = tuple([int(thinking)] + list(range(1, length)))
        return len(tokens), tokens

    def chat(self, payload):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.request_index += 1
            request_id = f"request-{self.request_index}"
        time.sleep(0.01)
        input_tokens = sum(len(message["content"]) for message in payload["messages"]) + 10
        with self.lock:
            self.active -= 1
        return {
            "id": request_id,
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "reasoning_content": "two short steps" if payload["chat_template_kwargs"]["enable_thinking"] else None,
                    "content": "answer",
                },
            }],
            "usage": {"prompt_tokens": input_tokens, "completion_tokens": 3},
        }


def args(*extra):
    return bench.parse_args([
        "--input-file", "input.csv", "--sample-size", "4",
        "--batch-sizes", "1", "2", "4", *extra,
    ])


def test_sample_is_reproducible_and_preserves_source_rows(tmp_path):
    path = tmp_path / "input.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["user_prompt_content"])
        writer.writerows([[f"question {i}" if i % 2 else ""] for i in range(100)])
    selected = bench.select_samples(path, "user_prompt_content", 32, 42)
    assert selected == bench.select_samples(path, "user_prompt_content", 32, 42)
    assert len({sample.source_row for sample in selected}) == 32
    assert all(sample.source_row % 2 for sample in selected)
    with pytest.raises(ValueError, match="only 50"):
        bench.select_samples(path, "user_prompt_content", 51, 42)


def test_prepare_samples_uses_server_tokenizer_for_both_lengths():
    samples = [bench.Sample(3, "abc")]
    client = FakeClient()
    prepared = bench.prepare_samples(samples, client, "model", True, "system", {3: 3})
    assert prepared[0].prompt_tokens == 3
    assert prepared[0].input_tokens == len("abc") + len("system") + 10
    assert prepared[0].input_token_ids[0] == 1
    assert prepared[0].messages[-1] == {"role": "user", "content": "abc"}


def test_payload_contains_chat_template_and_sampling_parameters():
    item = bench.PreparedSample(bench.Sample(0, "q"), [{"role": "user", "content": "q"}], 1, 11, (0,))
    payload = bench.make_chat_payload(args(), "served", item, True)
    assert payload["model"] == "served"
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["max_tokens"] == 8192
    assert payload["stream"] is False
    assert payload["skip_special_tokens"] is False


def test_run_batch_sends_simultaneous_requests_and_parses_usage():
    client = FakeClient()
    parsed = args("--no-warmup")
    samples = [bench.Sample(i, f"q{i}") for i in range(4)]
    lengths = {sample.source_row: len(sample.prompt) for sample in samples}
    prepared = bench.prepare_samples(samples, client, "model", True, "sys", lengths)
    results, elapsed = bench.run_batch(client, parsed, "model", prepared, True)
    assert client.max_active == 4
    assert elapsed >= 0.01
    assert [result.source_row for result in results] == [0, 1, 2, 3]
    assert all(result.output_tokens == 3 and result.reasoning == "two short steps"
               for result in results)


def test_benchmark_reuses_samples_and_reports_summary(capsys):
    parsed = args("--no-warmup", "--no-show-answers")
    client = FakeClient()
    samples = [bench.Sample(i, f"q{i}") for i in range(4)]
    summaries = bench.benchmark(parsed, samples, client, "model")
    assert len(summaries) == 6
    assert all(summary["samples"] == 4 for summary in summaries)
    assert [summary["batches"] for summary in summaries] == [4, 4, 2, 2, 1, 1]
    assert all(summary["mean_output_tokens"] == 3 for summary in summaries)
    terminal = capsys.readouterr().out
    assert "concurrent HTTP requests" in terminal
    assert "request_id" in terminal
    assert "Reasoning:" not in terminal


def test_thinking_template_must_change_tokenized_input():
    class IgnoredClient(FakeClient):
        def tokenize_chat(self, model, messages, thinking):
            return 2, (1, 2)

    parsed = args("--no-warmup", "--no-show-answers")
    with pytest.raises(ValueError, match="ignores enable_thinking"):
        bench.benchmark(parsed, [bench.Sample(i, "q") for i in range(4)],
                        IgnoredClient(), "model")


def test_parse_response_validates_server_token_usage():
    item = bench.PreparedSample(bench.Sample(2, "q"), [], 1, 5, (1, 2, 3, 4, 5))
    response = {
        "choices": [{"finish_reason": "stop", "message": {"content": "a"}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1},
    }
    with pytest.raises(RuntimeError, match="Input token mismatch"):
        bench.parse_chat_response(item, response)


def test_http_client_routes_and_model_discovery(monkeypatch):
    calls = []

    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            return False

        def read(self):
            return json.dumps(self.body).encode()

    def urlopen(request, timeout):
        calls.append((request.full_url, request.method, request.data, request.headers, timeout))
        if request.full_url.endswith("/models"):
            return Response({"data": [{"id": "served-model"}]})
        return Response({"count": 2, "tokens": [1, 2], "max_model_len": 32768})

    monkeypatch.setattr(bench.urllib.request, "urlopen", urlopen)
    client = bench.VLLMClient("http://localhost:8001/v1/", "EMPTY", 9)
    assert client.resolve_model(None) == "served-model"
    assert client.tokenize_prompt("served-model", "hi") == (2, (1, 2))
    assert calls[0][0] == "http://localhost:8001/v1/models"
    assert calls[1][0] == "http://localhost:8001/tokenize"
    assert calls[1][4] == 9
    assert calls[1][3]["Authorization"] == "Bearer EMPTY"
    assert json.loads(calls[1][2])["prompt"] == "hi"

    client_without_version = bench.VLLMClient("http://localhost:8001", "", 9)
    assert client_without_version.base_url == "http://localhost:8001/v1"
    assert client_without_version.server_url == "http://localhost:8001"


def test_model_name_is_validated(monkeypatch):
    client = bench.VLLMClient("http://localhost:8001/v1", "", 1)
    monkeypatch.setattr(client, "_request", lambda *args: {"data": [{"id": "one"}]})
    with pytest.raises(ValueError, match="available=.*one"):
        client.resolve_model("other")


@pytest.mark.parametrize("flags", [
    ["--sample-size", "0"], ["--sample-size", "3"], ["--batch-sizes", "0"],
    ["--batch-sizes", "1", "1"], ["--request-timeout", "0"],
])
def test_invalid_args(flags):
    with pytest.raises(SystemExit):
        bench.parse_args(["--input-file", "x", *flags])
