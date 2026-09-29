from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vllm_predictor import chat_benchmark as bench


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True
        return str(enable_thinking) + "|" + "|".join(message["content"] for message in messages)

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


class Engine:
    def __init__(self):
        self.calls = []

    def get_tokenizer(self):
        return Tokenizer()

    def generate(self, prompts, sampling, *, use_tqdm):
        assert use_tqdm is False
        self.calls.append((prompts, sampling.max_tokens))
        return [SimpleNamespace(
            finished=True, prompt_token_ids=prompt["prompt_token_ids"],
            outputs=[SimpleNamespace(token_ids=[1, 2, 3], finish_reason="length",
                                     text="<think>Brief reasoning</think> Answer")])
                for prompt in prompts]


def test_sample_is_reproducible_and_preserves_source_rows(tmp_path):
    path = tmp_path / "input.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["user_prompt_content"])
        writer.writerows([[f"question {i}" if i % 2 else ""] for i in range(100)])
    selected = bench.select_samples(path, "user_prompt_content", 32, 42)
    assert selected == bench.select_samples(path, "user_prompt_content", 32, 42)
    assert len({sample.source_row for sample in selected}) == 32
    assert all(sample.source_row % 2 and sample.prompt == f"question {sample.source_row}"
               for sample in selected)
    with pytest.raises(ValueError, match="only 50"):
        bench.select_samples(path, "user_prompt_content", 51, 42)
    with pytest.raises(ValueError, match="Missing prompt column"):
        bench.select_samples(path, "missing", 1, 42)


def test_template_modes_and_context_budget():
    sample = bench.Sample(9, "abc")
    off = bench.prepare_samples([sample], Tokenizer(), False, "system", 100, 10)[0]
    on = bench.prepare_samples([sample], Tokenizer(), True, "system", 100, 10)[0]
    assert off.prompt_tokens == on.prompt_tokens == 3
    assert off.input_ids != on.input_ids
    assert b"system" in bytes(off.input_ids)
    assert len(off.input_ids) > off.prompt_tokens
    with pytest.raises(ValueError, match="source_row=9"):
        bench.prepare_samples([sample], Tokenizer(), True, "system", 10, 10)


def test_all_batch_sizes_use_same_samples_and_exclude_warmup(monkeypatch, capsys):
    args = bench.parse_args(["--input-file", "input.csv", "--model", "model"])
    samples = [bench.Sample(i, f"prompt {i}") for i in range(32)]
    engine = Engine()
    sampling = SimpleNamespace(max_tokens=args.max_tokens)
    # Exactly two clock reads per generate; each call takes 2 seconds.
    ticks = iter(range(0, 10000, 2))
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(ticks))
    summaries = bench.benchmark(args, samples, engine, sampling)
    assert len(summaries) == 12
    assert sampling.max_tokens == 8192
    for summary in summaries:
        assert summary["samples"] == 32
        assert summary["batches"] == 32 // summary["batch_size"]
        assert summary["total_seconds"] == summary["batches"] * 2
        assert summary["mean_batch_seconds"] == 2
        assert summary["mean_output_tokens"] == 3
        assert summary["capped"] == 32
    assert sum(budget == 1 for _, budget in engine.calls) == 12
    measured = [(prompts, budget) for prompts, budget in engine.calls if budget != 1]
    assert sum(len(prompts) for prompts, _ in measured) == 384
    # Every formatted prompt appears once per batch-size setting.
    from collections import Counter
    counts = Counter(tuple(prompt["prompt_token_ids"])
                     for prompts, _ in measured for prompt in prompts)
    assert len(counts) == 64
    assert set(counts.values()) == {6}
    terminal = capsys.readouterr().out
    assert "Response (unabridged)" in terminal
    assert "<think>Brief reasoning</think> Answer" in terminal
    assert "Summary" in terminal and "output_tokens" in terminal


def test_partial_batch_and_quiet_answers(capsys):
    args = bench.parse_args(["--input-file", "x", "--model", "m", "--sample-size", "3",
                             "--batch-sizes", "2", "--no-warmup", "--no-show-answers"])
    engine = Engine()
    summaries = bench.benchmark(args, [bench.Sample(i, str(i)) for i in range(3)],
                                engine, SimpleNamespace(max_tokens=args.max_tokens))
    assert [len(prompts) for prompts, _ in engine.calls] == [2, 1, 2, 1]
    assert all(row["batches"] == 2 and row["samples"] == 3 for row in summaries)
    terminal = capsys.readouterr().out
    assert "requested=2 actual=1" in terminal
    assert "Response (unabridged)" not in terminal


def test_unsupported_thinking_template_fails():
    class IgnoredTemplate(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return "same input"

    args = bench.parse_args(["--input-file", "x", "--model", "m"])
    engine = SimpleNamespace(get_tokenizer=IgnoredTemplate)
    with pytest.raises(ValueError, match="ignores enable_thinking"):
        bench.benchmark(args, [bench.Sample(0, "prompt")], engine, None)


def test_engine_is_normal_generation(monkeypatch):
    fake = ModuleType("vllm")

    def make_llm(**settings):
        assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "0,1,2,3"
        return settings

    fake.LLM = make_llm
    fake.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "")
    args = bench.parse_args(["--input-file", "x", "--model", "m"])
    settings, sampling = bench.create_engine(args)
    assert "speculative_config" not in settings and "kv_transfer_config" not in settings
    assert "max_num_seqs" not in settings
    assert settings["block_size"] == 128
    assert not settings["enable_prefix_caching"] and not settings["enable_chunked_prefill"]
    assert sampling.skip_special_tokens is False


@pytest.mark.parametrize("flags", [["--sample-size", "8"], ["--batch-sizes", "0"],
                                  ["--batch-sizes", "1", "1"], ["--devices", "0,0,2,3"]])
def test_invalid_args(flags):
    with pytest.raises(SystemExit):
        bench.parse_args(["--input-file", "x", "--model", "m", *flags])
