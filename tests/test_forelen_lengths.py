from __future__ import annotations

import csv
import os
import shlex
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "data"))
import extract_forelen_lengths as lengths


def output(tokens, reason="stop"):
    return SimpleNamespace(finished=True, outputs=[SimpleNamespace(
        token_ids=tokens, text="This text must not be used to count tokens.",
        finish_reason=reason)])


def test_normal_engine_has_no_extraction_settings(monkeypatch):
    calls = {}
    fake = ModuleType("vllm")

    def llm(**kwargs):
        assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "2,3"
        calls.update(kwargs)
        return object()

    fake.LLM = llm
    fake.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "")
    args = lengths.parse_args(["--input-file", "input.csv", "--model", "qwen",
                              "--devices", "2,3", "--tensor-parallel-size", "2",
                              "--max-tokens", "100", "--temperature", "0"])
    _, sampling = lengths.create_local_llm(args)
    assert "speculative_config" not in calls
    assert "kv_transfer_config" not in calls
    assert calls["tensor_parallel_size"] == 2
    assert sampling["max_tokens"] == 100
    assert sampling["temperature"] == 0


def test_counts_tokens_and_preserves_capped_result():
    samples = [lengths.Sample(0, "first"), lengths.Sample(2, "third")]

    def generate(prompts, sampling_params, use_tqdm):
        assert prompts == ["first", "third"]
        return [output([1, 2]), output([3, 4, 5], "length")]

    records = lengths.process_batch(samples, SimpleNamespace(generate=generate), None)
    assert records == [
        {"source_row": 0, "completion_tokens": 2, "finish_reason": "stop"},
        {"source_row": 2, "completion_tokens": 3, "finish_reason": "length"},
    ]


@pytest.mark.parametrize("outputs", [[], [output([], "abort")],
                                    [SimpleNamespace(finished=False, outputs=[])]])
def test_rejects_incomplete_or_aborted_outputs(outputs):
    llm = SimpleNamespace(generate=lambda *args, **kwargs: outputs)
    with pytest.raises(RuntimeError):
        lengths.process_batch([lengths.Sample(0, "prompt")], llm, None)


def test_main_resume_limit_and_row_alignment(tmp_path, monkeypatch):
    source = tmp_path / "input.csv"
    source.write_text("user_prompt_content\nfirst\n\"\"\nthird\n", encoding="utf-8")
    calls = []

    def generate(prompts, *args, **kwargs):
        calls.extend(prompts)
        return [output([7, 8]) for _ in prompts]

    monkeypatch.setattr(lengths, "create_local_llm", lambda args: (
        SimpleNamespace(generate=generate), None))
    argv = ["--input-file", str(source), "--output-dir", str(tmp_path / "results"),
            "--model", "qwen", "--batch-size", "1"]
    assert lengths.main(argv + ["--limit", "1"]) == 0
    assert calls == ["first"]
    assert lengths.main(argv) == 0
    assert calls == ["first", "third"]

    def unexpected_engine(args):
        pytest.fail("A completed run must not load the model")

    monkeypatch.setattr(lengths, "create_local_llm", unexpected_engine)
    assert lengths.main(argv) == 0
    with (tmp_path / "results" / "samples.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["source_row"]) for row in rows] == [0, 2]
    assert all(row["source_file"] == str(source.resolve()) for row in rows)
    assert all(row["completion_tokens"] == "2" for row in rows)


def test_resume_rejects_incomplete_record(tmp_path):
    results = tmp_path / "samples.csv"
    results.write_text(
        "source_file,source_row,completion_tokens,finish_reason\ninput.csv,0,2\n",
        encoding="utf-8")
    with pytest.raises(ValueError):
        lengths.load_completed_rows(results, "input.csv")


@pytest.mark.parametrize("flags", [
    ["--max-tokens", "0"], ["--batch-size", "0"], ["--limit", "0"],
    ["--devices", "0,0,2,3"], ["--temperature", "-1"],
])
def test_invalid_arguments(flags):
    with pytest.raises(SystemExit):
        lengths.parse_args(["--input-file", "input.csv", "--model", "qwen", *flags])


def test_run_script_arguments_are_accepted():
    script = (ROOT / "scripts/data/run_extract_forelen_lengths.sh").read_text()
    command = script[script.index("python /home"):].replace("\\\n", " ")
    args = lengths.parse_args(shlex.split(command)[2:])
    assert args.max_tokens == 16384
    assert args.seed == 42
    assert args.limit is None
