from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))

from extract_forelen import (  # noqa: E402
    CSV_FIELDS,
    Sample,
    append_results,
    create_local_llm,
    extract_last_token_hidden_states,
    iter_csv_samples,
    parse_args,
    process_batch,
    write_shard,
)


def test_parse_args_has_requested_batch_and_shard_defaults(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--input-file",
            str(tmp_path / "forelen.csv"),
            "--model",
            "/models/qwen",
        ]
    )

    assert args.shard_size == 2048
    assert args.batch_size == 8
    assert args.devices == "0,1,2,3"
    assert args.tensor_parallel_size == 4
    assert args.block_size == 128
    assert args.enable_chunked_prefill is False
    assert args.enable_prefix_caching is False
    assert args.max_model_len == 32768
    assert args.gpu_memory_utilization == 0.9
    assert args.max_tokens == 1
    assert args.temperature == 1.0
    assert args.top_p == 0.95
    assert args.top_k == 20


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--max-tokens", "2"], "invalid choice"),
        (["--devices", "0,1,2"], "exactly four distinct"),
        (["--tensor-parallel-size", "2"], "must be 4"),
    ],
)
def test_parse_args_enforces_hidden_extraction_resources(
    tmp_path: Path,
    extra_args: list[str],
    message: str,
    capsys,
) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input-file",
                str(tmp_path / "forelen.csv"),
                "--model",
                "/models/qwen",
                *extra_args,
            ]
        )

    assert message in capsys.readouterr().err


def test_iter_csv_samples_reads_one_file_and_preserves_source_row(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.csv"
    with source.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("user_prompt_content", "other"))
        writer.writeheader()
        writer.writerow({"user_prompt_content": "first", "other": "x"})
        writer.writerow({"user_prompt_content": "", "other": "y"})
        writer.writerow({"user_prompt_content": "third", "other": "z"})

    samples = list(iter_csv_samples(source, "user_prompt_content"))

    assert samples == [Sample(0, "first"), Sample(2, "third")]


def test_create_local_llm_sets_generation_sampling_params(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = SimpleNamespace(llm=None, sampling=None)

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.llm = kwargs

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            calls.sampling = kwargs

    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    args = parse_args(
        [
            "--input-file",
            str(tmp_path / "forelen.csv"),
            "--model",
            "/models/qwen",
            "--hidden-states-dir",
            str(tmp_path / "connector"),
            "--seed",
            "123",
        ]
    )

    create_local_llm(args)

    assert calls.sampling == {
        "max_tokens": 1,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": 123,
    }
    assert calls.llm["speculative_config"]["method"] == "extract_hidden_states"
    assert calls.llm["kv_transfer_config"]["kv_role"] == "kv_producer"
    assert calls.llm["block_size"] == 128
    assert calls.llm["enable_chunked_prefill"] is False
    assert calls.llm["enable_prefix_caching"] is False
    assert calls.llm["max_model_len"] == 32768
    assert calls.llm["gpu_memory_utilization"] == 0.9


def test_extract_last_token_hidden_states(tmp_path: Path) -> None:
    source = tmp_path / "full.safetensors"
    token_ids = torch.tensor([11, 12, 13])
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    save_file({"token_ids": token_ids, "hidden_states": hidden_states}, source)
    Path(f"{source}.lock").touch()

    last_hidden_states = extract_last_token_hidden_states(source)

    assert torch.equal(last_hidden_states, hidden_states[-1])


def test_process_batch_generates_once_without_reading_output_lengths(
    tmp_path: Path,
) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"request-{index}.safetensors"
        save_file(
            {
                "token_ids": torch.tensor([1, 2]),
                "hidden_states": torch.full((2, 1, 4), float(index)),
            },
            path,
        )
        Path(f"{path}.lock").touch()
        paths.append(path)

    class FakeLLM:
        calls = 0

        def generate(self, prompts, sampling_params, use_tqdm):
            self.calls += 1
            assert prompts == ["a", "b"]
            assert use_tqdm is False
            return [
                SimpleNamespace(
                    kv_transfer_params={"hidden_states_path": str(path)},
                )
                for path in paths
            ]

    llm = FakeLLM()
    hidden_states = process_batch(
        [Sample(0, "a"), Sample(1, "b")], llm, object()
    )

    assert llm.calls == 1
    assert len(hidden_states) == 2
    assert torch.equal(hidden_states[0], torch.zeros(1, 4))
    assert torch.equal(hidden_states[1], torch.ones(1, 4))
    assert all(not path.exists() for path in paths)
    assert all(not Path(f"{path}.lock").exists() for path in paths)


def test_write_shard_and_csv_index(tmp_path: Path) -> None:
    samples = [Sample(3, "a"), Sample(9, "b")]
    hidden_states = [torch.zeros(1, 4), torch.ones(1, 4)]

    shard = write_shard(tmp_path, 0, hidden_states)
    results = tmp_path / "samples.csv"
    append_results(results, "/data/forelen.csv", samples, shard)

    assert shard.name == "shards-000000.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        assert tensors.get_tensor("hidden_states").shape == (2, 1, 4)
    with results.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == CSV_FIELDS
    assert rows == [
        {
            "source_file": "/data/forelen.csv",
            "source_row": "3",
            "hidden_states_path": str(shard),
            "hidden_states_row": "0",
        },
        {
            "source_file": "/data/forelen.csv",
            "source_row": "9",
            "hidden_states_path": str(shard),
            "hidden_states_row": "1",
        },
    ]
