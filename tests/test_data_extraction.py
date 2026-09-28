from __future__ import annotations

import argparse
import csv
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))

from extract_forelen import (  # noqa: E402
    CSV_FIELDS,
    GenerationResult,
    Sample,
    append_results,
    extract_last_token_hidden_states,
    generation_payload,
    iter_csv_samples,
    offline_hidden_states,
    parse_args,
    process_batch,
    write_shard,
)


def make_args(**overrides):
    values = {
        "url": "http://generation:8000",
        "model": "/models/qwen",
        "server_model": "qwen",
        "max_tokens": 2048,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": None,
        "timeout": 30.0,
        "retries": 0,
        "retry_backoff": 0.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_parse_args_has_requested_batch_and_shard_defaults(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--input-file",
            str(tmp_path / "forelen.csv"),
            "--url",
            "http://127.0.0.1:8000",
            "--model",
            "/models/qwen",
        ]
    )

    assert args.shard_size == 2048
    assert args.batch_size == 8
    assert args.offline_devices == "4,5,6,7"
    assert args.tensor_parallel_size == 4


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


def test_generation_payload_uses_online_sampling_and_distinct_row_seed() -> None:
    args = make_args(seed=100)

    payload = generation_payload(Sample(7, "hello"), args)

    assert payload["model"] == "qwen"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["max_tokens"] == 2048
    assert payload["seed"] == 107
    assert "kv_transfer_params" not in payload


def test_extract_last_token_hidden_states(tmp_path: Path) -> None:
    source = tmp_path / "full.safetensors"
    token_ids = torch.tensor([11, 12, 13])
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    save_file({"token_ids": token_ids, "hidden_states": hidden_states}, source)

    last_hidden_states = extract_last_token_hidden_states(source)

    assert torch.equal(last_hidden_states, hidden_states[-1])


def test_offline_hidden_states_deletes_batch_connector_files(
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
        def generate(self, prompts, sampling_params, use_tqdm):
            assert prompts == ["a", "b"]
            assert use_tqdm is False
            return [
                SimpleNamespace(
                    kv_transfer_params={"hidden_states_path": str(path)}
                )
                for path in paths
            ]

    result = offline_hidden_states(
        FakeLLM(), object(), [Sample(0, "a"), Sample(1, "b")]
    )

    assert len(result) == 2
    assert torch.equal(result[0], torch.zeros(1, 4))
    assert torch.equal(result[1], torch.ones(1, 4))
    assert all(not path.exists() for path in paths)
    assert all(not Path(f"{path}.lock").exists() for path in paths)


def test_process_batch_starts_online_work_before_offline_call(
    tmp_path: Path,
) -> None:
    online_started = threading.Event()
    allow_online_finish = threading.Event()
    hidden_file = tmp_path / "hidden.safetensors"

    def fake_request(*args, **kwargs):
        online_started.set()
        assert allow_online_finish.wait(timeout=2)
        return {
            "usage": {"completion_tokens": 17},
            "choices": [{"finish_reason": "stop"}],
        }

    class FakeLLM:
        def generate(self, prompts, sampling_params, use_tqdm):
            assert online_started.wait(timeout=2)
            save_file(
                {
                    "token_ids": torch.tensor([1]),
                    "hidden_states": torch.ones(1, 1, 4),
                },
                hidden_file,
            )
            allow_online_finish.set()
            return [
                SimpleNamespace(
                    kv_transfer_params={"hidden_states_path": str(hidden_file)}
                )
            ]

    with ThreadPoolExecutor(max_workers=1) as executor:
        hidden, generations = process_batch(
            [Sample(0, "prompt")],
            FakeLLM(),
            object(),
            make_args(),
            executor,
            request_fn=fake_request,
        )

    assert len(hidden) == 1
    assert generations == [GenerationResult(17, "stop")]


def test_write_shard_and_csv_index(tmp_path: Path) -> None:
    samples = [Sample(3, "a"), Sample(9, "b")]
    hidden_states = [torch.zeros(1, 4), torch.ones(1, 4)]
    generations = [GenerationResult(10, "stop"), GenerationResult(20, "length")]

    shard = write_shard(tmp_path, 0, hidden_states)
    results = tmp_path / "samples.csv"
    append_results(results, "/data/forelen.csv", samples, shard, generations)

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
            "completion_tokens": "10",
        },
        {
            "source_file": "/data/forelen.csv",
            "source_row": "9",
            "hidden_states_path": str(shard),
            "hidden_states_row": "1",
            "completion_tokens": "20",
        },
    ]
