from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))

from extract_forelen import (  # noqa: E402
    CSV_FIELDS,
    ExtractedSample,
    ExtractionConfig,
    ResultStore,
    Sample,
    ShardWriter,
    chat_payload,
    extract_last_token_hidden_states,
    iter_csv_samples,
    per_sample_seed,
    process_sample,
)


def test_iter_csv_samples_uses_only_prompt_column(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    with source.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("user_prompt_content", "response_content", "target_length"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "user_prompt_content": "first prompt",
                "response_content": "ignored",
                "target_length": 10,
            }
        )
        writer.writerow(
            {
                "user_prompt_content": "second prompt",
                "response_content": "also ignored",
                "target_length": 20,
            }
        )

    samples = list(iter_csv_samples(tmp_path))

    assert [sample.prompt for sample in samples] == ["first prompt", "second prompt"]
    assert [sample.row_index for sample in samples] == [0, 1]
    assert samples[0].sample_id != samples[1].sample_id


def test_extract_last_token_removes_only_token_dimension(tmp_path: Path) -> None:
    source = tmp_path / "full.safetensors"
    token_ids = torch.tensor([11, 12, 13])
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    save_file({"token_ids": token_ids, "hidden_states": hidden_states}, source)

    last_hidden_states, metadata = extract_last_token_hidden_states(source)

    assert metadata["hidden_states_shape"] == [2, 4]
    assert metadata["last_prompt_token_id"] == 13
    assert torch.equal(last_hidden_states, hidden_states[-1])


def test_rl_sampling_payload_is_not_greedy() -> None:
    payload = chat_payload(
        "prompt",
        "model",
        max_tokens=2048,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        seed=None,
        request_hidden_states=False,
    )

    assert payload["temperature"] > 0
    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 20
    assert "seed" not in payload
    assert "kv_transfer_params" not in payload


def test_per_sample_seed_is_reproducible_but_distinct() -> None:
    assert per_sample_seed(None, "1" * 24) is None
    first = per_sample_seed(42, "1" * 24)
    assert first == per_sample_seed(42, "1" * 24)
    assert first != per_sample_seed(42, "2" * 24)


def test_result_store_repairs_csv_from_completed_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    results = tmp_path / "samples.csv"
    shard = tmp_path / "shard.safetensors"
    save_file(
        {
            "hidden_states": torch.zeros(1, 4),
            "completion_tokens": torch.tensor([42], dtype=torch.int32),
        },
        shard,
    )
    completed = {
        "sample_id": "sample-1",
        "status": "completed",
        "hidden_states_path": str(shard),
        "hidden_states_row": 0,
        "completion_tokens": 42,
    }
    manifest.write_text(json.dumps(completed) + "\n", encoding="utf-8")

    ResultStore(manifest, results)

    with results.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == CSV_FIELDS
    assert len(rows) == 1
    assert rows[0]["hidden_states_path"] == completed["hidden_states_path"]
    assert rows[0]["hidden_states_row"] == "0"
    assert rows[0]["completion_tokens"] == "42"


def test_completed_record_is_not_resumable_when_feature_is_missing(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    completed = {
        "sample_id": "sample-missing",
        "status": "completed",
        "hidden_states_path": str(tmp_path / "missing.safetensors"),
        "hidden_states_row": 0,
        "completion_tokens": 42,
    }
    manifest.write_text(json.dumps(completed) + "\n", encoding="utf-8")
    store = ResultStore(manifest, tmp_path / "samples.csv")

    assert not store.is_materialized(completed)


def test_shard_writer_batches_samples_and_flushes_partial_shard(
    tmp_path: Path,
) -> None:
    store = ResultStore(tmp_path / "manifest.jsonl", tmp_path / "samples.csv")
    writer = ShardWriter(tmp_path / "output", shard_size=2, store=store)

    def extracted(index: int) -> ExtractedSample:
        return ExtractedSample(
            record={
                "sample_id": f"sample-{index}",
                "source_file": "train.csv",
                "row_index": index,
                "completion_tokens": 10 + index,
                "finish_reason": "stop",
                "truncated": False,
            },
            hidden_states=torch.full((4,), float(index)),
        )

    assert writer.add(extracted(0)) == []
    first_records = writer.add(extracted(1))
    assert [record["hidden_states_row"] for record in first_records] == [0, 1]
    assert writer.add(extracted(2)) == []
    final_records = writer.flush()

    shard_files = sorted((tmp_path / "output" / "shards").glob("*.safetensors"))
    assert len(shard_files) == 2
    assert final_records[0]["hidden_states_row"] == 0
    with safe_open(shard_files[0], framework="pt", device="cpu") as tensors:
        assert tensors.get_tensor("hidden_states").shape == (2, 4)
        assert tensors.get_tensor("completion_tokens").tolist() == [10, 11]
    with safe_open(shard_files[1], framework="pt", device="cpu") as tensors:
        assert tensors.get_tensor("hidden_states").shape == (1, 4)
        assert tensors.get_tensor("completion_tokens").tolist() == [12]


def test_process_sample_deletes_full_files_and_commits_shard(tmp_path: Path) -> None:
    prefill_file = tmp_path / "server-prefill.safetensors"
    generation_file = tmp_path / "server-generation.safetensors"
    source_tensors = {
        "token_ids": torch.tensor([21, 22]),
        "hidden_states": torch.arange(16, dtype=torch.float32).reshape(2, 1, 8),
    }
    save_file(source_tensors, prefill_file)
    save_file(source_tensors, generation_file)
    responses = iter(
        [
            {
                "id": "prefill-id",
                "kv_transfer_params": {"hidden_states_path": str(prefill_file)},
            },
            {
                "id": "generation-id",
                "usage": {"completion_tokens": 17},
                "choices": [{"finish_reason": "stop"}],
                "kv_transfer_params": {"hidden_states_path": str(generation_file)},
            },
        ]
    )
    request_count = 0

    def fake_request(*args, **kwargs):
        nonlocal request_count
        request_count += 1
        return next(responses)

    sample = Sample(
        sample_id="sample-2",
        source_file="train.csv",
        row_index=0,
        prompt="hello",
    )
    store = ResultStore(tmp_path / "manifest.jsonl", tmp_path / "samples.csv")
    config = ExtractionConfig(
        base_url="http://prefill",
        generation_base_url="http://generation",
        model="model",
        output_dir=tmp_path / "output",
    )

    extracted = process_sample(sample, config, request_fn=fake_request)

    assert isinstance(extracted, ExtractedSample)
    assert extracted.record["completion_tokens"] == 17
    assert request_count == 2
    assert not prefill_file.exists()
    assert not generation_file.exists()
    writer = ShardWriter(config.output_dir, shard_size=1, store=store)
    records = writer.add(extracted)
    assert len(records) == 1
    completed = records[0]
    shard_path = Path(completed["hidden_states_path"])
    assert shard_path.is_file()
    assert completed["hidden_states_row"] == 0
    assert list(store.history) == [sample.sample_id]
    assert all(
        "user_prompt_content" not in record
        for record in store.history[sample.sample_id]
    )
    with safe_open(shard_path, framework="pt", device="cpu") as tensors:
        assert torch.equal(
            tensors.get_tensor("hidden_states"),
            source_tensors["hidden_states"][-1].unsqueeze(0),
        )
        assert tensors.get_tensor("hidden_states").shape == (1, 1, 8)
        assert tensors.get_tensor("completion_tokens").tolist() == [17]
    with store.results_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == CSV_FIELDS
    assert rows == [
        {
            "hidden_states_path": completed["hidden_states_path"],
            "hidden_states_row": "0",
            "completion_tokens": "17",
        }
    ]

