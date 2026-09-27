from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_predictor.data_extraction import (  # noqa: E402
    CSV_FIELDS,
    ExtractionConfig,
    ResultStore,
    Sample,
    chat_payload,
    iter_csv_samples,
    per_sample_seed,
    process_sample,
    save_last_token_hidden_states,
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


def test_save_last_token_preserves_token_dimension(tmp_path: Path) -> None:
    source = tmp_path / "full.safetensors"
    destination = tmp_path / "compact" / "last.safetensors"
    token_ids = torch.tensor([11, 12, 13])
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    save_file({"token_ids": token_ids, "hidden_states": hidden_states}, source)

    metadata = save_last_token_hidden_states(source, destination)

    assert metadata["source_hidden_states_shape"] == [3, 2, 4]
    assert metadata["hidden_states_shape"] == [1, 2, 4]
    assert metadata["last_prompt_token_id"] == 13
    with safe_open(destination, framework="pt", device="cpu") as tensors:
        assert torch.equal(tensors.get_tensor("token_ids"), token_ids[-1:])
        assert torch.equal(
            tensors.get_tensor("hidden_states"), hidden_states[-1:]
        )


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
    )

    assert payload["temperature"] > 0
    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 20
    assert "seed" not in payload


def test_per_sample_seed_is_reproducible_but_distinct() -> None:
    assert per_sample_seed(None, "1" * 24) is None
    first = per_sample_seed(42, "1" * 24)
    assert first == per_sample_seed(42, "1" * 24)
    assert first != per_sample_seed(42, "2" * 24)


def test_result_store_repairs_csv_from_completed_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    results = tmp_path / "samples.csv"
    completed = {field: "" for field in CSV_FIELDS}
    completed.update(
        {
            "sample_id": "sample-1",
            "status": "completed",
            "hidden_states_shape": [1, 1, 4],
            "completion_tokens": 42,
        }
    )
    manifest.write_text(json.dumps(completed) + "\n", encoding="utf-8")

    ResultStore(manifest, results)

    with results.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "sample-1"
    assert rows[0]["completion_tokens"] == "42"


def test_completed_record_is_not_resumable_when_feature_is_missing(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    completed = {
        "sample_id": "sample-missing",
        "status": "completed",
        "hidden_states_path": str(tmp_path / "missing.safetensors"),
    }
    manifest.write_text(json.dumps(completed) + "\n", encoding="utf-8")
    store = ResultStore(manifest, tmp_path / "samples.csv")

    assert store.resumable_hidden_record("sample-missing") is None


def test_process_sample_is_resumable_and_deletes_full_files(tmp_path: Path) -> None:
    prefill_file = tmp_path / "server-prefill.safetensors"
    generation_file = tmp_path / "server-generation.safetensors"
    tensors = {
        "token_ids": torch.tensor([21, 22]),
        "hidden_states": torch.arange(16, dtype=torch.float32).reshape(2, 1, 8),
    }
    save_file(tensors, prefill_file)
    save_file(tensors, generation_file)
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
        prompt_sha256="digest",
    )
    store = ResultStore(tmp_path / "manifest.jsonl", tmp_path / "samples.csv")
    config = ExtractionConfig(
        base_url="http://prefill",
        generation_base_url="http://generation",
        model="model",
        output_dir=tmp_path / "output",
    )

    completed = process_sample(sample, config, store, request_fn=fake_request)
    skipped = process_sample(sample, config, store, request_fn=fake_request)

    assert completed["status"] == "completed"
    assert completed["completion_tokens"] == 17
    assert skipped["status"] == "skipped"
    assert request_count == 2
    assert not prefill_file.exists()
    assert not generation_file.exists()
    assert Path(completed["hidden_states_path"]).is_file()
    statuses = [record["status"] for record in store.history[sample.sample_id]]
    assert statuses == ["hidden_saved", "completed"]

