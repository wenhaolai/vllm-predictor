"""Online extraction tests using real .pt/safetensors files and a fake HTTP client."""

import csv
import sys
import threading
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vllm_predictor import extract_forelen_online as online


class FakeClient:
    def __init__(self, dumps, suffix=""):
        self.dumps = dumps
        self.suffix = suffix
        self.payloads = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def resolve_model(self, model):
        return model or "test-model"

    def tokenize_prompt(self, model, prompt):
        return len(prompt), tuple(range(len(prompt)))

    def tokenize_chat(self, model, messages, thinking):
        count = sum(len(message["content"]) for message in messages) + 10 + int(thinking)
        return count, tuple(range(count))

    def chat(self, payload):
        prompt = payload["messages"][-1]["content"]
        value = int(prompt[1:])
        request_id = f"request-{value}"
        hidden_id = request_id + self.suffix
        count, _ = self.tokenize_chat(payload["model"], payload["messages"], payload["chat_template_kwargs"]["enable_thinking"])
        with self.lock:
            self.payloads.append(payload)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.01 * (3 - value % 3))
        torch.save({
            "request_id": hidden_id, "num_prompt_tokens": count, "token_position": count - 1,
            "feature_type": "target_model_final_output_last_prompt_token",
            "hidden_state": torch.full((4,), value, dtype=torch.float16),
        }, self.dumps / f"{hidden_id}.pt")
        with self.lock:
            self.active -= 1
        return {
            "id": request_id,
            "usage": {"prompt_tokens": count, "completion_tokens": value + 1},
            "choices": [{"finish_reason": "length" if value == 3 else "stop", "message": {"content": "answer"}}],
        }


def setup_job(tmp_path, *extra):
    source = tmp_path / "input.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["user_prompt_content"])
        writer.writerows([["q0"], [""], ["q1"], ["q2"], ["q3"], [" "], ["q4"]])
    dumps = tmp_path / "server"
    dumps.mkdir()
    args = online.parse_args([
        "--base-url", "http://localhost:8001/v1", "--input-file", str(source),
        "--hidden-states-dir", str(dumps), "--output-dir", str(tmp_path / "output"),
        "--shard-size", "3", "--batch-size", "2", "--system-prompt", "",
        "--file-timeout", "0", *extra,
    ])
    return args, FakeClient(dumps, args.request_id_suffix)


@pytest.mark.parametrize("thinking", [False, True])
def test_nested_batches_csv_alignment_and_cleanup(tmp_path, thinking):
    """范围：N=3/n=2、尾批、并发乱序、原始行号、thinking、token 计数、分片映射与清理。"""
    args, client = setup_job(tmp_path, "--thinking" if thinking else "--no-thinking")
    unrelated = args.hidden_states_dir / "another-job.pt"
    unrelated.write_bytes(b"keep")
    manifest = online.extract(args, client)
    with manifest.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    assert [int(row["source_row"]) for row in records] == [0, 2, 3, 4, 6]
    assert [int(row["hidden_states_row"]) for row in records] == [0, 1, 2, 0, 1]
    assert 1 < client.max_active <= 2
    assert all(p["chat_template_kwargs"]["enable_thinking"] is thinking for p in client.payloads)
    assert all(p["max_tokens"] == 8192 for p in client.payloads)
    shards = sorted(args.output_dir.glob("*.safetensors"))
    assert [load_file(str(p))["hidden_states"].shape for p in shards] == [(3, 4), (2, 4)]
    for index, row in enumerate(records):
        assert row["request_id"] == f"request-{index}"
        assert int(row["prompt_length"]) == 2
        assert int(row["input_tokens"]) == 12 + int(thinking)
        assert int(row["output_tokens"]) == index + 1
        assert row["finish_reason"] == ("length" if index == 3 else "stop")
        tensor = load_file(row["hidden_states_path"])[row["hidden_states_key"]][int(row["hidden_states_row"])]
        torch.testing.assert_close(tensor, torch.full((4,), index, dtype=torch.float16))
    assert list(args.hidden_states_dir.iterdir()) == [unrelated]


def test_suffix_mapping_and_limit(tmp_path):
    """范围：显式 engine ID 后缀与非空样本 limit，CSV 同时保留返回 ID 和文件 ID。"""
    args, client = setup_job(tmp_path, "--request-id-suffix=-0", "--limit", "1")
    manifest = online.extract(args, client)
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["request_id"] == "request-0"
    assert rows[0]["hidden_request_id"] == "request-0-0"
    assert not list(args.hidden_states_dir.iterdir())


def test_failed_shard_preserves_all_source_files(tmp_path, monkeypatch):
    """范围：大文件写入失败时，已生成的小文件全部保留，不写入成功 CSV 记录。"""
    args, client = setup_job(tmp_path)
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(online, "write_shard", fail)
    with pytest.raises(OSError, match="disk full"):
        online.extract(args, client)
    assert len(list(args.hidden_states_dir.glob("*.pt"))) == 3
    with (args.output_dir / "results.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == []


def test_csv_flush_failure_preserves_source_files(tmp_path, monkeypatch):
    """范围：分片已保存但 CSV 持久化失败时，仍不能删除小文件。"""
    args, client = setup_job(tmp_path)
    original = online.os.fsync
    calls = []
    def fail_second(fd):
        calls.append(fd)
        if len(calls) == 2:
            raise OSError("manifest flush failed")
        return original(fd)
    monkeypatch.setattr(online.os, "fsync", fail_second)
    with pytest.raises(OSError, match="manifest flush failed"):
        online.extract(args, client)
    assert len(list(args.hidden_states_dir.glob("*.pt"))) == 3
    assert len(list(args.output_dir.glob("*.safetensors"))) == 1


def test_missing_and_mismatched_files_are_not_deleted(tmp_path):
    """范围：文件等待超时、ID/长度/层来源/维度错误均报错，保留原文件。"""
    with pytest.raises(FileNotFoundError):
        online.load_state(tmp_path, "missing", 4, 0)
    valid = {"request_id": "req", "num_prompt_tokens": 4, "token_position": 3,
             "feature_type": "target_model_final_output_last_prompt_token", "hidden_state": torch.ones(4)}
    for change in ({"request_id": "other"}, {"num_prompt_tokens": 9}, {"token_position": 4},
                   {"feature_type": "mtp"}, {"hidden_state": torch.ones(2, 4)}):
        path = tmp_path / "req.pt"
        torch.save(valid | change, path)
        with pytest.raises(ValueError):
            online.load_state(tmp_path, "req", 4, 0)
        assert path.exists()
    with pytest.raises(ValueError, match="Unsafe"):
        online.load_state(tmp_path, "../outside", 4, 0)


def test_shape_mismatch_does_not_publish_shard(tmp_path):
    """范围：维度或 dtype 不一致时拒绝合并，不能生成无效分片。"""
    for tensor in (torch.ones(5), torch.ones(4, dtype=torch.float16)):
        with pytest.raises(ValueError, match="matching"):
            online.write_shard(tmp_path / "bad.safetensors", [torch.ones(4), tensor], False)
    assert not list(tmp_path.iterdir())


def test_nonempty_and_nested_output_directories_are_rejected(tmp_path):
    """范围：拒绝覆盖旧结果，拒绝输出目录与 server 清空目录嵌套。"""
    args, client = setup_job(tmp_path)
    args.output_dir.mkdir()
    marker = args.output_dir / "existing"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="empty"):
        online.extract(args, client)
    assert marker.read_text() == "keep"
    args.output_dir = args.hidden_states_dir / "output"
    with pytest.raises(ValueError, match="separate"):
        online.extract(args, client)
