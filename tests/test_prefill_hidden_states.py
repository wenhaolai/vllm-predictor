"""Tests for model_runner_v1's final-prompt hidden-state export.

From the project root (PyTorch required; pytest is optional):
    python tests/test_prefill_hidden_states.py -v
    python -m pytest tests/test_prefill_hidden_states.py -v

Inside a matching vLLM/Ascend environment, import the real runner:
    python tests/test_prefill_hidden_states.py --native-runner -v
    python tests/test_prefill_hidden_states.py --native-runner --device npu:0 -v

The default CPU mode compiles the three production methods directly from the
local source AST, without importing the NPU dependency tree. It does not copy
their implementation. Native mode imports the class from the local checkout.
Both modes bypass __init__ and mock scheduler/batch metadata and the upstream
state updater. They do NOT test model execution, initialization/TP rank gating,
graph replay, or whether _prepare_inputs corrects asynchronous MTP counts.
The NPU option tests the helper's tensor operations/transfers on one device,
not a four-rank end-to-end inference run. No model weights are required.
"""

import argparse
import ast
import hashlib
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def _runner_source_path():
    return (
        Path(__file__).resolve().parents[1]
        / "vllm-ascend/vllm_ascend/worker/model_runner_v1.py"
    )


def _load_isolated_runner():
    """Compile unchanged production methods, stubbing only their parent class."""
    source_path = _runner_source_path()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    runner_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
    )
    method_names = {
        "_plan_prefill_hidden_states", "_save_prefill_hidden_states", "_update_states"
    }
    methods = [
        node for node in runner_node.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    if {node.name for node in methods} != method_names:
        raise AssertionError("Production export methods were removed or renamed")

    class StubGPUModelRunner:
        def _update_states(self, scheduler_output):
            raise AssertionError("Tests must mock the upstream state update")

    isolated_class = ast.ClassDef(
        name="NPUModelRunner",
        bases=[ast.Name(id="StubGPUModelRunner", ctx=ast.Load())],
        keywords=[], body=methods, decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            isolated_class,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "hashlib": hashlib, "StubGPUModelRunner": StubGPUModelRunner}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["NPUModelRunner"]


class TestPrefillHiddenStates(unittest.TestCase):
    native_runner = False
    test_device = "cpu"

    @classmethod
    def setUpClass(cls):
        if cls.test_device.startswith("npu"):
            importlib.import_module("torch_npu")
            torch.npu.set_device(cls.test_device)
        if cls.native_runner:
            checkout = _runner_source_path().parents[2]
            sys.path.insert(0, str(checkout))
            module = importlib.import_module("vllm_ascend.worker.model_runner_v1")
            if Path(module.__file__).resolve() != _runner_source_path().resolve():
                raise RuntimeError("Native mode imported a different checkout of model_runner_v1")
            cls.runner_class = module.NPUModelRunner
        else:
            cls.runner_class = _load_isolated_runner()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output_dir = Path(temporary.name)
        self.runner = self.runner_class.__new__(self.runner_class)
        self.runner.device = torch.device(self.test_device)
        self.runner._prefill_hidden_states_dir = self.output_dir
        self.runner._saved_prefill_request_ids = set()
        self.runner.use_async_scheduling = False
        self.runner.requests = {}

    def _batch(self, ids, prompt_lengths, counts, computed, cpu_computed=None):
        self.runner.input_batch = SimpleNamespace(
            req_ids=list(ids), num_prompt_tokens=list(prompt_lengths),
            num_computed_tokens_cpu=list(computed if cpu_computed is None else cpu_computed),
        )
        self.runner.num_computed_tokens = torch.tensor(
            computed, device=self.runner.device, dtype=torch.int32
        )
        return SimpleNamespace(num_scheduled_tokens=dict(zip(ids, counts)))

    def _states(self, rows, dtype=torch.float32):
        return torch.arange(rows * 4).reshape(rows, 4).to(device=self.runner.device, dtype=dtype)

    def _export(self, scheduler, hidden_states):
        plan = self.runner._plan_prefill_hidden_states(scheduler)
        self.runner._save_prefill_hidden_states(hidden_states, plan)

    def _path(self, req_id):
        return self.output_dir / (hashlib.sha256(req_id.encode("utf-8")).hexdigest() + ".pt")

    def _read(self, req_id):
        return torch.load(self._path(req_id), map_location="cpu", weights_only=True)

    def test_disabled_export_does_not_access_batch_or_create_files(self):
        """范围：未启用导出时，不读取请求状态、不访问隐藏张量、不产生文件。"""
        self.runner._prefill_hidden_states_dir = None
        plan = self.runner._plan_prefill_hidden_states(None)
        self.assertEqual(plan, [])
        self.runner._save_prefill_hidden_states(None, plan)
        self.assertEqual(list(self.output_dir.iterdir()), [])

    def test_unchunked_prefill_exports_last_prompt_token_and_metadata(self):
        """范围：完整 Prefill 的末 token、文件元数据、向量维度与去重记录。"""
        scheduler = self._batch(["request"], [5], [5], [0])
        states = self._states(5)
        self._export(scheduler, states)
        data = self._read("request")
        self.assertEqual(data["request_id"], "request")
        self.assertEqual(data["num_prompt_tokens"], 5)
        self.assertEqual(data["token_position"], 4)
        self.assertEqual(data["feature_type"], "target_model_final_output_last_prompt_token")
        self.assertEqual(data["hidden_state"].shape, (4,))
        torch.testing.assert_close(data["hidden_state"], states[4].cpu())
        self.assertEqual(self.runner._saved_prefill_request_ids, {"request"})

    def test_chunked_prefill_saves_only_on_final_chunk(self):
        """范围：连续三个 Prefill chunk，前两轮不落盘，最后一轮保存正确位置。"""
        for computed in (0, 2):
            scheduler = self._batch(["chunked"], [6], [2], [computed])
            self._export(scheduler, self._states(2))
            self.assertEqual(list(self.output_dir.iterdir()), [])
            self.assertEqual(self.runner._saved_prefill_request_ids, set())
        scheduler = self._batch(["chunked"], [6], [2], [4])
        states = self._states(2)
        self._export(scheduler, states)
        torch.testing.assert_close(self._read("chunked")["hidden_state"], states[1].cpu())

    def test_mixed_mtp_batch_uses_packed_token_offsets(self):
        """范围：四 token MTP Decode、未完成 Prefill、两个完成请求混排，排除 padding。"""
        scheduler = self._batch(
            ["decode", "partial", "last", "single"],
            [2, 6, 5, 1], [4, 2, 2, 1], [8, 0, 3, 0],
        )
        states = self._states(12)  # 9 real tokens + 3 padding rows.
        self._export(scheduler, states)
        self.assertFalse(self._path("decode").exists())
        self.assertFalse(self._path("partial").exists())
        torch.testing.assert_close(self._read("last")["hidden_state"], states[7].cpu())
        torch.testing.assert_close(self._read("single")["hidden_state"], states[8].cpu())
        self.assertEqual(len(list(self.output_dir.glob("*.pt"))), 2)

    def test_decode_only_batch_is_not_exported(self):
        """范围：C 等于或大于 prompt 长度的请求均不再次作为 Prefill 导出。"""
        scheduler = self._batch(["boundary", "later"], [3, 3], [1, 4], [3, 9])
        self._export(scheduler, self._states(5))
        self.assertEqual(list(self.output_dir.iterdir()), [])

    def test_device_counts_override_optimistic_cpu_counts(self):
        """范围：CPU 计数与修正后的设备计数冲突时，仅设备计数决定是否完成。"""
        scheduler = self._batch(
            ["incomplete", "complete"], [5, 5], [2, 2], [1, 3], cpu_computed=[4, 99]
        )
        states = self._states(4)
        self._export(scheduler, states)
        self.assertFalse(self._path("incomplete").exists())
        torch.testing.assert_close(self._read("complete")["hidden_state"], states[3].cpu())

    def test_plan_is_independent_of_mutated_batch_metadata(self):
        """范围：计划建立后，请求 ID、prompt 长度、调度字典的修改不污染快照。"""
        scheduler = self._batch(["original"], [3], [3], [0])
        plan = self.runner._plan_prefill_hidden_states(scheduler)
        self.runner.input_batch.req_ids[0] = "changed"
        self.runner.input_batch.num_prompt_tokens[0] = 999
        scheduler.num_scheduled_tokens["original"] = 999
        self.runner._save_prefill_hidden_states(self._states(3), plan)
        self.assertEqual(self._read("original")["num_prompt_tokens"], 3)
        self.assertFalse(self._path("changed").exists())

    def test_batch_reorder_recomputes_indices_on_next_step(self):
        """范围：跨轮请求换位后，重建计划，使用新位置保存最后一个 chunk。"""
        scheduler = self._batch(["prefill", "decode"], [6, 2], [2, 4], [0, 9])
        self._export(scheduler, self._states(6))
        scheduler = self._batch(["decode", "prefill"], [2, 6], [4, 4], [13, 2])
        states = self._states(8)
        self._export(scheduler, states)
        torch.testing.assert_close(self._read("prefill")["hidden_state"], states[7].cpu())

    def test_crossing_prompt_boundary_does_not_select_last_scheduled_token(self):
        """范围：S 超过剩余 prompt 长度时仍取 prompt 末位置，而非本轮最后位置。"""
        scheduler = self._batch(["cross"], [3], [4], [1])
        states = self._states(4)
        self._export(scheduler, states)
        torch.testing.assert_close(self._read("cross")["hidden_state"], states[1].cpu())

    def test_saved_request_does_not_shift_following_request_offset(self):
        """范围：跳过已保存请求时仍累计其 token 数，后续请求不会错位或覆盖旧文件。"""
        scheduler = self._batch(["saved"], [2], [2], [0])
        self._export(scheduler, self._states(2))
        original_bytes = self._path("saved").read_bytes()
        scheduler = self._batch(["saved", "new"], [2, 3], [2, 3], [0, 0])
        states = self._states(5)
        self._export(scheduler, states)
        self.assertEqual(self._path("saved").read_bytes(), original_bytes)
        torch.testing.assert_close(self._read("new")["hidden_state"], states[4].cpu())

    def test_empty_batch_and_zero_token_request_are_skipped(self):
        """范围：空 batch 和零调度量请求不会索引 hidden states 或产生文件。"""
        scheduler = self._batch([], [], [], [])
        self._export(scheduler, None)
        scheduler = self._batch(["zero"], [3], [0], [0])
        self._export(scheduler, None)
        self.assertEqual(list(self.output_dir.iterdir()), [])

    def test_saved_tensor_preserves_dtype_and_owns_only_one_row(self):
        """范围：FP32/FP16/BF16 保持 dtype，单请求不序列化整批 storage，源张量可安全复用。"""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                req_id = str(dtype)
                scheduler = self._batch([req_id], [3], [3], [0])
                states = self._states(3, dtype=dtype)
                expected = states[2].cpu().clone()
                self._export(scheduler, states)
                states.fill_(-1)
                feature = self._read(req_id)["hidden_state"]
                self.assertEqual(feature.dtype, dtype)
                self.assertEqual(feature.device.type, "cpu")
                self.assertEqual(feature.untyped_storage().nbytes(), feature.numel() * feature.element_size())
                torch.testing.assert_close(feature, expected)

    def test_request_id_cannot_escape_output_directory(self):
        """范围：含路径分隔符、绝对路径和 Unicode 的 ID 只生成输出目录内的哈希文件。"""
        ids = ["../../escape", "C:\\outside\\file", "/tmp/outside", "请求/一"]
        scheduler = self._batch(ids, [1] * 4, [1] * 4, [0] * 4)
        self._export(scheduler, self._states(4))
        self.assertEqual(len(list(self.output_dir.iterdir())), 4)
        for req_id in ids:
            self.assertEqual(self._read(req_id)["request_id"], req_id)
            self.assertEqual(self._path(req_id).parent, self.output_dir)

    def test_write_failure_cleans_temporary_file_and_allows_retry(self):
        """范围：写盘失败向上传播、清理半成品、不标记已保存，并允许同请求重试。"""
        scheduler = self._batch(["retry"], [2], [2], [0])

        def fail_save(payload, path):
            Path(path).write_bytes(b"incomplete")
            raise OSError("disk full")

        with patch.object(torch, "save", side_effect=fail_save):
            with self.assertRaisesRegex(OSError, "disk full"):
                self._export(scheduler, self._states(2))
        self.assertEqual(list(self.output_dir.iterdir()), [])
        self.assertNotIn("retry", self.runner._saved_prefill_request_ids)
        self._export(scheduler, self._states(2))
        self.assertTrue(self._path("retry").exists())

    def test_atomic_replace_failure_preserves_existing_file(self):
        """范围：最终替换失败时保留原文件、删除临时文件，不错误标记成功。"""
        scheduler = self._batch(["replace"], [2], [2], [0])
        self._path("replace").write_bytes(b"previous file")
        with patch.object(Path, "replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                self._export(scheduler, self._states(2))
        self.assertEqual(self._path("replace").read_bytes(), b"previous file")
        self.assertFalse(list(self.output_dir.glob("*.tmp")))
        self.assertNotIn("replace", self.runner._saved_prefill_request_ids)

    def test_finished_requests_clear_dedup_state_and_delegate_to_parent(self):
        """范围：仅清理已结束请求的去重记录，保留活动请求并原样调用上游状态更新。"""
        self.runner._saved_prefill_request_ids = {"finished", "active"}
        scheduler = SimpleNamespace(finished_req_ids={"finished", "unknown"}, scheduled_cached_reqs=None)
        sentinel = object()
        with patch.object(self.runner_class.__bases__[0], "_update_states", return_value=sentinel) as update:
            self.assertIs(self.runner._update_states(scheduler), sentinel)
            update.assert_called_once_with(scheduler)
        self.assertEqual(self.runner._saved_prefill_request_ids, {"active"})

    def test_cleanup_preserves_existing_async_rewind_guard(self):
        """范围：新增去重清理不破坏原有异步 KV 重算保护：回退请求的草稿长度清零。"""
        self.runner.use_async_scheduling = True
        self.runner.requests = {
            "rewind": SimpleNamespace(num_computed_tokens=8, prev_num_draft_len=3),
            "normal": SimpleNamespace(num_computed_tokens=4, prev_num_draft_len=3),
        }
        scheduler = SimpleNamespace(
            finished_req_ids=set(),
            scheduled_cached_reqs=SimpleNamespace(req_ids=["rewind", "normal", "missing"], num_computed_tokens=[2, 4, 0]),
        )
        with patch.object(self.runner_class.__bases__[0], "_update_states", return_value=None):
            self.runner._update_states(scheduler)
        self.assertEqual(self.runner.requests["rewind"].prev_num_draft_len, 0)
        self.assertEqual(self.runner.requests["normal"].prev_num_draft_len, 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--native-runner", action="store_true")
    parser.add_argument("--device", default="cpu")
    options, unittest_args = parser.parse_known_args()
    TestPrefillHiddenStates.native_runner = options.native_runner
    TestPrefillHiddenStates.test_device = options.device
    unittest.main(argv=[sys.argv[0], *unittest_args])
