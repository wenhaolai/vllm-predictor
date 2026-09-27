#!/usr/bin/env python3
"""Extract ForeLen last-token prefill features and fresh generation lengths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping


CSV_FIELDS = (
    "hidden_states_path",
    "hidden_states_row",
    "completion_tokens",
)


@dataclass(frozen=True)
class Sample:
    """One stable row from a source CSV file."""

    sample_id: str
    source_file: str
    row_index: int
    prompt: str


@dataclass(frozen=True)
class ExtractionConfig:
    """Server and output options shared by all samples."""

    base_url: str
    generation_base_url: str
    model: str
    output_dir: Path
    generation_max_tokens: int = 2048
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    timeout: float = 3600.0
    retries: int = 3
    retry_backoff: float = 2.0
    keep_server_hidden_states: bool = False


@dataclass(frozen=True)
class ExtractedSample:
    """A completed pair of requests waiting to be committed to a shard."""

    record: dict[str, Any]
    hidden_states: Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_sample_id(source_file: str, row_index: int, prompt: str) -> str:
    """Return a deterministic ID without persisting the prompt in metadata."""

    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    identity = f"{source_file}\0{row_index}\0{prompt_digest}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()[:24]


def iter_csv_samples(
    input_dir: Path,
    pattern: str = "*.csv",
    prompt_column: str = "user_prompt_content",
) -> Iterator[Sample]:
    """Yield prompts from all matching CSV files in deterministic order."""

    csv_files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files matching {pattern!r} were found in {input_dir}"
        )

    for csv_path in csv_files:
        relative_name = csv_path.relative_to(input_dir).as_posix()
        with csv_path.open("r", encoding="utf-8-sig", newline="") as input_file:
            reader = csv.DictReader(input_file)
            if not reader.fieldnames or prompt_column not in reader.fieldnames:
                raise ValueError(
                    f"{csv_path} does not contain required column {prompt_column!r}; "
                    f"columns={reader.fieldnames}"
                )
            for row_index, row in enumerate(reader):
                prompt = row.get(prompt_column)
                if prompt is None or not prompt.strip():
                    continue
                yield Sample(
                    sample_id=make_sample_id(relative_name, row_index, prompt),
                    source_file=relative_name,
                    row_index=row_index,
                    prompt=prompt,
                )


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    details = exc.read().decode("utf-8", errors="replace")
    return f"HTTP {exc.code}: {details}"


def post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    timeout: float,
    retries: int,
    retry_backoff: float,
) -> dict[str, Any]:
    """POST JSON with bounded exponential retry for transient failures."""

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    attempts = retries + 1
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            if not isinstance(result, dict):
                raise RuntimeError(f"Expected a JSON object from {url}")
            return result
        except urllib.error.HTTPError as exc:
            message = _http_error_message(exc)
            retryable = exc.code in {408, 409, 425, 429} or exc.code >= 500
            if attempt + 1 >= attempts or not retryable:
                raise RuntimeError(message) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            message = f"Request to {url} failed: {exc}"
            if attempt + 1 >= attempts:
                raise RuntimeError(message) from exc
        time.sleep(retry_backoff * (2**attempt))
    raise AssertionError("unreachable")


def chat_payload(
    prompt: str,
    model: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    frequency_penalty: float,
    repetition_penalty: float,
    seed: int | None,
    request_hidden_states: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "repetition_penalty": repetition_penalty,
        "stream": False,
    }
    if request_hidden_states:
        payload["kv_transfer_params"] = {"include_output_tokens": False}
    if seed is not None:
        payload["seed"] = seed
    return payload


def per_sample_seed(base_seed: int | None, sample_id: str) -> int | None:
    """Derive distinct, reproducible rollout seeds from an optional base seed."""

    if base_seed is None:
        return None
    return (base_seed + int(sample_id[:16], 16)) % (2**63 - 1)


def response_hidden_states_path(response: Mapping[str, Any]) -> Path:
    kv_params = response.get("kv_transfer_params") or {}
    path = kv_params.get("hidden_states_path")
    if not path:
        raise RuntimeError(
            "Response does not contain kv_transfer_params.hidden_states_path"
        )
    return Path(path)


def load_hidden_states(path: Path) -> dict[str, Any]:
    """Load using vLLM's synchronized connector, with a safetensors fallback."""

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            example_hidden_states_connector,
        )
    except ImportError:
        from safetensors import safe_open

        if not path.is_file():
            raise FileNotFoundError(f"Hidden-state file does not exist: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            return {
                "token_ids": tensors.get_tensor("token_ids"),
                "hidden_states": tensors.get_tensor("hidden_states"),
            }
    return example_hidden_states_connector.load_hidden_states(str(path))


def extract_last_token_hidden_states(source: Path) -> tuple[Any, dict[str, Any]]:
    """Load and return only the final prompt token on CPU.

    The token dimension is removed. Any remaining layer/hidden dimensions are
    retained so the shard format also works when a server exports more than one
    selected layer.
    """

    tensors = load_hidden_states(source)
    token_ids = tensors["token_ids"]
    hidden_states = tensors["hidden_states"]
    if token_ids.ndim != 1 or token_ids.shape[0] == 0:
        raise ValueError(f"Unexpected token_ids shape: {list(token_ids.shape)}")
    if hidden_states.ndim < 2 or hidden_states.shape[0] != token_ids.shape[0]:
        raise ValueError(
            "The leading dimensions of token_ids and hidden_states do not match: "
            f"{list(token_ids.shape)} vs {list(hidden_states.shape)}"
        )

    last_token_id = token_ids[-1].detach().cpu()
    last_hidden_states = hidden_states[-1].detach().cpu().contiguous()
    metadata = {
        "hidden_states_shape": list(last_hidden_states.shape),
        "prompt_tokens": int(token_ids.shape[0]),
        "last_prompt_token_id": int(last_token_id.item()),
    }
    return last_hidden_states, metadata


def _delete_server_file(path: Path, wait_seconds: float = 30.0) -> bool:
    """Delete one exact server-returned file after its data has been consumed."""

    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)


def _completion_metadata(response: Mapping[str, Any]) -> tuple[int, str | None]:
    usage = response.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int):
        raise RuntimeError(
            "Generation response does not contain integer usage.completion_tokens"
        )
    choices = response.get("choices") or []
    finish_reason = choices[0].get("finish_reason") if choices else None
    return completion_tokens, finish_reason


class ResultStore:
    """Durable concise manifest plus a shard-row CSV index."""

    def __init__(self, manifest_path: Path, results_path: Path) -> None:
        self.manifest_path = manifest_path
        self.results_path = results_path
        self._lock = threading.Lock()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load_manifest()
        self.latest = {
            sample_id: records[-1] for sample_id, records in self.history.items()
        }
        self._csv_keys = self._load_csv_keys()
        self._repair_csv_from_manifest()

    def _load_manifest(self) -> dict[str, list[dict[str, Any]]]:
        history: dict[str, list[dict[str, Any]]] = {}
        if not self.manifest_path.exists():
            return history
        with self.manifest_path.open("r", encoding="utf-8") as manifest:
            for line_number, line in enumerate(manifest, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {self.manifest_path}:{line_number}"
                    ) from exc
                sample_id = record.get("sample_id")
                if sample_id:
                    history.setdefault(sample_id, []).append(record)
        return history

    @staticmethod
    def _record_key(record: Mapping[str, Any]) -> tuple[str, int]:
        return (
            str(record["hidden_states_path"]),
            int(record["hidden_states_row"]),
        )

    def _load_csv_keys(self) -> set[tuple[str, int]]:
        if not self.results_path.exists() or self.results_path.stat().st_size == 0:
            return set()
        with self.results_path.open("r", encoding="utf-8", newline="") as results:
            reader = csv.DictReader(results)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise ValueError(
                    f"Unexpected columns in {self.results_path}: {reader.fieldnames}. "
                    "Use a new --output-dir for the sharded CSV format."
                )
            return {
                (row["hidden_states_path"], int(row["hidden_states_row"]))
                for row in reader
                if row.get("hidden_states_path")
                and row.get("hidden_states_row") is not None
            }

    def is_materialized(self, record: Mapping[str, Any]) -> bool:
        path = record.get("hidden_states_path")
        row = record.get("hidden_states_row")
        return (
            record.get("status") == "completed"
            and bool(path)
            and isinstance(row, int)
            and row >= 0
            and Path(str(path)).is_file()
        )

    def _append_csv_unlocked(self, records: Iterable[Mapping[str, Any]]) -> int:
        missing = [
            record
            for record in records
            if self._record_key(record) not in self._csv_keys
        ]
        if not missing:
            return 0
        write_header = (
            not self.results_path.exists() or self.results_path.stat().st_size == 0
        )
        with self.results_path.open("a", encoding="utf-8", newline="") as results:
            writer = csv.DictWriter(results, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            for record in missing:
                writer.writerow(
                    {
                        "hidden_states_path": record["hidden_states_path"],
                        "hidden_states_row": record["hidden_states_row"],
                        "completion_tokens": record["completion_tokens"],
                    }
                )
                self._csv_keys.add(self._record_key(record))
            results.flush()
            os.fsync(results.fileno())
        return len(missing)

    def _repair_csv_from_manifest(self) -> None:
        completed = []
        for records in self.history.values():
            for record in reversed(records):
                if self.is_materialized(record):
                    completed.append(record)
                    break
        with self._lock:
            self._append_csv_unlocked(completed)

    def append_manifest(self, record: Mapping[str, Any]) -> None:
        """Append one failure record; completed rows use commit_completed()."""

        materialized = dict(record)
        line = json.dumps(materialized, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            with self.manifest_path.open("a", encoding="utf-8") as manifest:
                manifest.write(line)
                manifest.flush()
                os.fsync(manifest.fileno())
            sample_id = materialized.get("sample_id")
            if sample_id:
                self.history.setdefault(sample_id, []).append(materialized)
                self.latest[sample_id] = materialized

    def commit_completed(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Commit one immutable shard's records with one fsync per index file."""

        materialized = [dict(record) for record in records]
        if not materialized:
            return
        lines = "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in materialized
        )
        with self._lock:
            with self.manifest_path.open("a", encoding="utf-8") as manifest:
                manifest.write(lines)
                manifest.flush()
                os.fsync(manifest.fileno())
            for record in materialized:
                sample_id = record.get("sample_id")
                if sample_id:
                    self.history.setdefault(sample_id, []).append(record)
                    self.latest[sample_id] = record
            self._append_csv_unlocked(materialized)


class ShardWriter:
    """Single-writer buffer for immutable safetensors dataset shards."""

    def __init__(self, output_dir: Path, shard_size: int, store: ResultStore) -> None:
        self.shards_dir = output_dir / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.store = store
        self._buffer: list[ExtractedSample] = []
        self._next_index = self._discover_next_index()

    def _discover_next_index(self) -> int:
        indices = []
        for path in self.shards_dir.glob("shard_*.safetensors"):
            try:
                indices.append(int(path.stem.removeprefix("shard_")))
            except ValueError:
                continue
        return max(indices, default=-1) + 1

    def add(self, extracted: ExtractedSample) -> list[dict[str, Any]]:
        self._buffer.append(extracted)
        if len(self._buffer) >= self.shard_size:
            return self.flush()
        return []

    def flush(self) -> list[dict[str, Any]]:
        if not self._buffer:
            return []

        import torch
        from safetensors.torch import save_file

        first_shape = tuple(self._buffer[0].hidden_states.shape)
        first_dtype = self._buffer[0].hidden_states.dtype
        for extracted in self._buffer[1:]:
            if tuple(extracted.hidden_states.shape) != first_shape:
                raise ValueError(
                    "Hidden-state shape changed within a shard: "
                    f"{first_shape} vs {tuple(extracted.hidden_states.shape)}"
                )
            if extracted.hidden_states.dtype != first_dtype:
                raise ValueError(
                    "Hidden-state dtype changed within a shard: "
                    f"{first_dtype} vs {extracted.hidden_states.dtype}"
                )

        hidden_states = torch.stack(
            [item.hidden_states for item in self._buffer], dim=0
        ).contiguous()
        completion_tokens = torch.tensor(
            [item.record["completion_tokens"] for item in self._buffer],
            dtype=torch.int32,
        )
        shard_path = (
            self.shards_dir / f"shard_{self._next_index:05d}.safetensors"
        ).resolve()
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{shard_path.stem}.",
                suffix=".safetensors",
                dir=self.shards_dir,
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
            save_file(
                {
                    "hidden_states": hidden_states,
                    "completion_tokens": completion_tokens,
                },
                temporary_name,
                metadata={
                    "selection": "last_prompt_token",
                    "samples": str(len(self._buffer)),
                },
            )
            os.replace(temporary_name, shard_path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)

        records = []
        for row, extracted in enumerate(self._buffer):
            records.append(
                {
                    **extracted.record,
                    "status": "completed",
                    "hidden_states_path": str(shard_path),
                    "hidden_states_row": row,
                }
            )
        self.store.commit_completed(records)
        self._buffer.clear()
        self._next_index += 1
        return records


def _base_record(sample: Sample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "source_file": sample.source_file,
        "row_index": sample.row_index,
    }


RequestFunction = Callable[..., dict[str, Any]]


def process_sample(
    sample: Sample,
    config: ExtractionConfig,
    *,
    request_fn: RequestFunction = post_json,
) -> ExtractedSample | dict[str, Any]:
    """Run both requests and return one in-memory training sample."""

    base = _base_record(sample)
    rollout_seed = per_sample_seed(config.seed, sample.sample_id)
    source_path: Path | None = None
    try:
        prefill_response = request_fn(
            f"{config.base_url.rstrip('/')}/v1/chat/completions",
            chat_payload(
                sample.prompt,
                config.model,
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                seed=None,
                request_hidden_states=True,
            ),
            timeout=config.timeout,
            retries=config.retries,
            retry_backoff=config.retry_backoff,
        )
        source_path = response_hidden_states_path(prefill_response)
        hidden_states, _hidden_metadata = extract_last_token_hidden_states(source_path)
    except Exception as exc:
        return {
            **base,
            "status": "hidden_failed",
            "created_at": utc_now(),
            "error": f"{type(exc).__name__}: {exc}"[:1000],
        }
    finally:
        if source_path is not None and not config.keep_server_hidden_states:
            _delete_server_file(source_path)

    try:
        generation_response = request_fn(
            f"{config.generation_base_url.rstrip('/')}/v1/chat/completions",
            chat_payload(
                sample.prompt,
                config.model,
                max_tokens=config.generation_max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                min_p=config.min_p,
                presence_penalty=config.presence_penalty,
                frequency_penalty=config.frequency_penalty,
                repetition_penalty=config.repetition_penalty,
                seed=rollout_seed,
                request_hidden_states=False,
            ),
            timeout=config.timeout,
            retries=config.retries,
            retry_backoff=config.retry_backoff,
        )
        completion_tokens, finish_reason = _completion_metadata(generation_response)

        try:
            generation_source = response_hidden_states_path(generation_response)
            if not config.keep_server_hidden_states:
                _delete_server_file(generation_source)
        except RuntimeError:
            pass

        ready = {
            **base,
            "created_at": utc_now(),
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "truncated": finish_reason == "length",
        }
        return ExtractedSample(record=ready, hidden_states=hidden_states)
    except Exception as exc:
        return {
            **base,
            "status": "generation_failed",
            "created_at": utc_now(),
            "error": f"{type(exc).__name__}: {exc}"[:1000],
        }


def pending_samples(
    samples: Iterable[Sample],
    store: ResultStore,
    limit: int | None,
) -> Iterator[Sample]:
    """Filter completed rows without materializing large source datasets."""

    yielded = 0
    for sample in samples:
        completed = next(
            (
                record
                for record in reversed(store.history.get(sample.sample_id, []))
                if store.is_materialized(record)
            ),
            None,
        )
        if completed is not None:
            continue
        if limit is not None and yielded >= limit:
            break
        yielded += 1
        yield sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read user_prompt_content from ForeLen CSVs, save the final prefill "
            "token hidden state, and measure a fresh completion length."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/laiwenhao/vllm-predictor/data"),
    )
    parser.add_argument("--glob", default="*.csv", dest="pattern")
    parser.add_argument("--prompt-column", default="user_prompt_content")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/laiwenhao/vllm-predictor/data/forelen_extracted"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--generation-base-url",
        default=None,
        help="Optional ordinary vLLM server; defaults to --base-url.",
    )
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature; must be > 0 (Qwen3.5 default: 0.6).",
    )
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional base seed. By default no seed is sent. When set, a stable "
            "distinct seed is derived for every source row."
        ),
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent samples. Start with 1, then increase after validation.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=2048,
        help="Number of samples per immutable safetensors shard (default: 2048).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many unfinished rows (useful for an MRE run).",
    )
    parser.add_argument(
        "--keep-server-hidden-states",
        action="store_true",
        help="Keep the server's full prompt-token files for debugging.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed sample instead of recording and continuing.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.shard_size < 1:
        parser.error("--shard-size must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0; greedy generation is not allowed")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.top_k < -1:
        parser.error("--top-k must be -1, 0, or a positive integer")
    if not 0 <= args.min_p <= 1:
        parser.error("--min-p must be in [0, 1]")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def report(record: dict[str, Any], counters: dict[str, int]) -> None:
    status = str(record["status"])
    counters[status] = counters.get(status, 0) + 1
    sample_id = record["sample_id"]
    if status == "completed":
        print(
            f"[{counters['processed']}] {sample_id} completed: "
            f"completion_tokens={record['completion_tokens']}, "
            f"finish_reason={record['finish_reason']}"
        )
    else:
        print(
            f"[{counters['processed']}] {sample_id} {status}: "
            f"{record.get('error', '')}",
            file=sys.stderr,
        )


def report_committed(
    records: list[dict[str, Any]], counters: dict[str, int]
) -> None:
    """Report one line per shard instead of one line per sample."""

    if not records:
        return
    counters["completed"] = counters.get("completed", 0) + len(records)
    print(
        f"[{counters['processed']}] committed {len(records)} samples to "
        f"{records[0]['hidden_states_path']} "
        f"(completed={counters['completed']})"
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    store = ResultStore(
        output_dir / "manifest.jsonl",
        output_dir / "samples.csv",
    )
    config = ExtractionConfig(
        base_url=args.base_url,
        generation_base_url=args.generation_base_url or args.base_url,
        model=args.model,
        output_dir=output_dir,
        generation_max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        timeout=args.timeout,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
        keep_server_hidden_states=args.keep_server_hidden_states,
    )
    samples = pending_samples(
        iter_csv_samples(
            args.input_dir.resolve(),
            pattern=args.pattern,
            prompt_column=args.prompt_column,
        ),
        store,
        args.limit,
    )
    shard_writer = ShardWriter(output_dir, args.shard_size, store)

    counters: dict[str, int] = {"processed": 0}
    failure_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending: set[Future[ExtractedSample | dict[str, Any]]] = set()
        exhausted = False
        stop_early = False
        while (pending or not exhausted) and not stop_early:
            while not exhausted and len(pending) < args.workers * 2:
                try:
                    sample = next(samples)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(executor.submit(process_sample, sample, config))
            if not pending:
                continue
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                counters["processed"] += 1
                if isinstance(result, ExtractedSample):
                    report_committed(shard_writer.add(result), counters)
                    continue

                store.append_manifest(result)
                report(result, counters)
                if result["status"] != "completed":
                    failure_count += 1
                    if args.fail_fast:
                        for outstanding in pending:
                            outstanding.cancel()
                        stop_early = True
                        break

    report_committed(shard_writer.flush(), counters)

    print(
        "Finished: "
        + ", ".join(f"{key}={value}" for key, value in sorted(counters.items()))
    )
    print(f"Manifest: {store.manifest_path}")
    print(f"Results:  {store.results_path}")
    return 1 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

