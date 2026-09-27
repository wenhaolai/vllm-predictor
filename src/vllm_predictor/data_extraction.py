"""Build a length-prediction dataset from a vLLM OpenAI-compatible server.

The hidden-state extraction feature writes all prompt-token activations to a
temporary safetensors file.  This module synchronously loads that file, keeps
only the last prompt token, and writes a compact safetensors feature file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping


CSV_FIELDS = (
    "sample_id",
    "source_file",
    "row_index",
    "prompt_sha256",
    "user_prompt_content",
    "model",
    "hidden_states_path",
    "hidden_states_shape",
    "prompt_tokens",
    "last_prompt_token_id",
    "completion_tokens",
    "finish_reason",
    "truncated",
    "prefill_request_id",
    "generation_request_id",
    "created_at",
)


@dataclass(frozen=True)
class Sample:
    """One stable row from a source CSV file."""

    sample_id: str
    source_file: str
    row_index: int
    prompt: str
    prompt_sha256: str


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_sample_id(source_file: str, row_index: int, prompt: str) -> tuple[str, str]:
    """Return a deterministic sample ID and the prompt SHA-256 digest."""

    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    identity = f"{source_file}\0{row_index}\0{prompt_digest}".encode("utf-8")
    sample_id = hashlib.sha256(identity).hexdigest()[:24]
    return sample_id, prompt_digest


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
                sample_id, prompt_digest = make_sample_id(
                    relative_name, row_index, prompt
                )
                yield Sample(
                    sample_id=sample_id,
                    source_file=relative_name,
                    row_index=row_index,
                    prompt=prompt,
                    prompt_sha256=prompt_digest,
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
        # Make it explicit that only prompt/prefill activations are requested.
        "kv_transfer_params": {"include_output_tokens": False},
    }
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


def save_last_token_hidden_states(source: Path, destination: Path) -> dict[str, Any]:
    """Save only the final prompt token while preserving the token dimension."""

    from safetensors.torch import save_file

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

    last_token_ids = token_ids[-1:].detach().cpu().contiguous()
    last_hidden_states = hidden_states[-1:].detach().cpu().contiguous()
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}.",
            suffix=".safetensors",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        save_file(
            {
                "token_ids": last_token_ids,
                "hidden_states": last_hidden_states,
            },
            temporary_name,
            metadata={
                "selection": "last_prompt_token",
                "source_hidden_states_shape": json.dumps(list(hidden_states.shape)),
            },
        )
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    return {
        "source_hidden_states_shape": list(hidden_states.shape),
        "hidden_states_shape": list(last_hidden_states.shape),
        "prompt_tokens": int(token_ids.shape[0]),
        "last_prompt_token_id": int(last_token_ids.item()),
    }


def feature_path(output_dir: Path, sample: Sample) -> Path:
    source_stem = Path(sample.source_file).stem
    safe_stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in source_stem
    )
    return (
        output_dir
        / "hidden_states"
        / safe_stem
        / f"{sample.row_index:08d}_{sample.sample_id}.safetensors"
    )


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
            # The connector writes asynchronously; the response can expose the
            # destination shortly before the writer creates it.
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
    """Thread-safe append-only manifest plus a deduplicated CSV projection."""

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
        self._csv_sample_ids = self._load_csv_sample_ids()
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

    def _load_csv_sample_ids(self) -> set[str]:
        if not self.results_path.exists() or self.results_path.stat().st_size == 0:
            return set()
        with self.results_path.open("r", encoding="utf-8", newline="") as results:
            reader = csv.DictReader(results)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise ValueError(
                    f"Unexpected columns in existing results file {self.results_path}: "
                    f"{reader.fieldnames}"
                )
            return {row["sample_id"] for row in reader if row.get("sample_id")}

    def _repair_csv_from_manifest(self) -> None:
        for records in self.history.values():
            for record in reversed(records):
                if record.get("status") == "completed":
                    self.append_csv(record)
                    break

    def append_manifest(self, record: Mapping[str, Any]) -> None:
        materialized = dict(record)
        line = json.dumps(materialized, ensure_ascii=False) + "\n"
        with self._lock:
            with self.manifest_path.open("a", encoding="utf-8") as manifest:
                manifest.write(line)
                manifest.flush()
                os.fsync(manifest.fileno())
            sample_id = materialized.get("sample_id")
            if sample_id:
                self.history.setdefault(sample_id, []).append(materialized)
                self.latest[sample_id] = materialized

    def append_csv(self, record: Mapping[str, Any]) -> bool:
        sample_id = str(record["sample_id"])
        with self._lock:
            if sample_id in self._csv_sample_ids:
                return False
            write_header = (
                not self.results_path.exists()
                or self.results_path.stat().st_size == 0
            )
            with self.results_path.open(
                "a", encoding="utf-8", newline=""
            ) as results:
                writer = csv.DictWriter(results, fieldnames=CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                row = {field: record.get(field) for field in CSV_FIELDS}
                if isinstance(row["hidden_states_shape"], list):
                    row["hidden_states_shape"] = json.dumps(
                        row["hidden_states_shape"]
                    )
                writer.writerow(row)
                results.flush()
                os.fsync(results.fileno())
            self._csv_sample_ids.add(sample_id)
            return True

    def resumable_hidden_record(self, sample_id: str) -> dict[str, Any] | None:
        for record in reversed(self.history.get(sample_id, [])):
            if record.get("status") == "completed":
                path = record.get("hidden_states_path")
                if path and Path(path).is_file():
                    return record
                # A completed label with a missing feature is not usable. Walk
                # farther back only in case an earlier valid feature exists.
                continue
            if record.get("status") in {"hidden_saved", "generation_failed"}:
                path = record.get("hidden_states_path")
                if path and Path(path).is_file():
                    return record
        return None


def _base_record(sample: Sample, config: ExtractionConfig) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "source_file": sample.source_file,
        "row_index": sample.row_index,
        "prompt_sha256": sample.prompt_sha256,
        "user_prompt_content": sample.prompt,
        "model": config.model,
    }


RequestFunction = Callable[..., dict[str, Any]]


def process_sample(
    sample: Sample,
    config: ExtractionConfig,
    store: ResultStore,
    *,
    request_fn: RequestFunction = post_json,
) -> dict[str, Any]:
    """Run the two-request workflow for one sample and persist every phase."""

    base = _base_record(sample, config)
    rollout_seed = per_sample_seed(config.seed, sample.sample_id)
    hidden_record = store.resumable_hidden_record(sample.sample_id)
    if hidden_record and hidden_record.get("status") == "completed":
        return {"sample_id": sample.sample_id, "status": "skipped"}

    if hidden_record is None:
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
                ),
                timeout=config.timeout,
                retries=config.retries,
                retry_backoff=config.retry_backoff,
            )
            source_path = response_hidden_states_path(prefill_response)
            compact_path = feature_path(config.output_dir, sample)
            hidden_metadata = save_last_token_hidden_states(
                source_path, compact_path
            )
            source_deleted = False
            if not config.keep_server_hidden_states:
                source_deleted = _delete_server_file(source_path)
            hidden_record = {
                **base,
                "status": "hidden_saved",
                "created_at": utc_now(),
                "prefill_request_id": prefill_response.get("id"),
                "source_hidden_states_path": str(source_path),
                "source_hidden_states_deleted": source_deleted,
                "hidden_states_path": str(compact_path),
                **hidden_metadata,
            }
            store.append_manifest(hidden_record)
        except Exception as exc:
            failure = {
                **base,
                "status": "hidden_failed",
                "phase": "prefill_hidden_states",
                "created_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}"[:4000],
            }
            store.append_manifest(failure)
            return failure

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
            ),
            timeout=config.timeout,
            retries=config.retries,
            retry_backoff=config.retry_backoff,
        )
        completion_tokens, finish_reason = _completion_metadata(generation_response)

        generation_hidden_path: str | None = None
        generation_hidden_deleted = False
        try:
            generation_source = response_hidden_states_path(generation_response)
            generation_hidden_path = str(generation_source)
            if not config.keep_server_hidden_states:
                generation_hidden_deleted = _delete_server_file(generation_source)
        except RuntimeError:
            # A separate, ordinary generation server may not extract activations.
            pass

        completed = {
            **base,
            "status": "completed",
            "created_at": utc_now(),
            "prefill_request_id": hidden_record.get("prefill_request_id"),
            "generation_request_id": generation_response.get("id"),
            "source_hidden_states_path": hidden_record.get(
                "source_hidden_states_path"
            ),
            "source_hidden_states_deleted": hidden_record.get(
                "source_hidden_states_deleted"
            ),
            "generation_hidden_states_path": generation_hidden_path,
            "generation_hidden_states_deleted": generation_hidden_deleted,
            "hidden_states_path": hidden_record["hidden_states_path"],
            "source_hidden_states_shape": hidden_record.get(
                "source_hidden_states_shape"
            ),
            "hidden_states_shape": hidden_record["hidden_states_shape"],
            "prompt_tokens": hidden_record["prompt_tokens"],
            "last_prompt_token_id": hidden_record["last_prompt_token_id"],
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "truncated": finish_reason == "length",
            "sampling": {
                "max_tokens": config.generation_max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "min_p": config.min_p,
                "presence_penalty": config.presence_penalty,
                "frequency_penalty": config.frequency_penalty,
                "repetition_penalty": config.repetition_penalty,
                "seed": rollout_seed,
            },
        }
        # Manifest is canonical.  On a crash between these two appends, startup
        # projects the completed manifest record back into the CSV.
        store.append_manifest(completed)
        store.append_csv(completed)
        return completed
    except Exception as exc:
        failure = {
            **base,
            "status": "generation_failed",
            "phase": "generation_length",
            "created_at": utc_now(),
            "error": f"{type(exc).__name__}: {exc}"[:4000],
            "prefill_request_id": hidden_record.get("prefill_request_id"),
            "hidden_states_path": hidden_record.get("hidden_states_path"),
            "source_hidden_states_path": hidden_record.get(
                "source_hidden_states_path"
            ),
            "source_hidden_states_deleted": hidden_record.get(
                "source_hidden_states_deleted"
            ),
            "source_hidden_states_shape": hidden_record.get(
                "source_hidden_states_shape"
            ),
            "hidden_states_shape": hidden_record.get("hidden_states_shape"),
            "prompt_tokens": hidden_record.get("prompt_tokens"),
            "last_prompt_token_id": hidden_record.get("last_prompt_token_id"),
        }
        store.append_manifest(failure)
        return failure


def pending_samples(
    samples: Iterable[Sample],
    store: ResultStore,
    limit: int | None,
) -> Iterator[Sample]:
    """Filter completed rows without materializing large source datasets."""

    yielded = 0
    for sample in samples:
        latest = store.latest.get(sample.sample_id)
        if latest and latest.get("status") == "completed":
            feature = latest.get("hidden_states_path")
            if feature and Path(feature).is_file():
                continue
        if limit is not None and yielded >= limit:
            break
        yielded += 1
        yield sample

