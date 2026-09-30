"""Collect generation lengths and runner-exported prompt states via vLLM HTTP.

The runner's .pt directory must be locally accessible (same host or shared
mount). Each outer group becomes one immutable safetensors shard; inner groups
are concurrent chat requests. Source .pt files are deleted only after both the
shard and its CSV rows have been written and flushed. A failed run preserves
uncommitted source files. Use a fresh output directory for each run.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TypeVar

import torch
from safetensors.torch import save_file

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")
CSV_FIELDS = (
    "source_file", "source_row", "request_id", "hidden_request_id",
    "prompt_length", "input_tokens", "output_tokens", "finish_reason",
    "thinking", "model", "hidden_states_path", "hidden_states_key", "hidden_states_row",
)


DEFAULT_SYSTEM_PROMPT = (
    "Provide a concise solution in 2-4 key steps, then give the final answer. "
    "Use fewer steps for simple questions. Keep each step to 1-2 sentences. "
    "Include only essential facts, formulas, calculations or justification. "
    "For multiple-choice questions, explain the basis for the selected option. "
    "Do not restate the question, repeat explanations, or list alternative solutions. "
    "Use the same language as the question."
)


@dataclass(frozen=True)
class Sample:
    source_row: int
    prompt: str


@dataclass(frozen=True)
class PreparedSample:
    sample: Sample
    messages: list[dict[str, str]]
    prompt_tokens: int
    input_tokens: int
    input_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class Result:
    source_row: int
    prompt_tokens: int
    input_tokens: int
    output_tokens: int
    finish_reason: str
    request_id: str


class VLLMClient:
    """Small dependency-free client for the vLLM OpenAI-compatible server."""

    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        supplied_url = base_url.rstrip("/")
        self.server_url = supplied_url[:-3] if supplied_url.endswith("/v1") else supplied_url
        self.base_url = f"{self.server_url}/v1"
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, url: str, method: str = "GET",
                 payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM request failed: {method} {url} HTTP {error.code}: {detail[:2000]}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Cannot reach vLLM server at {url}: {error.reason}") from error
        if not isinstance(result, dict):
            raise RuntimeError(f"Expected JSON object from {url}, got {type(result).__name__}")
        if "error" in result:
            raise RuntimeError(f"vLLM returned an error from {url}: {result['error']}")
        return result

    def resolve_model(self, requested_model: str | None) -> str:
        response = self._request(f"{self.base_url}/models")
        models = [item.get("id") for item in response.get("data", [])
                  if isinstance(item, dict) and item.get("id")]
        if not models:
            raise RuntimeError(f"No served model reported by {self.base_url}/models")
        if requested_model is None:
            return str(models[0])
        if requested_model not in models:
            raise ValueError(
                f"Model {requested_model!r} is not served by {self.base_url}; available={models}")
        return requested_model

    def tokenize_prompt(self, model: str, prompt: str) -> tuple[int, tuple[int, ...]]:
        response = self._request(
            f"{self.server_url}/tokenize", "POST",
            {"model": model, "prompt": prompt, "add_special_tokens": False},
        )
        return _parse_tokenize_response(response)

    def tokenize_chat(self, model: str, messages: list[dict[str, str]],
                      thinking: bool) -> tuple[int, tuple[int, ...]]:
        response = self._request(
            f"{self.server_url}/tokenize", "POST",
            {
                "model": model,
                "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": thinking},
            },
        )
        return _parse_tokenize_response(response)

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request(f"{self.base_url}/chat/completions", "POST", payload)


def _parse_tokenize_response(response: dict[str, Any]) -> tuple[int, tuple[int, ...]]:
    tokens = response.get("tokens")
    count = response.get("count")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise RuntimeError(f"Invalid /tokenize tokens: {tokens!r}")
    if not isinstance(count, int) or count != len(tokens):
        raise RuntimeError(f"Invalid /tokenize count={count!r}, token_count={len(tokens)}")
    return count, tuple(tokens)


def _messages(sample: Sample, system_prompt: str) -> list[dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": sample.prompt})
    return messages


def prepare_samples(samples: Sequence[Sample], client: VLLMClient, model: str,
                    thinking: bool, system_prompt: str,
                    prompt_lengths: dict[int, int]) -> list[PreparedSample]:
    prepared = []
    for sample in samples:
        messages = _messages(sample, system_prompt)
        input_count, input_ids = client.tokenize_chat(model, messages, thinking)
        prepared.append(PreparedSample(
            sample, messages, prompt_lengths[sample.source_row], input_count, input_ids))
    return prepared


def make_chat_payload(args: argparse.Namespace, model: str,
                      item: PreparedSample, thinking: bool,
                      max_tokens: int | None = None) -> dict[str, Any]:
    return {
        "model": model,
        "messages": item.messages,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "max_tokens": args.max_tokens if max_tokens is None else max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "skip_special_tokens": False,
        "stream": False,
    }


def parse_chat_response(item: PreparedSample, response: dict[str, Any]) -> Result:
    choices = response.get("choices")
    usage = response.get("usage")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(usage, dict):
        raise RuntimeError(f"Invalid chat completion response: {response}")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    prompt_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    if not isinstance(message, dict) or not isinstance(prompt_tokens, int) or not isinstance(output_tokens, int):
        raise RuntimeError(f"Missing message or token usage in response: {response}")
    if prompt_tokens != item.input_tokens:
        raise RuntimeError(
            f"Input token mismatch for source_row={item.sample.source_row}: "
            f"/tokenize={item.input_tokens}, completion usage={prompt_tokens}")
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str):
        raise RuntimeError(f"Missing finish_reason in response: {response}")
    return Result(
        source_row=item.sample.source_row,
        prompt_tokens=item.prompt_tokens,
        input_tokens=prompt_tokens,
        output_tokens=output_tokens,
        finish_reason=finish_reason,
        request_id=str(response.get("id", "")),
    )


def run_batch(client: VLLMClient, args: argparse.Namespace, model: str,
              batch: Sequence[PreparedSample], thinking: bool,
              max_tokens: int | None = None) -> tuple[list[Result], float]:
    """Release N HTTP calls together so vLLM can continuously batch them."""
    barrier = threading.Barrier(len(batch) + 1)

    def request_one(item: PreparedSample) -> Result:
        payload = make_chat_payload(args, model, item, thinking, max_tokens)
        barrier.wait()
        return parse_chat_response(item, client.chat(payload))

    with ThreadPoolExecutor(max_workers=len(batch), thread_name_prefix="vllm-request") as pool:
        futures = [pool.submit(request_one, item) for item in batch]
        started = time.perf_counter()
        barrier.wait()
        results = [future.result() for future in futures]
        elapsed = time.perf_counter() - started
    return results, elapsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--prompt-column", default="user_prompt_content")
    parser.add_argument("--base-url", required=True, help="vLLM server URL, with or without /v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--hidden-states-dir", type=Path, required=True,
                        help="Local/shared directory configured as server prefill_hidden_states_dir")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must be empty; never use the server dump directory")
    parser.add_argument("--shard-size", type=int, default=2048, help="N: maximum requests per safetensors shard")
    parser.add_argument("--batch-size", type=int, default=8, help="n: maximum concurrent chat requests")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--file-timeout", type=float, default=30)
    parser.add_argument("--request-id-suffix", default="",
                        help="Optional engine ID suffix, e.g. '-0'; filename is response id + suffix + .pt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    args = parser.parse_args(argv)
    if min(args.shard_size, args.batch_size, args.max_tokens) < 1:
        parser.error("shard-size, batch-size and max-tokens must be positive")
    if args.request_timeout <= 0 or args.file_timeout < 0:
        parser.error("request-timeout must be positive and file-timeout non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.temperature < 0 or not 0 < args.top_p <= 1 or not 0 <= args.min_p <= 1:
        parser.error("Require temperature >= 0, top-p in (0,1], min-p in [0,1]")
    if args.top_k < -1 or args.repetition_penalty <= 0:
        parser.error("Require top-k >= -1 and repetition-penalty > 0")
    if not -2 <= args.presence_penalty <= 2 or not -2 <= args.frequency_penalty <= 2:
        parser.error("Presence/frequency penalties must be in [-2,2]")
    return args


def iter_samples(path: Path, column: str) -> Iterator[Sample]:
    """Preserve zero-based CSV data-row positions, including gaps from empty rows."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"Missing column {column!r} in {path}")
        for index, row in enumerate(reader):
            prompt = row.get(column)
            if prompt and prompt.strip():
                yield Sample(index, prompt)


def batched(items: Iterable[T], size: int) -> Iterator[list[T]]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def load_state(directory: Path, request_id: str, input_tokens: int, timeout: float):
    """Read only the requested final-output vector, validating its identity."""
    if not request_id or request_id in (".", "..") or any(c in request_id for c in '/\\\x00:'):
        raise ValueError(f"Unsafe request_id: {request_id!r}")
    path = directory / f"{request_id}.pt"
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise FileNotFoundError(
                f"Missing hidden state {path}. Check the shared directory, server export setting "
                "and whether the engine request ID needs --request-id-suffix."
            )
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    if path.is_symlink() or path.resolve().parent != directory.resolve():
        raise ValueError(f"Hidden state must be a regular file inside {directory}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("request_id") != request_id:
        raise ValueError(f"request_id mismatch in {path}")
    if data.get("num_prompt_tokens") != input_tokens or data.get("token_position") != input_tokens - 1:
        raise ValueError(f"Prompt length/position mismatch in {path}: HTTP input_tokens={input_tokens}")
    if data.get("feature_type") != "target_model_final_output_last_prompt_token":
        raise ValueError(f"Unexpected feature_type in {path}")
    tensor = data.get("hidden_state")
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1 or not tensor.numel() or not tensor.is_floating_point():
        raise ValueError(f"Expected a nonempty floating-point [hidden_size] tensor in {path}")
    return tensor.contiguous(), path


def write_shard(path: Path, tensors: list[torch.Tensor], thinking: bool):
    if not tensors or any(t.shape != tensors[0].shape or t.dtype != tensors[0].dtype for t in tensors):
        raise ValueError("Shard tensors must have matching shapes and dtypes")
    temporary = path.with_suffix(".safetensors.tmp")
    try:
        save_file({"hidden_states": torch.stack(tensors).contiguous()}, str(temporary), metadata={
            "selection": "target_model_final_output_last_prompt_token",
            "thinking": str(thinking).lower(), "samples": str(len(tensors)),
        })
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def extract(args, client=None) -> Path:
    source = args.input_file.resolve()
    dumps = args.hidden_states_dir.resolve()
    output = args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not dumps.is_dir():
        raise FileNotFoundError(f"Server hidden-state directory must be mounted locally: {dumps}")
    if output == dumps or dumps in output.parents or output in dumps.parents:
        raise ValueError("Output and server dump directories must be separate, not nested")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}; use a new directory for a new run")
    client = client or VLLMClient(args.base_url, args.api_key, args.request_timeout)
    model = client.resolve_model(args.model)
    samples = iter_samples(source, args.prompt_column)
    if args.limit is not None:
        samples = islice(samples, args.limit)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "results.csv"
    seen_ids = set()
    with manifest.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for shard_index, group in enumerate(batched(samples, args.shard_size)):
            tensors, records, small_files = [], [], []
            shard_path = output / f"shards-{shard_index:06d}.safetensors"
            for batch_index, batch in enumerate(batched(group, args.batch_size)):
                lengths = {s.source_row: client.tokenize_prompt(model, s.prompt)[0] for s in batch}
                prepared = prepare_samples(batch, client, model, args.thinking, args.system_prompt, lengths)
                results, seconds = run_batch(client, args, model, prepared, args.thinking)
                for result in results:
                    if not result.request_id or result.request_id in seen_ids:
                        raise ValueError(f"Missing or duplicate response request_id: {result.request_id!r}")
                    seen_ids.add(result.request_id)
                    hidden_id = result.request_id + args.request_id_suffix
                    tensor, path = load_state(dumps, hidden_id, result.input_tokens, args.file_timeout)
                    records.append({
                        "source_file": str(source), "source_row": result.source_row,
                        "request_id": result.request_id, "hidden_request_id": hidden_id,
                        "prompt_length": result.prompt_tokens, "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens, "finish_reason": result.finish_reason,
                        "thinking": args.thinking, "model": model,
                        "hidden_states_path": str(shard_path), "hidden_states_key": "hidden_states",
                        "hidden_states_row": len(tensors),
                    })
                    tensors.append(tensor)
                    small_files.append(path)
                LOGGER.info("shard=%d batch=%d requests=%d elapsed=%.2fs", shard_index, batch_index, len(batch), seconds)
            write_shard(shard_path, tensors, args.thinking)
            writer.writerows(records)
            stream.flush()
            os.fsync(stream.fileno())
            # Never remove input files in a finally block: a failed shard/CSV
            # write must leave the original features available for recovery.
            for path in small_files:
                path.unlink()
            LOGGER.info("Committed %s: %d records; source .pt files removed", shard_path.name, len(records))
    return manifest


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args(argv)
    LOGGER.info("Server=%s N=%d n=%d thinking=%s", args.base_url, args.shard_size, args.batch_size, args.thinking)
    LOGGER.info("CSV: %s", extract(args))
    return 0
