"""Benchmark Qwen thinking modes through a running vLLM OpenAI server.

The script never loads a model. It sends concurrent HTTP requests to vLLM,
uses /tokenize for raw/template token counts, and uses usage returned by
/v1/chat/completions for the measured input/output token counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


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
    reasoning: str
    answer: str
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--prompt-column", default="user_prompt_content")
    parser.add_argument("--sample-size", type=int, default=32,
                        help="Random nonempty CSV rows reused in every configuration.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
                        help="Same instruction for both modes; pass an empty string to omit it.")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-answers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-url", default="http://localhost:8001/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--model", default=None,
                        help="Served model name; omit to use the first model from /v1/models.")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Safety ceiling, not a length target; capped results are marked.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Same request seed in both thinking modes.")
    args = parser.parse_args(argv)
    if args.sample_size < 1 or args.max_tokens < 1 or args.request_timeout <= 0:
        parser.error("--sample-size, --max-tokens and --request-timeout must be positive")
    if any(size < 1 for size in args.batch_sizes) or len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("--batch-sizes must contain distinct positive integers")
    if args.sample_size < max(args.batch_sizes):
        parser.error("--sample-size must be at least the largest batch size")
    if args.temperature < 0 or not 0 < args.top_p <= 1 or not 0 <= args.min_p <= 1:
        parser.error("Require temperature >= 0, top-p in (0, 1], min-p in [0, 1]")
    if args.top_k < -1 or args.repetition_penalty <= 0:
        parser.error("Require top-k >= -1 and repetition-penalty > 0")
    if not -2 <= args.presence_penalty <= 2 or not -2 <= args.frequency_penalty <= 2:
        parser.error("Presence and frequency penalties must be in [-2, 2]")
    return args


def select_samples(path: Path, column: str, size: int, seed: int) -> list[Sample]:
    """Uniform reservoir sample without loading the entire CSV into memory."""
    rng = random.Random(seed)
    samples: list[Sample] = []
    count = 0
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"Missing prompt column {column!r} in {path}")
        for source_row, row in enumerate(reader):
            prompt = row.get(column)
            if not prompt or not prompt.strip():
                continue
            sample = Sample(source_row, prompt)
            count += 1
            if len(samples) < size:
                samples.append(sample)
            else:
                index = rng.randrange(count)
                if index < size:
                    samples[index] = sample
    if count < size:
        raise ValueError(f"Requested {size} samples but only {count} nonempty prompts in {path}")
    return sorted(samples, key=lambda sample: sample.source_row)


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


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


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
    reasoning = _text(message.get("reasoning_content", message.get("reasoning")))
    return Result(
        source_row=item.sample.source_row,
        prompt_tokens=item.prompt_tokens,
        input_tokens=prompt_tokens,
        output_tokens=output_tokens,
        finish_reason=finish_reason,
        reasoning=reasoning,
        answer=_text(message.get("content")),
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


def benchmark(args: argparse.Namespace, samples: Sequence[Sample],
              client: VLLMClient, model: str) -> list[dict[str, Any]]:
    print("Tokenizing prompts through the server (excluded from timing)...", flush=True)
    prompt_lengths = {
        sample.source_row: client.tokenize_prompt(model, sample.prompt)[0]
        for sample in samples
    }
    prepared = {
        mode: prepare_samples(samples, client, model, mode, args.system_prompt, prompt_lengths)
        for mode in (False, True)
    }
    if all(off.input_token_ids == on.input_token_ids
           for off, on in zip(prepared[False], prepared[True])):
        raise ValueError("The served chat template ignores enable_thinking: inputs are identical")

    print("\nLengths are server tokenizer counts: prompt=raw CSV text, "
          "input=full chat request, output=all completion tokens.\n"
          "Batch seconds includes concurrent HTTP round trips and server generation; "
          "it excludes tokenization, warmup and printing.\n"
          "Each batch opens N simultaneous requests, allowing vLLM continuous batching.\n"
          "finish=length means the safety ceiling was reached.", flush=True)
    summaries = []
    for config_index, batch_size in enumerate(args.batch_sizes):
        modes = (False, True) if config_index % 2 == 0 else (True, False)
        for mode in modes:
            items = prepared[mode]
            print(f"\n=== think={mode} concurrent_requests={batch_size} ===", flush=True)
            if args.warmup:
                run_batch(client, args, model, items[:batch_size], mode, max_tokens=1)
                print("Warmup completed (excluded).", flush=True)
            results: list[Result] = []
            batch_seconds = []
            for offset in range(0, len(items), batch_size):
                batch = items[offset:offset + batch_size]
                records, elapsed = run_batch(client, args, model, batch, mode)
                results.extend(records)
                batch_seconds.append(elapsed)
                print(f"\nBatch {len(batch_seconds)}: requested={batch_size} actual={len(batch)} "
                      f"seconds={elapsed:.3f}", flush=True)
                print("source_row  prompt_tokens  input_tokens  output_tokens  finish  request_id",
                      flush=True)
                for record in records:
                    print(f"{record.source_row:10d} {record.prompt_tokens:14d} "
                          f"{record.input_tokens:13d} {record.output_tokens:14d} "
                          f"{record.finish_reason:7s} {record.request_id}", flush=True)
                if args.show_answers:
                    for item, record in zip(batch, records):
                        print(f"\n[source_row={record.source_row} think={mode}]\n"
                              f"Prompt:\n{item.sample.prompt}", flush=True)
                        if record.reasoning:
                            print(f"Reasoning:\n{record.reasoning}", flush=True)
                        print(f"Answer:\n{record.answer}", flush=True)
            total_seconds = sum(batch_seconds)
            total_output = sum(record.output_tokens for record in results)
            summaries.append({
                "thinking": mode,
                "batch_size": batch_size,
                "samples": len(results),
                "batches": len(batch_seconds),
                "total_seconds": total_seconds,
                "mean_batch_seconds": total_seconds / len(batch_seconds),
                "mean_prompt_tokens": sum(record.prompt_tokens for record in results) / len(results),
                "mean_input_tokens": sum(record.input_tokens for record in results) / len(results),
                "mean_output_tokens": total_output / len(results),
                "output_tokens_per_second": total_output / max(total_seconds, 1e-9),
                "capped": sum(record.finish_reason == "length" for record in results),
            })

    print("\n=== Summary (concurrent HTTP requests) ===")
    print("think batch samples batches prompt_avg input_avg output_avg batch_s_avg total_s out_tok/s capped")
    for row in summaries:
        print(f"{str(row['thinking']):5s} {row['batch_size']:5d} {row['samples']:7d} "
              f"{row['batches']:7d} {row['mean_prompt_tokens']:10.1f} "
              f"{row['mean_input_tokens']:9.1f} {row['mean_output_tokens']:10.1f} "
              f"{row['mean_batch_seconds']:11.3f} {row['total_seconds']:7.3f} "
              f"{row['output_tokens_per_second']:9.2f} {row['capped']:6d}", flush=True)
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_file = args.input_file.resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {input_file}")
    samples = select_samples(input_file, args.prompt_column, args.sample_size, args.sample_seed)
    client = VLLMClient(args.base_url, args.api_key, args.request_timeout)
    model = client.resolve_model(args.model)
    print(f"Server: {args.base_url}\nServed model: {model}\nDataset: {input_file}\n"
          f"Sample seed: {args.sample_seed}; source rows: {[sample.source_row for sample in samples]}\n"
          f"System prompt (both modes): {args.system_prompt!r}\n"
          f"Measured completions: {len(samples) * len(args.batch_sizes) * 2}", flush=True)
    benchmark(args, samples, client, model)
    return 0
