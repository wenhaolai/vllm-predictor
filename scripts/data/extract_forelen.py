#!/usr/bin/env python3
"""Build ForeLen length-prediction shards with online and offline vLLM.

The online OpenAI-compatible server generates full answers and supplies their
token lengths. One persistent offline ``vllm.LLM`` instance extracts prompt
hidden states on a disjoint set of NPUs. Online requests are submitted before
the synchronous offline call so both workloads run concurrently.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


CSV_FIELDS = (
    "source_file",
    "source_row",
    "hidden_states_path",
    "hidden_states_row",
    "completion_tokens",
)


@dataclass(frozen=True)
class Sample:
    """One non-empty prompt and its zero-based data-row position."""

    source_row: int
    prompt: str


@dataclass(frozen=True)
class GenerationResult:
    """The length metadata returned by the online generation server."""

    completion_tokens: int
    finish_reason: str | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract last-prompt-token hidden states with an offline vLLM "
            "instance while an online vLLM server generates length labels."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="One ForeLen CSV file to process.",
    )
    parser.add_argument("--prompt-column", default="user_prompt_content")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/forelen_extracted"),
    )
    parser.add_argument(
        "--url",
        required=True,
        help=(
            "Online generation server base URL, for example "
            "http://127.0.0.1:8001. A full /v1/chat/completions URL is also "
            "accepted."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model path or Hugging Face ID loaded once by the offline LLM.",
    )
    parser.add_argument(
        "--server-model",
        default=None,
        help="Online served model name; defaults to --model.",
    )
    parser.add_argument(
        "--offline-devices",
        default="4,5,6,7",
        help="ASCEND_RT_VISIBLE_DEVICES used by the offline process.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument(
        "--distributed-executor-backend",
        default="mp",
        choices=("mp", "ray", "uni", "external_launcher"),
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument(
        "--hidden-layer-ids",
        type=int,
        nargs="+",
        default=[64],
        help="Model layer indices exported by extract_hidden_states.",
    )
    parser.add_argument(
        "--hidden-states-dir",
        type=Path,
        default=Path("/dev/shm/vllm_forelen_hidden_states"),
        help="Temporary connector directory; returned files are deleted per batch.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=2048,
        help="Maximum records in each shards-XXXXXX.safetensors file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Prompts submitted together to the persistent offline LLM.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
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
        help="Optional base seed; source_row is added for each online request.",
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many unfinished rows.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)

    if args.shard_size < 1:
        parser.error("--shard-size must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be at least 1")
    if args.max_model_len < 1:
        parser.error("--max-model-len must be at least 1")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if not args.hidden_layer_ids or min(args.hidden_layer_ids) < 0:
        parser.error("--hidden-layer-ids must contain non-negative integers")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.top_k < -1:
        parser.error("--top-k must be -1, 0, or a positive integer")
    if not 0 <= args.min_p <= 1:
        parser.error("--min-p must be in [0, 1]")
    if args.retries < 0:
        parser.error("--retries must be non-negative")
    if args.retry_backoff < 0:
        parser.error("--retry-backoff must be non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def iter_csv_samples(input_file: Path, prompt_column: str) -> Iterator[Sample]:
    """Read one ForeLen CSV while retaining original zero-based data rows."""

    with input_file.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or prompt_column not in reader.fieldnames:
            raise ValueError(
                f"{input_file} does not contain required column {prompt_column!r}; "
                f"columns={reader.fieldnames}"
            )
        for source_row, row in enumerate(reader):
            prompt = row.get(prompt_column)
            if prompt is None or not prompt.strip():
                continue
            yield Sample(source_row=source_row, prompt=prompt)


def batched(items: Iterable[Sample], size: int) -> Iterator[list[Sample]]:
    """Yield lists of at most ``size`` without materializing the full CSV."""

    batch: list[Sample] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _chat_completions_url(url: str) -> str:
    normalized = url.rstrip("/")
    if normalized.endswith("/v1/chat/completions"):
        return normalized
    return f"{normalized}/v1/chat/completions"


def generation_payload(sample: Sample, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.server_model or args.model,
        "messages": [{"role": "user", "content": sample.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "repetition_penalty": args.repetition_penalty,
        "stream": False,
    }
    if args.seed is not None:
        payload["seed"] = (args.seed + sample.source_row) % (2**63 - 1)
    return payload


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
    """POST JSON, retrying only transport and transient HTTP failures."""

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
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
            if not retryable or attempt + 1 >= attempts:
                raise RuntimeError(message) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            message = f"Request to {url} failed: {exc}"
            if attempt + 1 >= attempts:
                raise RuntimeError(message) from exc
        time.sleep(retry_backoff * (2**attempt))
    raise AssertionError("unreachable")


RequestFunction = Callable[..., dict[str, Any]]


def request_generation(
    sample: Sample,
    args: argparse.Namespace,
    request_fn: RequestFunction = post_json,
) -> GenerationResult:
    response = request_fn(
        _chat_completions_url(args.url),
        generation_payload(sample, args),
        timeout=args.timeout,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
    )
    usage = response.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int):
        raise RuntimeError(
            "Online response does not contain integer usage.completion_tokens"
        )
    choices = response.get("choices") or []
    finish_reason = choices[0].get("finish_reason") if choices else None
    return GenerationResult(completion_tokens, finish_reason)


def create_offline_llm(args: argparse.Namespace) -> tuple[Any, Any]:
    """Set NPU visibility, then construct the one persistent offline engine."""

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.offline_devices
    args.hidden_states_dir.mkdir(parents=True, exist_ok=True)

    # Import only after ASCEND_RT_VISIBLE_DEVICES is fixed. The online server
    # is a separate process and should be launched on devices 0-3.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        distributed_executor_backend=args.distributed_executor_backend,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": args.hidden_layer_ids,
                }
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": str(args.hidden_states_dir.resolve()),
            },
        },
    )
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    return llm, sampling_params


def output_hidden_states_path(output: Any) -> Path:
    params = getattr(output, "kv_transfer_params", None) or {}
    path = params.get("hidden_states_path")
    if not path:
        raise RuntimeError(
            "Offline RequestOutput does not contain "
            "kv_transfer_params.hidden_states_path"
        )
    return Path(path)


def extract_last_token_hidden_states(path: Path) -> Any:
    """Load one connector file and retain only its last prompt token."""

    from safetensors import safe_open

    if not path.is_file():
        raise FileNotFoundError(f"Hidden-state file does not exist: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as tensors:
        token_ids = tensors.get_tensor("token_ids")
        hidden_states = tensors.get_tensor("hidden_states")
    if token_ids.ndim != 1 or token_ids.shape[0] == 0:
        raise ValueError(f"Unexpected token_ids shape: {list(token_ids.shape)}")
    if hidden_states.ndim < 2 or hidden_states.shape[0] != token_ids.shape[0]:
        raise ValueError(
            "The leading dimensions of token_ids and hidden_states do not match: "
            f"{list(token_ids.shape)} vs {list(hidden_states.shape)}"
        )
    return hidden_states[-1].detach().cpu().contiguous()


def delete_connector_file(path: Path) -> None:
    """Delete one temporary safetensors file and its synchronization lock."""

    Path(f"{path}.lock").unlink(missing_ok=True)
    path.unlink(missing_ok=True)


def offline_hidden_states(
    llm: Any,
    sampling_params: Any,
    samples: Sequence[Sample],
) -> list[Any]:
    """Run one offline batch, load final-token states, and clean all dumps."""

    outputs = llm.generate(
        [sample.prompt for sample in samples],
        sampling_params,
        use_tqdm=False,
    )
    if len(outputs) != len(samples):
        raise RuntimeError(
            f"Offline LLM returned {len(outputs)} outputs for {len(samples)} prompts"
        )

    paths: list[Path] = []
    try:
        for output in outputs:
            paths.append(output_hidden_states_path(output))
        return [extract_last_token_hidden_states(path) for path in paths]
    finally:
        for path in paths:
            delete_connector_file(path)


def process_batch(
    samples: Sequence[Sample],
    llm: Any,
    sampling_params: Any,
    args: argparse.Namespace,
    executor: ThreadPoolExecutor,
    request_fn: RequestFunction = post_json,
) -> tuple[list[Any], list[GenerationResult]]:
    """Overlap online generation requests with one offline LLM batch."""

    generation_futures: list[Future[GenerationResult]] = [
        executor.submit(request_generation, sample, args, request_fn)
        for sample in samples
    ]
    try:
        hidden_states = offline_hidden_states(llm, sampling_params, samples)
        generation_results = [future.result() for future in generation_futures]
    except BaseException:
        for future in generation_futures:
            future.cancel()
        raise
    return hidden_states, generation_results


def next_shard_index(output_dir: Path) -> int:
    indices: list[int] = []
    for path in output_dir.glob("shards-*.safetensors"):
        try:
            indices.append(int(path.stem.removeprefix("shards-")))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


def write_shard(
    output_dir: Path,
    shard_index: int,
    hidden_states: Sequence[Any],
) -> Path:
    """Atomically create one immutable hidden-state shard."""

    import torch
    from safetensors.torch import save_file

    if not hidden_states:
        raise ValueError("Cannot write an empty shard")
    first_shape = tuple(hidden_states[0].shape)
    first_dtype = hidden_states[0].dtype
    for tensor in hidden_states[1:]:
        if tuple(tensor.shape) != first_shape:
            raise ValueError(
                "Hidden-state shape changed within a shard: "
                f"{first_shape} vs {tuple(tensor.shape)}"
            )
        if tensor.dtype != first_dtype:
            raise ValueError(
                "Hidden-state dtype changed within a shard: "
                f"{first_dtype} vs {tensor.dtype}"
            )

    shard_path = (output_dir / f"shards-{shard_index:06d}.safetensors").resolve()
    stacked = torch.stack(list(hidden_states), dim=0).contiguous()
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{shard_path.stem}.",
            suffix=".safetensors",
            dir=output_dir,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        save_file(
            {"hidden_states": stacked},
            temporary_name,
            metadata={
                "selection": "last_prompt_token",
                "samples": str(len(hidden_states)),
            },
        )
        os.replace(temporary_name, shard_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return shard_path


def load_completed_rows(results_path: Path, source_file: str) -> set[int]:
    """Return materialized rows so an interrupted run can append safely."""

    if not results_path.exists() or results_path.stat().st_size == 0:
        return set()
    completed: set[int] = set()
    with results_path.open("r", encoding="utf-8", newline="") as results:
        reader = csv.DictReader(results)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(
                f"Unexpected columns in {results_path}: {reader.fieldnames}"
            )
        for row in reader:
            if row["source_file"] != source_file:
                continue
            shard = Path(row["hidden_states_path"])
            if shard.is_file():
                completed.add(int(row["source_row"]))
    return completed


def append_results(
    results_path: Path,
    source_file: str,
    samples: Sequence[Sample],
    shard_path: Path,
    generation_results: Sequence[GenerationResult],
) -> None:
    if len(samples) != len(generation_results):
        raise ValueError("Sample and generation result counts do not match")
    write_header = not results_path.exists() or results_path.stat().st_size == 0
    with results_path.open("a", encoding="utf-8", newline="") as results:
        writer = csv.DictWriter(results, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for hidden_states_row, (sample, generation) in enumerate(
            zip(samples, generation_results, strict=True)
        ):
            writer.writerow(
                {
                    "source_file": source_file,
                    "source_row": sample.source_row,
                    "hidden_states_path": str(shard_path),
                    "hidden_states_row": hidden_states_row,
                    "completion_tokens": generation.completion_tokens,
                }
            )
        results.flush()
        os.fsync(results.fileno())


def pending_samples(
    input_file: Path,
    prompt_column: str,
    completed_rows: set[int],
    limit: int | None,
) -> Iterator[Sample]:
    yielded = 0
    for sample in iter_csv_samples(input_file, prompt_column):
        if sample.source_row in completed_rows:
            continue
        if limit is not None and yielded >= limit:
            break
        yielded += 1
        yield sample


def count_pending_samples(
    input_file: Path,
    prompt_column: str,
    completed_rows: set[int],
    limit: int | None,
) -> int:
    return sum(
        1
        for _ in pending_samples(
            input_file,
            prompt_column,
            completed_rows,
            limit,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_file = args.input_file.resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"ForeLen CSV file does not exist: {input_file}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "samples.csv"
    source_file = str(input_file)
    completed_rows = load_completed_rows(results_path, source_file)
    pending_count = count_pending_samples(
        input_file,
        args.prompt_column,
        completed_rows,
        args.limit,
    )
    if pending_count == 0:
        print(f"No unfinished rows in {input_file}")
        return 0

    from tqdm import tqdm

    print(
        f"Loading offline model on Ascend devices {args.offline_devices}; "
        f"online generation server: {_chat_completions_url(args.url)}"
    )
    llm, sampling_params = create_offline_llm(args)
    shard_index = next_shard_index(output_dir)
    samples = pending_samples(
        input_file,
        args.prompt_column,
        completed_rows,
        args.limit,
    )

    with (
        ThreadPoolExecutor(
            max_workers=args.batch_size,
            thread_name_prefix="online-generation",
        ) as executor,
        tqdm(total=pending_count, unit="sample", desc=input_file.name) as progress,
    ):
        for shard_samples in batched(samples, args.shard_size):
            shard_hidden_states: list[Any] = []
            shard_generation_results: list[GenerationResult] = []
            for request_batch in batched(shard_samples, args.batch_size):
                hidden_states, generation_results = process_batch(
                    request_batch,
                    llm,
                    sampling_params,
                    args,
                    executor,
                )
                shard_hidden_states.extend(hidden_states)
                shard_generation_results.extend(generation_results)
                progress.update(len(request_batch))

            shard_path = write_shard(
                output_dir,
                shard_index,
                shard_hidden_states,
            )
            append_results(
                results_path,
                source_file,
                shard_samples,
                shard_path,
                shard_generation_results,
            )
            progress.set_postfix_str(shard_path.name)
            shard_index += 1

    print(f"Results: {results_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130) from None
