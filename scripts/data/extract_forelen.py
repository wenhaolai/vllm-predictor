#!/usr/bin/env python3
"""Build ForeLen length-prediction shards with one local vLLM engine.

Each local generation returns the generated token ids while the
``extract_hidden_states`` connector saves prompt-only hidden states. After a
batch completes, the final prompt-token state and generated length are stored
using the existing safetensors-shard and CSV manifest format.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


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
    """Length metadata returned by the local vLLM generation."""

    completion_tokens: int
    finish_reason: str | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Generate locally and extract last-prompt-token hidden states.")
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
        "--model",
        required=True,
        help="Model path or Hugging Face ID loaded once by the local LLM.",
    )
    parser.add_argument(
        "--devices",
        "--offline-devices",
        dest="devices",
        default="0,1,2,3",
        help="ASCEND_RT_VISIBLE_DEVICES used by the local LLM process.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument(
        "--distributed-executor-backend",
        default="mp",
        choices=("mp", "ray", "uni", "external_launcher"),
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
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
        help="Prompts submitted together to the persistent local LLM.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
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
        help="Optional sampling seed.",
    )
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
    if args.block_size < 1:
        parser.error("--block-size must be at least 1")
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


def create_local_llm(args: argparse.Namespace) -> tuple[Any, Any]:
    """Set NPU visibility and construct one persistent local engine."""

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.devices
    args.hidden_states_dir.mkdir(parents=True, exist_ok=True)

    # Import only after ASCEND_RT_VISIBLE_DEVICES is fixed.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        block_size=args.block_size,
        tensor_parallel_size=args.tensor_parallel_size,
        distributed_executor_backend=args.distributed_executor_backend,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_prefix_caching=args.enable_prefix_caching,
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
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    return llm, sampling_params


def output_hidden_states_path(output: Any) -> Path:
    params = getattr(output, "kv_transfer_params", None) or {}
    path = params.get("hidden_states_path")
    if not path:
        raise RuntimeError(
            "Local RequestOutput does not contain "
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


def generation_result(output: Any) -> GenerationResult:
    """Read the actual generated length from one local RequestOutput."""

    completions = getattr(output, "outputs", None) or []
    if len(completions) != 1:
        raise RuntimeError(
            "Expected exactly one completion per prompt, "
            f"but received {len(completions)}"
        )
    completion = completions[0]
    token_ids = getattr(completion, "token_ids", None)
    if token_ids is None:
        raise RuntimeError("Local completion does not contain token_ids")
    return GenerationResult(
        completion_tokens=len(token_ids),
        finish_reason=getattr(completion, "finish_reason", None),
    )


def process_batch(
    samples: Sequence[Sample],
    llm: Any,
    sampling_params: Any,
) -> tuple[list[Any], list[GenerationResult]]:
    """Generate once, then collect prompt states and generated lengths."""

    outputs = llm.generate(
        [sample.prompt for sample in samples],
        sampling_params,
        use_tqdm=False,
    )
    if len(outputs) != len(samples):
        raise RuntimeError(
            f"Local LLM returned {len(outputs)} outputs for {len(samples)} prompts"
        )

    paths: list[Path] = []
    try:
        for output in outputs:
            paths.append(output_hidden_states_path(output))
        hidden_states = [extract_last_token_hidden_states(path) for path in paths]
        generation_results = [generation_result(output) for output in outputs]
        return hidden_states, generation_results
    finally:
        for path in paths:
            delete_connector_file(path)


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

    print(f"Loading local model on Ascend devices {args.devices}")
    llm, sampling_params = create_local_llm(args)
    shard_index = next_shard_index(output_dir)
    samples = pending_samples(
        input_file,
        args.prompt_column,
        completed_rows,
        args.limit,
    )

    with tqdm(
        total=pending_count,
        unit="sample",
        desc=input_file.name,
    ) as progress:
        for shard_samples in batched(samples, args.shard_size):
            shard_hidden_states: list[Any] = []
            shard_generation_results: list[GenerationResult] = []
            for request_batch in batched(shard_samples, args.batch_size):
                hidden_states, generation_results = process_batch(
                    request_batch,
                    llm,
                    sampling_params,
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
