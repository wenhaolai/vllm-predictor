#!/usr/bin/env python3
"""Generate ForeLen answers normally and save token lengths in samples.csv.

Uses the same raw prompts and zero-based source rows as extract_forelen.py.
completion_tokens counts returned token IDs, not characters or re-tokenized
text. finish_reason=length indicates a capped answer rather than its natural
length. Keep the input, model and sampling settings unchanged when resuming;
use a new output directory for a different experiment.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from pathlib import Path
from typing import Any, Sequence

from extract_forelen import (
    Sample,
    batched,
    configure_logging,
    count_pending_samples,
    pending_samples,
)

LOGGER = logging.getLogger("extract_forelen_lengths")
CSV_FIELDS = ("source_file", "source_row", "completion_tokens", "finish_reason")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--prompt-column", default="user_prompt_content")
    parser.add_argument("--output-dir", type=Path, default=Path("data/forelen_lengths_2048"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--distributed-executor-backend", default="auto",
                        choices=("auto", "mp", "ray", "uni", "external_launcher"),
                        help="auto leaves backend selection to vLLM, as in the reference example.")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Maximum generated tokens per prompt; normal EOS stopping is enabled.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many unfinished rows; omit for all rows.")
    for name, default in (("enable-chunked-prefill", False),
                          ("enable-prefix-caching", False),
                          ("trust-remote-code", False), ("enforce-eager", True)):
        parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction,
                            default=default)
    args = parser.parse_args(argv)
    for name in ("tensor_parallel_size", "block_size", "max_model_len",
                 "batch_size", "max_tokens", "limit"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    devices = [device.strip() for device in args.devices.split(",")]
    if (any(not device.isdigit() for device in devices)
            or len(set(map(int, devices))) != len(devices)
            or len(devices) != args.tensor_parallel_size):
        parser.error("--devices must contain distinct NPU IDs matching --tensor-parallel-size")
    args.devices = ",".join(devices)
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if not args.temperature >= 0:
        parser.error("--temperature must be >= 0")
    if not 0 < args.top_p <= 1 or not 0 <= args.min_p <= 1:
        parser.error("--top-p must be in (0, 1] and --min-p in [0, 1]")
    if args.top_k < -1:
        parser.error("--top-k must be -1, 0, or a positive integer")
    if not -2 <= args.presence_penalty <= 2 or not -2 <= args.frequency_penalty <= 2:
        parser.error("--presence-penalty and --frequency-penalty must be in [-2, 2]")
    if not args.repetition_penalty > 0:
        parser.error("--repetition-penalty must be > 0")
    return args


def create_local_llm(args: argparse.Namespace) -> tuple[Any, Any]:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.devices
    # Import only after configuring NPU visibility. No extraction connectors.
    from vllm import LLM, SamplingParams

    engine_keys = ("model", "block_size", "tensor_parallel_size",
                   "distributed_executor_backend", "enable_chunked_prefill",
                   "enable_prefix_caching", "max_model_len", "gpu_memory_utilization",
                   "enforce_eager", "trust_remote_code")
    sampling_keys = ("max_tokens", "temperature", "top_p", "top_k", "min_p",
                     "presence_penalty", "frequency_penalty", "repetition_penalty", "seed")
    engine_kwargs = {key: getattr(args, key) for key in engine_keys}
    if args.distributed_executor_backend == "auto":
        del engine_kwargs["distributed_executor_backend"]
    sampling_kwargs = {key: getattr(args, key) for key in sampling_keys}
    LOGGER.info("Initializing LLM: devices=%s settings=%s", args.devices, engine_kwargs)
    LOGGER.info("Sampling settings: %s", sampling_kwargs)
    sampling_params = SamplingParams(**sampling_kwargs)
    # Match the reference engine's scheduler defaults. batch_size only controls
    # how many CSV prompts we submit in each generate() call.
    llm = LLM(**engine_kwargs)
    return llm, sampling_params


def process_batch(samples: Sequence[Sample], llm: Any,
                  sampling_params: Any) -> list[dict[str, Any]]:
    started_at = time.perf_counter()
    outputs = llm.generate([sample.prompt for sample in samples],
                           sampling_params, use_tqdm=False)
    if len(outputs) != len(samples):
        raise RuntimeError(f"LLM returned {len(outputs)} outputs for {len(samples)} prompts")
    records = []
    for sample, output in zip(samples, outputs):
        if not output.finished or len(output.outputs) != 1:
            raise RuntimeError(f"Expected one finished completion for source row {sample.source_row}")
        completion = output.outputs[0]
        if completion.finish_reason not in ("stop", "length"):
            raise RuntimeError(f"Unexpected finish reason: {completion.finish_reason!r}")
        records.append({"source_row": sample.source_row,
                        "completion_tokens": len(completion.token_ids),
                        "finish_reason": completion.finish_reason})
    LOGGER.info("Batch generated: samples=%d capped=%d elapsed=%.3fs", len(records),
                sum(record["finish_reason"] == "length" for record in records),
                time.perf_counter() - started_at)
    return records


def load_completed_rows(results_path: Path, source_file: str) -> set[int]:
    if not results_path.exists() or results_path.stat().st_size == 0:
        return set()
    completed = set()
    with results_path.open(encoding="utf-8", newline="") as results:
        reader = csv.DictReader(results)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"Unexpected CSV columns in {results_path}: {reader.fieldnames}")
        for row in reader:
            if (None in row or any(value is None for value in row.values())
                    or int(row["completion_tokens"]) < 0
                    or row["finish_reason"] not in ("stop", "length")):
                raise ValueError(f"Incomplete or invalid result in {results_path}: {row}")
            if row["source_file"] == source_file:
                completed.add(int(row["source_row"]))
    return completed


def append_results(results_path: Path, source_file: str,
                   records: Sequence[dict[str, Any]]) -> None:
    write_header = not results_path.exists() or results_path.stat().st_size == 0
    with results_path.open("a", encoding="utf-8", newline="") as results:
        writer = csv.DictWriter(results, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for record in records:
            writer.writerow({"source_file": source_file, **record})
        results.flush()
        os.fsync(results.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    input_file = args.input_file.resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"ForeLen CSV file does not exist: {input_file}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "samples.csv"
    source_file = str(input_file)
    completed = load_completed_rows(results_path, source_file)
    pending_count = count_pending_samples(input_file, args.prompt_column, completed, args.limit)
    if not pending_count:
        LOGGER.info("No unfinished rows: input=%s", input_file)
        return 0
    LOGGER.info("Dataset ready: input=%s completed=%d pending=%d", input_file,
                len(completed), pending_count)
    llm, sampling_params = create_local_llm(args)
    samples = pending_samples(input_file, args.prompt_column, completed, args.limit)
    saved = 0
    for batch in batched(samples, args.batch_size):
        records = process_batch(batch, llm, sampling_params)
        append_results(results_path, source_file, records)
        saved += len(records)
        LOGGER.info("Progress: %d/%d samples saved", saved, pending_count)
    LOGGER.info("Length extraction completed: results=%s", results_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user")
        raise SystemExit(130) from None
    except Exception:
        LOGGER.exception("Length extraction failed")
        raise
