"""Compare thinking on/off with identical CSV samples and different batch sizes.

All lengths are tokenizer token counts. prompt_tokens excludes chat formatting;
input_tokens is the actual engine input including the system message/template.
output_tokens includes all generated tokens (thinking and final answer).
Timing covers the blocking generate() call, not loading, templating or printing.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import time
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
    prompt_tokens: int
    input_ids: list[int]


@dataclass(frozen=True)
class Result:
    source_row: int
    prompt_tokens: int
    input_tokens: int
    output_tokens: int
    finish_reason: str
    text: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--prompt-column", default="user_prompt_content")
    parser.add_argument("--sample-size", type=int, default=32,
                        help="Random nonempty CSV rows reused in every configuration.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
                        help="Same system instruction for both modes; use an empty string to omit.")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
                        help="Run one untimed 1-token batch per mode and batch size.")
    parser.add_argument("--show-answers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--distributed-executor-backend", default="auto",
                        choices=("auto", "mp", "ray", "uni", "external_launcher"))
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Safety ceiling, not a length target; capped results are marked.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="Same sampling seed in both modes.")
    args = parser.parse_args(argv)
    for name in ("sample_size", "tensor_parallel_size", "block_size", "max_model_len", "max_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if any(size < 1 for size in args.batch_sizes) or len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("--batch-sizes must contain distinct positive integers")
    if args.sample_size < max(args.batch_sizes):
        parser.error("--sample-size must be at least the largest batch size")
    devices = [device.strip() for device in args.devices.split(",")]
    if (any(not device.isdigit() for device in devices)
            or len(set(map(int, devices))) != len(devices)
            or len(devices) != args.tensor_parallel_size):
        parser.error("--devices must contain distinct NPU IDs matching --tensor-parallel-size")
    args.devices = ",".join(devices)
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if not args.temperature >= 0 or not 0 < args.top_p <= 1 or not 0 <= args.min_p <= 1:
        parser.error("Require temperature >= 0, top-p in (0, 1], min-p in [0, 1]")
    if args.top_k < -1 or not args.repetition_penalty > 0:
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


def create_engine(args: argparse.Namespace) -> tuple[Any, Any]:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.devices
    from vllm import LLM, SamplingParams

    keys = ("model", "tensor_parallel_size", "block_size", "max_model_len",
            "gpu_memory_utilization", "enforce_eager", "trust_remote_code")
    settings = {key: getattr(args, key) for key in keys}
    # Prefix caching would favor samples repeated by earlier configurations.
    settings.update(enable_chunked_prefill=False, enable_prefix_caching=False)
    if args.distributed_executor_backend != "auto":
        settings["distributed_executor_backend"] = args.distributed_executor_backend
    sampling_keys = ("max_tokens", "temperature", "top_p", "top_k", "min_p",
                     "presence_penalty", "frequency_penalty", "repetition_penalty", "seed")
    sampling = SamplingParams(**{key: getattr(args, key) for key in sampling_keys},
                              skip_special_tokens=False)
    print(f"Engine: {settings}\nSampling: {sampling}", flush=True)
    started = time.perf_counter()
    llm = LLM(**settings)
    print(f"Model load time: {time.perf_counter() - started:.3f}s (excluded)", flush=True)
    return llm, sampling


def prepare_samples(samples: Sequence[Sample], tokenizer: Any, thinking: bool,
                    system_prompt: str, max_model_len: int, max_tokens: int) -> list[PreparedSample]:
    prepared = []
    for sample in samples:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": sample.prompt})
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        input_ids = tokenizer.encode(rendered, add_special_tokens=False)
        # Fail rather than silently changing generation budgets between modes.
        if len(input_ids) + max_tokens > max_model_len:
            raise ValueError(
                f"source_row={sample.source_row} think={thinking}: input={len(input_ids)} "
                f"+ max_tokens={max_tokens} exceeds max_model_len={max_model_len}")
        prepared.append(PreparedSample(
            sample, len(tokenizer.encode(sample.prompt, add_special_tokens=False)), input_ids))
    return prepared


def run_batch(llm: Any, sampling: Any,
              batch: Sequence[PreparedSample]) -> tuple[list[Result], float]:
    prompts = [{"prompt_token_ids": item.input_ids} for item in batch]
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    # generate() is blocking: all completions are available when it returns.
    elapsed = time.perf_counter() - start
    if len(outputs) != len(batch):
        raise RuntimeError("LLM output count does not match the submitted batch")
    results = []
    for item, output in zip(batch, outputs):
        if not output.finished or len(output.outputs) != 1:
            raise RuntimeError("Expected one finished completion per prompt")
        if list(output.prompt_token_ids or []) != item.input_ids:
            raise RuntimeError("Returned input tokens differ from submitted chat-template tokens")
        completion = output.outputs[0]
        if completion.finish_reason not in ("stop", "length"):
            raise RuntimeError(f"Unexpected finish reason: {completion.finish_reason}")
        results.append(Result(item.sample.source_row, item.prompt_tokens,
                              len(output.prompt_token_ids), len(completion.token_ids),
                              completion.finish_reason, completion.text))
    return results, elapsed


def benchmark(args: argparse.Namespace, samples: Sequence[Sample], llm: Any,
              sampling: Any) -> list[dict[str, Any]]:
    tokenizer = llm.get_tokenizer()
    prepared = {mode: prepare_samples(samples, tokenizer, mode, args.system_prompt,
                                     args.max_model_len, args.max_tokens)
                for mode in (False, True)}
    if all(off.input_ids == on.input_ids for off, on in zip(prepared[False], prepared[True])):
        raise ValueError("Chat template ignores enable_thinking: on/off inputs are identical")
    print("\nLengths in tokens: prompt=raw CSV prompt; input=full chat template; "
          "output=all generated tokens including thinking.\n"
          "Batch seconds=blocking generate() wall time; excludes model loading, "
          "template processing, warmup and printing.\n"
          "finish=length means capped, not a natural answer length.\n"
          "The batch size is the submitted request count, not guaranteed NPU concurrency.",
          flush=True)
    summaries = []
    for config_index, batch_size in enumerate(args.batch_sizes):
        # Alternate order to reduce systematic first-mode bias.
        modes = (False, True) if config_index % 2 == 0 else (True, False)
        for mode in modes:
            items = prepared[mode]
            print(f"\n=== think={mode} batch_size={batch_size} ===", flush=True)
            if args.warmup:
                warmup_params = copy.copy(sampling)
                warmup_params.max_tokens = 1
                run_batch(llm, warmup_params, items[:batch_size])
                print("Warmup completed (excluded).", flush=True)
            results = []
            batch_seconds = []
            for offset in range(0, len(items), batch_size):
                batch = items[offset:offset + batch_size]
                records, elapsed = run_batch(llm, sampling, batch)
                results.extend(records)
                batch_seconds.append(elapsed)
                print(f"\nBatch {len(batch_seconds)}: requested={batch_size} actual={len(batch)} "
                      f"seconds={elapsed:.3f}", flush=True)
                print("source_row  prompt_tokens  input_tokens  output_tokens  finish", flush=True)
                for record in records:
                    print(f"{record.source_row:10d} {record.prompt_tokens:14d} "
                          f"{record.input_tokens:13d} {record.output_tokens:14d} "
                          f"{record.finish_reason}", flush=True)
                if args.show_answers:
                    for item, record in zip(batch, records):
                        print(f"\n[source_row={record.source_row} think={mode}]\n"
                              f"Prompt:\n{item.sample.prompt}\n"
                              f"Response (unabridged):\n{record.text}", flush=True)
            total_seconds = sum(batch_seconds)
            total_output = sum(record.output_tokens for record in results)
            summaries.append({
                "thinking": mode, "batch_size": batch_size, "samples": len(results),
                "batches": len(batch_seconds), "total_seconds": total_seconds,
                "mean_batch_seconds": total_seconds / len(batch_seconds),
                "mean_prompt_tokens": sum(record.prompt_tokens for record in results) / len(results),
                "mean_input_tokens": sum(record.input_tokens for record in results) / len(results),
                "mean_output_tokens": total_output / len(results),
                "output_tokens_per_second": total_output / max(total_seconds, 1e-9),
                "capped": sum(record.finish_reason == "length" for record in results),
            })
    print("\n=== Summary (generation only) ===")
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
    samples = select_samples(args.input_file, args.prompt_column, args.sample_size, args.sample_seed)
    print(f"Dataset: {args.input_file.resolve()}\n"
          f"Sample seed: {args.sample_seed}; source rows: {[sample.source_row for sample in samples]}\n"
          f"System prompt (both modes): {args.system_prompt!r}\n"
          f"Measured completions: {len(samples) * len(args.batch_sizes) * 2}", flush=True)
    llm, sampling = create_engine(args)
    benchmark(args, samples, llm, sampling)
    return 0
