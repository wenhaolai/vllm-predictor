#!/usr/bin/env python3
"""Extract ForeLen last-token prefill features and fresh generation lengths."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vllm_predictor.data_extraction import (  # noqa: E402
    ExtractionConfig,
    ResultStore,
    iter_csv_samples,
    pending_samples,
    process_sample,
)


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
    elif status == "skipped":
        print(f"[{counters['processed']}] {sample_id} already completed")
    else:
        print(
            f"[{counters['processed']}] {sample_id} {status}: "
            f"{record.get('error', '')}",
            file=sys.stderr,
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

    counters: dict[str, int] = {"processed": 0}
    failure_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending: set[Future[dict[str, Any]]] = set()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < args.workers * 2:
                try:
                    sample = next(samples)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(executor.submit(process_sample, sample, config, store))
            if not pending:
                continue
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                record = future.result()
                counters["processed"] += 1
                report(record, counters)
                if record["status"] not in {"completed", "skipped"}:
                    failure_count += 1
                    if args.fail_fast:
                        for outstanding in pending:
                            outstanding.cancel()
                        raise RuntimeError("Stopping after the first failed sample")

    print(
        "Finished: "
        + ", ".join(f"{key}={value}" for key, value in sorted(counters.items()))
    )
    print(f"Manifest: {store.manifest_path}")
    print(f"Results:  {store.results_path}")
    return 1 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

