#!/usr/bin/env python3
"""CLI entry point for the thinking/batch-size benchmark."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vllm_predictor.chat_benchmark import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
