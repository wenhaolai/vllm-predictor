#!/usr/bin/env python3
"""CLI entry point for online ForeLen hidden-state extraction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vllm_predictor.extract_forelen_online import main

if __name__ == "__main__":
    raise SystemExit(main())
