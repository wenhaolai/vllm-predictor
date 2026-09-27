#!/usr/bin/env python3
"""Minimal online request for prompt/prefill hidden-state extraction."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {details}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--prompt", default="请解释什么是投机解码。")
    parser.add_argument(
        "--manifest",
        default=os.environ.get(
            "HIDDEN_STATES_MANIFEST",
            "/home/laiwenhao/vllm-predictor/outputs/hidden_states/manifest.jsonl",
        ),
    )
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }
    result = post_json(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        payload,
    )

    kv_params = result.get("kv_transfer_params") or {}
    hidden_states_path = kv_params.get("hidden_states_path")
    if not hidden_states_path:
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        raise RuntimeError("Response does not contain kv_transfer_params.hidden_states_path")

    path = Path(hidden_states_path)
    if not path.is_file():
        raise RuntimeError(f"Server returned a path that does not exist: {path}")

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            example_hidden_states_connector,
        )

        tensors = example_hidden_states_connector.load_hidden_states(str(path))
        token_shape = list(tensors["token_ids"].shape)
        hidden_shape = list(tensors["hidden_states"].shape)
    except ImportError:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as tensors:
            token_shape = list(tensors.get_tensor("token_ids").shape)
            hidden_shape = list(tensors.get_tensor("hidden_states").shape)

    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request_id": result.get("id"),
        "model": args.model,
        "prompt": args.prompt,
        "hidden_states_path": str(path),
        "token_ids_shape": token_shape,
        "hidden_states_shape": hidden_shape,
    }

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
