"""CLI: start a configured LLM's vLLM server in the foreground and record its
endpoint so runner.py finds it automatically -- no more re-exporting a
base-url env var every session.

    uv run -m experiments.serve_model --model-key gpt-oss-120b
    uv run -m experiments.serve_model --model-key qwen3-0.6b --port 8001

Streams vllm's own startup logs to the terminal; blocks once healthy.
Ctrl+C (or SIGTERM) stops the server and removes its entry from
experiments/.endpoints.json. Meant to run on the same machine (or a
filesystem shared with) wherever you run `experiments.runner` -- see
experiments/README.md if that's a different box than the GPU box.
"""

from __future__ import annotations

import argparse
import signal
import sys

from experiments.config import MODEL_REGISTRY, LLMModelConfig
from experiments.serving import ENDPOINTS_FILE, LocalVLLMServer, clear_endpoint, write_endpoint


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    llm_keys = sorted(k for k, v in MODEL_REGISTRY.items() if isinstance(v, LLMModelConfig))
    parser.add_argument("--model-key", required=True, choices=llm_keys)
    parser.add_argument("--port", type=int, default=None, help="Overrides the default port (8000)")
    return parser


def _raise_keyboard_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt()


def main() -> None:
    args = build_arg_parser().parse_args()
    model_config = MODEL_REGISTRY[args.model_key]

    server = LocalVLLMServer(
        model=model_config.model,
        port=args.port or 8000,
        extra_args=model_config.vllm_args,
        inherit_stdio=True,
    )

    # Convert SIGTERM into the same cleanup path as Ctrl+C, so a `kill` (not
    # just an interactive Ctrl+C) still tears down vllm's process group and
    # clears this model's entry from the endpoints file.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    print(f"Starting vllm serve {model_config.model!r} on port {server.port} (this can take minutes for large models)...")
    try:
        server.start()
    except Exception as e:
        print(f"Failed to start: {e}", file=sys.stderr)
        raise SystemExit(1)

    write_endpoint(model_config.key, server.base_url)
    print(f"Healthy after {server.startup_seconds:.1f}s -- wrote endpoint to {ENDPOINTS_FILE}")
    print(f"Serving {model_config.key} at {server.base_url}. Ctrl+C to stop.")
    try:
        server.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        server.stop()
        clear_endpoint(model_config.key)


if __name__ == "__main__":
    main()
