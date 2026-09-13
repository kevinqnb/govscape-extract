"""Small CLI queries submit.sh shells out to, so the bash side never parses
YAML itself. Each subcommand prints one tab-separated line to stdout.

    uv run -m experiments.submit_query experiment-info <experiment-id>
        -> <config_path>\\t<experiment_type>\\t<space-separated model_keys>

    uv run -m experiments.submit_query model-hardware <model_key>
        -> <device>\\t<gpu_count>\\t<min_vram_gb-or-empty>\\t<gpu_compute_capability-or-empty>\\t<max_walltime>
"""

from __future__ import annotations

import argparse

from experiments.config import MODEL_REGISTRY
from experiments.utils import find_experiment_config, load_experiment_spec


def experiment_info(experiment_id: str) -> str:
    spec = load_experiment_spec(find_experiment_config(experiment_id))
    model_keys = " ".join(spec.params["model_keys"])
    return f"{spec.config_path}\t{spec.params['experiment_type']}\t{model_keys}"


def model_hardware(model_key: str) -> str:
    hardware = MODEL_REGISTRY[model_key].hardware
    min_vram_gb = "" if hardware.min_vram_gb is None else hardware.min_vram_gb
    gpu_compute_capability = hardware.gpu_compute_capability or ""
    return f"{hardware.device}\t{hardware.gpu_count}\t{min_vram_gb}\t{gpu_compute_capability}\t{hardware.max_walltime}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiment_info_parser = subparsers.add_parser("experiment-info")
    experiment_info_parser.add_argument("experiment_id")

    model_hardware_parser = subparsers.add_parser("model-hardware")
    model_hardware_parser.add_argument("model_key")

    args = parser.parse_args()
    if args.command == "experiment-info":
        print(experiment_info(args.experiment_id))
    elif args.command == "model-hardware":
        print(model_hardware(args.model_key))


if __name__ == "__main__":
    main()
