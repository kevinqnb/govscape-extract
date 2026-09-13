"""submit.sh shells out to these two functions to read a config without
parsing YAML in bash -- pin their tab-separated output shape."""

from __future__ import annotations

from experiments.submit_query import experiment_info, model_hardware


def test_experiment_info_extraction_smoke():
    line = experiment_info("2026-09-12-extraction-smoke-01")
    config_path, experiment_type, model_keys = line.split("\t")
    assert config_path.endswith("2026-09-12-extraction-smoke-01.yaml")
    assert experiment_type == "extraction"
    assert model_keys == "gliner2-base"


def test_experiment_info_ground_truth_smoke():
    line = experiment_info("2026-09-12-ground-truth-smoke-01")
    _config_path, experiment_type, model_keys = line.split("\t")
    assert experiment_type == "ground_truth"
    assert model_keys == "gpt-5.6-terra claude-sonnet-5 gemini-3.7-flash"


def test_model_hardware_cpu_only_model_has_no_gpu_fields():
    device, gpu_count, min_vram_gb, gpu_compute_capability, max_walltime = model_hardware("gliner2-base").split("\t")
    assert device == "cpu"
    assert gpu_count == "0"
    assert min_vram_gb == ""
    assert gpu_compute_capability == ""
    assert max_walltime == "24:00:00"


def test_model_hardware_gpu_model_has_full_fields():
    device, gpu_count, min_vram_gb, gpu_compute_capability, max_walltime = model_hardware("qwen3-0.6b").split("\t")
    assert device == "cuda"
    assert gpu_count == "1"
    assert min_vram_gb == "4"
    assert gpu_compute_capability == "7.0"
    assert max_walltime == "24:00:00"


def test_model_hardware_hosted_api_has_no_device_and_no_gpu_fields():
    device, gpu_count, min_vram_gb, gpu_compute_capability, _max_walltime = model_hardware("gpt-5.6-terra").split("\t")
    assert device == "none"
    assert gpu_count == "0"
    assert min_vram_gb == ""
    assert gpu_compute_capability == ""
