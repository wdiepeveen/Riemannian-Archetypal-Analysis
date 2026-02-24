import os
import yaml

def load_experiment_config(config_path: str, experiment_name: str):
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    base_results_dir = raw.get("base_results_dir", "results/mnist/three_phases")
    experiments = raw["experiments"]
    if experiment_name not in experiments:
        raise KeyError(f"Experiment '{experiment_name}' not found in {config_path}")

    cfg = experiments[experiment_name]
    cfg["base_results_dir"] = base_results_dir
    return cfg
