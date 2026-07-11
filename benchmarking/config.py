import yaml
import numpy as np


def as_list(value):
    """Normalize a benchmark field that may be a scalar or a list into a list."""
    return list(value) if isinstance(value, list) else [value]


def parse_param_range(param_def):
    if isinstance(param_def, list):
        return param_def

    if isinstance(param_def, dict):
        if 'num_steps' in param_def:
            min_val = param_def['min']
            max_val = param_def['max']
            num_steps = param_def['num_steps']
            return [float(v) for v in np.linspace(min_val, max_val, int(num_steps))]

        if 'step' in param_def:
            min_val = param_def['min']
            max_val = param_def['max']
            step = param_def['step']

            if isinstance(min_val, int) and isinstance(max_val, int) and isinstance(step, int):
                return list(range(min_val, max_val + 1, step))

            n_steps = round((max_val - min_val) / step) + 1
            return [float(v) for v in np.linspace(min_val, max_val, n_steps)]

    return param_def


def load_dataset_config(config_path: str) -> dict:
    """Parse the datasets section of config.yaml.

    Returns a dict keyed by dataset name where each value maps
    param names to their expanded list of values, e.g.:
        {"california_housing": {"n_features": [1,2,...,8], "n_samples": [2000,4000,...]}}
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    datasets_dict = raw.get('datasets', {}) or {}
    result = {}
    for dataset_key, params in datasets_dict.items():
        result[dataset_key] = {}
        for param_name, param_def in (params or {}).items():
            result[dataset_key][param_name] = parse_param_range(param_def)
    return result


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    models_dict = raw.get('models', {})
    result = {}

    for model_key, params in models_dict.items():
        result[model_key] = {}
        for param_name, param_def in params.items():
            result[model_key][param_name] = parse_param_range(param_def)

    return result


def load_seed(config_path: str, default: int = 42) -> int:
    """A single benchmark-wide RNG seed (config.yaml -> benchmark.seed)."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    seed = (raw.get('benchmark', {}) or {}).get('seed', default)
    return as_list(seed)[0]
