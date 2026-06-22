#!/usr/bin/env python3
"""Prints the number of (dataset, model) task combinations for config-nn.yaml."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import ParameterGrid
from Models.config_parser import load_config, load_dataset_config

CONFIG = "configs/config-nn.yaml"
model_config = load_config(CONFIG)
nn_config = {k: v for k, v in model_config.items() if k == "pytorch_neural_network"}
model_runs = [p for pg in nn_config.values() for p in ParameterGrid(pg)]
dataset_runs = [p for pg in load_dataset_config(CONFIG).values() for p in ParameterGrid(pg)]
print(len(model_runs) * len(dataset_runs))
