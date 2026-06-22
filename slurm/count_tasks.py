#!/usr/bin/env python3
"""Prints the number of (dataset, model) task combinations for the current config."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import ParameterGrid
from Models.config_parser import load_config, load_dataset_config

CONFIG = "configs/config.yaml"
model_runs = [p for pg in load_config(CONFIG).values() for p in ParameterGrid(pg)]
dataset_runs = [p for pg in load_dataset_config(CONFIG).values() for p in ParameterGrid(pg)]
print(len(model_runs) * len(dataset_runs))
