import json
import os
import time

import numpy as np

from Xgboost import XGBModel


def train_surrogate_model(model, data_root, model_config_path, seed):
    # Load config
    # data_config = json.load(open(data_config_path, 'r'))

    model_config = json.load(open(model_config_path, 'r'))
    model_config['model'] = model

    surrogate_model = model(data_root, seed, model_config)

    surrogate_model.train()


if __name__ == "__main__":
    model = XGBModel
    data_root = "./data/dataset"
    model_config_path = "./config/xgb_configspace.json"
    seed = 6
    train_surrogate_model(model, data_root, model_config_path, seed)

