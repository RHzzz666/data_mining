import json
import os
import time

import numpy as np

from Xgboost import XGBModel


def train_surrogate_model(model, data_root, seed):

    surrogate_model = model(data_root, seed)

    surrogate_model.train()
    surrogate_model.predict()


if __name__ == "__main__":
    model = XGBModel
    data_root = 'data\\dataset'
    # model_config_path = 'config\\xgb_configspace.json'
    seed = 6
    train_surrogate_model(model, data_root, seed)

