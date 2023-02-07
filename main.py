import json
import os
import time

import numpy as np

from Xgboost import XGBModel
from Lgboost import LGBModel


def train_surrogate_model(model, data_root, seed):

    surrogate_model = model(data_root, seed)

    surrogate_model.train()
    surrogate_model.predict()


if __name__ == "__main__":
    # choose LGB or XGB
    model = XGBModel
    # model = LGBModel
    data_root = './data/dataset'
    seed = 6
    train_surrogate_model(model, data_root, seed)

