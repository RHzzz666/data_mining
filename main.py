import json
import os
import time

import numpy as np

from Xgboost import XGBModel
from Lgboost import LGBModel
from Adaboost import ADAModel


def train_surrogate_model(model, data_root, seed):

    surrogate_model = model(data_root, seed)

    surrogate_model.train()
    surrogate_model.predict()


if __name__ == "__main__":
    # choose LGB or XGB or ADA
    # model = ADAModel
    model = LGBModel
    # model = XGBModel
    data_root = './data/dataset'
    seed = 6
    train_surrogate_model(model, data_root, seed)
