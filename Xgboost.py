import logging
import os
import json

import numpy as np
from sklearn.model_selection import train_test_split

import xgboost as xgb
import torch
import torch.backends.cudnn as cudnn



class XGBModel:
    def __init__(self, data_root, seed, model_config):
        self.model = None
        self.data_root = data_root
        self.model_config = model_config
        # self.data_config = data_config
        self.seed = seed

        np.random.seed(seed)
        cudnn.benchmark = True
        torch.manual_seed(seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(seed)

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

        self.model_config["param:objective"] = "reg:squarederror"
        self.model_config["param:eval_metric"] = "rmse"

    def load_data(self, data_paths):
        hyps, val_accuracies, test_accuracies = [], [], []

        for data_path in data_paths:
            json_file = json.load(open(data_path, 'r'))

            hyp = json_file['x']
            val_accuracy = json_file['y']

        X = np.array(hyps)
        y = np.array(val_accuracies)

        return X, y

    def root_to_paths_train(self):
        root = os.path.join(self.data_root, 'train')
        paths = []
        i = 1
        while i < 46226:
            path = os.path.join(root, '{}.json'.format(i))
            paths.extend(path)
        return paths

    def parse_param_config(self):
        identifier = "param:"
        param_config = dict()
        for key, val in self.model_config.items():
            if key.startswith(identifier):
                param_config[key.replace(identifier, "")] = val
        return param_config

    def train(self):
        data_paths = self.root_to_paths_train()
        X, y = self.load_data(data_paths)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        param_config = self.parse_param_config()
        param_config["seed"] = self.seed

        self.model = xgb.train(param_config, dtrain, num_boost_round=self.model_config["param:num_rounds"],
                               early_stopping_rounds=self.model_config["early_stopping_rounds"],
                               verbose_eval=1,
                               evals=[(dval, 'val')])

        # train_pred, var_train = self.model.predict(dtrain), None
        # val_pred, var_val = self.model.predict(dval), None
        #
        # # self.save()
        #
        # fig_train = utils.scatter_plot(np.array(train_pred), np.array(y_train), xlabel='Predicted', ylabel='True',
        #                                title='')
        # fig_train.savefig(os.path.join(self.log_dir, 'pred_vs_true_train.jpg'))
        # plt.close()
        #
        # fig_val = utils.scatter_plot(np.array(val_pred), np.array(y_val), xlabel='Predicted', ylabel='True', title='')
        # fig_val.savefig(os.path.join(self.log_dir, 'pred_vs_true_val.jpg'))
        # plt.close()
        #
        # train_metrics = utils.evaluate_metrics(y_train, train_pred, prediction_is_first_arg=False)
        # valid_metrics = utils.evaluate_metrics(y_val, val_pred, prediction_is_first_arg=False)
        #
        # logging.info('train metrics: %s', train_metrics)
        # logging.info('valid metrics: %s', valid_metrics)

        # return valid_metrics

