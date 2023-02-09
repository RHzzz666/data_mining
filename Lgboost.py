import logging
import os
import json

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import lightgbm as lgb
import torch
import torch.backends.cudnn as cudnn
import utils

class LGBModel:
    def __init__(self, data_root, seed):
        self.model = None
        self.data_root = data_root
        # self.data_config = data_config
        self.seed = seed

        np.random.seed(seed)
        cudnn.benchmark = True
        torch.manual_seed(seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(seed)

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    def train(self):
        data_paths = self.root_to_paths_train()
        X, y = self.load_data(data_paths)

        # logging.info("shape of x: %s" % X.shape)
        # print(X.shape)

        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

        # dtrain = xgb.DMatrix(X_train, label=y_train)
        # dval = xgb.DMatrix(X_val, label=y_val)

        dtrain = lgb.Dataset(X, label=y)

        params = {
            'max_depth': 18,
            'num_leaves': 40,
            'max_bin': 336,
            'feature_fraction': 0.1531795056267644,
            'min_child_weight': 0.582192788716,
            'lambda_l1': 0.011489772418204284,
            'lambda_l2': 134.50743557010372,
            'boosting_type': "gbdt",
            'learning_rate': 0.02182249761978233,
            'objective': 'regression',
            'eval_metric': 'rmse',
            'seed': self.seed
        }

        logging.info("#####train####\n")

        self.model = lgb.train(params, dtrain, num_boost_round=30000)

        train_pred, var_train = self.model.predict(dtrain), None
        # val_pred, var_val = self.model.predict(dval), None

        # self.save()

        fig_train = utils.scatter_plot(np.array(train_pred), np.array(y), xlabel='Predicted', ylabel='True',
                                       title='')
        fig_train.savefig(os.path.join('./log', 'pred_vs_true_train_xgboost.jpg'))
        plt.close()

        train_metrics = utils.evaluate_metrics(y, train_pred, prediction_is_first_arg=False)

        logging.info('train metrics: %s', train_metrics)

    def predict(self):
        hyps = []
        data_paths = self.root_to_paths_test()
        for data_path in data_paths:
            json_file = json.load(open(data_path, 'r'))
            hyp = json_file['x']
            hyps.append(self.encode(hyp))

        X_pred = np.array(hyps)

        # dtest = lgb.Dataset(X_pred)
        ypred = self.model.predict(X_pred)
        # ypred = self.model.predict(dtest, iteration_range=(0, self.model.best_iteration + 1))
        my_pred = np.array(ypred)

        np.savetxt('202221044027_lgb.csv', my_pred, delimiter=',', encoding='utf-8')

    def load_data(self, data_paths):
        # hyp = hypermeter = archtecture
        hyps, val_accuracies = [], []

        for data_path in data_paths:
            json_file = json.load(open(data_path, 'r'))

            hyp = json_file['x']
            val_accuracy = json_file['y']

            # logging.info("x = %s" % hyp)
            # logging.info("y = %s" % val_accuracy)

            hyps.append(self.encode(hyp))
            val_accuracies.append(val_accuracy)
        # logging.info("encode_0: %s" % hyps[0])

        X = np.array(hyps)
        y = np.array(val_accuracies)

        return X, y

    def encode(self, hyp):
        x = []
        normal_matrix = np.array(hyp['normal']['adjacency_matrix']).ravel()

        normal_operators = []
        for op in hyp['normal']['operators']:
            if op == "sep_conv_3x3":
                normal_operators.append(0)
            if op == "sep_conv_5x5":
                normal_operators.append(1)
            if op == "dil_conv_3x3":
                normal_operators.append(2)
            if op == "dil_conv_5x5":
                normal_operators.append(3)
            if op == "max_pool_3x3":
                normal_operators.append(4)
            if op == "avg_pool_3x3":
                normal_operators.append(5)
            if op == "skip_connect":
                normal_operators.append(6)
            if op == "":
                normal_operators.append(7)

        reduce_matrix = np.array(hyp['reduce']['adjacency_matrix']).ravel()

        reduce_operators = []
        for op in hyp['reduce']['operators']:
            if op == "sep_conv_3x3":
                reduce_operators.append(0)
            if op == "sep_conv_5x5":
                reduce_operators.append(1)
            if op == "dil_conv_3x3":
                reduce_operators.append(2)
            if op == "dil_conv_5x5":
                reduce_operators.append(3)
            if op == "max_pool_3x3":
                reduce_operators.append(4)
            if op == "avg_pool_3x3":
                reduce_operators.append(5)
            if op == "skip_connect":
                reduce_operators.append(6)
            if op == "":
                reduce_operators.append(7)

        x.extend(normal_operators)
        x.extend(normal_matrix.tolist())
        x.extend(reduce_operators)
        x.extend(reduce_matrix.tolist())

        return x

    def root_to_paths_train(self):
        root = os.path.join(self.data_root, 'train')
        logging.info("root: %s" % root)
        paths = []
        i = 1
        while i < 46226:
            path = os.path.join(root, '{}.json'.format(i))
            paths.append(path)
            i += 1
        return paths

    def root_to_paths_test(self):
        root = os.path.join(self.data_root, 'test')
        paths = []
        i = 1
        while i < 5001:
            path = os.path.join(root, '{}.json'.format(i))
            paths.append(path)
            i += 1
        return paths