import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, r2_score


def scatter_plot(xs, ys, xlabel, ylabel, title):
    """
    Creates scatter plot of the predicted and groundtruth performance
    :param xs:
    :param ys:
    :param xlabel:
    :param ylabel:
    :param title:
    :return:
    """
    fig = plt.figure(figsize=(4, 3))
    plt.tight_layout()
    plt.grid(True, which='both', ls='-', alpha=0.5)
    plt.scatter(xs, ys, alpha=0.8, s=4)
    xs_min = xs.min()
    xs_max = xs.max()
    plt.plot(np.linspace(xs_min, xs_max), np.linspace(xs_min, xs_max), 'r', alpha=0.5)
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.title(title)
    return fig


def evaluate_metrics(y_true, y_pred, prediction_is_first_arg):
    """
    Create a dict with all evaluation metrics
    """

    if prediction_is_first_arg:
        y_true, y_pred = y_pred, y_true

    metrics_dict = dict()
    metrics_dict["mse"] = mean_squared_error(y_true, y_pred)
    metrics_dict["rmse"] = np.sqrt(metrics_dict["mse"])
    metrics_dict["r2"] = r2_score(y_true, y_pred)
    metrics_dict["kendall_tau"], p_val = kendalltau(y_true, y_pred)
    metrics_dict["kendall_tau_2_dec"], p_val = kendalltau(y_true, np.round(np.array(y_pred), decimals=2))
    metrics_dict["kendall_tau_1_dec"], p_val = kendalltau(y_true, np.round(np.array(y_pred), decimals=1))

    metrics_dict["spearmanr"] = spearmanr(y_true, y_pred).correlation

    return metrics_dict
