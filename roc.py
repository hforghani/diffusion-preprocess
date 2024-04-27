import datetime
import json
import os
from typing import Tuple
from matplotlib import pyplot
from sklearn import metrics

import numpy as np


def prepare_roc(fprs, tprs) -> Tuple[np.array, np.array]:
    """ Preprocess fpr and tpr values and sort them to calculate auc_roc or to plot ROC """
    # Every ROC curve must have 2 points <0,0> (no output) and <1,1> (returning all reference set as output).
    fprs, tprs = np.array(fprs), np.array(tprs)
    indexes = fprs.argsort()
    fprs = fprs[indexes]
    tprs = tprs[indexes]
    if (fprs[0], tprs[0]) != (0, 0):
        fprs = np.hstack((np.array([0]), fprs))
        tprs = np.hstack((np.array([0]), tprs))
    if (fprs[-1], tprs[-1]) != (1, 1):
        fprs = np.hstack((fprs, np.array([1])))
        tprs = np.hstack((tprs, np.array([1])))
    return fprs, tprs


def report_and_save_roc(fpr_list: list, tpr_list: list, method: str, criterion: str, dataset: str):
    """
    Save ROC plot as png and FPR-TPR values as json.
    """
    fpr, tpr = prepare_roc(fpr_list, tpr_list)
    auc_roc = metrics.auc(fpr, tpr)
    print(f"AUC-RoC = {auc_roc}")
    pyplot.figure()
    pyplot.plot(fpr, tpr)
    pyplot.axis((0, 1, 0, 1))
    pyplot.xlabel("fpr")
    pyplot.ylabel("tpr")
    results_path = 'results'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    base_name = f'{dataset}-{method}-{criterion}-roc-{datetime.datetime.now()}'.replace(" ", "-")
    pyplot.savefig(os.path.join(results_path, f'{base_name}.png'))
    # pyplot.show()
    with open(os.path.join(results_path, f'{base_name}.json'), "w") as f:
        json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist()}, f)
