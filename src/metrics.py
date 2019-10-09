import numpy as np
from sklearn.metrics import precision_score, fbeta_score


def avg_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average="macro")


def avg_fscore(y_true, y_pred):
    return fbeta_score(y_true, y_pred, average="macro", beta=0.5)
