from typing import Optional, Sequence, Tuple
import numpy as np

def _validate_1d_pair(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    yt_orig = np.asarray(y_true)
    yp_orig = np.asarray(y_pred)
    if yt_orig.ndim != 1 or yp_orig.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    yt = yt_orig.flatten()
    yp = yp_orig.flatten()
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length.")
    return yt, yp

class Metric:
    """
    Added Metric to streamline the decision tree process
    """
    def update(self, y_true, y_pred):
        raise NotImplementedError
    def result(self): 
        raise NotImplementedError
    def reset(self): 
        raise NotImplementedError

class Accuracy(Metric):
    def __init__(self):
        self.reset()
    def reset(self):
        self.correct = 0
        self.total = 0
    def update(self, y_true, y_pred):
        yt, yp = _validate_1d_pair(y_true, y_pred)
        self.correct += np.sum(yt == yp)
        self.total += yt.size
    def result(self):
        return self.correct / self.total if self.total > 0 else 0.0

class ConfusionMatrix(Metric):
    def __init__(self, labels=None):
        self.labels = labels
        self.matrix = None
        self.label_map = None
        if self.labels is not None:
            self.reset()

    def reset(self):
        if self.labels is not None:
            n = len(self.labels)
            self.matrix = np.zeros((n, n), dtype=int)
            self.label_map = {l: i for i, l in enumerate(self.labels)}
        else:
            self.matrix = None
            self.label_map = None

    def update(self, y_true, y_pred):
        yt, yp = _validate_1d_pair(y_true, y_pred)
        if self.labels is None:
            self.labels = np.unique(np.concatenate([yt, yp]))
            self.reset()
        
        for t, p in zip(yt, yp):
            if t in self.label_map and p in self.label_map:
                self.matrix[self.label_map[t], self.label_map[p]] += 1

    def result(self):
        return self.matrix

class RollingMetric(Metric):
    def __init__(self, metric_cls, window_size=10):
        self.metric_cls = metric_cls
        self.window_size = window_size
        self.history = []
    def update(self, y_true, y_pred):
        m = self.metric_cls()
        m.update(y_true, y_pred)
        self.history.append(m.result())
        if len(self.history) > self.window_size:
            self.history.pop(0)
    def result(self):
        return np.mean(self.history) if self.history else 0.0
    def reset(self):
        self.history = []

def accuracy(y_true, y_pred):
    yt, yp = _validate_1d_pair(y_true, y_pred)
    return float(np.mean(yt == yp))

def precision(y_true, y_pred, pos_label=1, zero_division=0):
    yt, yp = _validate_1d_pair(y_true, y_pred)
    tp = np.sum((yt == pos_label) & (yp == pos_label))
    fp = np.sum((yt != pos_label) & (yp == pos_label))
    denom = tp + fp
    if denom == 0:
        return float(zero_division)
    return tp / denom

def recall(y_true, y_pred, pos_label=1, zero_division=0):
    yt, yp = _validate_1d_pair(y_true, y_pred)
    tp = np.sum((yt == pos_label) & (yp == pos_label))
    fn = np.sum((yt == pos_label) & (yp != pos_label))
    denom = tp + fn
    if denom == 0:
        return float(zero_division)
    return tp / denom

def f1(y_true, y_pred, pos_label=1, zero_division=0):
    p = precision(y_true, y_pred, pos_label, zero_division)
    r = recall(y_true, y_pred, pos_label, zero_division)
    denom = p + r
    if denom == 0:
        return float(zero_division)
    return 2 * p * r / denom

def mse(y_true, y_pred):
    yt, yp = _validate_1d_pair(y_true, y_pred)
    return np.mean((yt - yp)**2)

def confusion_matrix(y_true, y_pred, labels=None):
    cm = ConfusionMatrix(labels=labels)
    cm.update(y_true, y_pred)
    return cm.result()

def roc_curve(y_true, y_score, pos_label=1):
    yt, ys = _validate_1d_pair(y_true, y_score)
    unique_y = np.unique(yt)
    if len(unique_y) < 2:
        raise ValueError("ROC curve requires both positive and negative classes.")
    
    order = np.argsort(-ys)
    ys = ys[order]
    yt = (yt[order] == pos_label).astype(int)
    
    tps = np.cumsum(yt)
    fps = np.cumsum(1 - yt)
    
    # Add a leading zero for the point (0,0)
    tpr = np.concatenate([[0], tps / tps[-1]])
    fpr = np.concatenate([[0], fps / fps[-1]])
    thresholds = np.concatenate([[np.inf], ys])
    
    return fpr, tpr, thresholds

def auc(fpr, tpr):
    if len(fpr) < 2:
        raise ValueError("AUC requires at least two points.")
    return np.trapezoid(tpr, fpr)
