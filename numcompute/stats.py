"""
stats.py
"""

import numpy as np


def mean(x, axis=None, ignore_nan=True):
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("Empty array")
    if ignore_nan:
        return np.nanmean(x, axis=axis)
    else:
        return np.mean(x, axis=axis)


def variance(x, axis=None, ddof=0, ignore_nan=True):
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("Empty array")
    if ignore_nan:
        return np.nanvar(x, axis=axis, ddof=ddof)
    else:
        return np.var(x, axis=axis, ddof=ddof)


def welford(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("Empty input")
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return (float(x[0]), 0.0) if n == 1 else (0.0, 0.0)
    mean = float(np.mean(x))
    variance = float(np.var(x, ddof=1))
    return mean, variance


def histogram(x, bins=10, range=None):
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("Empty array")
    hist, bin_edges = np.histogram(x, bins=bins, range=range)
    return hist, bin_edges


def quantile(x, q, axis=None):
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("Empty array")
    if np.any((np.asarray(q) < 0) | (np.asarray(q) > 1)):
        raise ValueError("q must be in [0, 1]")
    return np.quantile(x, q, axis=axis)


class StreamingStats:
    """
    Maintain running statistics for streaming data.
    """
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.count = np.zeros(n_features, dtype=int)
        self.mean = np.zeros(n_features)
        self.M2 = np.zeros(n_features)
        self.min = np.full(n_features, np.inf)
        self.max = np.full(n_features, -np.inf)

    def update(self, X_chunk: np.ndarray):
        X_chunk = np.asarray(X_chunk, dtype=float)
        
        for i in range(self.n_features):
            col = X_chunk[:, i]
            col = col[~np.isnan(col)]
            if col.size == 0:
                continue
            
            chunk_count = col.size
            chunk_mean = np.mean(col)
            chunk_m2 = np.sum((col - chunk_mean) ** 2)
            
            new_count = self.count[i] + chunk_count
            delta = chunk_mean - self.mean[i]
            
            self.mean[i] += delta * chunk_count / new_count
            self.M2[i] += chunk_m2 + delta**2 * self.count[i] * chunk_count / new_count
            self.count[i] = new_count
            
            self.min[i] = min(self.min[i], np.min(col))
            self.max[i] = max(self.max[i], np.max(col))

    @property
    def variance(self):
        var = np.zeros(self.n_features)
        mask = self.count > 0
        var[mask] = self.M2[mask] / self.count[mask]
        return var

    @property
    def std(self):
        return np.sqrt(self.variance)

def update_stats(X_chunk, state=None):
    if state is None:
        state = StreamingStats(X_chunk.shape[1])
    state.update(X_chunk)
    return state
