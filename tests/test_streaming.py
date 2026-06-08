import pytest
import numpy as np
from numcompute.stats import StreamingStats, update_stats
from numcompute.preprocessing import StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
from numcompute.metrics import Accuracy, RollingMetric
from numcompute.tree import DecisionTreeClassifier
from numcompute.ensemble import RandomForestClassifier
from numcompute.pipeline import Pipeline
from numcompute.stream import StreamTrainer

def test_streaming_stats_welford():
    stats = StreamingStats(n_features=1)
    data = [1, 2, 3, 4, 5]
    for x in data:
        stats.update(np.array([[x]]))
    
    assert stats.count == 5
    assert np.isclose(stats.mean[0], 3.0)
    assert np.isclose(stats.variance[0], 2.0)

def test_standard_scaler_partial_fit():
    scaler = StandardScaler()
    scaler.partial_fit(np.array([[1, 10]]))
    scaler.partial_fit(np.array([[3, 20]]))
    
    assert np.allclose(scaler.stats_.mean, [2, 15])
    
    X_test = np.array([[2, 15]])
    X_scaled = scaler.transform(X_test)
    assert np.allclose(X_scaled, [[0, 0]])

def test_imputer_streaming_mean():
    imputer = SimpleImputer(strategy="mean")
    imputer.partial_fit(np.array([[1], [3]])) # mean = 2
    
    X_miss = np.array([[np.nan]])
    X_filled = imputer.transform(X_miss)
    assert X_filled[0, 0] == 2.0

def test_one_hot_encoder_incremental():
    ohe = OneHotEncoder()
    ohe.partial_fit(np.array(["A", "B"]))
    assert len(ohe.categories_) == 2
    
    ohe.partial_fit(np.array(["C"]))
    assert len(ohe.categories_) == 3
    
    X_trans = ohe.transform(np.array(["A", "C"]))
    assert X_trans.shape == (2, 3)
    assert np.all(X_trans[0] == [1, 0, 0])
    assert np.all(X_trans[1] == [0, 0, 1])

def test_accuracy_metric_streaming():
    acc = Accuracy()
    acc.update([0, 1], [0, 0]) # 1/2
    acc.update([1, 1], [1, 1]) # 2/2
    # Total: 3/4 = 0.75
    assert acc.result() == 0.75

def test_decision_tree_fit_predict():
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y = np.array([0, 1, 0, 1])
    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X, y)
    preds = tree.predict(X)
    assert np.all(preds == y)

def test_random_forest_streaming():
    X = np.random.rand(20, 4)
    y = (X[:, 0] > 0.5).astype(int)
    
    rf = RandomForestClassifier(n_estimators=5)
    # perform training in chunks
    rf.partial_fit(X[:10], y[:10])
    rf.partial_fit(X[10:], y[10:])
    
    preds = rf.predict(X)
    assert preds.shape == (20,)

def test_pipeline_streaming():
    pipe = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler()),
        ('model', DecisionTreeClassifier())
    ])
    
    X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
    y = np.array([0, 0, 1])
    
    pipe.partial_fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == 3

def test_stream_trainer():
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=3))
    ])
    trainer = StreamTrainer(pipe)
    
    X = np.random.rand(30, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    
    trainer.fit_chunk(X[:10], y[:10])
    trainer.fit_chunk(X[10:20], y[10:20])
    trainer.fit_chunk(X[20:], y[20:])
    
    history = trainer.get_history()
    assert len(history['accuracy']) == 3
    assert history['chunk_idx'] == [0, 1, 2]

def test_numerical_stability_zeros():
    scaler = StandardScaler()
    scaler.fit(np.array([[1, 1], [1, 1]]))
    X_trans = scaler.transform(np.array([[1, 1]]))
    assert np.all(X_trans == 0)

def test_numerical_stability_nans():
    stats = StreamingStats(n_features=1)
    stats.update(np.array([[np.nan], [np.nan]]))
    assert stats.count == 0
    assert stats.mean[0] == 0
