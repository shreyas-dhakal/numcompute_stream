import numpy as np
from .tree import DecisionTreeClassifier

class EnsembleClassifier:
    """
    Implementation of random f orest with streaming updates
    """
    def __init__(self, n_estimators=10, max_depth=5, max_features='sqrt', **tree_params):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.tree_params = tree_params
        self.estimators = []
        self.classes_ = None

    def _get_max_features(self, n_features):
        """
        get max features based on the provided function.
        supports:
        sqrt
        log2
        """
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        return n_features

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        max_f = self._get_max_features(n_features)
        
        self.estimators = []
        for _ in range(self.n_estimators):
            indices = np.random.choice(len(X), len(X), replace=True)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features=max_f, **self.tree_params)
            tree.fit(X[indices], y[indices])
            self.estimators.append(tree)
        return self

    def partial_fit(self, X, y, classes=None):
        if self.classes_ is None:
            if classes is not None:
                self.classes_ = classes
            else:
                self.classes_ = np.unique(y)
        
        n_features = X.shape[1]
        max_f = self._get_max_features(n_features)

        if not self.estimators:
            for _ in range(self.n_estimators):
                tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features=max_f, **self.tree_params)
                self.estimators.append(tree)

        for tree in self.estimators:
            indices = np.random.choice(len(X), len(X), replace=True)
            tree.partial_fit(X[indices], y[indices], classes=self.classes_)
        
        return self

    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.estimators])
        y_pred = []
        for i in range(X.shape[0]):
            vals, counts = np.unique(all_preds[:, i], return_counts=True)
            y_pred.append(vals[np.argmax(counts)])
        return np.array(y_pred)

    def predict_probabilities(self, X):
        all_probas = np.array([tree.predict_probabilities(X) for tree in self.estimators])
        return np.mean(all_probas, axis=0)

class RandomForestClassifier(EnsembleClassifier):
    pass
