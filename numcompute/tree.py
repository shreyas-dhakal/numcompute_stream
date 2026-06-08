import numpy as np

class DecisionTreeClassifier:
    """
    Decision tree classifier with GINI index, depth-limited.
    Supports max_depth, min_samples_split and max_features 
    """
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.tree = None
        self.classes_ = None

    def _entropy(self, y):
        """
        Entropy calculation
        """
        if len(y) == 0: 
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-10))

    def _gini_index(self, y):
        """
        GINI index
        """
        if len(y) == 0: 
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs**2)

    def _best_split(self, X, y):
        """
        Find the best split
        """
        X = np.nan_to_num(X, nan=0.0)
        m, n = X.shape
        if m <= self.min_samples_split:
            return None, None

        best_gain = -1
        split_idx, split_val = None, None
        
        feat_indices = np.arange(n)
        if self.max_features:
            if isinstance(self.max_features, str):
                if self.max_features == 'sqrt':
                    k = int(np.sqrt(n))
                elif self.max_features == 'log2':
                    k = int(np.log2(n))
                else:
                    k = n
            elif isinstance(self.max_features, float):
                k = int(self.max_features * n)
            else:
                k = self.max_features
            k = max(1, min(k, n))
            feat_indices = np.random.choice(feat_indices, k, replace=False)

        current_score = self._gini_index(y) if self.criterion == 'gini' else self._entropy(y)

        for i in feat_indices:
            thresholds = np.unique(X[:, i])
            # incase of too many thresholds
            if len(thresholds) > 20:
                thresholds = np.percentile(thresholds, np.linspace(0, 100, 20))
                
            for t in thresholds:
                left_mask = X[:, i] <= t
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                left_score = self._gini_index(y[left_mask]) if self.criterion == 'gini' else self._entropy(y[left_mask])
                right_score = self._gini_index(y[right_mask]) if self.criterion == 'gini' else self._entropy(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                gain = current_score - (n_left/m * left_score + n_right/m * right_score)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = i
                    split_val = t
        
        return split_idx, split_val

    def _most_common_label(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def _build_tree(self, X, y, depth=0):
        """
        Function to build the tree
        """
        num_samples = len(y)
        unique_labels = np.unique(y)
        num_classes = len(unique_labels)

        if depth >= self.max_depth or num_classes <= 1 or num_samples < self.min_samples_split:
            return {'leaf': True, 'class': self._most_common_label(y)}

        idx, val = self._best_split(X, y)
        if idx is None:
            return {'leaf': True, 'class': self._most_common_label(y)}

        left_mask = X[:, idx] <= val
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return {'leaf': False, 'index': idx, 'value': val, 'left': left_tree, 'right': right_tree}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.tree = self._build_tree(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        if self.classes_ is None:
            if classes is not None:
                self.classes_ = classes
            else:
                self.classes_ = np.unique(y)
        
        if not hasattr(self, "_X_buffer"):
            self._X_buffer = X
            self._y_buffer = y
        else:
            self._X_buffer = np.vstack([self._X_buffer, X])
            self._y_buffer = np.concatenate([self._y_buffer, y])

        self.fit(self._X_buffer, self._y_buffer)
        return self

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['class']
        if x[node['index']] <= node['value']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        if self.tree is None:
            raise ValueError("DecisionTreeClassifier is not fitted yet.")
        return np.array([self._predict_one(x, self.tree) for x in X])

    def predict_proba(self, X):
        if self.tree is None:
            raise ValueError("DecisionTreeClassifier is not fitted yet.")
        preds = self.predict(X)
        proba = np.zeros((len(X), len(self.classes_)))
        for i, p in enumerate(preds):
            idx = np.where(self.classes_ == p)[0][0]
            proba[i, idx] = 1.0
        return proba
