import numpy as np
from .stats import StreamingStats

def _validate_array(X: np.ndarray, name: str = "X", check_nan: bool = False) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
      raise ValueError(f"{name} must be 2D.")
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if check_nan and np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values. Use SimpleImputer first.")
    return arr

class StandardScaler:
    def __init__(self) -> None:
      self.stats_ = None
      self.n_features_in_ = None
        
    def fit(self, X: np.ndarray) -> "StandardScaler":
      arr = _validate_array(X, name="X", check_nan=True)
      self.n_features_in_ = arr.shape[1]
      self.stats_ = StreamingStats(arr.shape[1])
      self.stats_.update(arr)
      return self

    def partial_fit(self, X: np.ndarray) -> "StandardScaler":
      arr = _validate_array(X, name="X", check_nan=True)
      if self.stats_ is None:
          self.n_features_in_ = arr.shape[1]
          self.stats_ = StreamingStats(arr.shape[1])
      
      if arr.shape[1] != self.n_features_in_:
          raise ValueError(f"X has {arr.shape[1]} features, but StandardScaler is fitted with {self.n_features_in_} features.")
      
      self.stats_.update(arr)
      return self

    def transform(self, X: np.ndarray) -> np.ndarray:
      if self.stats_ is None:
            raise ValueError("StandardScaler is not fitted yet.")
      arr = _validate_array(X, name="X", check_nan=True)
      
      if arr.shape[1] != self.n_features_in_:
          raise ValueError(f"X has {arr.shape[1]} features, but StandardScaler is fitted with {self.n_features_in_} features.")
          
      mean = self.stats_.mean
      std = self.stats_.std
      std = np.where(std == 0.0, 1.0, std)
      return (arr - mean) / std
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
      return self.fit(X).transform(X)

class MinMaxScaler:
    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of feature_range must be greater than min.")
        self.feature_range = feature_range
        self.stats_ = None
        self.n_features_in_ = None
  
    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        arr = _validate_array(X, name="X", check_nan=True)
        self.n_features_in_ = arr.shape[1]
        self.stats_ = StreamingStats(arr.shape[1])
        self.stats_.update(arr)
        return self

    def partial_fit(self, X: np.ndarray) -> "MinMaxScaler":
        arr = _validate_array(X, name="X", check_nan=True)
        if self.stats_ is None:
            self.n_features_in_ = arr.shape[1]
            self.stats_ = StreamingStats(arr.shape[1])
            
        if arr.shape[1] != self.n_features_in_:
             raise ValueError(f"X has {arr.shape[1]} features, but MinMaxScaler is fitted with {self.n_features_in_} features.")
             
        self.stats_.update(arr)
        return self
  
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.stats_ is None:
            raise ValueError("MinMaxScaler is not fitted yet.")
        arr = _validate_array(X, name="X", check_nan=True)
        
        if arr.shape[1] != self.n_features_in_:
             raise ValueError(f"X has {arr.shape[1]} features, but MinMaxScaler is fitted with {self.n_features_in_} features.")
             
        min_val = self.stats_.min
        max_val = self.stats_.max
        range_val = max_val - min_val
        range_val = np.where(range_val == 0.0, 1.0, range_val)
        a, b = self.feature_range
        return (arr - min_val) / range_val * (b - a) + a

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

class OneHotEncoder:
    def __init__(self) -> None:
        self.categories_ = None

    def fit(self, X: np.ndarray) -> "OneHotEncoder":
        arr = np.asarray(X).flatten()
        self.categories_ = np.unique(arr)
        return self

    def partial_fit(self, X: np.ndarray) -> "OneHotEncoder":
        arr = np.asarray(X).flatten()
        new_cats = np.unique(arr)
        if self.categories_ is None:
            self.categories_ = new_cats
        else:
            self.categories_ = np.unique(np.concatenate([self.categories_, new_cats]))
        return self
  
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.categories_ is None:
            raise ValueError("OneHotEncoder is not fitted yet.")
        arr = np.asarray(X).flatten()
        return (arr[:, None] == self.categories_).astype(int)
  
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

class SimpleImputer:
    def __init__(self, strategy: str = "mean", fill_value: float = 0):
        if strategy not in ["mean", "constant"]:
            raise ValueError(f"Invalid strategy: {strategy}. Only 'mean' and 'constant' are supported.")
        self.strategy = strategy
        self.fill_value = fill_value
        self.stats_ = None

    def fit(self, X: np.ndarray) -> "SimpleImputer":
        arr = _validate_array(X, name="X", check_nan=False)
        self.stats_ = StreamingStats(arr.shape[1])
        self.stats_.update(arr)
        return self

    def partial_fit(self, X: np.ndarray) -> "SimpleImputer":
        arr = _validate_array(X, name="X", check_nan=False)
        if self.stats_ is None:
            self.stats_ = StreamingStats(arr.shape[1])
        self.stats_.update(arr)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.stats_ is None and self.strategy != "constant":
            raise ValueError("SimpleImputer is not fitted yet.")
        arr = _validate_array(X, name="X", check_nan=False)
        if self.strategy == "mean":
            fill = self.stats_.mean
        elif self.strategy == "constant":
            fill = self.fill_value
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        
        mask = np.isnan(arr)
        X_out = arr.copy()
        if np.isscalar(fill):
            X_out[mask] = fill
        else:
            for i in range(arr.shape[1]):
                X_out[mask[:, i], i] = fill[i]
        return X_out
  
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

__all__ = ["StandardScaler", "MinMaxScaler", "OneHotEncoder", "SimpleImputer"]
