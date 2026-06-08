import numpy as np
import time
from .metrics import Accuracy

class StreamTrainer:
    """
    This class manages model, pipeline and logging
    """
    def __init__(self, pipeline, metric_aggregator=None):
        self.pipeline = pipeline
        self.metric_aggregator = metric_aggregator if metric_aggregator else Accuracy()
        self.history = {'chunk_idx': [],'accuracy': [],'time': [],'memory': []
        }
        self.cumulative_accuracy = Accuracy()

    def fit_chunk(self, X_chunk, y_chunk):
        """
        Fit chunk logic with per chunk accuracy, memory footprint and cumulative accuracy
        """
        start_time = time.time()

        try:
            y_pred = self.pipeline.predict(X_chunk)
            chunk_acc = Accuracy()
            chunk_acc.update(y_chunk, y_pred)
            acc_val = chunk_acc.result()
            self.cumulative_accuracy.update(y_chunk, y_pred)
        except Exception:
            acc_val = 0.0

        # update the model
        self.pipeline.partial_fit(X_chunk, y_chunk)
        
        end_time = time.time()
        
        # logging all the information
        idx = len(self.history['chunk_idx'])
        self.history['chunk_idx'].append(idx)
        self.history['accuracy'].append(acc_val)
        self.history['time'].append(end_time - start_time)
        self.history['memory'].append(X_chunk.size * 8 / 1024) 

        return acc_val

    def score_chunk(self, X_chunk, y_chunk):
        """
        Fit chunk logic with per chunk accuracy, memory footprint and cumulative accuracy
        """
        y_pred = self.pipeline.predict(X_chunk)
        acc = Accuracy()
        acc.update(y_chunk, y_pred)
        return acc.result()

    def get_history(self):
        return self.history
