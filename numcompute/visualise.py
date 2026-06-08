import matplotlib.pyplot as plt
import numpy as np

def _handle_save_and_show(save_path=None):
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

def plot_metric_over_time(metric_values, title="Metric Over Time", ylabel="Value", save_path=None):
    """
    Plot a metric (e.g., accuracy) for different chunks.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(metric_values, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel("Chunk Index")
    plt.ylabel(ylabel)
    plt.grid(True)
    _handle_save_and_show(save_path)

def compare_models(metrics_dict, title="Model Comparison", ylabel="Accuracy", save_path=None):
    """
    Compare multiple models on streaming metrics
    """
    plt.figure(figsize=(10, 5))
    for label, values in metrics_dict.items():
        plt.plot(values, label=label, marker='x')
    plt.title(title)
    plt.xlabel("Chunk Index")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    _handle_save_and_show(save_path)

def plot_predictions_vs_ground_truth(y_true, y_pred, title="Predictions vs Ground Truth", save_path=None):
    """
    Visualise predictions vs actuals on latest chunk
    """
    plt.figure(figsize=(10, 5))
    indices = np.arange(len(y_true))
    plt.scatter(indices, y_true, label="Actual", alpha=0.6, marker='o')
    plt.scatter(indices, y_pred, label="Predicted", alpha=0.6, marker='x')
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Class Label")
    plt.legend()
    plt.grid(True)
    _handle_save_and_show(save_path)
