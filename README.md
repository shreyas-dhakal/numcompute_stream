# NumCompute Stream

**End-to-end streaming machine learning framework, built from scratch with Python and NumPy.**

NumCompute Stream extends the NumCompute library, to support incremental, decision tree and model ensembling. It provides a robust framework for handling data streams, training models on chunks, and visualizing performance along with model comparison.

## Streaming & Ensembles

*   **Streaming**: All core components (scalers, imputers, models, metrics) support `.partial_fit()` or `.update()` for incremental learning.
*   **Decision Tree Classifier**: A depth-limited, Gini based decision tree implementation from scratch.
*   **Random Forest Ensemble**: An ensemble classifier that utilizes multiple decision trees using bagging, supporting both batch and streaming training.
*   **Streaming Metrics**: Metrics like `Accuracy` and `ConfusionMatrix` now accumulate results over time across data chunks.
*   **Visualisation Module**: Built-in `visualise.py` for plotting of metrics, model comparisons, and predictions.
*   **StreamTrainer**: A high-level manager for handling the training loop, logging performance, and monitoring memory footprint.

## Installation

To install NumCompute Stream, clone the repository and install it using pip:
```bash
git clone https://github.com/shreyas-dhakal/numcompute_stream
cd numcompute_stream
```
Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate
```
Install the required packages
```bash
pip3 install -e .
```


## API Overview

*   `numcompute.stream`: `StreamTrainer` for managing online learning workflows.
*   `numcompute.tree`: `DecisionTreeClassifier` with streaming support.
*   `numcompute.ensemble`: `RandomForestClassifier` (Ensemble methods).
*   `numcompute.visualise`: Reusable plotting functions for ML monitoring.
*   `numcompute.metrics`: Streaming-compatible classification and regression metrics.
*   `numcompute.preprocessing`: Incremental scalers and imputers.

## Demo

A demo is provided in `demo/stream_demo.ipyb`. It demonstrates:
1. Loading the Iris dataset via the custom I/O pipeline.
2. Splitting the dataset into chunks to simulate a stream.
3. Training a Random Forest pipeline incrementally.
4. Comparing performance against a single Decision Tree.
5. Visualizing the results and saving plots to disk.

 