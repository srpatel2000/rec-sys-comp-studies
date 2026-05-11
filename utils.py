"""
This file define functions that are used across the repository.
"""

import matplotlib.pyplot as plt
from config import GlobalConfig, SASRecModelConfig, GPT4RecModelConfig


def createDirectories(config):
    """Create directories for storing data, processed data, and outputs."""
    config.data_dir.mkdir(exist_ok=True)
    config.eda_dir.mkdir(exist_ok=True)
    config.trained_models_dir.mkdir(exist_ok=True)

    # create subdirectories for train/val/test splits + raw dataset + outputs
    (config.data_dir / "raw_dataset").mkdir(exist_ok=True)
    (config.data_dir / "train").mkdir(exist_ok=True)
    (config.data_dir / "val").mkdir(exist_ok=True)
    (config.data_dir / "test").mkdir(exist_ok=True)
    (config.data_dir / "outputs").mkdir(exist_ok=True)
    (config.data_dir / "outputs" / "baseline").mkdir(parents=True, exist_ok=True)

    # create subdirectories for trained models artifact and eval charts
    (config.trained_models_dir / "eval_metrics").mkdir(exist_ok=True)
    (config.trained_models_dir / "models").mkdir(exist_ok=True)



def createLineChart(x, y, title, x_label, y_label, output_path):
    """Helper function to create and save a line chart."""
    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(output_path)
    plt.close()


def createLineChartMulti(series, title, x_label, y_label, output_path):
    """Save a line chart with multiple (x, y) series and a legend.

    series: list of dicts with keys 'x', 'y', 'label' (each x,y same-length lists).
    """
    plt.figure(figsize=(8, 5))
    for s in series:
        plt.plot(s["x"], s["y"], marker="o", markersize=3, label=s["label"])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()