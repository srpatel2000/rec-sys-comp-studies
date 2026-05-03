"""
This file define functions that are used across the repository.
"""

import matplotlib

from config import GlobalConfig, SASRecModelConfig, GPT4RecModelConfig


def createDirectories(config):
    """Create directories for storing data, processed data, and outputs."""
    config.data_dir.mkdir(exist_ok=True)
    config.eda_dir.mkdir(exist_ok=True)
    config.outputs_dir.mkdir(exist_ok=True)

    # create subdirectories for train/val/test splits + raw dataset
    (config.data_dir / "raw_dataset").mkdir(exist_ok=True)
    (config.data_dir / "train").mkdir(exist_ok=True)
    (config.data_dir / "val").mkdir(exist_ok=True)
    (config.data_dir / "test").mkdir(exist_ok=True)

    return


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