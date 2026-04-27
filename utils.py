"""
This file define functions that are used across the repository.
"""

from config import GlobalConfig, SASRecModelConfig, GPT4RecModelConfig


def createDirectories(config):
    """Create directories for storing data, processed data, and outputs."""
    config.data_dir.mkdir(exist_ok=True)
    config.processed_dir.mkdir(exist_ok=True)
    config.outputs_dir.mkdir(exist_ok=True)

    # create subdirectories for train/val/test splits
    (config.data_dir / "raw_dataset").mkdir(exist_ok=True)
    (config.data_dir / "train").mkdir(exist_ok=True)
    (config.data_dir / "val").mkdir(exist_ok=True)
    (config.data_dir / "test").mkdir(exist_ok=True)
    
    return