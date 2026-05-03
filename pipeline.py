"""
Pipeline file to run the entire project end-to-end. 
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from config import GlobalConfig, SASRecModelConfig, GPT4RecModelConfig
from data_loading import *
from train_val_test_split import *
from eda import *
from utils import *
from custom_sasrec_funcs import runSASRecPipeline
from eval_metrics import evalPipeline

import time
import logging
import sys
from pathlib import Path
import pandas as pd

config = GlobalConfig()

# 1. Configuration & Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def instantiateDirs():
    """Step 1: Create necessary directories for data, processed data, and outputs."""
    logging.info("Setting up directories...")
    createDirectories(GlobalConfig)
    return

def extract():
    """Step 2: Extract data from HuggingFace and split into train/val/test sets."""

    project_root = Path(__file__).resolve().parent

    # loads reviews sample, item metadata, and joins them into one sequential dataset
    dense_file_path = project_root/"data"/"raw_dataset"/f"dense_sample_{config.samples}_users.csv"
    cs_file_path = project_root/"data"/"raw_dataset"/f"cold_start_sample_{config.samples}_users.csv"

    logging.info("Extracting samples for dense data...")
    if dense_file_path.is_file():
        dense_sample_df = pd.read_csv(dense_file_path)
    else:
        dense_sample_df = loadDataPipeline(config, cold_start=False, target_users=config.samples)
    
    logging.info("Extracting samples for cold start data...")
    if cs_file_path.is_file():
        cs_sample_df = pd.read_csv(cs_file_path)
    else:
        cs_sample_df = loadDataPipeline(config, cold_start=True, target_users=config.samples)

    return dense_sample_df, cs_sample_df

def runRawEDA(data, cold_start=False):
    """Step 3: Run EDA on the raw data."""

    logging.info("Running EDA on raw data...")

    plotInteractionDistribution(data, user_col="user_id", item_col="parent_asin", cold_start=cold_start)

    return data

def trainValTestSplit():
    """Step 4: Split the data into train, val, and test sets.
     
    For dense dataset: Reads in data from dense dataset and removes last interaction as test, 
    second to last as val, and rest as train.
    
    For cold start dataset: Reads in data from cold start dataset and trains on all users with 
    >=4 interactions and N-1th interactions for users with 2 reviews and N-2th interactions for 
    users with 3 reviews. Validate on users with 3 interactions on their N-1th interaction. Tests 
    on users with 2-3 interactions using last interaction."""

    logging.info("Splitting data into train/val/test sets...")
    
    if (config.data_dir / "train" / "dense_train.csv").exists() and \
            (config.data_dir / "val" / "dense_val.csv").exists() and \
                (config.data_dir / "test" / "dense_test.csv").exists() and \
                    (config.data_dir / "train" / "cold_start_train.csv").exists() and \
                        (config.data_dir / "val" / "cold_start_val.csv").exists() and \
                            (config.data_dir / "test" / "cold_start_test.csv").exists():
            logging.info("Train/val/test splits already exist in data directory.")
            return
    else:
        _, _, _, _, _, _ = createTrainValTestSplits()

def runEval(model_name, data_type, holdout_method):
    """Step 5: After models are run, run evaluation metrics and generate charts."""

    logging.info(f"Running evaluation on {data_type} {holdout_method} dataset using {model_name} model...")

    preds_file_path = config.outputs_dir / f"{data_type}_{model_name.lower()}_{holdout_method}_predictions.csv"
    try:
        predictions_df = pd.read_csv(preds_file_path)
    except FileNotFoundError:
        logging.error(f"Predictions file not found: {preds_file_path}. Make sure model is run and predictions are saved before running evaluation.")
        return

    predictions = predictions_df["top_k_asins"].tolist()
    actuals = predictions_df["ground_truth_asins"].tolist() # TODO: don't have in the current predictions_df
    step_size = 10
    rel_scores = predictions_df["top_k_rel_scores"].tolist() # TODO: don't have in the current predictions_df
    ideal_rel_scores = predictions_df["top_k_ideal_rel_scores"].tolist() # TODO: don't have in the current predictions_df

    evalPipeline(predictions, actuals, step_size, rel_scores, ideal_rel_scores)


def main():
    try:
        instantiateDirs()

        dense_sample_df, cs_sample_df = extract()

        runRawEDA(cs_sample_df, cold_start=True)
        runRawEDA(dense_sample_df)

        trainValTestSplit()

        ########## SASRec ##########
        runSASRecPipeline("dense")
        runSASRecPipeline("cold_start")
        runEval("SASRec", "dense", "val")
        runEval("SASRec", "cold_start", "val")
        runEval("SASRec", "dense", "test")
        runEval("SASRec", "cold_start", "test")

        ########## GPT4Rec ##########
        # runGPT4RecPipeline("dense")
        # runGPT4RecPipeline("cold_start")
        # runEval("GPT4Rec", "dense", "val")
        # runEval("GPT4Rec", "cold_start", "val")
        # runEval("GPT4Rec", "dense", "test")
        # runEval("GPT4Rec", "cold_start", "test")

        logging.info("Pipeline completed successfully.")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


