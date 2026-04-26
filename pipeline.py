"""
Pipeline file to run the entire project end-to-end. 
"""

from config import GlobalConfig, SASRecModelConfig, GPT4RecModelConfig
from data_loading import *
from eda import *
from utils import *

import time
import logging
import sys


# 1. Configuration & Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def instantiateDirs():
    """Step 1: Create necessary directories for data, processed data, and outputs."""
    logging.info("Setting up directories...")
    createDirectories(GlobalConfig)
    return

def extract():
    """Step 2: Extract data from HuggingFace"""
    logging.info("Extracting data...")

    loadAmazonReviews(GlobalConfig)
    loadAmazonReviews(GlobalConfig, cold_start=True)

    # itemmetadata = loadItemMetadata()
    # reviewsdata = loadAmazonReviews()
    return 

# def transform(data):
#     """Step 2: Clean, filter, or process the raw data"""
#     logging.info("Transforming data...")
#     # Example transformation: Uppercase the values
#     data["value"] = [v.lower() for v in data["value"]]
#     return data

# def load(data):
#     """Step 3: Save processed data to destination (SQL, S3, local file)"""
#     logging.info("Loading data to destination...")
#     # Your loading logic here
#     print(f"Final Data: {data}")

def main():
    try:
        instantiateDirs()
        extract()
        # processed_data = transform(raw_data)
        # load(processed_data)
        logging.info("Pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


