from config import GlobalConfig
from pathlib import Path
import logging
import pandas as pd


def splitDenseData(dense_df):
    """Split dense data: Nth interaction as test, N-1th interaction as val, every other interaction as train."""
    
    train_rows = []
    val_rows = []
    test_rows = []
    
    for user_id, group in dense_df.groupby("user_id"):
        # sort by timestamp to preserve interaction order
        group = group.sort_values("timestamp").reset_index(drop=True)
        n = len(group)

        if n < 5:
            # accounting for any users with less than 5 interactions (should be rare since this is the dense dataset)
            logging.warning(f"User {user_id} has less than 5 interactions ({n}). Removing from dense dataset.")
            continue
        
        train_rows.append(group.iloc[:-2])
        val_rows.append(group.iloc[-2:-1])
        test_rows.append(group.iloc[-1:])
    
    train_df = pd.concat(train_rows, ignore_index=True)
    val_df = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame(columns=dense_df.columns)
    test_df = pd.concat(test_rows, ignore_index=True)
    
    return train_df, val_df, test_df


def splitColdStartData(cs_df):
    """split cold start data: train all users with >=4 interactions, N-1th for 2-review users, N-2th for 3-review users.
    validate on users with 3 interactions on their N-1th. test on users with 2-3 interactions using last interaction."""
    
    train_rows = []
    val_rows = []
    test_rows = []
    
    for user_id, group in cs_df.groupby("user_id"):
        # sort by timestamp to preserve interaction order
        group = group.sort_values("timestamp").reset_index(drop=True)
        n = len(group)

        if n < 1:
            # accounting for any users with less than 1 interaction (should be rare)
            logging.warning(f"User {user_id} has less than 1 interaction ({n}). Removing from cold start dataset.")
            continue
        
        if n >= 4:
            train_rows.append(group)
        elif n == 3:
            train_rows.append(group.iloc[0:1])
            val_rows.append(group.iloc[1:2])
            test_rows.append(group.iloc[2:3])
        elif n == 2:
            # doesn't have enough interactions for val
            train_rows.append(group.iloc[0:1])
            test_rows.append(group.iloc[1:2])

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame(columns=cs_df.columns)
    test_df = pd.concat(test_rows, ignore_index=True)
    
    return train_df, val_df, test_df


def saveDataSplits(dense_train, dense_val, dense_test, cs_train, cs_val, cs_test):
    
    config = GlobalConfig()
    project_root = Path(__file__).resolve().parent
    
    splits = {
        "train": {"cold_start": cs_train, "dense": dense_train},
        "val": {"cold_start": cs_val, "dense": dense_val},
        "test": {"cold_start": cs_test, "dense": dense_test},
    }
    
    for split_type, datasets in splits.items():
        split_dir = project_root / "data" / split_type
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset_name, df in datasets.items():
            output_path = split_dir / f"{dataset_name}_{split_type}.csv"
            df.to_csv(output_path, index=False)
            logging.info(f"Saved {dataset_name} {split_type} split: {len(df)} rows to {output_path}")


def createTrainValTestSplits():
    """Orchestrate train/val/test splitting for both dense and cold start datasets."""
    
    config = GlobalConfig()
    project_root = Path(__file__).resolve().parent
    
    # load raw datasets from csv
    dense_path = project_root / "data" / "raw_dataset" / f"dense_sample_{config.samples}_users.csv"
    cs_path = project_root / "data" / "raw_dataset" / f"cold_start_sample_{config.samples}_users.csv"
    
    dense_df = pd.read_csv(dense_path)
    cs_df = pd.read_csv(cs_path)
    
    logging.info("Splitting dense dataset...")
    dense_train, dense_val, dense_test = splitDenseData(dense_df)
    
    logging.info("Splitting cold start dataset...")
    cs_train, cs_val, cs_test = splitColdStartData(cs_df)
    
    logging.info(f"Dense - train: {len(dense_train)}, val: {len(dense_val)}, test: {len(dense_test)}")
    logging.info(f"Cold Start - train: {len(cs_train)}, val: {len(cs_val)}, test: {len(cs_test)}")
    
    saveDataSplits(dense_train, dense_val, dense_test, cs_train, cs_val, cs_test)
    
    return dense_train, dense_val, dense_test, cs_train, cs_val, cs_test
