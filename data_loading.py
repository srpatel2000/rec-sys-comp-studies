import datasets
datasets.logging.set_verbosity_error()
from datasets import load_dataset
import pandas as pd
import logging
from config import GlobalConfig, SASRecModelConfig, GPT4RecModelConfig

logging.getLogger("filelock").setLevel(logging.WARNING)



# def check_cols(df, specified_cols):
#     """Check if the DataFrame contains the specified columns for user, item, and timestamp columns.
#     Raises error if specified columns are not found. """

#     for col in specified_cols:
#         if col in df.columns:
#             logging.info(f"Found column '{col}' in DataFrame.")
    
#     raise ValueError("DataFrame must contain specified user, item, and timestamp columns.")


# def concat_parts(parts, template_df):
#     if not parts:
#         return template_df.iloc[0:0].copy()
#     return pd.concat(parts, ignore_index=True)


# def sequential_split(source_df, user_col, time_col):
#     train_parts, val_parts, test_parts = [], [], []

#     for _, user_df in source_df.groupby(user_col, sort=False):
#         user_df = user_df.sort_values(time_col)
#         n = len(user_df)

#         if n >= 3:
#             train_parts.append(user_df.iloc[:-2])
#             val_parts.append(user_df.iloc[[-2]])
#             test_parts.append(user_df.iloc[[-1]])
#         elif n == 2:
#             val_parts.append(user_df.iloc[[0]])
#             test_parts.append(user_df.iloc[[1]])
#         elif n == 1:
#             test_parts.append(user_df.iloc[[0]])

#     train_df = concat_parts(train_parts, source_df)
#     val_df = concat_parts(val_parts, source_df)
#     test_df = concat_parts(test_parts, source_df)
#     return train_df, val_df, test_df


# def save_splits(train_df, val_df, test_df, out_dir):
#     out_dir.mkdir(parents=True, exist_ok=True)
#     train_df.to_csv(out_dir / "train.csv", index=False)
#     val_df.to_csv(out_dir / "val.csv", index=False)
#     test_df.to_csv(out_dir / "test.csv", index=False)


# def splitDataset(df, cold_start, config=GlobalConfig, split_ratio=0.8):
#     """
#     Function splits dataset into train, val, and test sets and saves them to the processed directory.
#     Train, val, and test are split by user sequence and will be different based on experiment type:
    
#     Cold Starts Experiment 1: Want to test model on users with interaction history size 2-4. Will 
#     train on all users with >=5 interactions.

#     Cold Starts Experiment 2: Want to test model on bottom 10% frequency items. Will leave the item 
#     in the vocabulary set, but will remove any interaction during training.

#     Non-cold start: Reads in data from dense dataset and removes last interaction as test, second to 
#     last as val, and rest as train.
#     """

#     user_col = check_cols(df, ["user_id"])
#     item_col = check_cols(df, ["parent_asin"])
#     time_col = check_cols(df, ["timestamp"])

#     df = df.sort_values([user_col, time_col]).reset_index(drop=True) # ensure sequential order by user, item

#     data_dir = config.data_dir

#     if cold_start:
#         ##### Experiment 1 `#####
#         # Train on users with >=5 interactions.
#         # Evaluate on users with 2-4 interactions.
#         user_counts = df[user_col].value_counts()
#         train_users = user_counts[user_counts >= 5].index
#         eval_users = user_counts[(user_counts >= 2) & (user_counts <= 4)].index

#         exp1_train = df[df[user_col].isin(train_users)].copy()
#         exp1_eval = df[df[user_col].isin(eval_users)].copy()
#         _, exp1_val, exp1_test = sequential_split(exp1_eval, user_col, time_col)

#         exp1_dir = base_processed_dir / "cold_start" / "experiment_1"
#         save_splits(exp1_train, exp1_val, exp1_test, exp1_dir)

#         ##### Experiment 2 #####
#         # Bottom 10% frequency items are test items.
#         # Remove their interactions from train, but keep them in val/test.
#         item_counts = df[item_col].value_counts()
#         freq_cutoff = item_counts.quantile(0.10)
#         low_freq_items = item_counts[item_counts <= freq_cutoff].index

#         exp2_train, exp2_val, exp2_test = sequential_split(df, user_col, time_col)
#         exp2_train = exp2_train[~exp2_train[item_col].isin(low_freq_items)].copy()
#         exp2_val = exp2_val[exp2_val[item_col].isin(low_freq_items)].copy()
#         exp2_test = exp2_test[exp2_test[item_col].isin(low_freq_items)].copy()

#         exp2_dir = base_processed_dir / "cold_start" / "experiment_2"
#         save_splits(exp2_train, exp2_val, exp2_test, exp2_dir)

#         return

#     else:
#         # want to return sequential interactions as test
#         # remove last interaction as test, second to last as val, and rest as train
#         train_df, val_df, test_df = sequential_split(df, user_col, time_col)
#         out_dir = base_processed_dir / "non_cold_start"
#         save_splits(train_df, val_df, test_df, out_dir)
#         return




def bottomFrequencyItems(df, item_col, quantile=0.10):
    item_counts = df[item_col].value_counts()
    freq_cutoff = item_counts.quantile(quantile)
    low_freq_items = item_counts[item_counts <= freq_cutoff].index
    return low_freq_items



# This file will load in the dataset for this project 
def loadAmazonReviews(config=GlobalConfig, 
                      core=5,
                      cold_start=False,
                      target_users=250000):
    
    categories = config.datasets

    streams = []
    keep_cols = ["user_id", "parent_asin", "timestamp"]

    for cat in categories:
        logging.info(f"Loading category '{cat}' from HuggingFace...")
        subset = f"0core_rating_only_{cat}" if cold_start else f"{core}core_rating_only_{cat}"
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            subset,
            trust_remote_code=True,
            streaming=True,
        )

        if hasattr(ds, "keys"):
            split_name = "full" if "full" in ds else next(iter(ds.keys()))
            ds = ds[split_name]

        ds = ds.select_columns(keep_cols)
        streams.append(ds)

    conc_dataset = datasets.interleave_datasets(streams)
    shuffled = conc_dataset.shuffle(seed=config.random_seed, buffer_size=100_000)

    # stream rows until we have target_users unique users (keeping all their interactions)
    user_interactions = {}
    logging.info(f"Streaming until {target_users} unique users collected...")

    for row in shuffled:
        uid = row.get("user_id")
        if uid is None:
            continue
        if uid not in user_interactions:
            if len(user_interactions) >= target_users:
                break
            user_interactions[uid] = []
        user_interactions[uid].append(row)

    logging.info(f"Collected {len(user_interactions)} unique users.")

    rows = [row for interactions in user_interactions.values() for row in interactions]
    dataset = datasets.Dataset.from_list(rows)

    logging.info(f"Total interactions: {len(dataset)}")
    return dataset


def loadItemMetadata(df):
    



def createUserItemPairings(cold_start=False):
    pass