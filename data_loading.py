import datasets
datasets.logging.set_verbosity_error()
from datasets import load_dataset
import pandas as pd
import logging
import random
import time
from config import GlobalConfig, SASRecModelConfig, GPT4RecModelConfig

logging.getLogger("filelock").setLevel(logging.WARNING) # don't want to see filelock warnings


def getHistoryBuckets(max_interactions=None):
    bucket_ranges = [
        (1, 4),
        (5, 9),
        (10, max_interactions),
    ]

    if max_interactions is None:
        bucket_ranges[-1] = (10, None)

    buckets = []
    for lower, upper in bucket_ranges:
        if upper is not None and lower > upper:
            continue

        label = f"{lower}+" if upper is None else f"{lower}-{upper}"
        buckets.append((label, lower, upper))

    return buckets


def sampleUsersByHistoryLength(user_counts, target_users, max_interactions, random_seed):
    buckets = getHistoryBuckets(max_interactions)
    bucket_users = {label: [] for label, _, _ in buckets}

    for user_id, count in user_counts.items():
        for label, lower, upper in buckets:
            if count < lower:
                continue
            if upper is not None and count > upper:
                continue

            bucket_users[label].append(user_id)
            break

    non_empty_labels = [label for label, users in bucket_users.items() if users]
    total_eligible = sum(len(bucket_users[label]) for label in non_empty_labels)

    if not non_empty_labels:
        return set(), bucket_users, {}

    sample_size = min(target_users, total_eligible)
    remaining_by_bucket = {label: len(bucket_users[label]) for label in non_empty_labels}
    selected_counts = {label: 0 for label in non_empty_labels}
    labels_to_fill = non_empty_labels[:]
    remaining_to_select = sample_size

    while remaining_to_select > 0 and labels_to_fill:
        base = remaining_to_select // len(labels_to_fill)
        extra = remaining_to_select % len(labels_to_fill)
        next_labels = []

        for index, label in enumerate(labels_to_fill):
            requested = base + (1 if index < extra else 0)
            take = min(requested, remaining_by_bucket[label])

            selected_counts[label] += take
            remaining_by_bucket[label] -= take
            remaining_to_select -= take

            if remaining_by_bucket[label] > 0:
                next_labels.append(label)

        labels_to_fill = next_labels

    rng = random.Random(random_seed)
    selected_users = set()
    for label, users in bucket_users.items():
        take = selected_counts.get(label, 0)
        if take > 0:
            selected_users.update(rng.sample(users, take))

    return selected_users, bucket_users, selected_counts


def getReviewStreams(config, core=5, cold_start=False):

    cfg = config() if isinstance(config, type) else config
    streams = []
    keep_cols = ["user_id", "parent_asin", "timestamp"]

    for cat in cfg.datasets:
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

        streams.append(ds.select_columns(keep_cols))

    return streams


def loadAmazonReviews(config=GlobalConfig, 
                      core=5,
                      cold_start=False,
                      target_users=250000,
                      max_interactions=None):
    
    cfg = config() if isinstance(config, type) else config
    total_start = time.perf_counter()

    first_pass_start = time.perf_counter()
    first_pass_streams = getReviewStreams(cfg, core=core, cold_start=cold_start)
    first_pass = datasets.interleave_datasets(first_pass_streams)

    user_counts = {}
    logging.info("Counting user histories for stratified sampling...")

    for row in first_pass:
        uid = row.get("user_id")
        if uid is None:
            continue

        user_counts[uid] = user_counts.get(uid, 0) + 1

    first_pass_minutes = (time.perf_counter() - first_pass_start) / 60
    logging.info(
        f"Finished history counting in {first_pass_minutes:.2f} minutes. "
        f"Counted {len(user_counts)} unique users."
    )

    sampling_start = time.perf_counter()
    selected_users, bucket_users, selected_counts = sampleUsersByHistoryLength(
        user_counts=user_counts,
        target_users=target_users,
        max_interactions=max_interactions,
        random_seed=cfg.random_seed,
    )
    sampling_seconds = time.perf_counter() - sampling_start

    for label, users in bucket_users.items():
        if not users:
            continue
        logging.info(
            f"Bucket {label}: selected {selected_counts.get(label, 0)} of {len(users)} eligible users."
        )

    logging.info(
        f"Finished stratified user sampling in {sampling_seconds:.2f} seconds. "
        f"Selected {len(selected_users)} users."
    )

    logging.info(f"Selected {len(selected_users)} users. Collecting full histories...")

    second_pass_start = time.perf_counter()
    second_pass_streams = getReviewStreams(cfg, core=core, cold_start=cold_start)
    second_pass = datasets.interleave_datasets(second_pass_streams)

    user_rows = {}
    for row in second_pass:
        uid = row.get("user_id")
        if uid not in selected_users:
            continue

        user_history = user_rows.setdefault(uid, [])
        user_history.append(row)

    second_pass_minutes = (time.perf_counter() - second_pass_start) / 60
    logging.info(
        f"Finished collecting full user histories in {second_pass_minutes:.2f} minutes. "
        f"Collected histories for {len(user_rows)} users."
    )

    dataset_start = time.perf_counter()
    rows = [row for history in user_rows.values() for row in history]

    dataset = datasets.Dataset.from_list(rows)
    dataset_seconds = time.perf_counter() - dataset_start
    logging.info(f"Built review dataset in {dataset_seconds:.2f} seconds.")

    if max_interactions is None:
        logging.info(
            f"Kept {len(user_rows)} users across stratified history buckets. "
            f"Total interactions: {len(dataset)}"
        )
    else:
        logging.info(
            f"Kept {len(user_rows)} users across stratified history buckets up to {max_interactions} interactions. "
            f"Total interactions: {len(dataset)}"
        )

    total_minutes = (time.perf_counter() - total_start) / 60
    logging.info(f"loadAmazonReviews completed in {total_minutes:.2f} minutes.")
    return dataset


def loadItemMetadata(reviews_df, config=GlobalConfig, max_scan=8_000_000):
    """Loads metadata for items present in reviews_df['parent_asin'].
    
    Creds to Copilot for the efficient streaming approach to load only relevant metadata from 
    HuggingFace instead of the full dataset."""

    total_start = time.perf_counter()
    item_id_start = time.perf_counter()

    if isinstance(reviews_df, pd.DataFrame):
        item_ids = set(reviews_df["parent_asin"].dropna().unique())
    else:
        item_ids = set(asin for asin in reviews_df.unique("parent_asin") if asin is not None)

    item_id_seconds = time.perf_counter() - item_id_start
    logging.info(
        f"Built item ID set with {len(item_ids)} unique items in {item_id_seconds:.2f} seconds."
    )

    if not item_ids:
        raise ValueError("No valid item IDs found.")

    keep_cols = ["parent_asin", "title", "main_category"]
    remaining = set(item_ids)
    rows = []

    for cat in config.datasets:
        category_start = time.perf_counter()
        found_before = len(item_ids) - len(remaining)

        logging.info(
            f"Loading metadata for '{cat}' "
            f"({len(item_ids) - len(remaining)}/{len(item_ids)} found so far)..."
        )

        subset = f"raw_meta_{cat}"
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            subset,
            streaming=True,
            trust_remote_code=True,
        )

        if hasattr(ds, "keys"):
            split_name = "full" if "full" in ds else next(iter(ds.keys()))
            ds = ds[split_name]

        ds = ds.select_columns(keep_cols)

        scanned = 0
        for batch in ds.iter(batch_size=1000):
            asins = batch["parent_asin"]

            for j, asin in enumerate(asins):
                if asin in remaining:
                    rows.append({
                        "parent_asin": batch["parent_asin"][j],
                        "title": batch["title"][j],
                        "main_category": batch["main_category"][j],
                    })
                    remaining.discard(asin)

            scanned += len(asins)

            if not remaining or scanned >= max_scan:
                break

        elapsed_minutes = (time.perf_counter() - category_start) / 60
        found_after = len(item_ids) - len(remaining)
        logging.info(
            f"Finished metadata for '{cat}' in {elapsed_minutes:.2f} minutes. "
            f"Scanned {scanned} rows and found {found_after - found_before} new items."
        )

        if not remaining:
            break

    total_minutes = (time.perf_counter() - total_start) / 60
    logging.info(
        f"Metadata loaded: {len(item_ids) - len(remaining)}/{len(item_ids)} items found in {total_minutes:.2f} minutes."
    )
    return pd.DataFrame(rows)


def createUserItemPairings(reviews_df, metadata_df):
    """Joins reviews_df and metadata_df on 'parent_asin' to create a sequential dataset 
    with user, item, timestamp, and metadata features."""

    merge_start = time.perf_counter()

    if not isinstance(reviews_df, pd.DataFrame):
        reviews_df = reviews_df.to_pandas()

    if not isinstance(metadata_df, pd.DataFrame):
        metadata_df = pd.DataFrame(metadata_df)

    merged_df = pd.merge(reviews_df, metadata_df, on="parent_asin", how="left")
    merge_seconds = time.perf_counter() - merge_start
    logging.info(f"Merged reviews and metadata in {merge_seconds:.2f} seconds.")
    return merged_df


def saveRawData(df, out_path=GlobalConfig.data_dir/"raw_dataset.csv"):
    """Saves the raw extracted data to a CSV file in the processed directory."""
    save_start = time.perf_counter()
    df.to_csv(out_path, index=False)
    save_seconds = time.perf_counter() - save_start
    logging.info(f"Raw data saved to {out_path} in {save_seconds:.2f} seconds.")


def loadDataPipeline(config=GlobalConfig, cold_start=False):
    """Full data loading pipeline that extracts reviews and metadata, creates sequential 
    dataset, and saves to processed dir."""

    total_start = time.perf_counter()

    reviews_start = time.perf_counter()
    reviews_df = loadAmazonReviews(config=config, cold_start=cold_start, target_users=config.samples)
    logging.info(f"Review loading section completed in {(time.perf_counter() - reviews_start) / 60:.2f} minutes.")

    metadata_start = time.perf_counter()
    metadata_df = loadItemMetadata(reviews_df, config=config)
    logging.info(f"Metadata loading section completed in {(time.perf_counter() - metadata_start) / 60:.2f} minutes.")

    pairing_start = time.perf_counter()
    sequential_df = createUserItemPairings(reviews_df, metadata_df)
    logging.info(f"Pairing section completed in {(time.perf_counter() - pairing_start) / 60:.2f} minutes.")

    save_start = time.perf_counter()
    saveRawData(sequential_df)
    logging.info(f"Save section completed in {(time.perf_counter() - save_start) / 60:.2f} minutes.")

    total_minutes = (time.perf_counter() - total_start) / 60
    logging.info(f"loadDataPipeline completed in {total_minutes:.2f} minutes.")

    return sequential_df