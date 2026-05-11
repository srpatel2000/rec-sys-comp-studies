"""
Random top-k recommender: uniform sample without replacement from the item catalog.

Writes the same prediction CSV schema as SASRec (after convertIDBackToRaw) so
`eval_metrics.evalPipeline` and file layouts under `data/outputs/baseline/` match SASRec.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import GlobalConfig, SASRecModelConfig

sasrec_defaults = SASRecModelConfig()


def _build_test_sequences(
    user_sequences: dict,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Same history extension as SASRec `predictTest` (val item appended per user)."""

    test_sequences: dict[int, list] = {}
    for uid in test_df["user_int_id"].unique():
        uid = int(uid)
        history = list(user_sequences.get(uid, []))
        user_val = val_df[val_df["user_int_id"] == uid]
        if len(user_val) > 0:
            history.append(int(user_val.iloc[0]["item_int_id"]))
        test_sequences[uid] = history
    return test_sequences


def _random_prediction_rows(
    target_df: pd.DataFrame,
    user_sequences: dict,
    itemnum: int,
    rng: np.random.Generator,
    num_preds: int,
) -> pd.DataFrame:
    """Rows aligned with `predictForUsers` / SASRec export (int columns before convert)."""

    if itemnum < 1:
        raise ValueError("itemnum must be >= 1")

    k = min(num_preds, itemnum)
    all_item_ids = np.arange(1, itemnum + 1, dtype=np.int32)

    rows = []
    for _, row in target_df.iterrows():
        uid = int(row["user_int_id"])
        gt_item = int(row["item_int_id"])
        history = user_sequences.get(uid, [])
        if len(history) == 0:
            continue

        pick = rng.choice(all_item_ids, size=k, replace=False)
        top_k_items = [int(x) for x in pick.tolist()]
        if k < num_preds:
            pad = int(top_k_items[0]) if top_k_items else 1
            top_k_items.extend([pad] * (num_preds - k))

        top_k_scores = sorted(rng.random(num_preds).tolist(), reverse=True)

        if gt_item in top_k_items:
            ground_truth_rank = top_k_items.index(gt_item) + 1
        else:
            ground_truth_rank = int(itemnum)

        rows.append(
            {
                "user_int_id": uid,
                "ground_truth_item": gt_item,
                "ground_truth_score": 0.0,
                "ground_truth_rank": ground_truth_rank,
                "top_k_items": top_k_items[:num_preds],
                "top_k_scores": top_k_scores,
            }
        )

    return pd.DataFrame(rows)


def runBaselinePipeline(data_type: str = "dense", config: Optional[GlobalConfig] = None) -> None:
    """
    Generate val/test prediction CSVs under `data/outputs/baseline/`, matching SASRec paths
    and columns (`dense_val_predictions.csv`, etc.).
    """

    from custom_sasrec_funcs import buildIDMappings, convertIDBackToRaw, perUserSequence

    config = config or GlobalConfig()
    num_preds = int(getattr(sasrec_defaults, "num_preds", 10))
    seed = int(getattr(config, "random_seed", 42))

    train_df = pd.read_csv(config.data_dir / "train" / f"{data_type}_train.csv")
    val_df = pd.read_csv(config.data_dir / "val" / f"{data_type}_val.csv")
    test_df = pd.read_csv(config.data_dir / "test" / f"{data_type}_test.csv")

    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    user_int_id, item_int_id = buildIDMappings(all_data)
    itemnum = len(item_int_id)

    user_sequences = perUserSequence(train_df.copy(), user_int_id, item_int_id)

    val_df = val_df.copy()
    test_df = test_df.copy()
    val_df["user_int_id"] = val_df["user_id"].map(user_int_id)
    val_df["item_int_id"] = val_df["parent_asin"].map(item_int_id)
    test_df["user_int_id"] = test_df["user_id"].map(user_int_id)
    test_df["item_int_id"] = test_df["parent_asin"].map(item_int_id)

    outputs_dir = Path(config.data_dir) / "outputs" / "baseline"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    val_rng = np.random.default_rng(seed)
    val_preds = _random_prediction_rows(val_df, user_sequences, itemnum, val_rng, num_preds)
    val_preds = convertIDBackToRaw(val_preds, user_int_id, item_int_id)

    test_sequences = _build_test_sequences(user_sequences, val_df, test_df)
    test_rng = np.random.default_rng(seed + 1_000_003)
    test_preds = _random_prediction_rows(test_df, test_sequences, itemnum, test_rng, num_preds)
    test_preds = convertIDBackToRaw(test_preds, user_int_id, item_int_id)

    column_order = [
        "user_int_id",
        "ground_truth_item",
        "ground_truth_score",
        "ground_truth_rank",
        "user_id",
        "ground_truth_asin",
        "top_k_asins",
        "ground_truth_asins",
        "top_k_scores",
    ]
    val_preds = val_preds[[c for c in column_order if c in val_preds.columns]]
    test_preds = test_preds[[c for c in column_order if c in test_preds.columns]]

    val_path = outputs_dir / f"{data_type}_val_predictions.csv"
    test_path = outputs_dir / f"{data_type}_test_predictions.csv"
    val_preds.to_csv(val_path, index=False)
    test_preds.to_csv(test_path, index=False)

    logging.info(
        "Random catalog baseline (%s): wrote %d val rows -> %s, %d test rows -> %s (catalog |I|=%d, k=%d)",
        data_type,
        len(val_preds),
        val_path,
        len(test_preds),
        test_path,
        itemnum,
        num_preds,
    )
