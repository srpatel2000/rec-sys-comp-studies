"""Data utilities for GPT4Rec over existing split CSVs."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


def build_id_mappings(all_data):
    """Build ID mappings for users and items."""
    
    users = all_data["user_id"].unique()
    items = all_data["parent_asin"].unique()
    user2id = {u: i for i, u in enumerate(users, start=1)}
    item2id = {a: i for i, a in enumerate(items, start=1)}
    return user2id, item2id


def build_user_histories(
    train_df: pd.DataFrame, user2id: Dict[str, int], item2id: Dict[str, int]
) -> Dict[int, List[int]]:
    df = train_df.copy()
    df["user_int_id"] = df["user_id"].map(user2id)
    df["item_int_id"] = df["parent_asin"].map(item2id)
    histories: Dict[int, List[int]] = {}
    for uid, g in df.groupby("user_int_id"):
        g = g.sort_values("timestamp")
        histories[int(uid)] = g["item_int_id"].astype(int).tolist()
    return histories


def build_item_texts(all_data: pd.DataFrame) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    for _, row in all_data.iterrows():
        asin = row["parent_asin"]
        title = row["title"] if "title" in all_data.columns else asin
        if pd.isna(title):
            title = asin
        texts[str(asin)] = str(title)
    return texts


@dataclass
class GPTExample:
    user_int_id: int
    history_item_int_ids: List[int]
    target_item_int_id: int
    history_titles: List[str]
    target_title: str


def build_examples_for_training(
    train_df: pd.DataFrame,
    user2id: Dict[str, int],
    item2id: Dict[str, int],
    item_text_by_asin: Dict[str, str],
) -> List[GPTExample]:
    examples: List[GPTExample] = []
    df = train_df.copy()
    df["user_int_id"] = df["user_id"].map(user2id)
    df["item_int_id"] = df["parent_asin"].map(item2id)
    for uid, g in df.groupby("user_int_id"):
        g = g.sort_values("timestamp")
        item_ids = g["item_int_id"].astype(int).tolist()
        asins = g["parent_asin"].astype(str).tolist()
        if len(item_ids) < 2:
            continue
        hist_ids = item_ids[:-1]
        target_id = item_ids[-1]
        hist_titles = [item_text_by_asin.get(a, a) for a in asins[:-1]]
        target_title = item_text_by_asin.get(asins[-1], asins[-1])
        examples.append(
            GPTExample(
                user_int_id=int(uid),
                history_item_int_ids=hist_ids,
                target_item_int_id=int(target_id),
                history_titles=hist_titles,
                target_title=target_title,
            )
        )
    return examples


def build_examples_for_holdout(
    target_df: pd.DataFrame,
    user_histories: Dict[int, List[int]],
    user2id: Dict[str, int],
    item2id: Dict[str, int],
    item_text_by_asin: Dict[str, str],
    int_to_item: Dict[int, str],
) -> List[GPTExample]:
    examples: List[GPTExample] = []
    for _, row in target_df.iterrows():
        uid = int(user2id[row["user_id"]])
        target_asin = str(row["parent_asin"])
        target_id = int(item2id[target_asin])
        hist = user_histories.get(uid, [])
        if len(hist) == 0:
            continue
        hist_titles = []
        for iid in hist:
            asin = int_to_item.get(int(iid), str(iid))
            hist_titles.append(item_text_by_asin.get(asin, asin))
        # Prefer actual title from asin map for target
        examples.append(
            GPTExample(
                user_int_id=uid,
                history_item_int_ids=list(hist),
                target_item_int_id=target_id,
                history_titles=hist_titles,
                target_title=item_text_by_asin.get(target_asin, target_asin),
            )
        )
    return examples


class GPTPromptDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        out = {k: v[idx] for k, v in self.encodings.items()}
        return {k: torch.tensor(v) for k, v in out.items()}
