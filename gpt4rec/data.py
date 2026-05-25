"""Data utilities for GPT4Rec over existing split CSVs."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

EXAMPLE_USER = "AHULIJDWTYYPBXFP5FFM75GFEZKA"


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

    if "user_id" in all_data.columns:
        mask = all_data["user_id"].astype(str) == EXAMPLE_USER
        if mask.any():
            u_asins = (
                all_data.loc[mask, "parent_asin"]
                .dropna()
                .astype(str)
                .unique()
            )
            preview = list(u_asins)[:5]
            logging.info(
                "[build_item_texts] example user %s: %d distinct items (showing up to 5)",
                EXAMPLE_USER,
                len(u_asins),
            )
            for a in preview:
                t = texts.get(a, "")
                shown = (t[:120] + "…") if len(t) > 120 else t
                logging.info("[build_item_texts]   %s -> %s", a, shown)
        else:
            logging.info(
                "[build_item_texts] example user %s not in all_data; skip preview",
                EXAMPLE_USER,
            )

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
        if EXAMPLE_USER in user2id and int(uid) == user2id[EXAMPLE_USER]:
            ex = examples[-1]
            print(
                f"[build_examples_for_training] example user {EXAMPLE_USER} (uid={uid}):\n"
                f"  history_item_int_ids={ex.history_item_int_ids}\n"
                f"  target_item_int_id={ex.target_item_int_id}\n"
                f"  history_titles={ex.history_titles}\n"
                f"  target_title={ex.target_title!r}",
                flush=True,
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
        if EXAMPLE_USER in user2id and uid == user2id[EXAMPLE_USER]:
            ex = examples[-1]
            print(
                f"[build_examples_for_holdout] example user {EXAMPLE_USER} "
                f"target_asin={target_asin}:\n"
                f"  history_item_int_ids={ex.history_item_int_ids}\n"
                f"  target_item_int_id={ex.target_item_int_id}\n"
                f"  history_titles={ex.history_titles}\n"
                f"  target_title={ex.target_title!r}",
                flush=True,
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
