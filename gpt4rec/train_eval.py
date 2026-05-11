"""Training, BM25 tuning, and evaluation helpers for GPT4Rec."""

import logging
import os
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Tokenizer

from eval_metrics import (
    append_train_eval_row,
    macro_metrics_at_k_items,
    refresh_combined_dense_cold_train_eval_plots,
    refresh_train_eval_plots,
)
from gpt4rec.model import GPT4RecCandidateRanker, GPT4RecGenerationModel
from gpt4rec.search import BM25SearchIndex, aggregate_candidates


def build_lm_encodings(
    tokenizer: GPT2Tokenizer, texts: List[str], max_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def train_generation_model(
    model: GPT4RecGenerationModel,
    tokenizer: GPT2Tokenizer,
    train_texts: List[str],
    args,
    device: torch.device,
    on_epoch_end: Optional[Callable[[int], None]] = None,
):
    """Fine-tune the LM on (prompt + target) strings; optional per-epoch callback."""

    if not train_texts:
        logging.warning("train_generation_model: no training texts; skipping.")
        if on_epoch_end:
            on_epoch_end(1)
        return

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    max_len = int(getattr(args, "maxlen", 128))
    batch_size = max(1, int(getattr(args, "batch_size", 8)))
    num_epochs = max(1, int(getattr(args, "num_epochs", 1)))

    input_ids_all, attn_all = build_lm_encodings(tokenizer, train_texts, max_len)
    labels = input_ids_all.clone()
    labels[attn_all == 0] = -100

    ds = TensorDataset(input_ids_all, attn_all, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for input_ids, attention_mask, labels_b in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_b = labels_b.to(device)
            optimizer.zero_grad()
            out = model.forward_train(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_b,
            )
            loss = out.loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        if n_batches:
            logging.info(
                "GPT4Rec LM epoch %d/%d mean loss=%.4f",
                epoch,
                num_epochs,
                total_loss / n_batches,
            )
        model.eval()
        if on_epoch_end is not None:
            on_epoch_end(epoch)
        model.train()


@torch.no_grad()
def evaluate_with_bm25(
    model: GPT4RecGenerationModel,
    tokenizer: GPT2Tokenizer,
    prompts: List[str],
    targets: List[int],
    search_index: BM25SearchIndex,
    ranker: GPT4RecCandidateRanker,
    args,
    device: torch.device,
    k1: float,
    b: float,
) -> pd.DataFrame:
    """Generate multi-query beam strings, BM25 aggregate, rank top-k item ints."""

    search_index.set_params(k1, b)
    model.eval()
    num_preds = max(1, int(getattr(args, "num_preds", 10)))
    num_queries = max(1, int(getattr(args, "num_queries_per_user", 5)))
    num_beams = max(int(getattr(args, "num_beams", 5)), num_queries)
    max_new = max(1, int(getattr(args, "max_query_tokens", 16)))
    search_top_k = max(10, int(getattr(args, "search_top_k", 100)))
    infer_bs = max(1, min(8, int(getattr(args, "batch_size", 8))))

    rows = []
    n = len(prompts)
    for start in range(0, n, infer_bs):
        batch_prompts = prompts[start : start + infer_bs]
        batch_targets = targets[start : start + infer_bs]
        enc = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=int(getattr(args, "maxlen", 128)),
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_len = int(input_ids.shape[1])

        gen = model.generate_queries(
            input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            num_return_sequences=num_queries,
            max_new_tokens=max_new,
        )

        for j, gt in enumerate(batch_targets):
            user_queries = []
            for r in range(num_queries):
                idx = j * num_queries + r
                if idx >= gen.shape[0]:
                    break
                tail = gen[idx, prompt_len:]
                q = tokenizer.decode(tail.tolist(), skip_special_tokens=True).strip()
                if q:
                    user_queries.append(q)
            if not user_queries:
                user_queries = [batch_prompts[j][:200]]

            cand_ids, cand_bm25 = aggregate_candidates(
                search_index, user_queries, top_k=search_top_k
            )
            if not cand_ids:
                rows.append({"ground_truth_item": int(gt), "top_k_items": [int(gt)] * num_preds})
                continue
            scored = ranker.score_candidates(cand_ids, cand_bm25)
            ranked = sorted(scored.keys(), key=lambda x: scored[x], reverse=True)
            topk = [int(x) for x in ranked[:num_preds]]
            if len(topk) < num_preds:
                pad = topk[0] if topk else int(gt)
                topk.extend([pad] * (num_preds - len(topk)))
            rows.append({"ground_truth_item": int(gt), "top_k_items": topk[:num_preds]})

    return pd.DataFrame(rows)


def tune_bm25_params(
    model: GPT4RecGenerationModel,
    tokenizer: GPT2Tokenizer,
    val_prompts: List[str],
    val_targets: List[int],
    search_index: BM25SearchIndex,
    ranker: GPT4RecCandidateRanker,
    args,
    device: torch.device,
) -> Tuple[float, float]:
    """Grid-search BM25 (k1, b) on val using macro NDCG@10 on int predictions."""

    best_k1 = float(getattr(args, "bm25_k1_default", 1.2))
    best_b = float(getattr(args, "bm25_b_default", 0.75))
    best_ndcg = -1.0
    k1_grid = list(getattr(args, "bm25_k1_grid", [0.8, 1.2, 1.6]))
    b_grid = list(getattr(args, "bm25_b_grid", [0.5, 0.75, 0.9]))

    for k1 in k1_grid:
        for b in b_grid:
            pred_df = evaluate_with_bm25(
                model,
                tokenizer,
                val_prompts,
                val_targets,
                search_index,
                ranker,
                args,
                device,
                float(k1),
                float(b),
            )
            m = macro_metrics_at_k_items(pred_df)
            if m is None:
                continue
            if float(m["ndcg"]) > best_ndcg:
                best_ndcg = float(m["ndcg"])
                best_k1, best_b = float(k1), float(b)

    return best_k1, best_b


def write_train_curves(curve_csv: str, plot_base: str, epoch: int, pred_df: pd.DataFrame):
    """Append val @10 row and refresh plots (SASRec-style artifacts)."""

    m = macro_metrics_at_k_items(pred_df)
    if m is None:
        return
    append_train_eval_row(curve_csv, epoch, m)
    refresh_train_eval_plots(curve_csv, plot_base)
    refresh_combined_dense_cold_train_eval_plots(os.path.dirname(curve_csv), "gpt4rec")
