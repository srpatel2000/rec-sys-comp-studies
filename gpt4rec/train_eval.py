"""Training, BM25 tuning, and evaluation helpers for GPT4Rec."""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from gpt4rec.runtime_tracking import GPT4RecRuntimeTracker
from gpt4rec.build_raptor import raptor_top_summary_items
from gpt4rec.search import BM25SearchIndex, aggregate_candidates, aggregate_candidates_on_items


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
    runtime_tracker: Optional[GPT4RecRuntimeTracker] = None,
):
    """Fine-tune the LM on (prompt + target) strings; optional per-epoch callback."""

    if not train_texts:
        logging.warning("train_generation_model: no training texts; skipping.")
        print("[GPT4Rec LM] no training texts; skipping optimization.", flush=True)
        if on_epoch_end:
            on_epoch_end(1)
        return

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    max_len = int(getattr(args, "maxlen", 128))
    batch_size = max(1, int(getattr(args, "batch_size", 8)))
    num_epochs = max(1, int(getattr(args, "num_epochs", 1))) 

    t_enc = time.perf_counter()
    input_ids_all, attn_all = build_lm_encodings(tokenizer, train_texts, max_len)
    labels = input_ids_all.clone()
    labels[attn_all == 0] = -100 # ensures that only real tokens are trained on

    ds = TensorDataset(input_ids_all, attn_all, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False) # splits data into batches
    if runtime_tracker is not None:
        runtime_tracker.log(
            "lm_tokenize_encode_and_build_dataloader",
            time.perf_counter() - t_enc,
            detail=f"max_len={max_len} batch_size={batch_size} n_train_texts={len(train_texts)}",
        )

    for epoch in range(1, num_epochs + 1):
        print(f"[GPT4Rec LM] epoch {epoch}/{num_epochs} start", flush=True)
        total_loss = 0.0
        n_batches = 0
        t_train = time.perf_counter()
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
        if runtime_tracker is not None:
            runtime_tracker.log(
                "lm_epoch_optimizer_forward_backward",
                time.perf_counter() - t_train,
                epoch=epoch,
                detail=f"n_batches={n_batches}",
            )
            logging.info(f"[GPT4Rec LM] epoch {epoch}/{num_epochs} optimizer forward backward time: {time.perf_counter() - t_train:.4f}")
        mean_loss = total_loss / n_batches if n_batches else 0.0
        if n_batches:
            logging.info(
                "GPT4Rec LM epoch %d/%d mean loss=%.4f",
                epoch,
                num_epochs,
                mean_loss,
            )
        model.eval()
        t_cb = time.perf_counter()
        if on_epoch_end is not None:
            on_epoch_end(epoch)
        if runtime_tracker is not None:
            runtime_tracker.log(
                "lm_epoch_on_epoch_end_callback",
                time.perf_counter() - t_cb,
                epoch=epoch,
                detail="includes in-training eval + write_train_curves when scheduled",
            )

        logging.info(
            f"[GPT4Rec LM] epoch {epoch}/{num_epochs} done (mean_loss={mean_loss:.4f})",
        )
        print(
            f"[GPT4Rec LM] epoch {epoch}/{num_epochs} done (mean_loss={mean_loss:.4f})",
            flush=True,
        )
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
    RA: Optional[Any] = None,  # built/loaded RAPTOR tree
    summary_to_items: Optional[Dict[int, List[int]]] = None, # layer-1 summary_idx -> item ids
    top_summaries: int = 15, # how many top layer-1 summaries to pull per query
    progress_label: Optional[str] = None,
    runtime_tracker: Optional[GPT4RecRuntimeTracker] = None,
    timing_detail: str = "",
) -> pd.DataFrame:
    """Generate multi-query beam strings, BM25 aggregate, rank top-k item ints."""

    t_setup = time.perf_counter()
    search_index.set_params(k1, b)
    model.eval()
    num_preds = max(1, int(getattr(args, "num_preds", 10)))
    num_queries = max(1, int(getattr(args, "num_queries_per_user", 5)))
    num_beams = max(int(getattr(args, "num_beams", 5)), num_queries)
    max_new = max(1, int(getattr(args, "max_query_tokens", 16)))
    search_top_k = max(10, int(getattr(args, "search_top_k", 100)))
    top_summaries = max(1, int(top_summaries))
    infer_bs = max(1, min(8, int(getattr(args, "batch_size", 8))))
    setup_sec = time.perf_counter() - t_setup

    rows = []
    n = len(prompts)
    detail = timing_detail or (progress_label or "")
    tok_sec = 0.0
    gen_sec = 0.0
    post_sec = 0.0

    for start in range(0, n, infer_bs):
        batch_prompts = prompts[start : start + infer_bs]
        batch_targets = targets[start : start + infer_bs]
        t0 = time.perf_counter()
        enc = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=int(getattr(args, "maxlen", 128)),
            return_tensors="pt",
        )
        tok_sec += time.perf_counter() - t0
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device) # tracks padding tokens to ignore during generation
        prompt_len = int(input_ids.shape[1])

        t0 = time.perf_counter()
        gen = model.generate_queries(
            input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            num_return_sequences=num_queries,
            max_new_tokens=max_new,
        )

        # example user prompt generation for viz
        if start == 0 and not getattr(evaluate_with_bm25, "_example_val_done", False):
            evaluate_with_bm25._example_val_done = True
            ex_p = "Previously, the customer has bought: Better Man. Gold. Gold. Invitation Only. I Want You Remastered. Greatest Love Songs. Chicago '85 The Movie. Perfect Moment. In the future, the customer wants to buy"
            ex = tokenizer([ex_p], truncation=True, max_length=int(getattr(args, "maxlen", 128)), return_tensors="pt")
            plen = ex["input_ids"].shape[1]
            ex_gen = model.generate_queries(ex["input_ids"].to(device), ex["attention_mask"].to(device), num_beams=num_beams, num_return_sequences=num_queries, max_new_tokens=max_new)
            print("[GPT4Rec example val prompt]", [tokenizer.decode(r[plen:], skip_special_tokens=True).strip() for r in ex_gen.tolist()], flush=True)

        gen_sec += time.perf_counter() - t0

        t0 = time.perf_counter()
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

            candidate_ids = set()
            if RA is not None and summary_to_items:
                for q in user_queries:
                    for iid in raptor_top_summary_items(
                        RA, q, summary_to_items, top_summaries=top_summaries
                    ):
                        candidate_ids.add(iid)

            if candidate_ids:
                cand_ids, cand_bm25 = aggregate_candidates_on_items(
                    search_index, user_queries, candidate_ids, top_k=search_top_k
                )
                # RAPTOR shortlist had no token overlap with the queries; retry on full catalog.
                if not cand_ids:
                    cand_ids, cand_bm25 = aggregate_candidates(
                        search_index, user_queries, top_k=search_top_k
                    )
            else:
                cand_ids, cand_bm25 = aggregate_candidates(
                    search_index, user_queries, top_k=search_top_k
                )

            scored = ranker.score_candidates(cand_ids, cand_bm25)
            ranked = sorted(scored.keys(), key=lambda x: scored[x], reverse=True)
            topk = [int(x) for x in ranked[:num_preds]]
            if len(topk) < num_preds:
                # Pad with a sentinel id (-1) so empty/short results count as a miss
                # rather than leaking the ground-truth item into the predictions.
                pad = topk[0] if topk else -1
                topk.extend([pad] * (num_preds - len(topk)))
            rows.append({"ground_truth_item": int(gt), "top_k_items": topk[:num_preds]})
        post_sec += time.perf_counter() - t0

    if runtime_tracker is not None:
        extra = (
            f"n_users={n} infer_bs={infer_bs} num_beams={num_beams} "
            f"num_queries={num_queries} search_top_k={search_top_k} k1={k1} b={b} "
            f"top_summaries={top_summaries}"
        )
        d = f"{detail}; {extra}" if detail else extra
        runtime_tracker.log("eval_run_setup_params", setup_sec, detail=d)
        runtime_tracker.log("eval_batch_prompt_tokenize", tok_sec, detail=d)
        runtime_tracker.log("eval_lm_beam_generate", gen_sec, detail=d)
        runtime_tracker.log(
            "eval_decode_queries_bm25_aggregate_rank",
            post_sec,
            detail=d,
        )
        runtime_tracker.log(
            "eval_with_bm25_total",
            setup_sec + tok_sec + gen_sec + post_sec,
            detail=d,
        )
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
    runtime_tracker: Optional[GPT4RecRuntimeTracker] = None,
    RA: Optional[Any] = None,
    summary_to_items: Optional[Dict[int, List[int]]] = None,
    top_summaries: int = 15,
) -> Tuple[float, float]:
    """Grid-search BM25 (k1, b) on val using macro NDCG@10 on int predictions."""

    t_tune = time.perf_counter()
    best_k1 = float(getattr(args, "bm25_k1_default", 1.2))
    best_b = float(getattr(args, "bm25_b_default", 0.75))
    best_ndcg = -1.0
    k1_grid = list(getattr(args, "bm25_k1_grid", [0.8, 1.2, 1.6]))
    b_grid = list(getattr(args, "bm25_b_grid", [0.5, 0.75, 0.9]))
    n_trials = len(k1_grid) * len(b_grid)
    print(
        f"[GPT4Rec BM25 tune] starting grid: {len(k1_grid)} k1 x {len(b_grid)} b = {n_trials} "
        f"full-val evaluate_with_bm25 passes ({len(val_prompts)} val users each).",
        flush=True,
    )

    trial = 0
    for k1 in k1_grid:
        for b in b_grid:
            trial += 1
            print(
                f"[GPT4Rec BM25 tune] trial {trial}/{n_trials}: k1={k1} b={b} ...",
                flush=True,
            )
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
                RA=RA,
                summary_to_items=summary_to_items,
                top_summaries=top_summaries,
                progress_label=f"BM25-tune-{trial}/{n_trials}-k1={k1}-b={b}",
                runtime_tracker=runtime_tracker,
                timing_detail=f"bm25_tune_trial_{trial}_of_{n_trials}",
            )
            m = macro_metrics_at_k_items(pred_df)
            if m is None:
                print(f"[GPT4Rec BM25 tune] trial {trial}: no metrics; skipping.", flush=True)
                continue
            ndcg = float(m["ndcg"])
            improved = ndcg > best_ndcg
            if improved:
                best_ndcg = ndcg
                best_k1, best_b = float(k1), float(b)
            print(
                f"[GPT4Rec BM25 tune] trial {trial}: ndcg={ndcg:.4f} "
                f"{'(new best)' if improved else ''} — best so far: ndcg={best_ndcg:.4f} k1={best_k1} b={best_b}",
                flush=True,
            )

    print(
        f"[GPT4Rec BM25 tune] finished. chosen k1={best_k1} b={best_b} (best ndcg={best_ndcg:.4f}).",
        flush=True,
    )
    if runtime_tracker is not None:
        runtime_tracker.log(
            "bm25_grid_search_outer_loop_total",
            time.perf_counter() - t_tune,
            detail=f"n_trials={n_trials} val_users={len(val_prompts)}",
        )
    return best_k1, best_b


def write_train_curves(curve_csv: str, plot_base: str, epoch: int, pred_df: pd.DataFrame):
    """Append val @10 row and refresh plots (SASRec-style artifacts)."""

    m = macro_metrics_at_k_items(pred_df)
    if m is None:
        return
    append_train_eval_row(curve_csv, epoch, m)
    refresh_train_eval_plots(curve_csv, plot_base)
    refresh_combined_dense_cold_train_eval_plots(os.path.dirname(curve_csv), "gpt4rec")
