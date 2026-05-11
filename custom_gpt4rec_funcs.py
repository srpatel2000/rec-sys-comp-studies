"""Custom GPT4Rec pipeline over existing split CSVs."""

import json
import logging
from pathlib import Path

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from custom_sasrec_funcs import buildIDMappings

from config import GlobalConfig
from gpt4rec.data import (
    build_examples_for_holdout,
    build_examples_for_training,
    build_item_texts,
    build_user_histories,
)
from gpt4rec.model import GPT4RecCandidateRanker, GPT4RecGenerationModel
from gpt4rec.prompting import build_history_prompt, build_train_text
from gpt4rec.search import BM25SearchIndex
from gpt4rec.train_eval import evaluate_with_bm25, train_generation_model, tune_bm25_params, write_train_curves


def int_to_asin_map(item2id):
    return {v: k for k, v in item2id.items()}


def to_eval_output(df, int_to_item):
    out = df.copy()
    out["top_k_asins"] = out["top_k_items"].apply(
        lambda ids: json.dumps([int_to_item.get(int(i), None) for i in ids])
    )
    out["ground_truth_asin"] = out["ground_truth_item"].map(int_to_item)
    out["ground_truth_asins"] = out["ground_truth_asin"].apply(lambda x: json.dumps([x]))
    return out


def resolve_gpt4rec_device(global_config: GlobalConfig) -> torch.device:
    """Pick torch device from GlobalConfig.use_gpu: CUDA, then Apple MPS, else CPU."""

    if not getattr(global_config, "use_gpu", True):
        logging.info("GPT4Rec: GlobalConfig.use_gpu=False; using CPU.")
        return torch.device("cpu")

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        logging.info("GPT4Rec: using CUDA device %s (%s).", idx, name)
        return torch.device("cuda", idx)

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        logging.info("GPT4Rec: CUDA not available; using Apple MPS (Metal).")
        return torch.device("mps")

    logging.warning(
        "GPT4Rec: use_gpu=True but neither CUDA nor MPS is available; using CPU."
    )
    return torch.device("cpu")


def runGPT4RecPipeline(data_type="dense"):
    """Run the full GPT4Rec pipeline: build ID mappings, create user histories, build model, train, and predict."""

    config = GlobalConfig()
    args = config.model_namespace("gpt4rec")

    logging.info(f"Running GPT4Rec pipeline for {data_type} dataset...")

    # load train/val/test CSVs
    train_df = pd.read_csv(config.data_dir / "train" / f"{data_type}_train.csv")
    val_df = pd.read_csv(config.data_dir / "val" / f"{data_type}_val.csv")
    test_df = pd.read_csv(config.data_dir / "test" / f"{data_type}_test.csv")

    # step 1: build ID mappings for all of the data
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    user2id, item2id = buildIDMappings(all_data) # use same methodology as SASRec to build mappings
    int_to_item = int_to_asin_map(item2id)

    # step 2: build item text mappings
    item_text_by_asin = build_item_texts(all_data)
    item_text_by_item_id = {
        item2id[asin]: txt for asin, txt in item_text_by_asin.items() if asin in item2id
    }

    user_hist = build_user_histories(train_df, user2id, item2id)
    train_examples = build_examples_for_training(train_df, user2id, item2id, item_text_by_asin)
    val_examples = build_examples_for_holdout(
        val_df, user_hist, user2id, item2id, item_text_by_asin, int_to_item
    )

    tokenizer = GPT2Tokenizer.from_pretrained(args.hf_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only: batch generation must pad on the left so the last real token aligns.
    tokenizer.padding_side = "left"
    lm = GPT2LMHeadModel.from_pretrained(args.hf_model_name)

    class _TmpCfg:
        pass

    tmp = _TmpCfg()
    tmp.num_users = len(user2id) + 1
    tmp.num_items = len(item2id) + 1
    tmp.vocab_size = tokenizer.vocab_size
    tmp.n_embd = lm.config.n_embd
    tmp.initializer_range = args.initializer_range

    model = GPT4RecGenerationModel(tmp, lm)
    device = resolve_gpt4rec_device(config)
    model.to(device)

    search_index = BM25SearchIndex(item_text_by_item_id, args.bm25_k1_default, args.bm25_b_default)
    ranker = GPT4RecCandidateRanker()

    val_prompts = [build_history_prompt(e.history_titles) for e in val_examples]
    val_targets = [e.target_item_int_id for e in val_examples]
    train_texts = [build_train_text(e.history_titles, e.target_title) for e in train_examples]

    # In-training style curve artifact parity
    curve_dir = config.trained_models_dir / "eval_metrics"
    curve_dir.mkdir(parents=True, exist_ok=True)
    curve_csv = str(curve_dir / f"gpt4rec_{data_type}_val_at10_train.csv")
    plot_base = f"gpt4rec_{data_type}_val_at10"
    if Path(curve_csv).exists():
        Path(curve_csv).unlink()

    val_cap = max(1, int(getattr(args, "val_eval_max_users", 512)))
    if len(val_prompts) > val_cap:
        idx = list(range(len(val_prompts)))[:val_cap]
        sampled_prompts = [val_prompts[i] for i in idx]
        sampled_targets = [val_targets[i] for i in idx]
    else:
        sampled_prompts = val_prompts
        sampled_targets = val_targets

    eval_every = max(1, int(getattr(args, "train_eval_every", 5)))

    def _epoch_eval(epoch: int):
        if not sampled_prompts:
            return
        if epoch % eval_every != 0 and epoch != 1:
            return
        default_pred = evaluate_with_bm25(
            model,
            tokenizer,
            sampled_prompts,
            sampled_targets,
            search_index,
            ranker,
            args,
            device,
            args.bm25_k1_default,
            args.bm25_b_default,
        )
        write_train_curves(curve_csv, plot_base, epoch, default_pred)

    train_generation_model(model, tokenizer, train_texts, args, device, on_epoch_end=_epoch_eval)

    best_k1, best_b = tune_bm25_params(
        model, tokenizer, val_prompts, val_targets, search_index, ranker, args, device
    )
    logging.info(f"BM25 tuned params for {data_type}: k1={best_k1}, b={best_b}")

    # Final val/test predictions
    val_final = evaluate_with_bm25(
        model, tokenizer, val_prompts, val_targets, search_index, ranker, args, device, best_k1, best_b
    )
    val_out = to_eval_output(val_final, int_to_item)

    # test prompts include val target appended when available
    test_hist = dict(user_hist)
    if not val_df.empty:
        for _, r in val_df.iterrows():
            uid = user2id.get(r["user_id"])
            iid = item2id.get(r["parent_asin"])
            if uid is None or iid is None:
                continue
            test_hist.setdefault(uid, []).append(iid)
    test_examples = build_examples_for_holdout(
        test_df, test_hist, user2id, item2id, item_text_by_asin, int_to_item
    )
    test_prompts = [build_history_prompt(e.history_titles) for e in test_examples]
    test_targets = [e.target_item_int_id for e in test_examples]
    test_final = evaluate_with_bm25(
        model, tokenizer, test_prompts, test_targets, search_index, ranker, args, device, best_k1, best_b
    )
    test_out = to_eval_output(test_final, int_to_item)

    outputs_dir = config.data_dir / "outputs" / "gpt4rec"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    val_out.to_csv(outputs_dir / f"{data_type}_val_predictions.csv", index=False)
    test_out.to_csv(outputs_dir / f"{data_type}_test_predictions.csv", index=False)
    logging.info(f"GPT4Rec pipeline complete for {data_type}. predictions saved to {outputs_dir}")

    return val_out, test_out

