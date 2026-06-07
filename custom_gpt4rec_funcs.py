"""Custom GPT4Rec pipeline over existing split CSVs."""

import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from custom_sasrec_funcs import buildIDMappings

from config import GlobalConfig, GPT4RecModelConfig
from eval_metrics import eval_artifact_path
from gpt4rec.data import (
    build_examples_for_holdout,
    build_examples_for_training,
    build_item_texts,
    build_user_histories,
)
from gpt4rec.model import GPT4RecCandidateRanker, GPT4RecGenerationModel
from gpt4rec.prompting import build_history_prompt, build_train_text
from gpt4rec.runtime_tracking import GPT4RecRuntimeTracker
from gpt4rec.search import BM25SearchIndex
from gpt4rec.train_eval import evaluate_with_bm25, train_generation_model, tune_bm25_params, write_train_curves

from gpt4rec.build_raptor import (
    GPT4RecSummarizationModel,
    SBertEmbeddingModel,
    build_raptor_tree,
    catalog_document_for_raptor,
    initialize_raptor,
    load_raptor_tree,
    save_raptor_tree,
    summary_to_items_from_tree,
)

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
    tracker = GPT4RecRuntimeTracker(data_type)

    logging.info(f"Running GPT4Rec pipeline for {data_type} dataset...")

    t0 = time.perf_counter()
    train_df = pd.read_csv(config.data_dir / "train" / f"{data_type}_train.csv")
    val_df = pd.read_csv(config.data_dir / "val" / f"{data_type}_val.csv")
    test_df = pd.read_csv(config.data_dir / "test" / f"{data_type}_test.csv")
    tracker.log(
        "io_load_train_val_test_csv",
        time.perf_counter() - t0,
        detail=f"rows train={len(train_df)} val={len(val_df)} test={len(test_df)}",
    )

    t0 = time.perf_counter()
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    user2id, item2id = buildIDMappings(all_data)  # use same methodology as SASRec to build mappings
    int_to_item = int_to_asin_map(item2id)
    tracker.log(
        "data_concat_and_id_mappings",
        time.perf_counter() - t0,
        detail=f"n_users={len(user2id)} n_items={len(item2id)}",
    )

    t0 = time.perf_counter()
    item_text_by_asin = build_item_texts(all_data)
    item_text_by_item_id = {
        item2id[asin]: txt for asin, txt in item_text_by_asin.items() if asin in item2id
    }
    tracker.log(
        "data_build_item_title_texts",
        time.perf_counter() - t0,
        detail=f"n_asin_texts={len(item_text_by_asin)}",
    )

    t0 = time.perf_counter()
    user_hist = build_user_histories(train_df, user2id, item2id)
    train_examples = build_examples_for_training(train_df, user2id, item2id, item_text_by_asin)
    val_examples = build_examples_for_holdout(
        val_df, user_hist, user2id, item2id, item_text_by_asin, int_to_item
    )
    tracker.log(
        "data_user_histories_and_train_val_examples",
        time.perf_counter() - t0,
        detail=f"n_train_examples={len(train_examples)} n_val_examples={len(val_examples)}",
    )

    t0 = time.perf_counter()
    tokenizer = GPT2Tokenizer.from_pretrained(args.hf_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only: batch generation must pad on the left so the last real token aligns.
    tokenizer.padding_side = "left"
    tracker.log(
        "hf_tokenizer_from_pretrained",
        time.perf_counter() - t0,
        detail=str(args.hf_model_name),
    )

    t0 = time.perf_counter()
    lm = GPT2LMHeadModel.from_pretrained(args.hf_model_name)

    class TmpCfg:
        pass

    tmp = TmpCfg()
    tmp.num_users = len(user2id) + 1
    tmp.num_items = len(item2id) + 1
    tmp.vocab_size = tokenizer.vocab_size
    tmp.n_embd = lm.config.n_embd # in order to deal with shape mismatch between lm and gpt4rec model
    tmp.initializer_range = args.initializer_range

    model = GPT4RecGenerationModel(tmp, lm)
    tracker.log(
        "hf_lm_from_pretrained_and_wrap_gpt4rec_generation_model",
        time.perf_counter() - t0,
        detail=str(args.hf_model_name),
    )

    t0 = time.perf_counter()
    device = resolve_gpt4rec_device(config)
    model.to(device)
    tracker.log(
        "device_resolve_and_model_to_device",
        time.perf_counter() - t0,
        detail=str(device),
    )

    t0 = time.perf_counter()
    search_index = BM25SearchIndex(item_text_by_item_id, args.bm25_k1_default, args.bm25_b_default)
    ranker = GPT4RecCandidateRanker()
    tracker.log(
        "bm25_index_build_and_ranker_init",
        time.perf_counter() - t0,
        detail=f"n_indexed_items={len(search_index.doc_tokens)}",
    )

    # Retrieval: per query, take top-N layer-1 summaries (default 15),
    # union all items under them, then BM25 on that shortlist.
    raptor_model_config = GPT4RecModelConfig()
    t0 = time.perf_counter()
    raptor_path = config.trained_models_dir / "raptor" / f"raptor_catalog_{data_type}.pkl"
    raptor_prefilter_k = int(getattr(args, "raptor_prefilter_k", 50))
    raptor_top_summaries = int(getattr(args, "raptor_top_summaries", 15))
    raptor_embed = SBertEmbeddingModel()
    if raptor_path.exists():
        print(f"Loading RAPTOR tree from {raptor_path}", flush=True)
        RA = load_raptor_tree(str(raptor_path), raptor_embed, raptor_prefilter_k)
    else:
        RA = initialize_raptor(
            GPT4RecSummarizationModel(raptor_model_config),
            raptor_embed,
            leaf_top_k=raptor_prefilter_k,
        )
        document = catalog_document_for_raptor(item_text_by_item_id)
        build_raptor_tree(RA, document)
        save_raptor_tree(RA, raptor_path)
    summary_to_items = summary_to_items_from_tree(RA)
    n_summaries = len(summary_to_items)
    summary_sizes = [len(v) for v in summary_to_items.values()]
    avg_summary_size = (sum(summary_sizes) / n_summaries) if n_summaries else 0
    if not summary_to_items:
        logging.warning(
            "RAPTOR layer-1 summaries are empty; BM25 will search the full catalog. "
            "Delete %s and rerun to rebuild the tree with item tags.",
            raptor_path,
        )
    tracker.log(
        "raptor_initialize_and_build_tree",
        time.perf_counter() - t0,
        detail=(
            f"n_summary_nodes={n_summaries} "
            f"avg_items_per_summary={avg_summary_size:.1f} "
            f"top_summaries={raptor_top_summaries}"
        ),
    )
    print(
        f"[GPT4Rec pipeline:{data_type}] RAPTOR enabled — "
        f"layer-1 prefilter (top_summaries={raptor_top_summaries}, tr_top_k={raptor_prefilter_k}).",
        flush=True,
    )

    t0 = time.perf_counter()
    val_prompts = [build_history_prompt(e.history_titles) for e in val_examples]
    val_targets = [e.target_item_int_id for e in val_examples]
    train_texts = [build_train_text(e.history_titles, e.target_title) for e in train_examples]
    tracker.log(
        "prompting_build_history_prompts_train_texts",
        time.perf_counter() - t0,
        detail=f"n_val_prompts={len(val_prompts)} n_train_texts={len(train_texts)}",
    )

    # In-training style curve artifact parity
    t_curve = time.perf_counter()
    curve_dir = config.trained_models_dir / "eval_metrics"
    curve_dir.mkdir(parents=True, exist_ok=True)
    curve_csv = str(curve_dir / f"gpt4rec_{data_type}_val_at10_train.csv")
    plot_base = f"gpt4rec_{data_type}_val_at10"

    artifact_csv = eval_artifact_path(curve_csv)
    if Path(artifact_csv).exists():
        Path(artifact_csv).unlink()

    val_cap = max(1, int(getattr(args, "val_eval_max_users", 512)))
    if len(val_prompts) > val_cap:
        idx = list(range(len(val_prompts)))[:val_cap]
        sampled_prompts = [val_prompts[i] for i in idx]
        sampled_targets = [val_targets[i] for i in idx]
    else:
        sampled_prompts = val_prompts
        sampled_targets = val_targets
    tracker.log(
        "in_train_eval_metrics_init_and_val_subsample",
        time.perf_counter() - t_curve,
        detail=f"val_cap={val_cap} sampled_val_users={len(sampled_prompts)}",
    )

    eval_every = max(1, int(getattr(args, "train_eval_every", 5)))
    num_epochs = max(1, int(getattr(args, "num_epochs", 1)))
    print(
        f"[GPT4Rec pipeline:{data_type}] LM training: num_epochs={num_epochs}, "
        f"train_eval_every={eval_every} (in-training val runs on epoch 1 and when epoch % {eval_every} == 0). "
        f"Sampled val users for in-training eval: {len(sampled_prompts)} / {len(val_prompts)}.",
        flush=True,
    )

    def epoch_eval(epoch: int):
        if not sampled_prompts:
            return # no sampled prompts for in-training eval
        if epoch % eval_every != 0 and epoch != 1:
            return # not time to evaluate
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
            RA=RA,
            summary_to_items=summary_to_items,
            top_summaries=raptor_top_summaries,
            progress_label=f"in-train-val-epoch-{epoch}",
            runtime_tracker=tracker,
            timing_detail=f"in_train_val_epoch_{epoch}",
        )
        t_w = time.perf_counter()
        write_train_curves(curve_csv, plot_base, epoch, default_pred)
        tracker.log(
            "in_train_write_train_curves_and_plots",
            time.perf_counter() - t_w,
            epoch=epoch,
            detail="eval_metrics append + png refresh",
        )

    print(f"[GPT4Rec pipeline:{data_type}] starting LM fine-tuning...", flush=True)
    train_generation_model(
        model, tokenizer, train_texts, args, device, on_epoch_end=epoch_eval, runtime_tracker=tracker
    )
    print(
        f"[GPT4Rec pipeline:{data_type}] LM fine-tuning finished; starting BM25 hyperparameter search...",
        flush=True,
    )

    # removing grid search due to time constraints
    # best_k1, best_b = tune_bm25_params(
    #     model,
    #     tokenizer,
    #     val_prompts,
    #     val_targets,
    #     search_index,
    #     ranker,
    #     args,
    #     device,
    #     runtime_tracker=tracker,
    #     RA=RA,
    #     summary_to_items=summary_to_items,
    #     top_summaries=raptor_top_summaries,
    # )
    # logging.info(f"BM25 tuned params for {data_type}: k1={best_k1}, b={best_b}")

    best_k1 = args.bm25_k1_default
    best_b = args.bm25_b_default

    # Final val/test predictions
    print(
        f"[GPT4Rec pipeline:{data_type}] final full validation eval "
        f"({len(val_prompts)} users, tuned k1={best_k1}, b={best_b})...",
        flush=True,
    )
    val_final = evaluate_with_bm25(
        model,
        tokenizer,
        val_prompts,
        val_targets,
        search_index,
        ranker,
        args,
        device,
        best_k1,
        best_b,
        RA=RA,
        summary_to_items=summary_to_items,
        top_summaries=raptor_top_summaries,
        progress_label=f"final-val-{data_type}",
        runtime_tracker=tracker,
        timing_detail="final_full_validation",
    )
    print(f"[GPT4Rec pipeline:{data_type}] final validation eval complete.", flush=True)
    val_out = to_eval_output(val_final, int_to_item)

    t_testprep = time.perf_counter()
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
    tracker.log(
        "test_split_prompting_build_examples_and_prompts",
        time.perf_counter() - t_testprep,
        detail=f"n_test_examples={len(test_examples)}",
    )
    print(
        f"[GPT4Rec pipeline:{data_type}] final test eval ({len(test_prompts)} users)...",
        flush=True,
    )
    test_final = evaluate_with_bm25(
        model,
        tokenizer,
        test_prompts,
        test_targets,
        search_index,
        ranker,
        args,
        device,
        best_k1,
        best_b,
        RA=RA,
        summary_to_items=summary_to_items,
        top_summaries=raptor_top_summaries,
        progress_label=f"final-test-{data_type}",
        runtime_tracker=tracker,
        timing_detail="final_full_test",
    )
    print(f"[GPT4Rec pipeline:{data_type}] final test eval complete.", flush=True)
    test_out = to_eval_output(test_final, int_to_item)

    outputs_dir = config.data_dir / "outputs" / "gpt4rec"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    val_path = outputs_dir / f"{data_type}_val_predictions.csv"
    test_path = outputs_dir / f"{data_type}_test_predictions.csv"
    print(f"[GPT4Rec pipeline:{data_type}] writing {val_path} and {test_path} ...", flush=True)
    t_io = time.perf_counter()
    val_out.to_csv(val_path, index=False)
    test_out.to_csv(test_path, index=False)
    tracker.log(
        "io_write_val_test_prediction_csvs",
        time.perf_counter() - t_io,
        detail=str(outputs_dir),
    )

    timing_csv = tracker.save(config.trained_models_dir)
    logging.info(f"GPT4Rec pipeline complete for {data_type}. predictions saved to {outputs_dir}")
    logging.info("GPT4Rec runtime component CSV: %s", timing_csv)

    return val_out, test_out

