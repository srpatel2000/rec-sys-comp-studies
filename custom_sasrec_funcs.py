"""This file contains all of the custom functions to be able to feed in my data
to the SASRec model implementation from https://github.com/kang205/SASRec"""

import json
import pandas as pd
import numpy as np
import time
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silences info and warning logs
import tensorflow.compat.v1 as tf

from config import GlobalConfig, SASRecModelConfig
from eval_metrics import K_EVAL, append_train_eval_row, macro_metrics_at_k_items, refresh_train_eval_plots
from sas_rec.sampler import WarpSampler
from sas_rec.model import Model

sasrec_config = SASRecModelConfig()


def buildIDMappings(data):
    """Build user and item ID mappings to convert raw IDs to integer indices
    to be later fed into tensor flow embedding_lookup."""

    logging.info("Building ID mappings for users and items...")

    unique_users = data["user_id"].unique()
    unique_items = data["parent_asin"].unique()

    user_int_id = {user: idx for idx, user in enumerate(unique_users,start=1)}  # start=1 to reserve 0 for padding
    item_int_id = {item: idx for idx, item in enumerate(unique_items,start=1)} 

    return user_int_id, item_int_id


def perUserSequence(data, user_int_id, item_int_id):
    """For each user, create a sequence of item interactions sorted by timestamp."""

    logging.info("Creating per-user interaction sequences...")

    data["user_int_id"] = data["user_id"].map(user_int_id)
    data["item_int_id"] = data["parent_asin"].map(item_int_id)

    user_sequences = {}
    for user_id, group in data.groupby("user_int_id"):
        sorted_group = group.sort_values("timestamp")
        user_sequences[user_id] = sorted_group["item_int_id"].tolist()

    return user_sequences


def trainingBatches(user_sequences, usernum, itemnum, batch_size, maxlen): 
    """Create training batches using sas_rec's WarpSampler (multiprocess background sampler).
    Returns the sampler object — call sampler.next_batch() in your training loop."""

    logging.info("Creating training batches with WarpSampler...")

    sampler = WarpSampler(user_sequences, usernum, itemnum, batch_size=batch_size, maxlen=maxlen, n_workers=1)
    return sampler


def buildSASRecModel(usernum, itemnum, args):
    """Build the SASRec model using the provided user and item counts and hyperparameters."""

    logging.info("Building SASRec model...")

    model = Model(usernum, itemnum, args)

    return model


def trainSASRec(
    sess,
    model,
    sampler,
    user_sequences,
    args,
    val_eval_df=None,
    itemnum=None,
    maxlen=None,
    data_type="",
    curve_dir=None,
):
    """Train the SASRec model using the training batches from the WarpSampler.
    Creds to: https://github.com/recommenders-team/recommenders/blob/main/examples/00_quick_start/sasrec_amazon.ipynb"""

    logging.info("Training SASRec model...")

    num_batches = max(1, len([u for u in user_sequences if len(user_sequences[u]) > 1]) // args.batch_size)
    train_start = time.perf_counter()

    curve_csv = None
    plot_base = None
    eval_every = max(1, int(getattr(args, "train_eval_every", 5)))
    val_cap = max(1, int(getattr(args, "val_eval_max_users", 512)))

    if curve_dir and data_type and val_eval_df is not None and len(val_eval_df) > 0 and itemnum and maxlen:
        os.makedirs(curve_dir, exist_ok=True)
        curve_csv = os.path.join(curve_dir, f"sasrec_{data_type}_val_at10_train.csv")
        plot_base = f"sasrec_{data_type}_val_at10"
        if os.path.isfile(curve_csv):
            os.remove(curve_csv)
    elif curve_dir and data_type:
        logging.info("Skipping in-training val @10 curve: no val rows for this dataset.")

    for epoch in range(1, args.num_epochs + 1):
        for _ in range(num_batches):
            u, seq, pos, neg = sampler.next_batch()
            sess.run(model.train_op, {
                model.u: u,
                model.input_seq: seq,
                model.pos: pos,
                model.neg: neg,
                model.is_training: True,
            })
        if epoch % eval_every == 0 or epoch == 1:
            elapsed = (time.perf_counter() - train_start) / 60
            logging.info(f"epoch {epoch}/{args.num_epochs} ({elapsed:.1f} min)")

        if curve_csv and (epoch % eval_every == 0 or epoch == 1):
            sample = val_eval_df
            if len(sample) > val_cap:
                sample = sample.sample(val_cap, random_state=42)
            pred_df = predictForUsers(
                model, sess, user_sequences, sample, itemnum, maxlen, verbose=False,
            )
            m = macro_metrics_at_k_items(pred_df, k=K_EVAL)
            if m is not None:
                append_train_eval_row(curve_csv, epoch, m)
                refresh_train_eval_plots(curve_csv, plot_base)
                logging.info(
                    f"val @10: P={m['precision']:.4f} R={m['recall']:.4f} NDCG={m['ndcg']:.4f} (n={m['n_users']})"
                )

    sampler.close()
    train_minutes = (time.perf_counter() - train_start) / 60
    logging.info(f"training complete in {train_minutes:.2f} minutes")


def predictForUsers(
    model, sess, user_sequences, target_df, itemnum, maxlen, pred_batch_size=256, verbose=True,
):
    """Score each target user's ground-truth item plus sampled negatives.
    target_df must have columns: user_int_id, item_int_id (the ground-truth item)."""

    if verbose:
        logging.info("Predicting for users...")

    predictions = []

    valid_users = []
    seqs = []
    ground_truth_items = []

    for _, row in target_df.iterrows():
        uid = int(row["user_int_id"])
        gt_item = int(row["item_int_id"])
        history = user_sequences.get(uid, [])
        if len(history) == 0:
            continue

        seq = np.zeros([maxlen], dtype=np.int32)
        start = max(0, len(history) - maxlen)
        trimmed = history[start:]
        seq[maxlen - len(trimmed):] = trimmed

        valid_users.append(uid)
        seqs.append(seq)
        ground_truth_items.append(gt_item)

    # score in batches — score all items, then extract ground-truth rank
    all_items = np.arange(1, itemnum + 1)

    for batch_start in range(0, len(valid_users), pred_batch_size):
        batch_end = min(batch_start + pred_batch_size, len(valid_users))
        u_batch = valid_users[batch_start:batch_end]
        seq_batch = np.array(seqs[batch_start:batch_end])
        gt_batch = ground_truth_items[batch_start:batch_end]

        scores = model.predict(sess, u_batch, seq_batch, all_items)

        for i, uid in enumerate(u_batch):
            gt_item = gt_batch[i]
            gt_score = float(scores[i][gt_item - 1])  # items are 1-indexed
            rank = int((scores[i] >= gt_score).sum())  # rank of ground-truth item
            order = np.argsort(-scores[i])[: sasrec_config.num_preds]
            top_k_items = (order + 1).tolist()
            top_k_scores = scores[i][order].astype(float).tolist()

            predictions.append({
                "user_int_id": uid,
                "ground_truth_item": gt_item,
                "ground_truth_score": gt_score,
                "ground_truth_rank": rank,
                "top_k_items": top_k_items,
                "top_k_scores": top_k_scores,
            })

    return pd.DataFrame(predictions)


def predictVal(model, sess, user_sequences, val_df, itemnum, maxlen):
    """predict using only train sequences as history."""

    logging.info("Predicting for validation set...")

    return predictForUsers(model, sess, user_sequences, val_df, itemnum, maxlen)


def predictTest(model, sess, user_sequences, val_df, test_df, itemnum, maxlen):
    """predict with val item appended to train sequence."""

    logging.info("Predicting for test set with val item appended to history...")

    test_sequences = {}
    for uid in test_df["user_int_id"].unique():
        history = list(user_sequences.get(uid, []))
        user_val = val_df[val_df["user_int_id"] == uid]
        if len(user_val) > 0:
            history.append(int(user_val.iloc[0]["item_int_id"]))
        test_sequences[uid] = history

    return predictForUsers(model, sess, test_sequences, test_df, itemnum, maxlen)


def convertIDBackToRaw(predictions_df, user_int_id, item_int_id):
    """convert integer IDs back to raw user_id and parent_asin strings."""

    # invert the mappings
    int_to_user = {v: k for k, v in user_int_id.items()}
    int_to_item = {v: k for k, v in item_int_id.items()}

    predictions_df["user_id"] = predictions_df["user_int_id"].map(int_to_user)
    predictions_df["ground_truth_asin"] = predictions_df["ground_truth_item"].map(int_to_item)
    predictions_df["top_k_asins"] = predictions_df["top_k_items"].apply(
        lambda ids: json.dumps([int_to_item.get(int(i)) for i in ids]),
    )
    predictions_df["ground_truth_asins"] = predictions_df["ground_truth_asin"].apply(
        lambda x: json.dumps([x]),
    )
    predictions_df["top_k_scores_list"] = predictions_df["top_k_scores"].apply(json.dumps)

    predictions_df.drop(columns=["top_k_items", "top_k_scores"], inplace=True)
    predictions_df.rename(columns={"top_k_scores_list": "top_k_scores"}, inplace=True)

    return predictions_df


def runSASRecPipeline(data_type="dense"):
    """run the full SASRec pipeline: build ID mappings, create user sequences, build model, train, and predict.
    data_type: 'dense' or 'cold_start' to specify which dataset to run on."""

    config = GlobalConfig()
    args = config.model_namespace("sasrec")

    logging.info(f"Running SASRec pipeline for {data_type} dataset...")

    # load train/val/test CSVs
    train_df = pd.read_csv(config.data_dir / "train" / f"{data_type}_train.csv")
    val_df = pd.read_csv(config.data_dir / "val" / f"{data_type}_val.csv")
    test_df = pd.read_csv(config.data_dir / "test" / f"{data_type}_test.csv")

    # step 1: build ID mappings from all data
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    user_int_id, item_int_id = buildIDMappings(all_data)
    usernum = len(user_int_id)
    itemnum = len(item_int_id)

    # step 2: build per-user train sequences
    user_sequences = perUserSequence(train_df.copy(), user_int_id, item_int_id)

    # encode val/test dataframes with int IDs
    val_df["user_int_id"] = val_df["user_id"].map(user_int_id)
    val_df["item_int_id"] = val_df["parent_asin"].map(item_int_id)
    test_df["user_int_id"] = test_df["user_id"].map(user_int_id)
    test_df["item_int_id"] = test_df["parent_asin"].map(item_int_id)

    # step 3: build model and initialize session BEFORE starting sampler
    tf.reset_default_graph()
    model = buildSASRecModel(usernum, itemnum, args)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    logging.info("Session created and variables initialized.")

    # step 4: create sampler AFTER session is ready (avoids multiprocessing deadlock on MacOS)
    sampler = trainingBatches(user_sequences, usernum, itemnum, args.batch_size, args.maxlen)

    curve_dir = str(config.trained_models_dir / "eval_metrics")
    trainSASRec(
        sess,
        model,
        sampler,
        user_sequences,
        args,
        val_eval_df=val_df,
        itemnum=itemnum,
        maxlen=args.maxlen,
        data_type=data_type,
        curve_dir=curve_dir,
    )

    # step 6: val predictions
    val_preds = predictVal(model, sess, user_sequences, val_df, itemnum, args.maxlen)
    val_preds = convertIDBackToRaw(val_preds, user_int_id, item_int_id)

    # step 7: test predictions
    test_preds = predictTest(model, sess, user_sequences, val_df, test_df, itemnum, args.maxlen)
    test_preds = convertIDBackToRaw(test_preds, user_int_id, item_int_id)

    outputs_dir = config.data_dir / "outputs" / "sasrec"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    val_preds.to_csv(outputs_dir / f"{data_type}_val_predictions.csv", index=False)
    test_preds.to_csv(outputs_dir / f"{data_type}_test_predictions.csv", index=False)

    sess.close()
    logging.info(f"SASRec pipeline complete for {data_type}. predictions saved to {outputs_dir}")

    return val_preds, test_preds 