"""Helper functions to compute evaluation metrics for recommendation outputs."""

import csv
import json
import math
import os
import matplotlib.pyplot as plt

import pandas as pd

from utils import createLineChart, createLineChartMulti

K_EVAL = 10
# Prepended to curve CSVs, per-epoch PNGs, combined PNGs, and post-hoc eval snapshots.
EVAL_ARTIFACT_PREFIX = "v3"


def eval_artifact_path(path):
    """Map a logical artifact path to the on-disk name for the current experiment batch."""

    path = os.path.expanduser(str(path))
    d, b = os.path.split(path)
    if not b:
        return path
    if b.startswith(EVAL_ARTIFACT_PREFIX):
        return path
    return os.path.join(d if d else ".", EVAL_ARTIFACT_PREFIX + b)


def parse_list_cell(cell):
    if pd.isna(cell):
        return None
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        return json.loads(cell)
    return None


def usable_prediction_rows(predictions_df):
    """Each row: (ranked_asins list, frozenset of relevant ASINs)."""

    rows = []
    for _, row in predictions_df.iterrows():
        ranked = parse_list_cell(row["top_k_asins"])
        gt = parse_list_cell(row["ground_truth_asins"])
        if not ranked or not gt or len(ranked) < K_EVAL:
            continue
        gset = frozenset(x for x in gt if x is not None)
        if not gset:
            continue
        rows.append((ranked, gset))
    return rows


def rows_from_int_predictions(pred_df, k=K_EVAL):
    """In-memory pred frame: top_k_items list, ground_truth_item int."""

    rows = []
    for _, row in pred_df.iterrows():
        ranked = row["top_k_items"]
        if not isinstance(ranked, (list, tuple)) or len(ranked) < k:
            continue
        gt = int(row["ground_truth_item"])
        rows.append((list(ranked), frozenset([gt])))
    return rows


def macro_at_k(rows, k):
    """Return (precision, recall, ndcg) macro means or None if no rows.

    Precision/recall use **distinct** relevant items found in the top-`k` list
    (set intersection), so duplicate item ids in the ranking cannot inflate
    recall above 1. DCG credits each relevant item at its **first** rank only.
    """

    if not rows:
        return None
    n = len(rows)
    prec = 0.0
    rec = 0.0
    ndcg = 0.0
    for ranked, gset in rows:
        clipped = ranked[:k]
        rel_found = len(set(clipped) & gset)
        prec += rel_found / k
        rec += rel_found / len(gset)

        dcg = 0.0
        credited = set()
        for j, item in enumerate(clipped):
            if item in gset and item not in credited:
                credited.add(item)
                dcg += 1.0 / math.log2(j + 2)
        m = min(k, len(gset))
        idcg = sum(1.0 / math.log2(j + 2) for j in range(m))
        ndcg += (dcg / idcg) if idcg > 0 else 0.0

    return prec / n, rec / n, ndcg / n


def macro_metrics_at_k_items(pred_df, k=K_EVAL):
    """Macro P/R/NDCG@k from a predictForUsers DataFrame (int item ids). Returns dict or None."""

    rows = rows_from_int_predictions(pred_df, k=k)
    out = macro_at_k(rows, k)
    if out is None:
        return None
    p, r, n = out
    return {"precision": p, "recall": r, "ndcg": n, "k": k, "n_users": len(rows)}


def macro_metrics_at_k_asins(predictions_df, k=K_EVAL):
    """Same as macro_metrics_at_k_items for exported CSV (JSON ASIN columns)."""

    rows = usable_prediction_rows(predictions_df)
    out = macro_at_k(rows, k)
    if out is None:
        return None
    p, r, n = out
    return {"precision": p, "recall": r, "ndcg": n, "k": k, "n_users": len(rows)}


def append_train_eval_row(csv_path, epoch, metrics_dict):
    """Append one epoch row; write header if new file."""

    csv_path = eval_artifact_path(csv_path)
    new_file = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["epoch", "precision_at_10", "recall_at_10", "ndcg_at_10", "n_users"])
        w.writerow([
            epoch,
            metrics_dict["precision"],
            metrics_dict["recall"],
            metrics_dict["ndcg"],
            metrics_dict.get("n_users", ""),
        ])


def refresh_train_eval_plots(csv_path, plot_basename):
    """Re-read training curve CSV and overwrite three epoch vs metric charts."""

    csv_path = eval_artifact_path(csv_path)
    if not plot_basename.startswith(EVAL_ARTIFACT_PREFIX):
        plot_basename = f"{EVAL_ARTIFACT_PREFIX}{plot_basename}"
    if not os.path.isfile(csv_path):
        return
    df = pd.read_csv(csv_path)
    if df.empty or "epoch" not in df.columns:
        return
    out_dir = os.path.dirname(csv_path)
    x = df["epoch"].tolist()
    for col, name in [
        ("precision_at_10", "precision"),
        ("recall_at_10", "recall"),
        ("ndcg_at_10", "ndcg"),
    ]:
        if col not in df.columns:
            continue
        y = df[col].tolist()
        title = f"{name.capitalize()}@10 vs epoch ({os.path.basename(plot_basename)})"
        path = os.path.join(out_dir, f"{plot_basename}_{name}_at10_vs_epoch.png")
        createLineChart(x, y, title, "Epoch", name.capitalize(), path)


def refresh_combined_dense_cold_train_eval_plots(curve_dir, model_prefix):
    """Overlay dense vs cold_start val@10 training curves (one PNG per metric).

    Expects CSVs: ``v2_{model_prefix}_dense_val_at10_train.csv`` and
    ``v2_{model_prefix}_cold_start_val_at10_train.csv`` under ``curve_dir``.
    Writes: ``v2_{model_prefix}_val_at10_combined_<metric>_at10_vs_epoch.png``.
    """
    curve_dir = os.path.expanduser(str(curve_dir))
    if not curve_dir or not os.path.isdir(curve_dir):
        return

    paths_labels = [
        (
            eval_artifact_path(os.path.join(curve_dir, f"{model_prefix}_dense_val_at10_train.csv")),
            "dense",
        ),
        (
            eval_artifact_path(os.path.join(curve_dir, f"{model_prefix}_cold_start_val_at10_train.csv")),
            "cold_start",
        ),
    ]
    metric_cols = [
        ("precision_at_10", "precision", "Precision"),
        ("recall_at_10", "recall", "Recall"),
        ("ndcg_at_10", "ndcg", "NDCG"),
    ]

    for col, short, y_title in metric_cols:
        series = []
        for csv_path, split_label in paths_labels:
            if not os.path.isfile(csv_path):
                continue
            df = pd.read_csv(csv_path)
            if df.empty or "epoch" not in df.columns or col not in df.columns:
                continue
            series.append({
                "x": df["epoch"].tolist(),
                "y": df[col].tolist(),
                "label": split_label,
            })
        if len(series) == 0:
            continue
        title = f"{y_title}@10 vs epoch ({model_prefix}, val)"
        out_path = os.path.join(
            curve_dir,
            f"{EVAL_ARTIFACT_PREFIX}{model_prefix}_val_at10_combined_{short}_at10_vs_epoch.png",
        )
        createLineChartMulti(series, title, "Epoch", y_title, out_path)


def evalPipeline(predictions_df, export_prefix="", k=K_EVAL):
    """Post-hoc: macro P/R/NDCG@k on saved predictions CSV; one-row summary CSV only."""

    m = macro_metrics_at_k_asins(predictions_df, k=k)
    out_dir = os.path.join("trained_models", "eval_metrics")
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{export_prefix}_" if export_prefix else ""

    if m is None:
        return

    snap = os.path.join(out_dir, f"{EVAL_ARTIFACT_PREFIX}{prefix}eval_at10_posthoc.csv")
    with open(snap, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "precision_macro", "recall_macro", "ndcg_macro", "n_users"])
        w.writerow([m["k"], m["precision"], m["recall"], m["ndcg"], m["n_users"]])
