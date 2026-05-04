"""Helper functions to compute evaluation metrics for recommendation outputs."""

import csv
import json
import math
import os

import pandas as pd

from utils import createLineChart

K_EVAL = 10


def _parse_list_cell(cell):
    if pd.isna(cell):
        return None
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        return json.loads(cell)
    return None


def _usable_prediction_rows(predictions_df):
    """Each row: (ranked_asins list, frozenset of relevant ASINs)."""

    rows = []
    for _, row in predictions_df.iterrows():
        ranked = _parse_list_cell(row["top_k_asins"])
        gt = _parse_list_cell(row["ground_truth_asins"])
        if not ranked or not gt or len(ranked) < K_EVAL:
            continue
        gset = frozenset(x for x in gt if x is not None)
        if not gset:
            continue
        rows.append((ranked, gset))
    return rows


def _rows_from_int_predictions(pred_df, k=K_EVAL):
    """In-memory pred frame: top_k_items list, ground_truth_item int."""

    rows = []
    for _, row in pred_df.iterrows():
        ranked = row["top_k_items"]
        if not isinstance(ranked, (list, tuple)) or len(ranked) < k:
            continue
        gt = int(row["ground_truth_item"])
        rows.append((list(ranked), frozenset([gt])))
    return rows


def _macro_at_k(rows, k):
    """Return (precision, recall, ndcg) macro means or None if no rows."""

    if not rows:
        return None
    n = len(rows)
    prec = 0.0
    rec = 0.0
    ndcg = 0.0
    for ranked, gset in rows:
        clipped = ranked[:k]
        hits = sum(1 for x in clipped if x in gset)
        prec += hits / k
        rec += hits / len(gset)

        dcg = 0.0
        for j, item in enumerate(clipped):
            rel = 1.0 if item in gset else 0.0
            dcg += rel / math.log2(j + 2)
        m = min(k, len(gset))
        idcg = sum(1.0 / math.log2(j + 2) for j in range(m))
        ndcg += (dcg / idcg) if idcg > 0 else 0.0

    return prec / n, rec / n, ndcg / n


def macro_metrics_at_k_items(pred_df, k=K_EVAL):
    """Macro P/R/NDCG@k from a predictForUsers DataFrame (int item ids). Returns dict or None."""

    rows = _rows_from_int_predictions(pred_df, k=k)
    out = _macro_at_k(rows, k)
    if out is None:
        return None
    p, r, n = out
    return {"precision": p, "recall": r, "ndcg": n, "k": k, "n_users": len(rows)}


def macro_metrics_at_k_asins(predictions_df, k=K_EVAL):
    """Same as macro_metrics_at_k_items for exported CSV (JSON ASIN columns)."""

    rows = _usable_prediction_rows(predictions_df)
    out = _macro_at_k(rows, k)
    if out is None:
        return None
    p, r, n = out
    return {"precision": p, "recall": r, "ndcg": n, "k": k, "n_users": len(rows)}


def append_train_eval_row(csv_path, epoch, metrics_dict):
    """Append one epoch row; write header if new file."""

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


def evalPipeline(predictions_df, export_prefix="", k=K_EVAL):
    """Post-hoc: macro P/R/NDCG@k on saved predictions CSV; one-row summary CSV only."""

    m = macro_metrics_at_k_asins(predictions_df, k=k)
    out_dir = os.path.join("trained_models", "eval_metrics")
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{export_prefix}_" if export_prefix else ""

    if m is None:
        return

    snap = os.path.join(out_dir, f"{prefix}eval_at10_posthoc.csv")
    with open(snap, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "precision_macro", "recall_macro", "ndcg_macro", "n_users"])
        w.writerow([m["k"], m["precision"], m["recall"], m["ndcg"], m["n_users"]])
