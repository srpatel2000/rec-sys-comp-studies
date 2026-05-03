"""This file contains the helper functions to compute evaluation metrics for the recommendation system models."""

import numpy as np
import pandas as pd 
import math
import os

from utils import createLineChart

def evalPrecisionatK(predictions, actuals, step_size):
    """Compute Precision@K for the given predictions and true labels."""

    # TODO: save this to a csv file instead of printing to console
    # create a chart with eval over multiple values of k (complete this for all of the metrics in this file)

    results = {}

    actual_set = set(actuals)
    for i in range(step_size, len(predictions) + 1, step_size):
        pred_set = set(predictions[:i])
        precision = len(pred_set.intersection(actual_set)) / len(pred_set)
        results[i] = precision
    
    return results


def evalRecallatK(predictions, actuals, step_size):
    """Compute Recall@K for the given predictions and true labels."""

    results = {}

    actual_set = set(actuals)
    for i in range(step_size, len(predictions) + 1, step_size):
        pred_set = set(predictions[:i])
        recall = len(pred_set.intersection(actual_set)) / len(actual_set)
        results[i] = recall

    return results


def evalNDCGatK(predictions, actuals, step_size, rel_scores, ideal_rel_scores):
    """Compute NDCG@K for the given predictions and true labels."""

    results = {}

    for i in range(step_size, len(predictions) + 1, step_size):

        # step 1: compute dcg@k
        dcg_k = 0
        for j in range(0, i):
            dcg_k += rel_scores[j] / math.log2((j+1) + 1) # j+1 because of 0-indexing, +1 for log base change

        # step 2: compute idcg@k
        idcg_k = 0
        for j in range(0, i):
            idcg_k += ideal_rel_scores[j] / math.log2((j+1) + 1)
        
        # step 3: compute NDCG@K
        ndcg_k = dcg_k / idcg_k 
        results[i] = ndcg_k

    return results


def evalPipeline(predictions, actuals, step_size, rel_scores, ideal_rel_scores):
    """Compute all evaluation metrics for the given predictions and true labels."""

    precision_results = evalPrecisionatK(predictions, actuals, step_size)
    recall_results = evalRecallatK(predictions, actuals, step_size)
    ndcg_results = evalNDCGatK(predictions, actuals, step_size, rel_scores, ideal_rel_scores)

    # collect metrics
    metrics = {
        "precision": precision_results,
        "recall": recall_results,
        "ndcg": ndcg_results,
    }

    # ensure output directory exists
    out_dir = os.path.join("trained_models", "eval_metrics")
    os.makedirs(out_dir, exist_ok=True)

    num_preds = len(predictions)

    # create a line chart for each metric
    for metric_name, results in metrics.items():
        x = list(results.keys())
        y = list(results.values())
        title = f"{metric_name.capitalize()}@K across top-{num_preds} predictions (step={step_size})"
        x_label = "K"
        y_label = metric_name.capitalize()
        output_path = os.path.join(out_dir, f"{metric_name}_{num_preds}preds_ss{step_size}.png")
        createLineChart(x, y, title, x_label, y_label, output_path)
