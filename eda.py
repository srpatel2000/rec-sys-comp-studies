# This file contains the code to perform exploratory data analysis on the datasets

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import GlobalConfig


def _print_interaction_count_stats(label: str, counts: pd.Series) -> None:
    """Print min/max range, mean, and median for a series of interaction counts."""

    if counts.empty:
        print(f"  {label}: no entities with interactions.")
        return
    cmin = int(counts.min())
    cmax = int(counts.max())
    mean = float(counts.mean())
    median = float(counts.median())
    print(
        f"  {label}: n={len(counts):,}, interactions per entity "
        f"min={cmin:,}, max={cmax:,}, mean={mean:.4f}, median={median:.4f}"
    )


def percColdStartUsers(df, user_col, min_interactions=2, max_interactions=3):
    """Calculate the percentage of cold start users in the dataset based on interaction range."""

    user_counts = df[user_col].value_counts()
    cold_start_users = user_counts[(user_counts >= min_interactions) & (user_counts <= max_interactions)].index
    perc_cold_start = len(cold_start_users) / len(user_counts)

    print(f"Percentage of cold start users (with {min_interactions}-{max_interactions} interactions): {perc_cold_start:.2%}")

    return perc_cold_start


def plotInteractionDistribution(df, user_col, item_col, cold_start=False):
    """Plot the distribution of interactions per user and per item."""

    user_counts = df[user_col].value_counts()
    item_counts = df[item_col].value_counts()

    dataset_tag = "cold_start" if cold_start else "dense"
    print(f"[EDA] {dataset_tag} dataset — interaction count summary")
    print(f"  total interaction rows: {len(df):,}, unique users: {len(user_counts):,}, unique items: {len(item_counts):,}")
    _print_interaction_count_stats("per user", user_counts)
    _print_interaction_count_stats("per item", item_counts)

    # custom bins (forces 2–4 bucket)
    if cold_start == False:
        init = np.arange(5, 101, 5)
        bin_edges = init
    else:
        bin_edges = np.array([0, 2, 4, 6, 8, 10])
        bin_edges = np.append(bin_edges, np.arange(12, 101, 2))

    counts, _ = np.histogram(user_counts, bins=bin_edges)

    plt.figure(figsize=(12, 5))
    ax = sns.histplot(user_counts, bins=bin_edges, kde=False)

    # annotate first few bins with range + count
    for i in range(len(counts[:2])):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        x = (left + right) / 2
        y = counts[i]
        if y > 0:
            label = f"{int(left)}–{int(right)}\n({y})"
            ax.text(x, y, label, ha='center', va='bottom', fontsize=9)

    title_suffix = " (Cold Start)" if cold_start else ""
    plt.title(f"Distribution of Interactions per User{title_suffix}")
    plt.xlabel("Number of Interactions (capped at 100 for visualization)")
    plt.ylabel("Count of Users")
    plt.ylim(0, 30000) 

    plt.savefig(GlobalConfig.eda_dir / f"interaction_distribution_per_user{'_cold_start' if cold_start else ''}.png")
    plt.close()


    # ---- Interactions per item ----
    counts, _ = np.histogram(item_counts, bins=bin_edges)

    plt.figure(figsize=(12, 5))
    ax = sns.histplot(item_counts, bins=bin_edges, kde=False)

    # Annotate each bin with range + count
    for i in range(len(counts[:2])):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        x = (left + right) / 2
        y = counts[i]
        if y > 0:
            label = f"{int(left)}–{int(right)}\n({y})"
            ax.text(x, y, label, ha='center', va='bottom', fontsize=9)

    plt.title(f"Distribution of Interactions per Item{title_suffix}")
    plt.xlabel("Number of Interactions (capped at 100 for visualization)")
    plt.ylabel("Count of Items")
    plt.ylim(0, 180000)

    plt.savefig(GlobalConfig.eda_dir / f"interaction_distribution_per_item{'_cold_start' if cold_start else ''}.png")
    plt.close()


def _resolve_gpt4rec_runtime_csv(
    data_type: str,
    runtime_csv: Optional[Path] = None,
    trained_models_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Return path to a runtime CSV if it exists (explicit path or latest per data_type)."""

    if runtime_csv is not None:
        path = Path(runtime_csv)
        return path if path.is_file() else None

    root = trained_models_dir or GlobalConfig.trained_models_dir
    pattern = f"gpt4rec_runtime_{data_type}_*.csv"
    matches = sorted(root.glob(f"time_tracking/gpt4rec/{pattern}"))
    if not matches:
        return None
    return matches[-1]


def _gpt4rec_runtime_section(component: str, detail: str) -> Optional[str]:
    """
    Map a tracker component row to a coarse pipeline section.
    Returns None for rows that are sub-steps of a parent total (avoid double counting).
    """

    detail = detail or ""
    if component == "pipeline_end_to_end_wall":
        return None

    if component.startswith("eval_") and component != "eval_with_bm25_total":
        return None
    if component in ("lm_epoch_on_epoch_end_callback", "in_train_write_train_curves_and_plots"):
        return None

    if component.startswith("io_") or component.startswith("data_"):
        return "Data preparation"
    if component == "test_split_prompting_build_examples_and_prompts":
        return "Data preparation"

    if component in (
        "hf_tokenizer_from_pretrained",
        "hf_lm_from_pretrained_and_wrap_gpt4rec_generation_model",
        "device_resolve_and_model_to_device",
        "bm25_index_build_and_ranker_init",
        "prompting_build_history_prompts_train_texts",
        "in_train_eval_metrics_init_and_val_subsample",
    ):
        return "Model & retrieval setup"

    if component in (
        "lm_tokenize_encode_and_build_dataloader",
        "lm_epoch_optimizer_forward_backward",
    ):
        return "LM fine-tuning"

    if component == "eval_with_bm25_total":
        if "in_train_val_epoch" in detail:
            return "In-training validation"
        if "bm25_tune" in detail:
            return None  # use bm25_grid_search_outer_loop_total instead
        if "final_full_validation" in detail:
            return "Final validation"
        if "final_full_test" in detail:
            return "Final test"

    if component == "bm25_grid_search_outer_loop_total":
        return "BM25 tuning (validation)"

    return "Other"


def plotGpt4RecRuntimeBySection(
    data_type: str = "cold_start",
    runtime_csv: Optional[Path] = None,
    trained_models_dir: Optional[Path] = None,
    config: GlobalConfig = GlobalConfig,
) -> Optional[Path]:
    """
    Bar chart of GPT4Rec wall time by coarse pipeline section (aggregated from runtime CSV).

    Skips plotting if the runtime CSV does not exist.
  """

    csv_path = _resolve_gpt4rec_runtime_csv(data_type, runtime_csv, trained_models_dir)
    if csv_path is None:
        print(
            f"[EDA] GPT4Rec runtime plot skipped for {data_type}: "
            f"no runtime CSV found under time_tracking/gpt4rec/."
        )
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[EDA] GPT4Rec runtime plot skipped for {data_type}: CSV is empty ({csv_path}).")
        return None

    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0.0)
    df["detail"] = df["detail"].fillna("").astype(str)

    section_seconds: Dict[str, float] = {}
    for _, row in df.iterrows():
        section = _gpt4rec_runtime_section(str(row["component"]), row["detail"])
        if section is None:
            continue
        section_seconds[section] = section_seconds.get(section, 0.0) + float(row["duration_sec"])

    if not section_seconds:
        print(f"[EDA] GPT4Rec runtime plot skipped for {data_type}: no mappable components.")
        return None

    order = [
        "Data preparation",
        "Model & retrieval setup",
        "LM fine-tuning",
        "In-training validation",
        "BM25 tuning (validation)",
        "Final validation",
        "Final test",
        "Other",
    ]
    labels = [s for s in order if s in section_seconds]
    labels += [s for s in section_seconds if s not in labels]
    minutes = [section_seconds[s] / 60.0 for s in labels]
    total_min = sum(minutes)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(labels))))
    colors = sns.color_palette("husl", n_colors=len(labels))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, minutes, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Wall time (minutes)")
    run_id = df["run_id"].iloc[0] if "run_id" in df.columns and len(df) else ""
    ax.set_title(
        f"GPT4Rec runtime by section — {data_type}"
        + (f" (run {run_id})" if run_id else "")
        + f"\nTotal plotted: {total_min:.1f} min (excl. end-to-end wall & eval sub-steps)"
    )
    for i, m in enumerate(minutes):
        ax.text(m, i, f" {m:.1f}", va="center", ha="left", fontsize=9)

    config.eda_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.eda_dir / f"gpt4rec_runtime_by_section_{data_type}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[EDA] Saved GPT4Rec runtime plot: {out_path}")
    return out_path


def _resolve_sasrec_runtime_csv(
    data_type: str,
    runtime_csv: Optional[Path] = None,
    trained_models_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Return path to a SASRec runtime CSV if it exists."""

    if runtime_csv is not None:
        path = Path(runtime_csv)
        return path if path.is_file() else None

    root = trained_models_dir or GlobalConfig.trained_models_dir
    pattern = f"sasrec_runtime_{data_type}_*.csv"
    matches = sorted(root.glob(f"time_tracking/sasrec/{pattern}"))
    if not matches:
        return None
    return matches[-1]


def _sasrec_runtime_section(component: str, detail: str) -> Optional[str]:
    """Map SASRec tracker rows to coarse pipeline sections."""

    _ = detail
    if component == "pipeline_end_to_end_wall":
        return None

    if component.startswith("io_") or component.startswith("data_"):
        return "Data preparation"

    if component in (
        "sasrec_model_build_and_session_init",
        "warpsampler_init",
        "in_train_eval_metrics_init",
    ):
        return "Model & sampler setup"

    if component == "sasrec_epoch_optimizer_forward_backward":
        return "Training (mini-batch updates)"

    if component == "in_train_val_predict_total":
        return "In-training validation"

    if component == "final_val_predict_total":
        return "Final validation"

    if component == "final_test_predict_total":
        return "Final test"

    return "Other"


def plotSasRecRuntimeBySection(
    data_type: str = "cold_start",
    runtime_csv: Optional[Path] = None,
    trained_models_dir: Optional[Path] = None,
    config: GlobalConfig = GlobalConfig,
) -> Optional[Path]:
    """
    Bar chart of SASRec wall time by coarse pipeline section (aggregated from runtime CSV).

    Skips plotting if the runtime CSV does not exist.
    """

    csv_path = _resolve_sasrec_runtime_csv(data_type, runtime_csv, trained_models_dir)
    if csv_path is None:
        print(
            f"[EDA] SASRec runtime plot skipped for {data_type}: "
            f"no runtime CSV found under time_tracking/sasrec/."
        )
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[EDA] SASRec runtime plot skipped for {data_type}: CSV is empty ({csv_path}).")
        return None

    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0.0)
    df["detail"] = df["detail"].fillna("").astype(str)

    section_seconds: Dict[str, float] = {}
    for _, row in df.iterrows():
        section = _sasrec_runtime_section(str(row["component"]), row["detail"])
        if section is None:
            continue
        section_seconds[section] = section_seconds.get(section, 0.0) + float(row["duration_sec"])

    if not section_seconds:
        print(f"[EDA] SASRec runtime plot skipped for {data_type}: no mappable components.")
        return None

    order = [
        "Data preparation",
        "Model & sampler setup",
        "Training (mini-batch updates)",
        "In-training validation",
        "Final validation",
        "Final test",
        "Other",
    ]
    labels = [s for s in order if s in section_seconds]
    labels += [s for s in section_seconds if s not in labels]
    minutes = [section_seconds[s] / 60.0 for s in labels]
    total_min = sum(minutes)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(labels))))
    colors = sns.color_palette("husl", n_colors=len(labels))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, minutes, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Wall time (minutes)")
    run_id = df["run_id"].iloc[0] if "run_id" in df.columns and len(df) else ""
    ax.set_title(
        f"SASRec runtime by section — {data_type}"
        + (f" (run {run_id})" if run_id else "")
        + f"\nTotal plotted: {total_min:.1f} min (excl. end-to-end wall)"
    )
    for i, m in enumerate(minutes):
        ax.text(m, i, f" {m:.1f}", va="center", ha="left", fontsize=9)

    config.eda_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.eda_dir / f"sasrec_runtime_by_section_{data_type}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[EDA] Saved SASRec runtime plot: {out_path}")
    return out_path


# def plotSampleUserItemInteractionGraph(df, user_col, item_col, sample_size=100):
#     """Plot a sample user-item interaction graph."""

#     sample_df = df.sample(n=sample_size, random_state=GlobalConfig.random_seed)
#     interaction_matrix = pd.crosstab(sample_df[user_col], sample_df[item_col])

#     plt.figure(figsize=(12, 8))
#     sns.heatmap(interaction_matrix, cmap="YlGnBu", cbar=False)
#     plt.title("Sample User-Item Interaction Graph")
#     plt.xlabel("Items")
#     plt.ylabel("Users")
#     plt.show()

