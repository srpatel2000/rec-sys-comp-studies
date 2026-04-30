# This file contains the code to perform exploratory data analysis on the datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import GlobalConfig


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
    max_val = user_counts.max()

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
    item_counts = df[item_col].value_counts()

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

