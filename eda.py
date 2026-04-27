# This file contains the code to perform exploratory data analysis on the datasets

def percColdStartUsers(df, user_col, min_interactions=2, max_interactions=4):
    """Calculate the percentage of cold start users in the dataset based on interaction range."""

    user_counts = df[user_col].value_counts()
    cold_start_users = user_counts[(user_counts >= min_interactions) & (user_counts <= max_interactions)].index
    perc_cold_start = len(cold_start_users) / len(user_counts)

    print(f"Percentage of cold start users (with {min_interactions}-{max_interactions} interactions): {perc_cold_start:.2%}")

    return perc_cold_start