from typing import Callable

import pandas as pd


def recency_score(row, max_timestamp):
    # Normalize recency: 1 = most recent, 0 = oldest
    return (row.name.timestamp() / max_timestamp) if max_timestamp else 0


def workout_score(row, max_timestamp):
    return (
        0.4 * row.get("energy", 0)
        + 0.3 * row.get("danceability", 0)
        + 0.2 * row.get("attention_span", 0)
        + 0.1 * recency_score(row, max_timestamp)
    )


def commute_score(row, max_timestamp):
    return (
        0.5 * (not row.get("skipped", False))
        + 0.3 * recency_score(row, max_timestamp)
        + 0.2 * row.get("attention_span", 0)
    )


def focus_score(row, max_timestamp):
    return (
        0.5 * row.get("instrumentalness", 0)
        + 0.3 * (1 - row.get("speechiness", 0))
        + 0.2 * row.get("attention_span", 0)
    )


def get_scoring_function(playlist_name: str) -> Callable:
    if playlist_name == "Workout":
        return workout_score
    elif playlist_name == "Commute":
        return commute_score
    elif playlist_name == "Focus":
        return focus_score
    else:
        return lambda row, max_timestamp: 0


def score(df: pd.DataFrame, name_title_case: str, max_timestamp: float) -> pd.DataFrame:
    """
    Score the dataframe using the scoring function and return the dataframe with the score column.
    """
    scoring_fn = get_scoring_function(name_title_case)
    df["score"] = df.apply(lambda row: scoring_fn(row, max_timestamp), axis=1)
    return df
