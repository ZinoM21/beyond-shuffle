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


def artist_loyalty_score(row, max_timestamp):
    return row.get("artist_loyalty", 0)


def vacation_score(row, max_timestamp):
    return recency_score(row, max_timestamp)


def background_score(row, max_timestamp):
    return row.get("attention_span", 0)


def sing_along_score(row, max_timestamp):
    return row.get("attention_span", 0)


def evening_chill_score(row, max_timestamp):
    return 0.5 * (1 - row.get("energy", 1.0)) + 0.5 * row.get("valence", 0.0)


def spring_vibes_score(row, max_timestamp):
    return row.get("valence", 0.0)


def weekend_party_score(row, max_timestamp):
    return (
        0.4 * row.get("energy", 0)
        + 0.4 * row.get("danceability", 0)
        + 0.2 * recency_score(row, max_timestamp)
    )


def popular_favorites_score(row, max_timestamp):
    return 0.5 * row.get("personal_popularity", 0) + 0.5 * row.get("popularity", 0)


def get_scoring_function(playlist_name: str) -> Callable:
    if playlist_name == "Workout":
        return workout_score
    elif playlist_name == "Commute":
        return commute_score
    elif playlist_name == "Focus":
        return focus_score
    elif playlist_name == "Artist Loyalty":
        return artist_loyalty_score
    elif playlist_name == "Vacation":
        return vacation_score
    elif playlist_name == "Background":
        return background_score
    elif playlist_name == "Sing Along":
        return sing_along_score
    elif playlist_name == "Evening Chill":
        return evening_chill_score
    elif playlist_name == "Spring Vibes":
        return spring_vibes_score
    elif playlist_name == "Weekend Party":
        return weekend_party_score
    elif playlist_name == "Popular Favorites":
        return popular_favorites_score
    else:
        return lambda row, max_timestamp: 0


def score(df: pd.DataFrame, scoring_fn, max_timestamp: float) -> pd.DataFrame:
    """
    Score the dataframe using the provided scoring function and return the dataframe with the score column.
    """
    df["score"] = df.apply(lambda row: scoring_fn(row, max_timestamp), axis=1)
    return df
