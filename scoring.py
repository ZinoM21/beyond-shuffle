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


def summer_memories_score(row, max_timestamp):
    # Emphasize attention span, personal popularity, and recency
    return (
        0.4 * row.get("attention_span", 0)
        + 0.3 * row.get("personal_popularity", 0)
        + 0.2 * recency_score(row, max_timestamp)
        + 0.1 * row.get("valence", 0)
    )


def travel_memories_score(row, max_timestamp):
    # Emphasize vacation status, energy, and recency
    return (
        0.5 * int(row.get("is_vacation", False))
        + 0.2 * row.get("energy", 0)
        + 0.2 * row.get("attention_span", 0)
        + 0.1 * recency_score(row, max_timestamp)
    )


def late_night_reflections_score(row, max_timestamp):
    # Emphasize acousticness, low valence, and attention span
    return (
        0.4 * row.get("acousticness", 0)
        + 0.3 * (1 - row.get("valence", 1))
        + 0.2 * row.get("attention_span", 0)
        + 0.1 * recency_score(row, max_timestamp)
    )


def firsts_milestones_score(row, max_timestamp):
    # Emphasize recency of first play, attention span, and personal popularity
    return (
        0.5 * (1 - row.get("first_played_months_ago", 99) / 24)  # more recent = higher
        + 0.3 * row.get("attention_span", 0)
        + 0.2 * row.get("personal_popularity", 0)
    )


def heartbreak_healing_score(row, max_timestamp):
    # Emphasize low valence, low energy, and high attention span
    return (
        0.4 * (1 - row.get("valence", 1))
        + 0.3 * (1 - row.get("energy", 1))
        + 0.3 * row.get("attention_span", 0)
    )


def semester_abroad_us_score(row, max_timestamp):
    # Emphasize vacation status, US country, attention span, and recency
    is_us = str(row.get("country", "")).upper() == "US"
    return (
        0.6 * int(is_us)
        + 0.4 * row.get("attention_span", 0)
    )


def get_scoring_function(playlist_name: str) -> Callable:
    match playlist_name:
        case "Workout":
            return workout_score
        case "Commute":
            return commute_score
        case "Focus":
            return focus_score
        case "Artist Loyalty":
            return artist_loyalty_score
        case "Vacation":
            return vacation_score
        case "Background":
            return background_score
        case "Sing Along":
            return sing_along_score
        case "Evening Chill":
            return evening_chill_score
        case "Spring Vibes":
            return spring_vibes_score
        case "Weekend Party":
            return weekend_party_score
        case "Popular Favorites":
            return popular_favorites_score
        case "Summer Memories":
            return summer_memories_score
        case "Travel Memories":
            return travel_memories_score
        case "Late Night Reflections":
            return late_night_reflections_score
        case "Firsts & Milestones":
            return firsts_milestones_score
        case "Heartbreak & Healing":
            return heartbreak_healing_score
        case "Semester Abroad (US)":
            return semester_abroad_us_score
        case _:
            return lambda row, max_timestamp: 0


def score(df: pd.DataFrame, scoring_fn, max_timestamp: float) -> pd.DataFrame:
    """
    Score the dataframe using the provided scoring function and return the dataframe with the score column.
    """
    df["score"] = df.apply(lambda row: scoring_fn(row, max_timestamp), axis=1)
    return df
