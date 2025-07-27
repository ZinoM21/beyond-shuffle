from typing import Dict, List

import pandas as pd

from constants import WEIGHTS
from pattern_finder import DetectedPattern, Habit


def _calculate_popularity(pattern_tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a contextual popularity score for each track in a pattern
    using a weighted sum of normalized plays, skip rate, and attention span.
    - Total Plays: 0.5 weight (normalized against the most played track in the pattern)
    - Skip Rate: 0.3 weight (inverted, so fewer skips are better)
    - Attention Span: 0.2 weight (using the track's mean attention span)
    If attention span data is missing for a track, it's filled with the average of other
    tracks in the pattern to ensure fairness.
    """
    # 1. Calculate base stats per track
    track_stats = (
        pattern_tracks.groupby("spotify_track_uri")
        .agg(
            total_plays=("spotify_track_uri", "count"),
            total_skips=("skipped", "sum"),
        )
        .reset_index()
    )

    if track_stats.empty:
        pattern_tracks["contextual_popularity_score"] = pd.Series(dtype=float)
        return pattern_tracks

    # 2. Calculate components for the weighted sum

    # Play count component (normalized)
    max_plays = track_stats["total_plays"].max()
    normalized_plays = track_stats["total_plays"] / max_plays if max_plays > 0 else 0

    # Non-skip rate component
    track_stats["skip_rate"] = track_stats["total_skips"] / track_stats["total_plays"]
    non_skip_rate = 1 - track_stats["skip_rate"]

    # Attention span component
    if (
        "attention_span" in pattern_tracks.columns
        and pattern_tracks["attention_span"].notna().any()
    ):
        attention_stats = (
            pattern_tracks.dropna(subset=["attention_span"])
            .groupby("spotify_track_uri")["attention_span"]
            .mean()
            .rename("mean_attention_span")
        )
        track_stats = track_stats.merge(
            attention_stats, on="spotify_track_uri", how="left"
        )

        # Fill NA values with the global mean attention span
        global_mean_attention = track_stats["mean_attention_span"].mean()
        fill_value = global_mean_attention if pd.notna(global_mean_attention) else 0
        track_stats["mean_attention_span"].fillna(fill_value, inplace=True)

        attention_score = track_stats["mean_attention_span"]
    else:
        attention_score = 0

    # 3. Calculate final weighted score
    track_stats["contextual_popularity_score"] = (
        WEIGHTS["count"] * normalized_plays
        + WEIGHTS["skip_rate"] * non_skip_rate
        + WEIGHTS["attention_span"] * attention_score
    )

    return pattern_tracks.merge(
        track_stats[["spotify_track_uri", "contextual_popularity_score"]],
        on="spotify_track_uri",
        how="left",
    )


def _filter_and_sort_habit_tracks(
    p: Habit, tracks_with_popularity: pd.DataFrame, num_songs: int
) -> pd.DataFrame:
    """
    Filters tracks for a Habit pattern to only those that strongly exhibit
    the habit's characteristic and then sorts them by popularity.
    """
    tracks_to_show = tracks_with_popularity.copy()
    feature, direction = list(p.contributing_features.items())[0]

    if direction == "High":
        threshold = tracks_to_show[feature].quantile(0.75)
        tracks_to_show = tracks_to_show[tracks_to_show[feature] > threshold]
    elif direction == "Low":
        threshold = tracks_to_show[feature].quantile(0.25)
        tracks_to_show = tracks_to_show[tracks_to_show[feature] < threshold]

    return (
        tracks_to_show.sort_values("contextual_popularity_score", ascending=False)
        .drop_duplicates(subset=["track", "artist"])
        .head(num_songs)
    )


def _filter_and_sort_generic_tracks(
    tracks_with_popularity: pd.DataFrame, num_songs: int
) -> pd.DataFrame:
    """
    Sorts tracks for a generic pattern by contextual popularity.
    """
    return (
        tracks_with_popularity.sort_values(
            "contextual_popularity_score", ascending=False
        )
        .drop_duplicates(subset=["track", "artist"])
        .head(num_songs)
    )


def process_patterns(
    detected_patterns: List[DetectedPattern], num_songs: int
) -> Dict[str, pd.DataFrame]:
    """
    Processes a list of detected patterns to calculate track popularity
    and extract the top N songs for each pattern.

    Args:
        detected_patterns: A list of DetectedPattern objects.
        num_songs: The number of top songs to extract for each pattern's playlist.

    Returns:
        A dictionary mapping pattern names to a DataFrame of their top tracks.
    """
    top_tracks_map = {}

    for p in detected_patterns:
        # Calculate contextual popularity for all tracks in the pattern
        tracks_with_popularity = _calculate_popularity(p.tracks.copy())

        # Filter and sort tracks based on pattern type
        if isinstance(p, Habit):
            top_tracks = _filter_and_sort_habit_tracks(
                p, tracks_with_popularity, num_songs
            )
        else:
            top_tracks = _filter_and_sort_generic_tracks(
                tracks_with_popularity, num_songs
            )

        top_tracks_map[p.name] = top_tracks

    return top_tracks_map
