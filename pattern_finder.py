from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd


@dataclass
class DetectedPattern:
    """A base class for a detected pattern in listening history."""

    name: str
    description: str
    score: float
    tracks: pd.DataFrame
    contributing_features: Dict[str, Any] = field(default_factory=dict)
    pattern_type: str = "Generic"


@dataclass
class Period(DetectedPattern):
    """Represents a significant, anomalous listening period of 2 days or more."""

    pattern_type: str = "Period"
    start_date: Any = None
    end_date: Any = None


@dataclass
class Habit(DetectedPattern):
    """Represents a recurring, cyclical listening habit."""

    pattern_type: str = "Habit"
    time_slot: Tuple = ()  # e.g., ('Wednesday', 'Afternoon')


def calculate_baseline(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates the baseline (average) listening behavior from the dataframe.

    Args:
        df: The full listening history dataframe.

    Returns:
        A dictionary containing baseline values for key features.
    """
    baseline = {}

    # Sort by datetime
    df = df.sort_values("datetime").reset_index(drop=True)

    # For categorical features, the baseline is the most common value (mode)
    categorical_features = [
        "country",
        "day_of_week",
        "season",
        "platform",
        "time_of_day",
    ]
    for col in categorical_features:
        if col in df.columns and not df[col].empty:
            baseline[col] = df[col].mode()[0]

    # For numerical audio features, the baseline is the mean and standard deviation
    numerical_features = [
        "tempo",
        "speechiness",
        "liveness",
        "acousticness",
        "energy",
        "danceability",
        "loudness",
        "valence",
        "instrumentalness",
        "popularity",
    ]
    for col in numerical_features:
        if col in df.columns:
            baseline[col] = {"mean": df[col].mean(), "std": df[col].std()}

    return baseline


def find_periods(df: pd.DataFrame, baseline: Dict[str, Any]) -> List[Period]:
    """
    Finds significant, anomalous listening periods (2+ days).

    This is done by using a sliding window to find periods where a feature
    or combination of features deviates significantly and consistently from the baseline.

    Args:
        df: The user's listening history.
        baseline: The pre-calculated baseline profile.

    Returns:
        A list of detected Period objects.
    """
    patterns = []

    # --- Configuration for Detection ---
    # For Periods, we ONLY look for significant country changes.
    MIN_DURATION_DAYS = 7  # A period must be at least a week long
    WINDOW_SIZE = 100
    STEP_SIZE = 50
    CONSISTENCY_THRESHOLD = 0.7
    MIN_TRACKS_FOR_PLAYLIST = 30

    # Reset index to make 'datetime' a column, then sort
    if "datetime" not in df.columns:
        df = df.reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    detected_windows = []

    # --- Detect ONLY Categorical (Country) Anomalies ---
    baseline_country = baseline.get("country")
    if baseline_country:
        for i in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
            window = df.iloc[i : i + WINDOW_SIZE]
            mode_result = window["country"].mode()
            if not mode_result.empty:
                window_mode = mode_result[0]
                if (
                    window_mode != "ZZ"
                    and window_mode != baseline_country
                    and (window["country"] == window_mode).mean()
                    >= CONSISTENCY_THRESHOLD
                ):
                    detected_windows.append(
                        {
                            "start_index": i,
                            "end_index": i + WINDOW_SIZE,
                            "anomalies": {"country": window_mode},
                        }
                    )

    if not detected_windows:
        return []

    # --- Merge Consecutive Windows with the SAME set of anomalies ---
    merged_periods = []
    # Sort windows by their anomaly set to group them for merging
    detected_windows.sort(key=lambda x: frozenset(x["anomalies"].items()))

    if not detected_windows:
        return []
    current_period = detected_windows[0]
    for i in range(1, len(detected_windows)):
        next_window = detected_windows[i]

        if (
            frozenset(next_window["anomalies"].items())
            == frozenset(current_period["anomalies"].items())
            and next_window["start_index"] < current_period["end_index"] + STEP_SIZE
        ):
            current_period["end_index"] = next_window["end_index"]
        else:
            merged_periods.append(current_period)
            current_period = next_window
    merged_periods.append(current_period)

    # --- Score, Filter, and Create Final Period Objects ---
    for period_data in merged_periods:
        full_period_tracks = df.iloc[
            period_data["start_index"] : period_data["end_index"]
        ]

        # --- Filter for ONLY the tracks that contributed to the anomaly ---
        contributing_tracks = full_period_tracks.copy()
        for feature, value in period_data["anomalies"].items():
            if value in ["High", "Low"]:
                baseline_stats = baseline[feature]
                if value == "High":
                    contributing_tracks = contributing_tracks[
                        contributing_tracks[feature]
                        > baseline_stats["mean"] + (1.5 * baseline_stats["std"])
                    ]
                else:
                    contributing_tracks = contributing_tracks[
                        contributing_tracks[feature]
                        < baseline_stats["mean"] - (1.5 * baseline_stats["std"])
                    ]
            else:
                contributing_tracks = contributing_tracks[
                    contributing_tracks[feature] == value
                ]

        # --- Remove duplicates and check if playlist is large enough ---
        unique_contributing_tracks = contributing_tracks.drop_duplicates(
            subset=["track", "artist"], keep="first"
        )
        if len(unique_contributing_tracks) < MIN_TRACKS_FOR_PLAYLIST:
            continue

        start_date = full_period_tracks.iloc[0]["datetime"].date()
        end_date = full_period_tracks.iloc[-1]["datetime"].date()
        duration_days = (end_date - start_date).days + 1

        if duration_days < MIN_DURATION_DAYS:
            continue

        # Scoring: higher score for more combined features
        num_features = len(period_data["anomalies"])
        score = len(unique_contributing_tracks) * duration_days * (num_features**2)
        if "country" in period_data["anomalies"]:
            score *= 5
        if "valence" in period_data["anomalies"]:
            score *= 3

        # --- Naming and Description ---
        desc_parts = []
        for feature, value in period_data["anomalies"].items():
            if value in ["High", "Low"]:
                desc_parts.append(f"{value} {feature.capitalize()}")
            else:
                desc_parts.append(f"Country changed to {value}")

        name = " & ".join(desc_parts)
        desc = f"A {duration_days}-day period from {start_date} to {end_date} defined by {name}."

        patterns.append(
            Period(
                name=name,
                description=desc,
                score=score,
                tracks=unique_contributing_tracks,
                contributing_features=period_data["anomalies"],
                start_date=start_date,
                end_date=end_date,
            )
        )

    return patterns


def find_habits(df: pd.DataFrame, baseline: Dict[str, Any]) -> List[Habit]:
    """
    Finds recurring, cyclical listening habits, including combined feature habits.

    A habit is detected by grouping tracks by time slots (e.g., weekday afternoons)
    and checking if the audio features in that slot consistently deviate from the baseline.

    Args:
        df: The user's listening history.
        baseline: The pre-calculated baseline profile.

    Returns:
        A list of detected Habit objects.
    """
    habits = []

    # --- Configuration for Habit Detection ---
    NUMERICAL_FEATURES_TO_CHECK = [
        "valence",
        "energy",
        "danceability",
        "acousticness",
        "tempo",
    ]
    MIN_TRACKS_PER_SLOT = (
        50  # Minimum tracks needed to consider a time slot for a habit
    )
    STD_DEV_THRESHOLD = 1.1  # Lower threshold for habits as they are less intense
    MAX_FEATURE_COMBINATION = 3  # Max number of features to combine for a habit

    # --- Group by Day of Week and Time of Day ---
    grouped = df.groupby(["day_of_week", "time_of_day"])

    for (day, time_of_day), group in grouped:
        if len(group) < MIN_TRACKS_PER_SLOT:
            continue

        # Find all single-feature deviations in this time slot
        slot_deviations = []
        for feature in NUMERICAL_FEATURES_TO_CHECK:
            baseline_stats = baseline.get(feature)
            if not baseline_stats:
                continue

            slot_mean = group[feature].mean()
            is_high = slot_mean > baseline_stats["mean"] + (
                STD_DEV_THRESHOLD * baseline_stats["std"]
            )
            is_low = slot_mean < baseline_stats["mean"] - (
                STD_DEV_THRESHOLD * baseline_stats["std"]
            )

            if is_high or is_low:
                slot_deviations.append(
                    {
                        "feature": feature,
                        "direction": "High" if is_high else "Low",
                        "magnitude": abs(slot_mean - baseline_stats["mean"])
                        / baseline_stats["std"],
                    }
                )

        # Now, create habits from combinations of these deviations
        from itertools import combinations

        for i in range(1, min(len(slot_deviations), MAX_FEATURE_COMBINATION) + 1):
            for combo in combinations(slot_deviations, i):

                contributing_features = {
                    dev["feature"]: dev["direction"] for dev in combo
                }
                total_magnitude = sum(dev["magnitude"] for dev in combo)

                # Scoring: based on number of tracks, magnitude, and number of combined features
                score = (
                    len(group) * total_magnitude * (i**2)
                )  # i**2 to heavily favor more complex patterns

                feature_names = " & ".join(
                    [
                        f"{dev['direction']} {dev['feature'].capitalize()}"
                        for dev in combo
                    ]
                )
                name = f"{feature_names} on {day} {time_of_day}s"
                desc = f"A recurring habit of listening to music with these traits on {day} {time_of_day}s."

                habits.append(
                    Habit(
                        name=name,
                        description=desc,
                        score=score,
                        tracks=group,
                        contributing_features=contributing_features,
                        time_slot=(day, time_of_day),
                    )
                )

    return habits


def find_patterns(df: pd.DataFrame) -> List[DetectedPattern]:
    """
    Main orchestrator function to find all types of patterns.

    Args:
        df: The user's listening history.

    Returns:
        A list of all detected patterns, sorted by significance.
    """
    print("Calculating user's baseline listening profile...")
    baseline = calculate_baseline(df)

    all_patterns = []

    print("Finding significant listening periods...")
    all_patterns.extend(find_periods(df, baseline))
    print("Finding recurring habits...")
    all_patterns.extend(find_habits(df, baseline))

    print(f"Baseline profile calculated: {baseline}")

    # Sort patterns by score
    all_patterns.sort(key=lambda p: p.score, reverse=True)

    return all_patterns
