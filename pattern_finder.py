from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd

from constants import (
    ANOMALY_THRESHOLD,
    MIN_DURATION_DAYS,
    MIN_TRACKS_FOR_PLAYLIST,
    MIN_TRACKS_PER_SLOT,
    NUMERICAL_FEATURES_TO_CHECK,
    STEP_SIZE_DAYS,
    WINDOW_SIZE_DAYS,
)


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


def _create_habit(
    name: str,
    description: str,
    score: float,
    tracks: pd.DataFrame,
    feature: str,
    direction: str,
    time_slot: Tuple,
) -> Habit:
    """Helper function to create a Habit object."""
    day, time_of_day = time_slot
    name = f"{direction} {feature.capitalize()} on {day} {time_of_day}"

    return Habit(
        name=name,
        description=description,
        score=score,
        tracks=tracks,
        contributing_features={feature: direction},
        time_slot=time_slot,
    )


def _detect_categorical_anomalies(
    df: pd.DataFrame, baseline_country: str
) -> List[Dict]:
    """
    Scans through listening data to find anomalous windows based on categorical features.
    It uses a sliding window approach and returns windows where listening habits
    deviate from the baseline.
    """
    detected_windows = []
    df_sorted = df.sort_index()
    start_date, end_date = df_sorted.index.min(), df_sorted.index.max()
    current_date = start_date

    while current_date + pd.Timedelta(days=WINDOW_SIZE_DAYS - 1) <= end_date:
        window_end_date = current_date + pd.Timedelta(days=WINDOW_SIZE_DAYS - 1)
        window = df_sorted.loc[current_date:window_end_date]

        if window.empty:
            current_date += pd.Timedelta(days=STEP_SIZE_DAYS)
            continue

        anomalies = {}
        # Country anomaly
        if "country" in window.columns:
            country_counts = window["country"].value_counts(normalize=True)
            for country, percentage in country_counts.items():
                if (
                    country != "ZZ"
                    and country != baseline_country
                    and percentage >= ANOMALY_THRESHOLD
                ):
                    anomalies["country"] = country

        if anomalies:
            detected_windows.append(
                {
                    "start_date": window.index.min(),
                    "end_date": window.index.max(),
                    "anomalies": anomalies,
                }
            )
        current_date += pd.Timedelta(days=STEP_SIZE_DAYS)

    return detected_windows


def _merge_consecutive_windows(detected_windows: List[Dict]) -> List[Dict]:
    """Merges consecutive or overlapping windows with the same anomaly."""
    if not detected_windows:
        return []

    # Sort by start date to ensure correct merging order
    detected_windows.sort(key=lambda x: x["start_date"])

    merged_periods = []
    current_period = detected_windows[0]

    for next_window in detected_windows[1:]:
        # Check for same anomaly and overlapping/consecutive windows
        time_gap = next_window["start_date"] - current_period["end_date"]
        if frozenset(next_window["anomalies"].items()) == frozenset(
            current_period["anomalies"].items()
        ) and time_gap <= pd.Timedelta(days=STEP_SIZE_DAYS):
            # Merge by extending the end date
            current_period["end_date"] = max(
                current_period["end_date"], next_window["end_date"]
            )
        else:
            # Found a new, separate period
            merged_periods.append(current_period)
            current_period = next_window

    merged_periods.append(current_period)
    return merged_periods


def _create_period_from_data(
    period_data: Dict, df: pd.DataFrame, baseline: Dict[str, Any]
) -> Period:
    """Creates a Period object from raw period data."""
    start_date = period_data["start_date"]
    end_date = period_data["end_date"]
    period_df = df.loc[start_date:end_date].copy()

    # Determine the main anomaly. For now, we just take the first one.
    # This could be expanded to handle multiple anomalies more gracefully.
    main_anomaly = next(iter(period_data["anomalies"].items()), (None, None))
    anomaly_type, anomaly_value = main_anomaly

    # Filter for only the tracks that contributed to the anomaly
    contributing_tracks = period_df[period_df[anomaly_type] == anomaly_value]

    unique_contributing_tracks = contributing_tracks.drop_duplicates(
        subset=["track", "artist"], keep="first"
    )

    duration_days = (end_date - start_date).days + 1

    # Scoring
    num_features = len(period_data["anomalies"])
    score = len(unique_contributing_tracks) * duration_days * (num_features**2)
    if "country" in period_data["anomalies"]:
        score *= 5

    # Naming and Description
    name = f"Travel to {anomaly_value} ({start_date.date()} - {end_date.date()})"
    desc = f"A {duration_days}-day period from {start_date.date()} to {end_date.date()} defined by {name}."

    print(
        f"Found period with {len(contributing_tracks)} streams, i.e. {len(unique_contributing_tracks)} unique tracks"
    )

    return Period(
        name=name,
        description=desc,
        score=score,
        tracks=contributing_tracks,
        contributing_features=period_data["anomalies"],
        start_date=start_date,
        end_date=end_date,
    )


def calculate_baseline(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates the baseline (average) listening behavior from the dataframe.

    Args:
        df: The full listening history dataframe.

    Returns:
        A dictionary containing baseline values for key features.
    """
    baseline = {}
    df = df.sort_values("datetime").reset_index(drop=True)

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

    for col in NUMERICAL_FEATURES_TO_CHECK:
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
    df_local = df.copy()

    # Ensure df_local has a DatetimeIndex.
    if not isinstance(df_local.index, pd.DatetimeIndex):
        if "datetime" in df_local.columns:
            df_local["datetime"] = pd.to_datetime(df_local["datetime"])
            df_local = df_local.set_index("datetime")
        else:
            # Fallback for when the index is datetime-like but not of the correct type.
            df_local = df_local.reset_index()
            if "datetime" not in df_local.columns and "index" in df_local.columns:
                df_local = df_local.rename(columns={"index": "datetime"})

            if "datetime" in df_local.columns:
                df_local["datetime"] = pd.to_datetime(df_local["datetime"])
                df_local = df_local.set_index("datetime")
            else:
                raise ValueError(
                    "DataFrame must have a 'datetime' column or a DatetimeIndex."
                )

    df_local = df_local.sort_index()

    baseline_country = baseline.get("country")
    if not isinstance(baseline_country, str):
        baseline_country = ""
    detected_windows = _detect_categorical_anomalies(df_local, baseline_country)

    if not detected_windows:
        return []

    merged_periods = _merge_consecutive_windows(detected_windows)

    final_periods = []
    for period in merged_periods:
        period = _create_period_from_data(period, df_local, baseline)

        if (
            len(period.tracks) >= MIN_TRACKS_FOR_PLAYLIST
            and (period.end_date - period.start_date).days + 1 >= MIN_DURATION_DAYS
        ):
            final_periods.append(period)

    return final_periods


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
    grouped = df.groupby(["day_of_week", "time_of_day"])

    for feature in NUMERICAL_FEATURES_TO_CHECK:
        slot_means = grouped[feature].mean().dropna()
        if slot_means.empty:
            continue

        highest_slot_name = slot_means.idxmax()
        lowest_slot_name = slot_means.idxmin()

        assert isinstance(highest_slot_name, tuple)
        assert isinstance(lowest_slot_name, tuple)

        high_group = grouped.get_group(highest_slot_name)
        if len(high_group) >= MIN_TRACKS_PER_SLOT:
            day, time_of_day = highest_slot_name
            habits.append(
                _create_habit(
                    name=f"Highest {feature.capitalize()} Habit",
                    description=f"Your most consistently high-{feature} music is listened to on {day} {time_of_day}s.",
                    score=len(high_group) * slot_means.max(),
                    tracks=high_group,
                    feature=feature,
                    direction="High",
                    time_slot=highest_slot_name,
                )
            )

        low_group = grouped.get_group(lowest_slot_name)
        if len(low_group) >= MIN_TRACKS_PER_SLOT:
            day, time_of_day = lowest_slot_name
            habits.append(
                _create_habit(
                    name=f"Lowest {feature.capitalize()} Habit",
                    description=f"Your most consistently low-{feature} music is listened to on {day} {time_of_day}s.",
                    score=len(low_group) * slot_means.min(),
                    tracks=low_group,
                    feature=feature,
                    direction="Low",
                    time_slot=lowest_slot_name,
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
