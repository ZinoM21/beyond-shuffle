from typing import Any, Dict, List

import pandas as pd

from constants import MIN_TRACKS_FOR_PLAYLIST, PERIOD_MIN_DAYS
from constants import (
    CATEGORICAL_FEATURES_TO_CHECK,
    NUMERICAL_FEATURES_TO_CHECK,
    HABIT_MIN_CLUSTER_SHARE,
    HABIT_MIN_CLUSTER_WEEKS,
    HABIT_FEATURE_ZSCORE_THRESHOLD,
    HABIT_BEHAVIORAL_FEATURES,
    HABIT_BEHAVIORAL_ZSCORE_THRESHOLD,
    HABIT_MIN_NUM_FEATURES,
    HABIT_MIN_STREAMS_PER_SLOT,
)
from pf_periods import (
    create_period_from_data,
    detect_categorical_anomalies,
    merge_consecutive_windows,
)
from pf_habits import (
    cluster_audio_profiles,
    compute_slot_feature_stats,
    format_slot_name,
    refine_slot_name_with_device,
    select_habit_slots,
    slot_key_builders,
)
from pf_types import DetectedPattern, Habit, Period


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

    for col in CATEGORICAL_FEATURES_TO_CHECK:
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
    detected_windows = detect_categorical_anomalies(df_local, baseline_country)

    if not detected_windows:
        return []

    merged_periods = merge_consecutive_windows(detected_windows)

    final_periods = []
    for period in merged_periods:
        period = create_period_from_data(period, df_local, baseline)

        if (
            len(period.tracks) >= MIN_TRACKS_FOR_PLAYLIST
            and (period.end_date - period.start_date).days + 1 >= PERIOD_MIN_DAYS
        ):
            final_periods.append(period)

    return final_periods


def _slot_key_builders(_: pd.DataFrame):
    # Backwards compatibility shim for external calls; delegate to pf_habits
    return slot_key_builders()


def _compute_slot_feature_stats(df: pd.DataFrame, slot_col: str) -> pd.DataFrame:
    return compute_slot_feature_stats(df, slot_col)


def _select_habit_slots(stats: pd.DataFrame):
    return select_habit_slots(stats)


def _extract_slot_group(df_with_slot: pd.DataFrame, slot_key):
    return df_with_slot[df_with_slot["_slot"] == slot_key]


def _format_slot_name(schema: str, slot_key):
    return format_slot_name(schema, slot_key)


def _refine_slot_name_with_device(df_slot: pd.DataFrame, schema: str, base_name: str):
    return refine_slot_name_with_device(df_slot, base_name)


def find_habits(df: pd.DataFrame, baseline: Dict[str, Any]) -> List[Habit]:
    """
    Multi-feature recurring-slot habit detection.

    Strategy:
    - Build several slot schemas (without using time_of_day) like day_of_week+hour, +platform, and season+hour.
    - For each schema, compute per-slot means for audio features and compare to global means via z-scores.
    - Select slots with sufficient streams, recurrence across weeks, and multiple deviating features.
    - Emit habits for those slots with contributing feature directions.
    """
    df_local = df.copy()
    # Ensure datetime index for resampling/feature extraction
    if not isinstance(df_local.index, pd.DatetimeIndex):
        if "datetime" in df_local.columns:
            df_local["datetime"] = pd.to_datetime(df_local["datetime"])
            df_local = df_local.set_index("datetime")
        else:
            df_local = df_local.sort_index()

    habits: List[Habit] = []
    builders = _slot_key_builders(df_local)

    for schema, build in builders.items():
        df_schema = build(df_local)
        if "_slot" not in df_schema.columns:
            continue
        # Cluster audio profiles globally to use discrete style buckets in addition to z-scores
        df_schema = df_schema.copy()
        df_schema["_audio_cluster"] = cluster_audio_profiles(df_schema)

        stats = _compute_slot_feature_stats(df_schema, "_slot")
        selected_slots = _select_habit_slots(stats)

        for slot_key in selected_slots:
            slot_group = _extract_slot_group(df_schema, slot_key)
            # Require that a single audio cluster dominates within the slot and recurs across weeks
            cluster_counts = slot_group["_audio_cluster"].value_counts(normalize=True)
            dominant_cluster = None
            dominant_share = 0.0
            if not cluster_counts.empty:
                dominant_cluster = cluster_counts.index[0]
                dominant_share = float(cluster_counts.iloc[0])
            if (
                dominant_cluster is None
                or dominant_cluster == -1
                or dominant_share < HABIT_MIN_CLUSTER_SHARE
                or slot_group.index.to_series().dt.isocalendar().week.nunique()
                < HABIT_MIN_CLUSTER_WEEKS
            ):
                continue

            # Determine contributing features and directions
            contributing: Dict[str, str] = {}
            for feat in NUMERICAL_FEATURES_TO_CHECK:
                if feat not in slot_group.columns:
                    continue
                mu = df_local[feat].mean()
                sigma = df_local[feat].std() or 0.0
                if sigma == 0 or pd.isna(sigma):
                    continue
                slot_mu = slot_group[feat].mean()
                z = (slot_mu - mu) / sigma
                if abs(z) >= HABIT_FEATURE_ZSCORE_THRESHOLD:
                    contributing[feat] = "High" if z > 0 else "Low"

            # Add behavioral feature directions
            for feat in HABIT_BEHAVIORAL_FEATURES:
                if feat not in slot_group.columns:
                    continue
                mu = df_local[feat].mean()
                sigma = df_local[feat].std() or 0.0
                if sigma == 0 or pd.isna(sigma):
                    continue
                slot_mu = slot_group[feat].mean()
                z = (slot_mu - mu) / sigma
                if abs(z) >= HABIT_BEHAVIORAL_ZSCORE_THRESHOLD:
                    direction = "High" if z > 0 else "Low"
                    contributing[feat] = direction

            if len(contributing) < HABIT_MIN_NUM_FEATURES:
                continue

            if len(slot_group) < HABIT_MIN_STREAMS_PER_SLOT:
                continue

            slot_name = _format_slot_name(schema, slot_key)
            slot_name = _refine_slot_name_with_device(slot_group, schema, slot_name)
            # Enrich name with strong recurring anchors beyond day: add "at HH:MM" if not already present
            if ":" not in slot_name:
                # If hour exists in the slot group, infer a stable hour
                if "hour" in slot_group.columns:
                    hour_mode = int(slot_group["hour"].mode().iloc[0])
                    slot_name = f"{slot_name} at {hour_mode:02d}:00"
            description = f"Recurring listening pattern."

            habits.append(
                Habit(
                    name=slot_name,
                    description=description,
                    tracks=slot_group,
                    contributing_features=contributing,
                    time_slot=slot_key,
                    slot_schema=schema,
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

    return all_patterns
