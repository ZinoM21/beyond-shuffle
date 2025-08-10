from typing import Any, Dict, List

import pandas as pd

from constants import PERIOD_FEATURE_ZSCORE_THRESHOLD, STEP_SIZE_DAYS, WINDOW_SIZE_DAYS
from pf_types import Period


def detect_categorical_anomalies(df: pd.DataFrame, baseline_country: str) -> List[Dict]:
    """Sliding-window categorical anomaly detection for periods (country)."""
    detected_windows: List[Dict] = []
    df_sorted = df.sort_index()
    start_date, end_date = df_sorted.index.min(), df_sorted.index.max()
    current_date = start_date

    while current_date + pd.Timedelta(days=WINDOW_SIZE_DAYS - 1) <= end_date:
        window_end_date = current_date + pd.Timedelta(days=WINDOW_SIZE_DAYS - 1)
        window = df_sorted.loc[current_date:window_end_date]

        if window.empty:
            current_date += pd.Timedelta(days=STEP_SIZE_DAYS)
            continue

        anomalies: Dict[str, Any] = {}
        if "country" in window.columns:
            country_counts = window["country"].value_counts(normalize=True)
            for country, percentage in country_counts.items():
                if (
                    country != "ZZ"
                    and country != baseline_country
                    and percentage >= PERIOD_FEATURE_ZSCORE_THRESHOLD
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


def merge_consecutive_windows(detected_windows: List[Dict]) -> List[Dict]:
    if not detected_windows:
        return []
    detected_windows.sort(key=lambda x: x["start_date"])  # in place
    merged_periods: List[Dict] = []
    current_period = detected_windows[0]
    for next_window in detected_windows[1:]:
        time_gap = next_window["start_date"] - current_period["end_date"]
        if frozenset(next_window["anomalies"].items()) == frozenset(
            current_period["anomalies"].items()
        ) and time_gap <= pd.Timedelta(days=STEP_SIZE_DAYS):
            current_period["end_date"] = max(
                current_period["end_date"], next_window["end_date"]
            )
        else:
            merged_periods.append(current_period)
            current_period = next_window
    merged_periods.append(current_period)
    return merged_periods


def create_period_from_data(
    period_data: Dict, df: pd.DataFrame, baseline: Dict[str, Any]
) -> Period:
    start_date = period_data["start_date"]
    end_date = period_data["end_date"]
    period_df = df.loc[start_date:end_date].copy()

    anomaly_type, anomaly_value = next(
        iter(period_data["anomalies"].items()), (None, None)
    )
    contributing_tracks = period_df[period_df[anomaly_type] == anomaly_value]

    duration_days = (end_date - start_date).days + 1
    name = f"Travel to {anomaly_value} ({start_date.date()} - {end_date.date()})"
    desc = f"A {duration_days}-day period defined by {name}."

    return Period(
        name=name,
        description=desc,
        tracks=contributing_tracks,
        contributing_features=period_data["anomalies"],
        start_date=start_date,
        end_date=end_date,
    )
