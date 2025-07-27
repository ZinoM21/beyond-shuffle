import numpy as np
import pandas as pd
from pandas import DataFrame


def get_time_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


def get_season(month: int) -> str:
    if 3 <= month <= 5:
        return "Spring"
    elif 6 <= month <= 8:
        return "Summer"
    elif 9 <= month <= 11:
        return "Fall"
    else:
        return "Winter"


def compute_attention_span(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the attention_span column for the DataFrame.
    - attention_span = ms_played / duration_ms, capped at 1.
    - Only keep rows where ms_played > 0.
    - If duration_ms is missing or zero, attention_span will be pd.NA (undefined).
    """
    df = df.copy()
    # Set attention_span to NA where duration_ms is missing or zero
    mask_valid = df["duration_ms"].notna() & (df["duration_ms"] != 0)
    df["attention_span"] = pd.NA
    df.loc[mask_valid, "attention_span"] = (
        df.loc[mask_valid, "ms_played"] / df.loc[mask_valid, "duration_ms"]
    )
    df = df[df["ms_played"] > 0]
    df.loc[df["attention_span"] > 1, "attention_span"] = 1
    return df


def add_skipping_behavior(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add skipping behavior to the DataFrame.
    This version decides the skip logic for each row based on whether 'attention_span' is available for that row.
    - If 'attention_span' is NA: skipped if played for < 30 seconds.
    - If 'attention_span' is not NA:
        - If 'reason_end' column exists and is 'fwdbtn': skipped if played for < 30 seconds.
        - Otherwise (no 'reason_end' or not 'fwdbtn'): skipped if 'attention_span' < 0.25 and played for < 30 seconds.
    """
    df = input_df.copy()
    short_play = df["ms_played"] < (30 * 1000)

    # --- Define the two possible outcomes for 'skipped' based on the logic ---

    # Outcome 1: Logic when attention_span IS available for a row.
    if "reason_end" in df.columns:
        # If reason_end is 'fwdbtn', that determines the skip.
        # Otherwise, fall back to the low_attention logic.
        low_attention_skip = (df["attention_span"] < 0.25) & short_play
        skipped_if_not_na = np.where(
            df["reason_end"] == "fwdbtn", short_play, low_attention_skip
        )
    else:
        # If reason_end doesn't exist, just use attention_span.
        skipped_if_not_na = (df["attention_span"] < 0.25) & short_play

    # Outcome 2: Logic when attention_span is NOT available for a row.
    skipped_if_na = short_play

    # --- Use np.where to choose between the two outcomes based on the condition ---
    if "attention_span" in df.columns:
        # The condition is whether attention_span is NA for each row.
        is_na_condition = df["attention_span"].isna()
        skipped = np.where(is_na_condition, skipped_if_na, skipped_if_not_na)
    else:
        # If the attention_span column doesn't exist, all are treated as NA.
        skipped = skipped_if_na

    output_df = input_df.copy()
    output_df["skipped"] = pd.Series(skipped, index=df.index, dtype=bool)
    return output_df


def compute_personal_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the personal popularity of each track.
    """
    df = df.copy()
    df["personal_popularity"] = df.groupby("track")["track"].transform("count")
    return df


def compute_artist_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the popularity of each artist.
    """
    df = df.copy()
    df["artist_popularity"] = df.groupby("artist")["artist"].transform("count")
    return df


def compute_artist_loyalty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the max number of consecutive plays of the same artist.
    """
    df = df.copy()
    if "artist" in df.columns:
        artist_change = (df["artist"] != df["artist"].shift()).cumsum()
        df["artist_loyalty"] = df.groupby(artist_change).cumcount() + 1
    else:
        df["artist_loyalty"] = 0
    return df


# def compute_vacation_status(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Compute the vacation status of the user.
#     """
#     df = df.copy()
#     if "country" in df.columns:
#         home_country = df["country"].mode().iloc[0]
#         df["is_vacation"] = df["country"] != home_country
#     else:
#         df["is_vacation"] = False
#     return df


def compute_first_played_months_ago(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, compute how many months ago the track was first played (relative to the row's timestamp).
    Adds 'first_played_months_ago' column.
    """
    df = df.copy()
    if "track" in df.columns:
        # Find the first play timestamp for each track
        first_play = df.groupby("track").apply(lambda x: x.index.min())
        # Map to each row
        df["first_played_timestamp"] = df["track"].map(first_play)
        # Compute months difference
        months_ago = (df.index.to_series() - df["first_played_timestamp"]).dt.days // 30
        df["first_played_months_ago"] = months_ago
    else:
        df["first_played_months_ago"] = 99
    return df


def feature_engineering(df: DataFrame) -> DataFrame:
    """
    Engineers features needed for context-based playlists.
    """
    df = df.copy()
    print("Starting feature engineering...")

    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Time-based features
    df.loc[:, "time_of_day"] = pd.Series(df.index.hour, index=df.index).map(
        get_time_of_day
    )
    df.loc[:, "day_of_week_nr"] = pd.Series(df.index.dayofweek, index=df.index)
    df.loc[:, "day_of_week"] = pd.Series(df.index.day_name(), index=df.index)
    df.loc[:, "is_weekday"] = df["day_of_week_nr"] < 5
    df.loc[:, "month"] = pd.Series(df.index.month, index=df.index)
    df.loc[:, "season"] = df["month"].map(get_season)
    df.loc[:, "year"] = pd.Series(df.index.year, index=df.index)
    df.loc[:, "session_gap_s"] = (
        df.index.to_series().diff().dt.total_seconds().fillna(0)
    )

    df = compute_attention_span(df)

    df = add_skipping_behavior(df)

    df = compute_personal_popularity(df)

    df = compute_artist_popularity(df)

    df = compute_artist_loyalty(df)

    # df = compute_vacation_status(df)

    df = compute_first_played_months_ago(df)

    print("Feature engineering complete.")

    return df
