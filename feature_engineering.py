import pandas as pd


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
    """
    df = df.copy()
    df["attention_span"] = df["ms_played"].div(df["duration_ms"].values)
    df = df[df["ms_played"] > 0]
    df.loc[df["attention_span"] > 1, "attention_span"] = 1
    return df


def add_skipping_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add skipping behavior to the DataFrame.
    """
    df = df.copy()
    if "reason_end" in df.columns:
        df.loc[:, "skipped"] = df["reason_end"] == "fwdbtn"
    else:
        df.loc[:, "skipped"] = (df["attention_span"] < 0.1) & (
            df["listening_time_in_s"] < 30
        )
    return df


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


def compute_vacation_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the vacation status of the user.
    """
    df = df.copy()
    if "country" in df.columns:
        home_country = df["country"].mode().iloc[0]
        df["is_vacation"] = df["country"] != home_country
    else:
        df["is_vacation"] = False
    return df


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


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
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

    df = compute_vacation_status(df)

    df = compute_first_played_months_ago(df)

    print("Feature engineering complete.")

    return df
