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


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features needed for context-based playlists.
    """
    df = df.copy()
    print("Starting feature engineering for context playlists...")
    df.loc[:, "time_of_day"] = df.index.hour.map(get_time_of_day)
    df.loc[:, "day_of_week_nr"] = df.index.dayofweek
    df.loc[:, "day_of_week"] = df.index.day_name()
    df.loc[:, "is_weekday"] = df["day_of_week_nr"] < 5
    df.loc[:, "month"] = df.index.month
    df.loc[:, "season"] = df["month"].map(get_season)
    df.loc[:, "session_gap_s"] = (
        df.index.to_series().diff().dt.total_seconds().fillna(0)
    )

    df.loc[:, "attention_span"] = df["relation_listening_lenght"]

    if "reason_end" in df.columns:
        df.loc[:, "skipped"] = df["reason_end"] == "fwdbtn"
    else:
        # Fallback for skipped tracks if 'reason_end' is not available
        df.loc[:, "skipped"] = (df["attention_span"] < 0.1) & (
            df["listening_time_in_s"] < 30
        )

    print("Feature engineering complete.")
    return df


def model_data(
    df: pd.DataFrame,
    exclude_devices: list[str] | None = None,
    df_audio_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge, clean, and process streaming data DataFrames.
    Args:
        df (pd.DataFrame): Streaming data
        exclude_devices (list[str], optional): List of devices to exclude
        df_audio_features (pd.DataFrame, optional): Audio features data for every track from the Spotify API /audio-features
    Returns:
        modeled_data (pd.DataFrame): The processed DataFrame
    """
    if df_audio_features is not None:
        print(
            f"""\nModeling streaming data: \nstreams: {len(df)} \nof which are unique tracks: {len(df['spotify_track_uri'].unique())} \nunique audio features: {len(df_audio_features['spotify_track_uri'].unique())}"""
        )
        print("\nConcatenating audio features to streaming data set ...")
        if "spotify_track_uri" not in df_audio_features.columns:
            if "uri" in df_audio_features.columns:
                df_audio_features.rename(
                    {"uri": "spotify_track_uri"}, axis=1, inplace=True
                )
            else:
                raise ValueError(
                    "Neither 'spotify_track_uri' nor 'uri' column found in audio features DataFrame"
                )
        df_merged = pd.merge(df, df_audio_features, on="spotify_track_uri", how="left")
    else:
        print(f"Modeling data with {len(df)} rows")
        df_merged = df

    print("Converting timestamp to datetime ...")
    df_merged["datetime"] = pd.to_datetime(df_merged["ts"])
    df_indexed = df_merged.set_index("datetime")
    print("Sorting data set by timestamp ...")
    df_sorted = df_indexed.sort_index()
    print("Renaming columns ...")
    df_renamed = df_sorted.rename(
        columns={
            "ms_played": "listening_time_in_ms",
            "conn_country": "country",
            "ip_addr_decrypted": "ip_address",
            "master_metadata_track_name": "track",
            "master_metadata_album_artist_name": "artist",
            "master_metadata_album_album_name": "album",
            "episode_name": "podcast_episode",
            "episode_show_name": "podcast_show",
        }
    )
    print("Dropping irrelevant columns ...")
    columns_to_drop = [
        "offline_timestamp",
        "ts",
        "podcast_episode",
        "podcast_show",
        "spotify_episode_uri",
        "audiobook_title",
        "audiobook_uri",
        "audiobook_chapter_title",
        "audiobook_chapter_uri",
    ]
    optional_columns_to_drop = ["time_signature", "track_href", "analysis_url"]
    columns_to_drop.extend(
        [col for col in optional_columns_to_drop if col in df_renamed.columns]
    )
    df_renamed_relevant = df_renamed.drop(columns=columns_to_drop)

    print("Renaming existing columns & adding new ones ...")
    df_renamed_relevant["listening_time_in_s"] = (
        df_renamed_relevant["listening_time_in_ms"] / 1000
    )
    df_renamed_relevant["listening_time_in_min"] = (
        df_renamed_relevant["listening_time_in_s"] / 60
    )
    df_renamed_relevant["listening_time_in_h"] = (
        df_renamed_relevant["listening_time_in_min"] / 60
    )
    df_renamed_relevant["song_lenght_in_s"] = df_renamed_relevant["duration_ms"] / 1000
    df_renamed_relevant["song_lenght_in_min"] = (
        df_renamed_relevant["song_lenght_in_s"] / 60
    )
    df_renamed_relevant["song_lenght_in_h"] = (
        df_renamed_relevant["song_lenght_in_min"] / 60
    )
    df_renamed_relevant["relation_listening_lenght"] = df_renamed_relevant[
        "listening_time_in_ms"
    ].div(df_renamed_relevant["duration_ms"].values)
    df_renamed_relevant = df_renamed_relevant[
        df_renamed_relevant["listening_time_in_ms"] > 0
    ]
    df_renamed_relevant.loc[
        df_renamed_relevant["relation_listening_lenght"] > 1,
        "relation_listening_lenght",
    ] = 1
    print("Clearing data set from unwanted tracks ...")
    df_renamed_relevant_cleared = df_renamed_relevant.loc[
        df_renamed_relevant["track"] != "Solace Album Mix"
    ]
    print("Replacing special characters ...")
    df_renamed_devices = df_renamed_relevant_cleared.replace("\\$", "S", regex=True)

    print("Renaming platforms ...")
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*HTC, HTC One_M8.*$)", "HTC One M8", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*SM-G900F.*$)", "Samsung Galaxy S5", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*SM-A520F.*$)", "Samsung Galaxy A5", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*SM-G930F.*$)", "Samsung Galaxy S7", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*SM-G950F.*$)", "Samsung Galaxy S8", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*SM-G973F.*$)", "Samsung Galaxy S10", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*SM-T520.*$)", "Samsung Galaxy Tab Pro 10.1", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*rockchip, rk3288.*$)", "Android Tablet", regex=True
    )

    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*iPad4,5.*$)", "iPad Mini 2", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*iPad5,3.*$)", "iPad Air 2", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*iPad6,4.*$)", "iPad Pro 9.7", regex=True
    )

    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*iPhone5,2.*$)", "iPhone 5", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*iPhone6,2.*$)", "iPhone 5s", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*iPhone8,1.*$)", "iPhone 6s", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*iPhone9,3.*$)", "iPhone 7", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*iPhone10,6.*$)", "iPhone X", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*iPhone11,2.*$)", "iPhone XS", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*iPhone11,8.*$)", "iPhone XR", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*iPhone12,3.*$)", "iPhone 11 Pro", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*iPhone16,1.*$)", "iPhone 15 Pro", regex=True
    )

    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*arm 2.*$)", "MacBook", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*x86 4.*$)", "MacBook", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*Windows 7.*$)", "Windows 7", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*Windows 10.*$)", "Windows 10", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*sony_tv;ps3.*$)", "Playstation 3", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*sony_tv;ps4.*$)", "Playstation 4", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*sony_tv;ps5.*$)", "Playstation 5", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*microsoft;xbox_one.*$)", "XBox One S", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*Partner amazon_salmon Amazon;Echo_Show_5.*$)",
        "Amazon Echo Show 5",
        regex=True,
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*Partner amazon_salmon Amazon;Echo_Dot.*$)", "Amazon Echo Dot", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*Partner amazon_fireos Amazon;Echo_Dot.*$)", "Amazon Echo Dot", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*Partner android_tv Amazon;AFTSSS.*$)", "Amazon Fire TV Stick", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*Partner android_tv Sky;IP100.*$)", "Sky Receiver", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*Partner google cast_tv;Chromecast.*$)", "Google Chromecast", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*Partner google cast;Chromecast_Audio.*$)", "Google Chromecast", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*Partner sonos_ppc Sonos.*$)", "Sonos Amp", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*ppc 0.*$)", "Sonos Amp", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*Partner sonos_imx6 Sonos;PLAY1.*$)", "Sonos One", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*Partner sonos_imx6 Sonos;Play1.*$)", "Sonos One", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*Partner sonos_a53 Sonos;One.*$)", "Sonos One", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*Partner ti_sitara_am3x Yamaha;CRX-N470D.*$)",
        "Yamaha MusicCast",
        regex=True,
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*Partner frontier_jupiter hama;ir26.*$)", "Hama Speaker", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*Bose;Soundtouch.*$)", "Bose Soundtouch", regex=True
    )
    df_renamed_devices["platform"] = df_renamed_devices.platform.str.replace(
        r"(^.*Partner android_tv Sony;BRAVIA4KGB.*$)", "Sony Smart TV", regex=True
    )

    if exclude_devices is not None:
        print(f"Excluding devices: {exclude_devices}")
        df_cleared = df_renamed_devices[
            ~df_renamed_devices["platform"].isin(exclude_devices)
        ]
        # With a little bit of research, we found that the following correlations between the name & ID of a platform:
        # iPhone 5 == 'iPhone5,2'
        # iPhone 7 == 'iPhone9,3'
        # iPhone XS == 'iPhone11,2'
        # Samsung A5 = 'samsung, SM-A520F'

    else:
        df_cleared = df_renamed_devices

    # Perform feature engineering to prepare for context-based analysis
    df_featured = feature_engineering(df_cleared)

    return df_featured
