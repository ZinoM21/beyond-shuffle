import pandas as pd

from feature_engineering import feature_engineering


def concat_with_audio_features(
    df: pd.DataFrame, df_audio_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Concatenate streaming data with audio features.
    """
    print("\nConcatenating audio features to streaming data set ...")
    if "spotify_track_uri" not in df_audio_features.columns:
        if "uri" in df_audio_features.columns:
            df_audio_features.rename({"uri": "spotify_track_uri"}, axis=1, inplace=True)
        else:
            raise ValueError(
                "Neither 'spotify_track_uri' nor 'uri' column found in audio features DataFrame"
            )
    df = pd.merge(df, df_audio_features, on="spotify_track_uri", how="left")
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop irrelevant columns from the DataFrame.
    """
    df = df.copy()
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
        [col for col in optional_columns_to_drop if col in df.columns]
    )
    df = df.drop(columns=columns_to_drop)
    return df


def rename_devices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename devices to a more readable format.
    """
    df = df.copy()
    print("Renaming platforms ...")

    replacements = [
        (r"(^.*HTC, HTC One_M8.*$)", "HTC One M8"),
        (r"(^.*SM-G900F.*$)", "Samsung Galaxy S5"),
        (r"(^.*SM-A520F.*$)", "Samsung Galaxy A5"),
        (r"(^.*SM-G930F.*$)", "Samsung Galaxy S7"),
        (r"(^.*SM-G950F.*$)", "Samsung Galaxy S8"),
        (r"(^.*SM-G973F.*$)", "Samsung Galaxy S10"),
        (r"(^.*SM-T520.*$)", "Samsung Galaxy Tab Pro 10.1"),
        (r"(^.*rockchip, rk3288.*$)", "Android Tablet"),
        (r"(^.*iPad4,5.*$)", "iPad Mini 2"),
        (r"(^.*iPad5,3.*$)", "iPad Air 2"),
        (r"(^.*iPad6,4.*$)", "iPad Pro 9.7"),
        (r"(^.*iPhone3,1.*$)", "iPhone 4"),
        (r"(^.*iPhone3,2.*$)", "iPhone 4 GSM Rev A"),
        (r"(^.*iPhone3,3.*$)", "iPhone 4 CDMA"),
        (r"(^.*iPhone4,1.*$)", "iPhone 4S"),
        (r"(^.*iPhone5,1.*$)", "iPhone 5 (GSM)"),
        (r"(^.*iPhone5,2.*$)", "iPhone 5 (GSM+CDMA)"),
        (r"(^.*iPhone5,3.*$)", "iPhone 5C (GSM)"),
        (r"(^.*iPhone5,4.*$)", "iPhone 5C (Global)"),
        (r"(^.*iPhone6,1.*$)", "iPhone 5S (GSM)"),
        (r"(^.*iPhone6,2.*$)", "iPhone 5S (Global)"),
        (r"(^.*iPhone7,1.*$)", "iPhone 6 Plus"),
        (r"(^.*iPhone7,2.*$)", "iPhone 6"),
        (r"(^.*iPhone8,1.*$)", "iPhone 6s"),
        (r"(^.*iPhone8,2.*$)", "iPhone 6s Plus"),
        (r"(^.*iPhone8,4.*$)", "iPhone SE (GSM)"),
        (r"(^.*iPhone9,1.*$)", "iPhone 7"),
        (r"(^.*iPhone9,2.*$)", "iPhone 7 Plus"),
        (r"(^.*iPhone9,3.*$)", "iPhone 7"),
        (r"(^.*iPhone9,4.*$)", "iPhone 7 Plus"),
        (r"(^.*iPhone10,1.*$)", "iPhone 8"),
        (r"(^.*iPhone10,2.*$)", "iPhone 8 Plus"),
        (r"(^.*iPhone10,3.*$)", "iPhone X Global"),
        (r"(^.*iPhone10,4.*$)", "iPhone 8"),
        (r"(^.*iPhone10,5.*$)", "iPhone 8 Plus"),
        (r"(^.*iPhone10,6.*$)", "iPhone X GSM"),
        (r"(^.*iPhone11,2.*$)", "iPhone XS"),
        (r"(^.*iPhone11,4.*$)", "iPhone XS Max"),
        (r"(^.*iPhone11,6.*$)", "iPhone XS Max Global"),
        (r"(^.*iPhone11,8.*$)", "iPhone XR"),
        (r"(^.*iPhone12,1.*$)", "iPhone 11"),
        (r"(^.*iPhone12,3.*$)", "iPhone 11 Pro"),
        (r"(^.*iPhone12,5.*$)", "iPhone 11 Pro Max"),
        (r"(^.*iPhone12,8.*$)", "iPhone SE 2nd Gen"),
        (r"(^.*iPhone13,1.*$)", "iPhone 12 Mini"),
        (r"(^.*iPhone13,2.*$)", "iPhone 12"),
        (r"(^.*iPhone13,3.*$)", "iPhone 12 Pro"),
        (r"(^.*iPhone13,4.*$)", "iPhone 12 Pro Max"),
        (r"(^.*iPhone14,2.*$)", "iPhone 13 Pro"),
        (r"(^.*iPhone14,3.*$)", "iPhone 13 Pro Max"),
        (r"(^.*iPhone14,4.*$)", "iPhone 13 Mini"),
        (r"(^.*iPhone14,5.*$)", "iPhone 13"),
        (r"(^.*iPhone14,6.*$)", "iPhone SE 3rd Gen"),
        (r"(^.*iPhone14,7.*$)", "iPhone 14"),
        (r"(^.*iPhone14,8.*$)", "iPhone 14 Plus"),
        (r"(^.*iPhone15,2.*$)", "iPhone 14 Pro"),
        (r"(^.*iPhone15,3.*$)", "iPhone 14 Pro Max"),
        (r"(^.*iPhone15,4.*$)", "iPhone 15"),
        (r"(^.*iPhone15,5.*$)", "iPhone 15 Plus"),
        (r"(^.*iPhone16,1.*$)", "iPhone 15 Pro"),
        (r"(^.*iPhone16,2.*$)", "iPhone 15 Pro Max"),
        (r"(^.*iPhone17,1.*$)", "iPhone 16 Pro"),
        (r"(^.*iPhone17,2.*$)", "iPhone 16 Pro Max"),
        (r"(^.*iPhone17,3.*$)", "iPhone 16"),
        (r"(^.*iPhone17,4.*$)", "iPhone 16 Plus"),
        (r"(^.*iPhone17,5.*$)", "iPhone 16e"),
        (r"(^.*arm 2.*$)", "MacBook ARM"),
        (r"(^.*x86 4.*$)", "MacBook Intel"),
        (r"(^.*Windows 7.*$)", "Windows 7"),
        (r"(^.*Windows 10.*$)", "Windows 10"),
        (r"(^.*sony_tv;ps3.*$)", "Playstation 3"),
        (r"(^.*sony_tv;ps4.*$)", "Playstation 4"),
        (r"(^.*sony_tv;ps5.*$)", "Playstation 5"),
        (r"(^.*microsoft;xbox_one.*$)", "XBox One S"),
        (r"(^.*Partner amazon_salmon Amazon;Echo_Show_5.*$)", "Amazon Echo Show 5"),
        (r"(^.*Partner amazon_salmon Amazon;Echo_Dot.*$)", "Amazon Echo Dot"),
        (r"(^.*Partner amazon_fireos Amazon;Echo_Dot.*$)", "Amazon Echo Dot"),
        (r"(^.*Partner android_tv Amazon;AFTSSS.*$)", "Amazon Fire TV Stick"),
        (r"(^.*Partner android_tv Sky;IP100.*$)", "Sky Receiver"),
        (r"(^.*Partner google cast_tv;Chromecast.*$)", "Google Chromecast"),
        (r"(^.*Partner google cast;Chromecast_Audio.*$)", "Google Chromecast"),
        (r"(^.*Partner sonos_ppc Sonos.*$)", "Sonos Amp"),
        (r"(^.*ppc 0.*$)", "Sonos Amp"),
        (r"(^.*Partner sonos_imx6 Sonos;PLAY1.*$)", "Sonos One"),
        (r"(^.*Partner sonos_imx6 Sonos;Play1.*$)", "Sonos One"),
        (r"(^.*Partner sonos_a53 Sonos;One.*$)", "Sonos One"),
        (r"(^.*Partner ti_sitara_am3x Yamaha;CRX-N470D.*$)", "Yamaha MusicCast"),
        (r"(^.*Partner frontier_jupiter hama;ir26.*$)", "Hama Speaker"),
        (r"(^.*Bose;Soundtouch.*$)", "Bose Soundtouch"),
        (r"(^.*Partner android_tv Sony;BRAVIA4KGB.*$)", "Sony Smart TV"),
    ]

    patterns, names = zip(*replacements)
    df["platform"] = df["platform"].replace(list(patterns), list(names), regex=True)

    return df


def drop_devices(df: pd.DataFrame, exclude_devices: list[str]) -> pd.DataFrame:
    """
    Drop devices from the DataFrame.
    """
    df = df.copy()
    print("Excluding devices:")
    for device in exclude_devices:
        print(f"- {device}")
    df = df[~df["platform"].isin(exclude_devices)]
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

    print(
        f"""\nModeling streaming data: \nstreams: {len(df)} \nof which are unique tracks: {len(df['spotify_track_uri'].unique())}"""
    )

    if df_audio_features is not None:
        print(
            f"""unique audio features: {len(df_audio_features['spotify_track_uri'].unique())}"""
        )
        df = concat_with_audio_features(df, df_audio_features)

    print("Converting timestamp to datetime ...")
    df["datetime"] = pd.to_datetime(df["ts"])
    df_indexed = df.set_index("datetime")

    print("Sorting data set by timestamp ...")
    df_sorted = df_indexed.sort_index()

    print("Renaming columns ...")
    df_renamed = df_sorted.rename(
        columns={
            "conn_country": "country",
            "ip_addr_decrypted": "ip_address",
            "master_metadata_track_name": "track",
            "master_metadata_album_artist_name": "artist",
            "master_metadata_album_album_name": "album",
            "episode_name": "podcast_episode",
            "episode_show_name": "podcast_show",
        }
    )

    df = drop_irrelevant_columns(df_renamed)

    print("Replacing special characters ...")
    df = df.replace("\\$", "S", regex=True)

    df = rename_devices(df)

    if exclude_devices is not None:
        df = drop_devices(df, exclude_devices)

    df = feature_engineering(df)

    return df
