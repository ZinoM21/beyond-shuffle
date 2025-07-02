import pandas as pd
from typing import Dict, List

# --- Feature Functions ---
def is_weekday(row) -> bool:
    """Return True if the row is a weekday."""
    return row.get('is_weekday', False)

def time_of_day(row, periods=None) -> bool:
    """Return True if the row's time_of_day is in the given periods."""
    if periods is None:
        periods = []
    return row.get('time_of_day') in periods

def device_contains(row, devices=None) -> bool:
    """Return True if the platform contains any of the given devices (case-insensitive)."""
    if devices is None:
        devices = []
    platform = str(row.get('platform', ''))
    return any(device.lower() in platform.lower() for device in devices)

def not_skipped(row) -> bool:
    """Return True if the track was not skipped."""
    return not row.get('skipped', False)

def high_energy(row, threshold=0.7) -> bool:
    return row.get('energy', 0.0) > threshold

def high_danceability(row, threshold=0.7) -> bool:
    return row.get('danceability', 0.0) > threshold

def low_speechiness(row, threshold=0.3) -> bool:
    return row.get('speechiness', 1.0) < threshold

def high_instrumentalness(row, threshold=0.5) -> bool:
    return row.get('instrumentalness', 0.0) > threshold

def high_attention_span(row, threshold=0.9) -> bool:
    """Return True if the listening time over the track duration is greater than the threshold (default: 0.9 = 90% of the track duration)"""
    return row.get('attention_span', 0.0) > threshold

# --- Playlist Condition Registry ---
# Each playlist type is a list of (feature_function, kwargs) tuples
def playlist_conditions():
    return {
        'Commute': [
            (is_weekday, {}),
            (time_of_day, {'periods': ['Morning', 'Evening']}),
            (device_contains, {'devices': ['iPhone', 'Android']}),
            (not_skipped, {}),
        ],
        'Workout': [
            (high_energy, {'threshold': 0.7}),
            (high_danceability, {'threshold': 0.7}),
            (not_skipped, {}),
            (time_of_day, {'periods': ['Afternoon', 'Evening']}),
        ],
        'Focus': [
            (low_speechiness, {'threshold': 0.3}),
            (high_instrumentalness, {'threshold': 0.5}),
            (high_attention_span, {'threshold': 0.9}),
            (not_skipped, {}),
        ],
    }


def ensure_required_columns(df: pd.DataFrame) -> None:
    required_columns = [
        ("energy", 0.0),
        ("danceability", 0.0),
        ("speechiness", 0.0),
        ("instrumentalness", 0.0),
        ("skipped", False),
        ("attention_span", 0.0),
        ("platform", ""),
        ("time_of_day", ""),
        ("is_weekday", False),
    ]
    for col, default in required_columns:
        if col not in df.columns:
            df[col] = default


def get_max_per_artist(num_songs: int, max_per_artist: int | None) -> int:
    if max_per_artist is not None:
        return max_per_artist
    return max(1, int(0.15 * num_songs))


def filter_by_conditions(
    df: pd.DataFrame, conditions: List[Tuple[Callable, Dict]]
) -> pd.DataFrame:
    mask = df.apply(
        lambda row: all(func(row, **kwargs) for func, kwargs in conditions), axis=1
    )
    return df[mask].copy()


def get_unique_tracks(tracks: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    unique_tracks = []
    for t in tracks:
        if t not in seen:
            unique_tracks.append(t)
            seen.add(t)
    return unique_tracks


def generate_playlists(
    df: pd.DataFrame,
    playlists_to_generate: str | List[str],
    num_songs: int,
    max_per_artist: int | None = None,
) -> Dict[str, List[str]]:
    """
    Generates context-based playlists from a feature-rich dataframe using modular feature functions.

    Args:
        df: The dataframe containing the listening history.
        playlists_to_generate: The name(s) of the playlist(s) to generate.
        num_songs: The number of songs to generate.
        max_per_artist: The maximum number of songs per artist.

    Returns:
        A dictionary of playlist names and their corresponding tracks.
    """
    print("Generating context playlists...")

    ensure_required_columns(df)

    if isinstance(playlists_to_generate, str):
        playlists_to_generate = [playlists_to_generate]

    available_playlists = playlist_conditions()
    generated_playlists = {}

    max_timestamp = df.index.max().timestamp() if not df.empty else 1
    max_per_artist = get_max_per_artist(num_songs, max_per_artist)

    for name in playlists_to_generate:
        name_title_case = name.title()
        if name_title_case in available_playlists:
            conditions = available_playlists[name_title_case]
            filtered_df = filter_by_conditions(df, conditions)

            scored_df = score(filtered_df, name_title_case, max_timestamp)
            sorted_df = scored_df.sort_values("score", ascending=False)

            tracks = list(zip(sorted_df["track"], sorted_df["artist"]))
            unique_tracks = get_unique_tracks(tracks)
            diverse_tracks = enforce_artist_diversity(
                unique_tracks, max_per_artist=max_per_artist
            )
            selected_tracks = diverse_tracks[
                :num_songs
            ]  # select the top num_songs of all tracks

            generated_playlists[name_title_case] = selected_tracks
            print(
                f"Playlist '{name_title_case}' generated with {len(selected_tracks)} tracks. (max {max_per_artist} per artist)"
            )
        else:
            print(f"Warning: Playlist type '{name}' not recognized. Skipped.")

    return generated_playlists
