from typing import Callable, Dict, List, Tuple

import pandas as pd

from diversity import enforce_artist_diversity
from presets import PLAYLIST_PRESETS
from scoring import score


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


def ensure_required_columns(df):
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
        ("artist_loyalty", 0),
        ("personal_popularity", 0),
        ("popularity", 0),
        ("valence", 0.0),
        ("season", ""),
    ]
    for col, default in required_columns:
        if col not in df.columns:
            df[col] = default


def generate_playlists(
    df: pd.DataFrame,
    playlists_to_generate: str | List[str],
    num_songs: int,
    max_per_artist: int | None = None,
) -> Dict[str, List[str]]:
    """
    Generate playlists using preset conditions and scoring. Deduplicate tracks and enforce artist diversity.
    """
    print("Generating context playlists with recommender logic...")
    ensure_required_columns(df)

    if isinstance(playlists_to_generate, str):
        playlists_to_generate = [playlists_to_generate]
        
    presets = PLAYLIST_PRESETS()
    generated_playlists = {}
    max_timestamp = df.index.max().timestamp() if not df.empty else 1
    max_per_artist = get_max_per_artist(num_songs, max_per_artist)
    for name in playlists_to_generate:
        name_title_case = name.title()
        if name_title_case in presets:
            preset = presets[name_title_case]
            conditions = preset["conditions"]
            scoring_fn = preset["scoring"]
            playlist_df = filter_by_conditions(df, conditions)
            playlist_df = score(playlist_df, scoring_fn, max_timestamp)
            playlist_df = playlist_df.sort_values("score", ascending=False)
            tracks = list(zip(playlist_df["track"], playlist_df["artist"]))
            unique_tracks = get_unique_tracks(tracks)
            unique_tracks = enforce_artist_diversity(
                unique_tracks, max_per_artist=max_per_artist
            )
            generated_playlists[name_title_case] = unique_tracks[:num_songs]
            print(
                f"Playlist '{name_title_case}' generated with {len(generated_playlists[name_title_case])} tracks. (max {max_per_artist} per artist)"
            )
        else:
            print(f"Warning: Playlist type '{name}' not recognized. Skipped.")
    return generated_playlists
