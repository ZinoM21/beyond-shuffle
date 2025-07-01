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

# --- Main Playlist Generation Function ---
def generate_context_playlists(df: pd.DataFrame, playlists_to_generate: str | List[str], num_songs: int) -> Dict[str, List[str]]:
    """
    Generates context-based playlists from a feature-rich dataframe using modular feature functions.
    """
    print("Generating context playlists...")

    # Ensure required columns exist (add defaults if missing)
    for col, default in [
        ('energy', 0.0),
        ('danceability', 0.0),
        ('speechiness', 0.0),
        ('instrumentalness', 0.0),
        ('skipped', False),
        ('attention_span', 0.0),
        ('platform', ''),
        ('time_of_day', ''),
        ('is_weekday', False),
    ]:
        if col not in df.columns:
            df[col] = default

    if isinstance(playlists_to_generate, str):
        playlists_to_generate = [playlists_to_generate]

    available_playlists = playlist_conditions()
    generated_playlists = {}

    for name in playlists_to_generate:
        name_title_case = name.title()
        if name_title_case in available_playlists:
            conditions = available_playlists[name_title_case]
            # Build a mask by applying all feature functions
            mask = df.apply(
                lambda row: all(
                    func(row, **kwargs) for func, kwargs in conditions
                ), axis=1
            )
            playlist_df = df[mask]
            # Prioritize tracks by play count within the context
            top_tracks = playlist_df['track'].value_counts().head(num_songs).index.tolist()
            # For each top track, get the most common artist for that track in the filtered context
            playlist_tracks = []
            for track in top_tracks:
                artist = (
                    playlist_df[playlist_df['track'] == track]['artist']
                    .mode()
                    .iloc[0] if not playlist_df[playlist_df['track'] == track]['artist'].empty else 'Unknown Artist'
                )
                playlist_tracks.append((track, artist))
            generated_playlists[name_title_case] = playlist_tracks
            print(f"Playlist '{name_title_case}' generated with {len(playlist_tracks)} tracks.")
        else:
            print(f"Warning: Playlist type '{name}' not recognized. Skipped.")

    return generated_playlists

# --- Add more feature functions and playlist types as needed for extensibility --- 