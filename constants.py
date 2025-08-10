EXCLUDE_DEVICES = [
    "iPhone 5 (GSM+CDMA)",
    "iPhone 7",
    "iPhone XS",
    "Samsung Galaxy A5",
    "Android Tablet",
    "Hama Speaker",
    "android",
    "playstation",
]

AUDIO_FEATURES_PATHS = {
    "data/audio-features/reccobeats/tracks_with_audio_features.csv": "spotify_track_uri",
    "data/audio-features/beatport_track_features.csv": "track_id",
    "data/audio-features/figueroa_features.csv": "id",
    "data/audio-features/million_song_dataset_spotify_lastfm_2023.csv": "spotify_id",
    "data/audio-features/orlandi_weekly_top_songs_features.csv": "id",
    "data/audio-features/pandya_features.csv": "track_id",
    "data/audio-features/tomigelo_april_2019_features.csv": "track_id",
    "data/audio-features/tomigelo_november_2018_features.csv": "track_id",
}

WEIGHTS = {
    "count": 0.5,
    "skip_rate": 0.3,
    "attention_span": 0.2,
}

# --- Constants for period detection ---
WINDOW_SIZE_DAYS = 7  # Days
STEP_SIZE_DAYS = 1  # Days
ANOMALY_THRESHOLD = 0.6  # % of tracks in window that must have anomaly
MIN_TRACKS_FOR_PLAYLIST = 15
MIN_DURATION_DAYS = 2

NUMERICAL_FEATURES_TO_CHECK = [
    "tempo",
    "speechiness",
    "liveness",
    "acousticness",
    "energy",
    "danceability",
    "loudness",
    "valence",
    "instrumentalness",
]

MIN_TRACKS_PER_SLOT = 10
