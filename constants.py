DATA_PATH = "./data/out/enriched_data.parquet"

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

AUDIO_FEATURES_PATH = "./data/recco-audio-features/tracks_with_audio_features.csv"
AUDIO_FEATURES_PATHS = {
    # "data/recco-audio-features/tracks_with_audio_features.csv": "spotify_track_uri",
    "data/test-features/dataset.csv": "track_id",
    "data/test-features/million-song-dataset-spotify-lastfm.csv": "spotify_id",
    "data/test-features/spotify_top_songs_audio_features.csv": "id",
    "data/test-features/SpotifyAudioFeaturesNov2018.csv": "track_id",
    "data/test-features/SpotifyAudioFeaturesApril2019.csv": "track_id",
    "data/test-features/tracks_features.csv": "id",
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
