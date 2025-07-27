WEIGHTS = {
    "count": 0.5,
    "skip_rate": 0.3,
    "attention_span": 0.2,
}

# --- Constants for period detection ---
WINDOW_SIZE_DAYS = 7  # Days
STEP_SIZE_DAYS = 1  # Days
ANOMALY_THRESHOLD = 0.4  # % of tracks in window that must have anomaly
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
    "popularity",
]

MIN_TRACKS_PER_SLOT = 10

