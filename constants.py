EXCLUDE_DEVICES = [
    # "iPhone 5 (GSM+CDMA)",
    # "iPhone 7",
    # "iPhone XS",
    # "Samsung Galaxy A5",
    # "Android Tablet",
    # "Hama Speaker",
    # "android",
    # "playstation",
]


# Enrichment data sources
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

# Candidate selection config
CANDIDATE_SELECTION_WEIGHTS = {
    "count": 0.5,
    "skip_rate": 0.3,
    "attention_span": 0.2,
}

# Period detection config
PERIOD_MIN_DAYS = 2
WINDOW_SIZE_DAYS = 4
STEP_SIZE_DAYS = 1
PERIOD_FEATURE_ZSCORE_THRESHOLD = 0.6
MIN_TRACKS_FOR_PLAYLIST = 15

CATEGORICAL_FEATURES_TO_CHECK = [
    "country",
    "platform",
    "day_of_week",
]

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

# Habit config
HABIT_MIN_WEEKS = 6
HABIT_MIN_STREAMS_PER_SLOT = 100
HABIT_FEATURE_ZSCORE_THRESHOLD = 0.8
HABIT_MIN_NUM_FEATURES = 3
HABIT_MAX_SLOTS_PER_SCHEMA = 3
HABIT_TOP_PLATFORMS = 1


HABIT_AUDIO_CLUSTER_K = 5
HABIT_MIN_CLUSTER_SHARE = 0.6
HABIT_MIN_CLUSTER_WEEKS = 6

# Behavioral features for habit detection (computed in feature_engineering)
HABIT_BEHAVIORAL_FEATURES = [
    "attention_span",
    "skipped",
    "session_length",
    "session_duration_min",
]
HABIT_BEHAVIORAL_ZSCORE_THRESHOLD = 0.5

# Artist concentration threshold within a slot (share of top artist streams)
HABIT_TOP_ARTIST_SHARE = 0.9

# Minimum share of a device/group within a slot to include it in the habit name
HABIT_MIN_DEVICE_SHARE = 0.1
