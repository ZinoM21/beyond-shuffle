from typing import Dict

from feature_heuristics import (
    device_contains,
    high_attention_span,
    high_danceability,
    high_energy,
    high_instrumentalness,
    high_personal_popularity,
    high_popularity,
    high_valence,
    is_vacation,
    is_weekday,
    low_energy,
    low_speechiness,
    not_skipped,
    season,
    time_of_day,
)
from scoring import (
    artist_loyalty_score,
    background_score,
    commute_score,
    evening_chill_score,
    focus_score,
    popular_favorites_score,
    sing_along_score,
    spring_vibes_score,
    vacation_score,
    weekend_party_score,
    workout_score,
)


def PLAYLIST_PRESETS() -> Dict[str, Dict]:
    return {
        "Commute": {
            "heuristics": [
                (is_weekday, {}),
                (time_of_day, {"periods": ["Morning", "Evening"]}),
                (device_contains, {"devices": ["iPhone", "Android"]}),
                (not_skipped, {}),
            ],
            "scoring": commute_score,
        },
        "Workout": {
            "heuristics": [
                (high_energy, {}),
                (high_danceability, {}),
                (not_skipped, {}),
                (time_of_day, {"periods": ["Afternoon", "Evening"]}),
            ],
            "scoring": workout_score,
        },
        "Focus": {
            "heuristics": [
                (low_speechiness, {}),
                (high_instrumentalness, {}),
                (high_attention_span, {}),
                (not_skipped, {}),
            ],
            "scoring": focus_score,
        },
        "Artist Loyalty": {
            "heuristics": [
                (lambda row: row.get("artist_loyalty", 0) >= 2, {}),
            ],
            "scoring": artist_loyalty_score,
        },
        "Vacation": {
            "heuristics": [
                (is_vacation, {}),
            ],
            "scoring": vacation_score,
        },
        "Background": {
            "heuristics": [
                (device_contains, {"devices": ["Home Pod", "Amazon Echo", "Echo Dot"]}),
            ],
            "scoring": background_score,
        },
        "Sing Along": {
            "heuristics": [
                (device_contains, {"devices": ["iPhone", "Android"]}),
                (high_attention_span, {}),
            ],
            "scoring": sing_along_score,
        },
        "Evening Chill": {
            "heuristics": [
                (time_of_day, {"periods": ["Evening"]}),
                (low_energy, {}),
                (high_valence, {}),
            ],
            "scoring": evening_chill_score,
        },
        "Spring Vibes": {
            "heuristics": [
                (season, {"season": ["Spring"]}),
            ],
            "scoring": spring_vibes_score,
        },
        "Weekend Party": {
            "heuristics": [
                (lambda row: not is_weekday(row), {}),
                (high_danceability, {}),
                (high_energy, {}),
            ],
            "scoring": weekend_party_score,
        },
        "Popular Favorites": {
            "heuristics": [
                (high_personal_popularity, {}),
                (high_popularity, {}),
            ],
            "scoring": popular_favorites_score,
        },
    }
