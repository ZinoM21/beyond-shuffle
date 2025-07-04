from typing import Dict

from conditons import (
    device_contains,
    high_attention_span,
    high_danceability,
    high_energy,
    high_instrumentalness,
    is_weekday,
    low_speechiness,
    not_skipped,
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
            "conditions": [
                (is_weekday, {}),
                (time_of_day, {"periods": ["Morning", "Evening"]}),
                (device_contains, {"devices": ["iPhone", "Android"]}),
                (not_skipped, {}),
            ],
            "scoring": commute_score,
        },
        "Workout": {
            "conditions": [
                (high_energy, {"threshold": 0.7}),
                (high_danceability, {"threshold": 0.7}),
                (not_skipped, {}),
                (time_of_day, {"periods": ["Afternoon", "Evening"]}),
            ],
            "scoring": workout_score,
        },
        "Focus": {
            "conditions": [
                (low_speechiness, {"threshold": 0.3}),
                (high_instrumentalness, {"threshold": 0.5}),
                (high_attention_span, {"threshold": 0.9}),
                (not_skipped, {}),
            ],
            "scoring": focus_score,
        },
        "Artist Loyalty": {
            "conditions": [
                (lambda row: row.get("artist_loyalty", 0) >= 2, {}),
            ],
            "scoring": artist_loyalty_score,
        },
        "Vacation": {
            "conditions": [
                (lambda row: row.get("is_vacation", False), {}),
            ],
            "scoring": vacation_score,
        },
        "Background": {
            "conditions": [
                (device_contains, {"devices": ["Home Pod", "Amazon Echo", "Echo Dot"]}),
            ],
            "scoring": background_score,
        },
        "Sing Along": {
            "conditions": [
                (device_contains, {"devices": ["iPhone", "Android"]}),
                (high_attention_span, {"threshold": 0.95}),
            ],
            "scoring": sing_along_score,
        },
        "Evening Chill": {
            "conditions": [
                (time_of_day, {"periods": ["Evening"]}),
                (lambda row: row.get("energy", 1.0) < 0.5, {}),
                (lambda row: row.get("valence", 0.0) > 0.5, {}),
            ],
            "scoring": evening_chill_score,
        },
        "Spring Vibes": {
            "conditions": [
                (lambda row: row.get("season", "") == "Spring", {}),
            ],
            "scoring": spring_vibes_score,
        },
        "Weekend Party": {
            "conditions": [
                (lambda row: not row.get("is_weekday", True), {}),
                (high_danceability, {"threshold": 0.7}),
                (high_energy, {"threshold": 0.7}),
            ],
            "scoring": weekend_party_score,
        },
        "Popular Favorites": {
            "conditions": [
                (lambda row: row.get("personal_popularity", 0) >= 3, {}),
                (lambda row: row.get("popularity", 0) >= 60, {}),
            ],
            "scoring": popular_favorites_score,
        },
    }
