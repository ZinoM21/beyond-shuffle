from typing import Dict

from feature_heuristics import (
    country,
    device_contains,
    high_acousticness,
    high_attention_span,
    high_danceability,
    high_energy,
    high_personal_popularity,
    high_popularity,
    is_first_played,
    is_vacation,
    is_weekday,
    low_energy,
    low_valence,
    not_skipped,
    season,
    time_of_day,
    year_is,
)
from scoring import (
    commute_score,
    firsts_milestones_score,
    heartbreak_healing_score,
    late_night_reflections_score,
    popular_favorites_score,
    semester_abroad_us_score,
    summer_memories_score,
    travel_memories_score,
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
                (device_contains, {"devices": ["iPhone", "ios"]}),
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
        # "Focus": {
        #     "heuristics": [
        #         (low_speechiness, {}),
        #         (high_instrumentalness, {}),
        #         (high_attention_span, {}),
        #         (not_skipped, {}),
        #     ],
        #     "scoring": focus_score,
        # },
        # "Artist Loyalty": {
        #     "heuristics": [
        #         (lambda row: row.get("artist_loyalty", 0) >= 2, {}),
        #     ],
        #     "scoring": artist_loyalty_score,
        # },
        "Vacation": {
            "heuristics": [
                (is_vacation, {}),
            ],
            "scoring": vacation_score,
        },
        # "Background": {
        #     "heuristics": [
        #         (device_contains, {"devices": ["Home Pod", "Amazon Echo", "Echo Dot"]}),
        #     ],
        #     "scoring": background_score,
        # },
        # "Sing Along": {
        #     "heuristics": [
        #         (device_contains, {"devices": ["iPhone", "Android"]}),
        #         (high_attention_span, {}),
        #     ],
        #     "scoring": sing_along_score,
        # },
        # "Evening Chill": {
        #     "heuristics": [
        #         (time_of_day, {"periods": ["Evening"]}),
        #         (low_energy, {}),
        #         (high_valence, {}),
        #     ],
        #     "scoring": evening_chill_score,
        # },
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
        "Summer Memories": {
            "heuristics": [
                (season, {"season": ["Summer"]}),
                (high_attention_span, {}),
                (high_personal_popularity, {"threshold": 5}),
            ],
            "scoring": summer_memories_score,
        },
        "Travel Memories": {
            "heuristics": [
                (is_vacation, {}),
                (device_contains, {"devices": ["iPhone", "Android", "iPad"]}),
                (high_energy, {}),
            ],
            "scoring": travel_memories_score,
        },
        "Late Night Reflections": {
            "heuristics": [
                (time_of_day, {"periods": ["Night"]}),
                (low_valence, {"threshold": 0.4}),
                (high_acousticness, {"threshold": 0.6}),
                (high_attention_span, {"threshold": 0.8}),
            ],
            "scoring": late_night_reflections_score,
        },
        "Firsts & Milestones": {
            "heuristics": [
                (is_first_played, {"months": 2}),
                (high_attention_span, {"threshold": 0.7}),
            ],
            "scoring": firsts_milestones_score,
        },
        "Heartbreak & Healing": {
            "heuristics": [
                (low_valence, {"threshold": 0.3}),
                (low_energy, {"threshold": 0.4}),
                (high_attention_span, {"threshold": 0.8}),
            ],
            "scoring": heartbreak_healing_score,
        },
        "Semester Abroad": {
            "heuristics": [
                # (
                #     device_contains,
                #     {"devices": ["iPhone", "ios", "Mac", "osx"]},
                # ),
                (country, {"country": "US"}),
                (season, {"season": ["Summer", "Fall", "Winter"]}),
                (year_is, {"year": 2024}),
                (high_attention_span, {"threshold": 0.7}),
            ],
            "scoring": semester_abroad_us_score,
        },
    }
