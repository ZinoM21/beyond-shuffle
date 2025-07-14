def is_weekday(row) -> bool:
    return row.get("is_weekday", False)


def time_of_day(row, periods=None) -> bool:
    if periods is None:
        periods = []
    return row.get("time_of_day") in periods


def season(row, season=None) -> bool:
    if season is None:
        season = []
    return row.get("season") in season


def device_contains(row, devices=None) -> bool:
    if devices is None:
        devices = []
    platform = str(row.get("platform", ""))
    return any(device.lower() in platform.lower() for device in devices)


def is_vacation(row) -> bool:
    return row.get("is_vacation", False)


def not_skipped(row) -> bool:
    return not row.get("skipped", False)


def high_attention_span(row, threshold=0.9) -> bool:
    return row.get("attention_span", 0.0) > threshold


def high_personal_popularity(row, threshold=3) -> bool:
    return row.get("personal_popularity", 0) >= threshold


def high_popularity(row, threshold=60) -> bool:
    return row.get("popularity", 0) >= threshold


## Audio Features


def high_energy(row, threshold=0.7) -> bool:
    return row.get("energy", 0.0) > threshold


def low_energy(row, threshold=0.5) -> bool:
    return row.get("energy", 1.0) < threshold


def high_valence(row, threshold=0.5) -> bool:
    return row.get("valence", 0.0) > threshold


def low_valence(row, threshold=0.5) -> bool:
    return row.get("valence", 1.0) < threshold


def high_loudness(row, threshold=0.7) -> bool:
    return row.get("loudness", 0.0) > threshold


def low_loudness(row, threshold=0.3) -> bool:
    return row.get("loudness", 1.0) < threshold


def high_tempo(row, threshold=130) -> bool:
    return row.get("tempo", 0) > threshold


def low_tempo(row, threshold=120) -> bool:
    return row.get("tempo", 0) < threshold


def high_acousticness(row, threshold=0.7) -> bool:
    return row.get("acousticness", 0.0) > threshold


def high_danceability(row, threshold=0.7) -> bool:
    return row.get("danceability", 0.0) > threshold


def high_speechiness(row, threshold=0.7) -> bool:
    return row.get("speechiness", 0.0) > threshold


def low_speechiness(row, threshold=0.3) -> bool:
    return row.get("speechiness", 1.0) < threshold


def high_instrumentalness(row, threshold=0.5) -> bool:
    return row.get("instrumentalness", 0.0) > threshold


# Heuristic: is_first_played (track first played within X months of its first appearance)
def is_first_played(row, months=2) -> bool:
    # Assumes 'first_played_months_ago' is computed in feature_engineering
    return row.get("first_played_months_ago", 99) <= months


def country(row, country=None) -> bool:
    if country is None:
        return False
    return str(row.get("country", "")).upper() == str(country).upper()


def year_is(row, year=None) -> bool:
    if year is None:
        return False
    return int(row.get("year", 0)) == int(year)
