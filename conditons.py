def is_weekday(row) -> bool:
    return row.get("is_weekday", False)


def time_of_day(row, periods=None) -> bool:
    if periods is None:
        periods = []
    return row.get("time_of_day") in periods


def device_contains(row, devices=None) -> bool:
    if devices is None:
        devices = []
    platform = str(row.get("platform", ""))
    return any(device.lower() in platform.lower() for device in devices)


def not_skipped(row) -> bool:
    return not row.get("skipped", False)


def high_energy(row, threshold=0.7) -> bool:
    return row.get("energy", 0.0) > threshold


def high_danceability(row, threshold=0.7) -> bool:
    return row.get("danceability", 0.0) > threshold


def low_speechiness(row, threshold=0.3) -> bool:
    return row.get("speechiness", 1.0) < threshold


def high_instrumentalness(row, threshold=0.5) -> bool:
    return row.get("instrumentalness", 0.0) > threshold


def high_attention_span(row, threshold=0.9) -> bool:
    return row.get("attention_span", 0.0) > threshold
