def enforce_artist_diversity(tracks, max_per_artist=2):
    artist_count = {}
    diverse_tracks = []
    for track, artist in tracks:
        if artist_count.get(artist, 0) < max_per_artist:
            diverse_tracks.append((track, artist))
            artist_count[artist] = artist_count.get(artist, 0) + 1
    return diverse_tracks
