import pandas as pd
from typing import Dict, List

def generate_context_playlists(df: pd.DataFrame, playlists_to_generate: str | List[str], num_songs: int) -> Dict[str, List[str]]:
    """
    Generates context-based playlists from a feature-rich dataframe.
    """
    print("Generating context playlists...")
    
    # Ensure required columns exist, providing empty Series for those that don't to avoid errors
    for col in ['energy', 'danceability', 'speechiness', 'instrumentalness', 'skipped']:
        if col not in df.columns:
            df[col] = pd.Series([False] * len(df) if col == 'skipped' else [0.0] * len(df), index=df.index)

    available_playlists = {
        'Commute': (
            (df['is_weekday']) &
            (df['time_of_day'].isin(['Morning', 'Evening'])) &
            (df['platform'].str.contains('iPhone|Android', case=False, na=False)) &
            (~df['skipped'])
        ),
        'Workout': (
            (df['energy'] > 0.7) &
            (df['danceability'] > 0.7) &
            (~df['skipped']) &
            (df['time_of_day'].isin(['Afternoon', 'Evening']))
        ),
        'Study/Focus': (
            (df['speechiness'] < 0.3) &
            (df['instrumentalness'] > 0.5) &
            (df['attention_span'] > 0.9) &
            (~df['skipped'])
        ),
    }

    generated_playlists = {}
    
    if isinstance(playlists_to_generate, str):
        playlists_to_generate = [playlists_to_generate]

    for name in playlists_to_generate:
        name_title_case = name.title()
        if name_title_case in available_playlists:
            conditions = available_playlists[name_title_case]
            playlist_df = df[conditions]
            
            # Prioritize tracks by play count within the context
            top_tracks = playlist_df['track'].value_counts().head(num_songs).index.tolist()
            # For each top track, get the most common artist for that track in the filtered context
            playlist_tracks = []
            for track in top_tracks:
                artist = (
                    playlist_df[playlist_df['track'] == track]['artist']
                    .mode()
                    .iloc[0] if not playlist_df[playlist_df['track'] == track]['artist'].empty else 'Unknown Artist'
                )
                playlist_tracks.append((track, artist))
            generated_playlists[name_title_case] = playlist_tracks
            print(f"Playlist '{name_title_case}' generated with {len(playlist_tracks)} tracks.")
        else:
            print(f"Warning: Playlist type '{name}' not recognized. Skipped.")

    return generated_playlists 