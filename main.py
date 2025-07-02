import os
import sys
import warnings

import click
import pandas as pd

from data_import import load_streaming_data
from data_modelling import model_data
from generate_context_playlists import generate_playlists

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

DATA_PATH = './data/out/enriched_data.parquet'
AUDIO_FEATURES_PATH = './data/recco-audio-features/tracks_with_audio_features.csv'
EXCLUDE_DEVICES = ['iPhone 5', 'iPhone 7', 'iPhone XS', 'Samsung Galaxy A5', 'Android Tablet', 'Sony Smart TV']

@click.command(context_settings={"show_default": True})
@click.option('-si' ,'--skip-import', is_flag=True, help=f'Skip data import and modeling, load modeled data from parquet file in {DATA_PATH}')
@click.option('-lo', '--load-only', is_flag=True, help='Only import & model data, then exit (no playlist generation)')
@click.option('-p', '--playlists', multiple=True, default=["Commute"], metavar='[PLAYLIST ...]', help='One or more playlist types to generate (e.g., Commute, Workout, Study/Focus)')
@click.option('-n', '--num-songs', default=20, show_default=True, help='Number of songs per playlist')
@click.option('--max-per-artist', default=None, type=int, help='Maximum number of songs per artist in a playlist (default: auto, ~15% of playlist size)')
@click.version_option()
@click.help_option()
@click.pass_context
def main(ctx, skip_import, load_only, playlists, num_songs, max_per_artist):
    """
    Spotify Streaming History Analysis

    Example usage:

        python main.py --skip-import -p Commute Workout -n 15

    This will load the modeled data, generate both Commute and Workout playlists with 15 songs each, and print them to the terminal.
    """
    if skip_import:
        if not os.path.exists(DATA_PATH):
            click.echo(f"Error: {DATA_PATH} not found. Cannot skip import.", err=True)
            sys.exit(1)
        click.echo(f"Skipping import and loading data from {DATA_PATH}")
        data_df = pd.read_parquet(DATA_PATH)
    else:
        raw_data_df = load_streaming_data()
        audio_features_df = pd.read_csv(AUDIO_FEATURES_PATH)
        data_df = model_data(raw_data_df, EXCLUDE_DEVICES, audio_features_df)
        data_df.to_parquet(DATA_PATH)
        click.echo(f"Modeled data saved to {DATA_PATH}\n")

    if load_only:
        click.echo("Load only mode enabled. Exiting.")
        sys.exit(0)

    playlists_to_generate = list(playlists)
    generated_playlists = generate_playlists(data_df, playlists_to_generate, num_songs, max_per_artist)

    click.echo("\n--- Generated Playlists ---")
    for name, tracks in generated_playlists.items():
        click.echo(f"\n🎵 {name} Playlist ({len(tracks)} songs):")
        for i, (track, artist) in enumerate(tracks, 1):
            click.echo(f"  {i}. {track} — {artist}")
    click.echo("\n---------------------------\n")

    click.echo("Done! 🎉 Check the ./data/out/ directory for the results.")

if __name__ == '__main__':
    main()
