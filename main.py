import os
import sys
import warnings

import click
import pandas as pd

from constants import EXCLUDE_DEVICES
from data_import import load_streaming_data
from data_modelling import model_data
from feature_engineering import feature_engineering
from generate_context_playlists import generate_playlists
from pattern_finder import find_patterns
from pattern_processing import process_patterns
from reporting import display_patterns, display_statistics

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

DATA_PATH = "./data/out/enriched_data.parquet"
AUDIO_FEATURES_PATH = "./data/recco-audio-features/tracks_with_audio_features.csv"


def load_data(skip_import: bool) -> pd.DataFrame:
    """Loads and processes the data, either from source or from a cached file."""
    if not skip_import:
        raw_data_df = load_streaming_data()
        audio_features_df = pd.read_csv(AUDIO_FEATURES_PATH)
        modeled_data = model_data(raw_data_df, EXCLUDE_DEVICES, audio_features_df)
        data = feature_engineering(modeled_data)
        data.to_parquet(DATA_PATH)
        click.echo(f"Modeled data saved to {DATA_PATH}\n")
    else:
        if not os.path.exists(DATA_PATH):
            click.echo(f"Error: {DATA_PATH} not found. Cannot skip import.", err=True)
            sys.exit(1)
        click.echo(f"Skipping import and loading data from {DATA_PATH}")
        data = pd.read_parquet(DATA_PATH)
        click.echo(f"Data loaded from {DATA_PATH}\n")
    return data


@click.group(context_settings={"show_default": True})
@click.version_option()
@click.help_option()
def cli():
    """
    Spotify Streaming History Analysis

    This tool can either generate context-based playlists from predefined presets
    or dynamically find memory-evoking patterns in your listening history.
    """
    pass


@cli.command("generate")
@click.option(
    "-si",
    "--skip-import",
    is_flag=True,
    help=f"Skip data import and modeling, load modeled data from parquet file in {DATA_PATH}",
)
@click.option(
    "-lo",
    "--load-only",
    is_flag=True,
    help="Only import & model data, then exit (no playlist generation)",
)
@click.option(
    "-p",
    "--playlists",
    multiple=True,
    default=["Commute"],
    metavar="[PLAYLIST ...]",
    help="One or more playlist types to generate (e.g., Commute, Workout, Artist_Loyalty, Evening_Chill). Use underscores for spaces in playlist names.",
)
@click.option(
    "-n",
    "--num-songs",
    default=20,
    show_default=True,
    help="Number of songs per playlist",
)
@click.option(
    "--max-per-artist",
    default=None,
    type=int,
    help="Maximum number of songs per artist in a playlist (default: auto, ~15% of playlist size)",
)
def generate(skip_import, load_only, playlists, num_songs, max_per_artist):
    """Generate playlists from predefined presets."""
    data = load_data(skip_import)

    if load_only:
        click.echo("Data loaded. Exiting.")
        return

    playlists_to_generate = [p.replace("_", " ") for p in playlists]
    generated_playlists = generate_playlists(
        data, playlists_to_generate, num_songs, max_per_artist
    )

    click.echo("\n--- Generated Playlists ---")
    for name, tracks in generated_playlists.items():
        click.echo(f"\n🎵 {name} Playlist ({len(tracks)} songs):")
        for i, (track, artist) in enumerate(tracks, 1):
            click.echo(f"  {i}. {track} — {artist}")
    click.echo("\n---------------------------\n")

    click.echo("Done! 🎉 Check the ./data/out/ directory for the results.")


@cli.command("find-patterns")
@click.option(
    "-si",
    "--skip-import",
    is_flag=True,
    help=f"Skip data import and modeling, load modeled data from parquet file in {DATA_PATH}",
)
@click.option(
    "--num-songs",
    default=20,
    help="Number of top songs to include in pattern playlists.",
)
def patterns(skip_import, num_songs):
    """
    Finds and displays listening patterns (periods and habits) from the data.
    """

    df = load_data(skip_import)

    click.echo("\nFinding listening patterns...")
    detected_patterns = find_patterns(df)

    if not detected_patterns:
        click.echo(
            "\nNo significant patterns found yet. Implementation is in progress."
        )
        return

    click.echo("\nProcessing patterns...")
    top_tracks_map = process_patterns(detected_patterns, num_songs)
    display_patterns(detected_patterns, top_tracks_map)
    display_statistics(detected_patterns)
    


if __name__ == "__main__":
    cli()
