import os
import sys
import warnings
from collections import Counter

import click
import pandas as pd

from data_import import load_streaming_data
from data_modelling import model_data
from feature_engineering import feature_engineering
from generate_context_playlists import generate_playlists
from pattern_finder import DetectedPattern, Habit, Period, find_patterns

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

DATA_PATH = "./data/out/enriched_data.parquet"
AUDIO_FEATURES_PATH = "./data/recco-audio-features/tracks_with_audio_features.csv"
EXCLUDE_DEVICES = [
    "iPhone 5 (GSM+CDMA)",
    "iPhone 7",
    "iPhone XS",
    "Samsung Galaxy A5",
    "Android Tablet",
    "Hama Speaker",
    "android",
    "playstation",
]


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


@cli.command()
@click.option(
    "-si",
    "--skip-import",
    is_flag=True,
    help=f"Skip data import and modeling, load modeled data from parquet file in {DATA_PATH}",
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
def generate(skip_import, playlists, num_songs, max_per_artist):
    """Generate playlists from predefined presets."""
    data = load_data(skip_import)

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
def patterns(skip_import):
    """Find and display memory-evoking listening patterns."""
    data = load_data(skip_import)

    click.echo("Finding listening patterns...")

    # This will call the main function in your new module
    detected_patterns = find_patterns(data)

    if not detected_patterns:
        click.echo(
            "\nNo significant patterns found yet. Implementation is in progress."
        )
        return

    click.echo("\n--- Detected Listening Patterns ---")
    for pattern in detected_patterns:
        click.echo(f"\n🎵 Pattern: {pattern.name} ({pattern.pattern_type})")
        click.echo(f"   Description: {pattern.description}")
        click.echo(f"   Score: {pattern.score:.2f}")
        click.echo(f"   Contributing Features: {pattern.contributing_features}")
        click.echo(f"   Tracks ({len(pattern.tracks)}):")
        for i, row in pattern.tracks.head(5).iterrows():
            click.echo(f"     - {row['track']} — {row['artist']}")
        if len(pattern.tracks) > 5:
            click.echo("     ...")
    click.echo("\n------------------------------------\n")

    # --- Statistics Section ---
    periods = [p for p in detected_patterns if isinstance(p, Period)]
    habits = [p for p in detected_patterns if isinstance(p, Habit)]

    click.echo("\n--- Pattern Statistics ---")
    click.echo(f"Total Patterns Found: {len(detected_patterns)}")
    click.echo(f"  - Periods: {len(periods)}")
    click.echo(f"  - Habits: {len(habits)}")

    def get_pattern_description(p: DetectedPattern) -> str:
        """Helper to create a granular description for a pattern."""
        desc_parts = []
        # Sort to ensure consistent ordering for counting
        for feature, value in sorted(p.contributing_features.items()):
            if str(value) in ["High", "Low"]:
                desc_parts.append(f"{value} {feature.capitalize()}")
            else:
                desc_parts.append(f"{feature.capitalize()} changed to {value}")
        return " & ".join(desc_parts)

    if periods:
        click.echo("\nPeriod Breakdown:")
        period_types = Counter(get_pattern_description(p) for p in periods)
        for type, count in period_types.most_common(15):
            click.echo(f"  - {type}: {count}")

    if habits:
        click.echo("\nHabit Breakdown by Time Slot:")
        habit_slots = Counter(f"{p.time_slot[0]} {p.time_slot[1]}s" for p in habits)
        for slot, count in habit_slots.most_common(5):
            click.echo(f"  - {slot}: {count} habits")

    click.echo("\n--------------------------\n")


if __name__ == "__main__":
    cli()
