import os
import sys
import warnings
from collections import Counter

import click
import pandas as pd

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


def load_data(data_path: str) -> pd.DataFrame:
    """Loads data from a parquet file."""
    if not os.path.exists(data_path):
        click.echo(f"Error: {data_path} not found. Cannot skip import.", err=True)
        sys.exit(1)
    return pd.read_parquet(data_path)


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
@click.option(
    "--num-songs",
    default=20,
    help="Number of top songs to include in pattern playlists.",
)
def patterns(skip_import, num_songs):
    """
    Finds and displays listening patterns (periods and habits) from the data.
    """
    data_path = "./data/out/enriched_data.parquet"
    if skip_import:
        df = load_data(data_path)
        # Ensure datetime and track_uri are columns
        df = df.reset_index()
    else:
        # Fallback or error if trying to run without the enriched file
        click.echo(
            "Error: This command requires the enriched_data.parquet file. Please run with --skip-import.",
            err=True,
        )
        return

    click.echo("\nFinding listening patterns...")
    detected_patterns = find_patterns(df)

    if not detected_patterns:
        click.echo(
            "\nNo significant patterns found yet. Implementation is in progress."
        )
        return

    click.echo("\n--- Detected Listening Patterns ---\n")
    for p in sorted(detected_patterns, key=lambda x: x.score, reverse=True):
        pattern_type = p.__class__.__name__
        click.echo(f"🎵 Pattern: {p.name} ({pattern_type})")
        click.echo(f"   Description: {p.description}")
        click.echo(f"   Score: {p.score:.2f}")
        click.echo(f"   Contributing Features: {p.contributing_features}")

        # --- Calculate Contextual Popularity Score ---
        pattern_tracks = p.tracks.copy()
        pattern_tracks["is_skipped"] = pattern_tracks["ms_played"] < 10000
        track_stats = (
            pattern_tracks.groupby("spotify_track_uri")
            .agg(
                total_plays=("spotify_track_uri", "count"),
                total_skips=("is_skipped", "sum"),
            )
            .reset_index()
        )
        track_stats["skip_rate"] = (
            track_stats["total_skips"] / track_stats["total_plays"]
        )
        track_stats["contextual_popularity_score"] = track_stats["total_plays"] * (
            1 - track_stats["skip_rate"]
        )
        pattern_tracks = pattern_tracks.merge(
            track_stats[["spotify_track_uri", "contextual_popularity_score"]],
            on="spotify_track_uri",
            how="left",
        )

        # --- Filter and Sort Tracks ---
        tracks_to_show = pattern_tracks

        # For habits, filter tracks to only those that strongly exhibit the habit's characteristic
        if isinstance(p, Habit):
            feature, direction = list(p.contributing_features.items())[0]
            if direction == "High":
                threshold = tracks_to_show[feature].quantile(0.75)
                tracks_to_show = tracks_to_show[tracks_to_show[feature] > threshold]
            elif direction == "Low":
                threshold = tracks_to_show[feature].quantile(0.25)
                tracks_to_show = tracks_to_show[tracks_to_show[feature] < threshold]

        # Sort tracks by the new contextual popularity and take the top N
        top_tracks = (
            tracks_to_show.sort_values("contextual_popularity_score", ascending=False)
            .drop_duplicates(subset=["track", "artist"])
            .head(num_songs)
        )

        click.echo(f"   Tracks ({len(top_tracks)} of {len(p.tracks)}):")
        for _, row in top_tracks.iterrows():
            click.echo(f"     - {row['track']} — {row['artist']}")
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
