import os
import sys
import warnings

import click
import pandas as pd

from candidate_selection import select_candidates
from data_import import load_streaming_data
from data_modelling import model_data
from feature_engineering import feature_engineering
from pattern_finder import find_patterns
from reporting import display_patterns, display_statistics

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def load_and_model_data(skip_import: bool, input_folder_name: str) -> pd.DataFrame:
    """Loads and processes the data, either from source or from a cached file.

    Args:
        skip_import: When True, load modeled data from cached parquet instead of importing raw JSON.
        input_folder: Folder name under ./data containing the input JSON files (defaults handled by caller).
    """
    path = f"./data/{input_folder_name}/enriched.parquet"
    if not skip_import:
        raw_data_df = load_streaming_data(input_folder_name)
        modeled_data = model_data(raw_data_df)
        data = feature_engineering(modeled_data)
        data.to_parquet(path)
        click.echo(f"Modeled data saved to {path}\n")
    else:
        if not os.path.exists(path):
            click.echo(f"Error: {path} not found. Cannot skip import.", err=True)
            sys.exit(1)
        click.echo(f"Skipping import and loading data from {path}")
        data = pd.read_parquet(path)
        click.echo(f"Data loaded from {path}\n")
    return data


@click.group(context_settings={"show_default": True})
@click.version_option()
@click.help_option()
def cli():
    """
    Spotify Streaming History Analysis

    This tool can find memory-evoking patterns in your listening history.
    """
    pass


@cli.command("find-patterns")
@click.option(
    "-io",
    "--import-only",
    is_flag=True,
    help="Only import & model data, then exit (no pattern finding)",
)
@click.option(
    "-si",
    "--skip-import",
    is_flag=True,
    help="Skip data import and modeling, load modeled data from parquet file.",
)
@click.option(
    "-in",
    "--input-folder",
    metavar="FOLDER",
    help="Name of the folder under ./data containing input JSON files",
)
@click.option(
    "-n",
    "--num-songs",
    default=20,
    help="Number of top songs to include in pattern playlists.",
)
def patterns(import_only, skip_import, input_folder, num_songs):
    """
    Finds and displays listening patterns (periods and habits) from the data.
    """

    df = load_and_model_data(skip_import, input_folder)

    if import_only:
        click.echo("Data loaded. Exiting.")
        return

    click.echo("\nFinding listening patterns...")
    detected_patterns = find_patterns(df)

    if not detected_patterns:
        click.echo(
            "\nNo significant patterns found yet. Implementation is in progress."
        )
        return

    click.echo("\nProcessing patterns...")
    top_tracks_map = select_candidates(detected_patterns, num_songs)
    display_patterns(detected_patterns, top_tracks_map)
    display_statistics(detected_patterns)


if __name__ == "__main__":
    cli()
