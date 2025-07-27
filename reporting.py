from collections import Counter
from typing import List

import click
import pandas as pd

from pattern_finder import DetectedPattern, Habit, Period


def display_patterns(patterns: List[DetectedPattern], top_tracks_map: dict):
    """Displays the detected listening patterns and their top tracks."""
    click.echo("\n--- Detected Listening Patterns ---\n")
    if not patterns:
        click.echo("No significant patterns found.")
        return

    for p in sorted(patterns, key=lambda x: x.score, reverse=True):
        pattern_type = p.__class__.__name__
        click.echo(f"🎵 Pattern: {p.name} ({pattern_type})")
        click.echo(f"   Description: {p.description}")
        click.echo(f"   Score: {p.score:.2f}")
        click.echo(f"   Contributing Features: {p.contributing_features}")

        top_tracks = top_tracks_map.get(p.name, pd.DataFrame())
        if not top_tracks.empty:
            click.echo(f"   Tracks ({len(top_tracks)} of {len(p.tracks)}):")
            for _, row in top_tracks.iterrows():
                click.echo(f"     - {row['track']} — {row['artist']}")
            click.echo("     ...")
        click.echo("\n------------------------------------\n")


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


def display_statistics(patterns: List[DetectedPattern]):
    """Displays statistics about the found patterns."""
    if not patterns:
        return

    periods = [p for p in patterns if isinstance(p, Period)]
    habits = [p for p in patterns if isinstance(p, Habit)]

    click.echo("\n--- Pattern Statistics ---")
    click.echo(f"Total Patterns Found: {len(patterns)}")
    click.echo(f"  - Periods: {len(periods)}")
    click.echo(f"  - Habits: {len(habits)}")

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
