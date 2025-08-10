# Beyond Shuffle: Composing Context-Responsive Playlists Through Behavioral Patterns with Spotify Data

A code base for my bachelor thesis of the same name.

### Table of contents

- [Description](#description)
- [Technology Stack](#technology-stack)
- [Install & Run](#how-to-install-and-run-the-application)
- [Configuration via constants.py](#configuration)
- [CLI usage (main.py)](#cli)
- [Feature examples](#feature-examples)
- [Credits](#credits)

## Description

Based on Spotify GDPR streaming-history exports, this project discovers listening patterns (periods and habits) and turns them into context‑responsive playlists that support memory evocation and self‑reflection.

## Technology Stack

- Python 3
- Notebooks for exploration
- Core packages (see `requirements.txt`): pandas, numpy, pyarrow, tqdm, requests, python-dateutil, seaborn, click

## How to Install and Run the application

### 1) Install Python 3

Check with `python3 --version`. Upgrade if needed.

### 2) Create and activate a virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

Deactivate with `deactivate`.

### 3) Install dependencies

```
pip install -r requirements.txt
```

### 4) Prepare your data

Place your Spotify streaming-history JSON files under a new folder inside `./data/`, e.g. `./data/userX/`. The pipeline will read raw JSON from that folder and cache modeled data at `./data/<folder>/enriched.parquet`.

## Configuration

Key settings live in `constants.py`. Adjust as needed:

- EXCLUDE_DEVICES: Optional list of device names to ignore during analysis. Example to enable:
  - e.g. uncomment entries or add your own: "iPhone 7", "playstation".
- AUDIO_FEATURES_PATHS: Mapping of enrichment CSV paths → ID column name to join on. Multiple sources can be enabled at once. Defaults include Reccobeats, Beatport, Million Song Dataset, and others.
- CANDIDATE_SELECTION_WEIGHTS: Weights used to rank tracks within detected patterns. Defaults: count=0.5, skip_rate=0.3, attention_span=0.2.
- Period detection: PERIOD_MIN_DAYS, WINDOW_SIZE_DAYS, STEP_SIZE_DAYS, PERIOD_FEATURE_ZSCORE_THRESHOLD, MIN_TRACKS_FOR_PLAYLIST.
- Features to analyze: CATEGORICAL_FEATURES_TO_CHECK, NUMERICAL_FEATURES_TO_CHECK.
- Habit detection: HABIT_MIN_WEEKS, HABIT_MIN_STREAMS_PER_SLOT, HABIT_FEATURE_ZSCORE_THRESHOLD, HABIT_MIN_NUM_FEATURES, HABIT_MAX_SLOTS_PER_SCHEMA, HABIT_TOP_PLATFORMS, audio‑cluster settings (HABIT_AUDIO_CLUSTER_K, HABIT_MIN_CLUSTER_SHARE, HABIT_MIN_CLUSTER_WEEKS), and naming thresholds (HABIT_TOP_ARTIST_SHARE, HABIT_MIN_DEVICE_SHARE).

Tip: after tweaking constants, re‑run the CLI. If you rely on the cached parquet, pass `--skip-import` to avoid re‑parsing JSON.

## CLI

The entrypoint is `main.py`, which exposes a Click command group with the `find-patterns` subcommand.

General form:

```
python3 main.py find-patterns [OPTIONS]
```

Options:

- --import-only, -io: Only import & model data, then exit (no pattern finding).
- --skip-import, -si: Skip import/modeling and load modeled data from `./data/<folder>/enriched.parquet`.
- --input-folder, -in FOLDER: Name of the folder under `./data/` containing your input JSON (and/or the cached parquet).
- --num-songs, -n INT: Number of top songs to include per detected pattern playlist (default: 20).

Practical examples:

```
# Full pipeline from raw JSON in ./data/user1
python3 main.py find-patterns --input-folder user1

# Only build the modeled cache ./data/user1/enriched.parquet and exit
python3 main.py find-patterns --input-folder user1 --import-only

# Use the cached parquet (skips JSON parsing) and return 30 songs per pattern
python3 main.py find-patterns --input-folder user1 --skip-import --num-songs 30
```

### Feature examples

- number of streams → personal popularity of track
- time of day → eating, commuting, evening chill, etc
- day of week → weekday & weekend
- season → spring, fall, winter
- session gap → time between consecutive plays
- attention span → time played vs duration
- device → HomePod for background, iPhone to sing along, etc.
- reason start & end → e.g., filter out skipped
- connection country → on vacation
- “artist loyalty” → consecutive plays of an artist / within a timeframe
- audio enrichment (e.g., via Reccobeats, Beatport, MSD) → speechiness, energy, danceability, valence, popularity

## Credits

Thanks a lot to [CODE University](https://code.berlin)!
