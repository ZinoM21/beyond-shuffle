# Beyond Shuffle: Composing Context-Responsive Playlists Through Behavioral Patterns with Spotify Data

A code base for my bachelor thesis of the same name.

### Table of contents

- [Description](#description)
- [Tech Stack](#technology-stack)
- [How to Install and Run the application](#how-to-install-and-run-the-application)
- [Credits](#credits)

## Description

Based on the streaming history GDPR data request available with Spotify Europe, this project aimed to develop a software system that transforms a user's music streaming history into context-responsive playlists that specifically serve this purpose of memory evocation and self-reflection.

## Technology Stack

- Language Interpreter / Engine: Python3 & IPython Notebook
- Packages:
  - pandas
  - numpy
  - pyarrow
  - tqdm
  - requests
  - python-dateutil
  - matplotlib-inline
  - seaborn

## How to Install and Run the application

### 1. Install the latest python3 version

You should install a version of python3. On Mac, it should come pre-installed but you should consider updating if you do not have a version of python3 but python2 (check your version by running `python3 --version` in the command line).

### 2. Create a virtual environment

Because the project will have some required packages like flask and we do not want to interfere with all the packages on your computer, you have to create a virtual environment. Enter `python3 -m venv venv` into the command line.

> Note: It is crucial that you do this step while being in the right directory. This should be a copy of this GitHub repository on your computer.

To activate your virtual environment, type `source venv/bin/activate` on Mac & Linux.

(Be sure deactivate the virtual environment when you want to use the command line like normal again. Run `deactivate` for that. If the comamnd line doesn't show "(venv)" it means, the environment isn't active.)

### 3. Install all required packages

> Be sure that your virtual environment is activated!

Then run `pip install -r requirements.txt`

This will install all packages that are used in this project automatically. No need to install extra packages.

### 4. Configuration

In `constants.py`, you can set the following constants at the top of the file if you want:

- `AUDIO_FEATURES_PATH`: path to the CSV file with audio features for your songs (default: `./data/audio-features/reccobeats/tracks_with_audio_features.csv`). Note: the API is deprecated, so future compatibility is uncertain.
- `EXCLUDE_DEVICES`: a list of device names to exclude from your analysis (e.g., if you shared your account with someone else).

### 5. Run

Run `main.py` to generate context-responsive playlists:

```
python3 main.py
```

### 6. CLI

You can run `main.py` from the terminal with the following CLI flags:

- `--skip-import`: Skip data import and modeling, and load modeled data from the parquet file specified in `DATA_PATH`.
- `--load-only`: Only import and model data, then exit (no playlist generation).
- `--playlists <name-of-playlist> [PLAYLIST ...]`: Specify one or more playlist types to generate (e.g., `Commute`, `Workout`, `Study/Focus`). Default is `Commute`.
- `--num-songs <int>`: Number of songs per playlist (default: 20).

**Example usage:**

```
python3 main.py --skip-import --playlists Commute Workout --num-songs 15
```

This will load the modeled data, generate both Commute and Workout playlists with 15 songs each, and print them to the terminal.


### Feature examples

- number of streams → personal popularity of track
- time of day → eating, commuting, evening chill, etc
- day of week → weekday & weekend
- season → spring, fall, winter
- session gap → time between consecutive plays
- attention span → time played vs duration
- device → Home Pod for background, iPhone to sing along, etc.
- reason start & end → e.g. filter out skipped
- connection country → on vacation
- “artist loyalty” → e.g. consecutive plays of an artist / in a timeframe
- shuffle?
- Enriched with Recco beats:
- speechiness / lyric density
- popularity

## Credits

Thanks a lot to [CODE University](https://code.berlin) & our lecturer Kristian!
