import glob
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

# -------------------- Robustness utilities --------------------
OUTAGE_RETRY_SECONDS = 30
OUTAGE_EXIT_AFTER_SECONDS = 30 * 60


class NetworkOutageExceededError(Exception):
    pass


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def sanitize_for_filename(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text))
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def robust_get(url: str, timeout: Optional[int] = None) -> requests.Response:
    """Perform GET with retry on network outages.

    - Retries every 30s on ConnectionError/Timeout
    - Gives up after 30 minutes and raises NetworkOutageExceededError
    - Does NOT special-case HTTP 429; caller keeps existing rate-limit logic
    """
    start = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            return requests.get(url, timeout=timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            elapsed = time.monotonic() - start
            if elapsed >= OUTAGE_EXIT_AFTER_SECONDS:
                raise NetworkOutageExceededError(
                    f"Network unavailable for {int(elapsed)}s while requesting {url}"
                ) from e
            wait_left = OUTAGE_EXIT_AFTER_SECONDS - elapsed
            wait = OUTAGE_RETRY_SECONDS
            if wait > wait_left:
                wait = max(1, int(wait_left))
            print(
                f"Network issue on attempt {attempt} for {url}: {e}. "
                f"Retrying in {wait}s (will give up after {int(wait_left)}s)."
            )
            time.sleep(wait)


def save_checkpoint_csv(records: List[Dict[str, Any]], filepath: str) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    ensure_dir(os.path.dirname(filepath))
    df.to_csv(filepath, index=False)


# Shared temp dir for checkpoints used by both phases
TMP_DIR = "./data/audio-features/reccobeats/tmp"
ensure_dir(TMP_DIR)

## Phase 1:
# -------------------- FETCH TRACK INFOS WHERE AUDIO FEATURES ARE MISSING --------------------
user_parquet_paths = []
user_parquet_paths.extend(glob.glob("./data/user*/enriched.parquet"))

if len(user_parquet_paths) == 0:
    print("No enriched parquet files found under ./data/user*/")
    exit()

dfs = []
for path in sorted(user_parquet_paths):
    try:
        df = pd.read_parquet(path)
        dfs.append(df)
    except Exception as e:
        print(f"Failed to read parquet {path}: {e}")

if len(dfs) == 0:
    print("No data loaded from enriched parquet files.")
    exit()

streams_all = pd.concat(dfs, ignore_index=True)

if "acousticness" in streams_all.columns:
    streams = streams_all[streams_all["acousticness"].isna()].copy()
else:
    # If column not present, assume all rows are missing audio features
    streams = streams_all.copy()

streams = streams.reset_index(drop=True)

num_unique = (
    streams["spotify_track_uri"].nunique()
    if "spotify_track_uri" in streams.columns
    else 0
)
print(
    f"Discovered {len(user_parquet_paths)} enriched parquet files. Loaded {len(streams_all)} streams, {len(streams)} streams missing audio features, {num_unique} unique tracks"
)
START_INDEX = 0
END_INDEX = 0
BATCH_SIZE = 40

# Extract all track URIs
uris = streams["spotify_track_uri"]

# Count occurrences of each uri
uris_by_occurence = uris.value_counts()
print(f"Total unique tracks: {len(uris_by_occurence)}")

# Prepare batches (use ALL unique tracks)
spotify_ids = [
    uri.split(":")[-1]
    for uri in uris_by_occurence.index[
        START_INDEX : END_INDEX if END_INDEX > 1 else None
    ]
]
id_batches = [
    spotify_ids[i : i + BATCH_SIZE] for i in range(0, len(spotify_ids), BATCH_SIZE)
]
print(
    f"{f'Limiting to {END_INDEX-START_INDEX} tracks. ' if END_INDEX > 1 else ''}Total batches of {BATCH_SIZE}: {len(id_batches)}"
)


RECCO_BASE_URL = "https://api.reccobeats.com/v1"

recco_track_infos: List[Dict[str, Any]] = []
total_batches = len(id_batches)
batch_idx = 0

# checkpoint setup (every 5%)
TMP_DIR = "./data/audio-features/reccobeats/tmp"
ensure_dir(TMP_DIR)
checkpoint_stride_batches = (
    max(1, math.ceil(total_batches * 0.05)) if total_batches > 0 else 1
)
last_checkpoint_at_batch = 0

try:
    # Use tqdm manual mode to update progress even if we retry a batch
    with tqdm(total=total_batches, desc="Fetching ReccoBeats track info") as pbar:
        while batch_idx < total_batches:
            batch = id_batches[batch_idx]
            ids_param = ",".join(batch)
            url = f"{RECCO_BASE_URL}/track?ids={ids_param}"
            try:
                response = robust_get(url)
                pbar.set_postfix_str(
                    f"Batch {batch_idx}, status: {response.status_code}"
                )
                if response.status_code == 200:
                    data = response.json()
                    if (
                        isinstance(data, dict)
                        and "content" in data
                        and isinstance(data["content"], list)
                    ):
                        recco_track_infos.extend(data["content"])
                    batch_idx += 1
                    pbar.update(1)
                    time.sleep(0.05)
                elif response.status_code == 429:
                    retry_after_header = response.headers.get("Retry-After")
                    if retry_after_header is None:
                        raise ValueError(f"No Retry-After header for batch {batch_idx}")
                    match = re.search(r"\d+", str(retry_after_header))
                    if match is None:
                        raise ValueError(
                            f"Could not extract number of seconds from Retry-After header for batch {batch_idx}. The header was: {retry_after_header}"
                        )
                    retry_after = int(match.group(0))
                    pbar.set_postfix_str(
                        f"Rate limit for batch {batch_idx}: waiting {retry_after}s"
                    )
                    time.sleep(retry_after)
                    continue  # Retry the same batch
                else:
                    raise ValueError(
                        f"Request failed for batch {batch_idx}: {response.status_code}"
                    )
            except Exception as e:
                print(f"Exception for batch {batch_idx}: {e}")
                batch_idx += 1
                pbar.update(1)
                time.sleep(0.05)

            # periodic checkpoint every 5% of batches progressed
            progressed_batches = batch_idx
            if (
                progressed_batches - last_checkpoint_at_batch
                >= checkpoint_stride_batches
            ):
                checkpoint_path = os.path.join(
                    TMP_DIR,
                    f"track_infos_checkpoint_{progressed_batches}_of_{total_batches}.csv",
                )
                save_checkpoint_csv(recco_track_infos, checkpoint_path)
                last_checkpoint_at_batch = progressed_batches
except NetworkOutageExceededError as e:
    # save and exit
    error_suffix = sanitize_for_filename(type(e).__name__)
    err_path = f"./data/audio-features/reccobeats/track_infos.error_{error_suffix}.csv"
    save_checkpoint_csv(recco_track_infos, err_path)
    print(str(e))
    raise SystemExit(1)
except Exception as e:
    error_suffix = sanitize_for_filename(type(e).__name__)
    err_path = f"./data/audio-features/reccobeats/track_infos.error_{error_suffix}.csv"
    save_checkpoint_csv(recco_track_infos, err_path)
    print(f"Unhandled error; progress saved to {err_path}")
    raise SystemExit(1)

# Save track info to CSV
if recco_track_infos:
    print(
        f"Found {len(recco_track_infos)} ReccoBeats track infos out of {len(uris_by_occurence)} unique tracks"
    )
    df_recco_track_infos = pd.DataFrame(recco_track_infos)
    dir_path = "./data/audio-features/reccobeats"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    suffix = f"_{START_INDEX}_{END_INDEX if END_INDEX > 1 else len(recco_track_infos)}"
    file_name = f"{dir_path}/track_infos{suffix}.csv"
    print(f"Saving ReccoBeats track info to {file_name} ...")
    df_recco_track_infos.to_csv(file_name, index=False)
    print("✅ Done")
else:
    print("No ReccoBeats track info fetched.")


## Phase 2:
# -------------------- LOAD TRACK INFO FOR RECCOBEATS AUDIO FEATURES FROM CSV FILES --------------------
CHOSEN_FILE = 1

recco_track_files = glob.glob("./data/audio-features/reccobeats/track_infos_*_*.csv")
recco_track_files.sort()  # ensure consistent ordering

if CHOSEN_FILE < 1 or CHOSEN_FILE > len(recco_track_files):
    print(
        f"Invalid file index. Please select a number between 1 and {len(recco_track_files)}"
    )
    exit()

selected_file = recco_track_files[CHOSEN_FILE - 1]

match = re.search(r"track_infos_(\d+)_(\d+)\.csv", os.path.basename(selected_file))
if not match:
    # Allow arbitrary filenames; default to generic index range
    start_from_track = 821
    end_to_track = 0
    print(
        "Filename does not match track_infos_<start>_<end>.csv; numbering will start at 1."
    )
else:
    start_from_track = int(match.group(1))
    end_to_track = int(match.group(2))
print(f"Processing file: {selected_file}")
print(f"Start index: {start_from_track}, End index: {end_to_track}")

# Load track info from selected CSV
df_tracks = pd.read_csv(selected_file)
tracks_for_features = df_tracks.to_dict("records")


# DEFINE FUNCTION TO FETCH AUDIO FEATURES
RECCO_BASE_URL = "https://api.reccobeats.com/v1"


def fetch_audio_feature(track):
    recco_id = track.get("id")
    if not recco_id:
        return None
    url = f"{RECCO_BASE_URL}/track/{recco_id}/audio-features"
    try:
        response = robust_get(url, timeout=10)
        if response.status_code != 200:
            if response.status_code == 429:
                retry_after_header = response.headers.get("Retry-After")
                if retry_after_header is None:
                    raise ValueError(f"No Retry-After header for track {recco_id}")
                match_retry = re.search(r"\d+", str(retry_after_header))
                if match_retry is None:
                    raise ValueError(
                        f"Could not extract number of seconds from Retry-After header for track {recco_id}. The header was: {retry_after_header}"
                    )
                retry_after = int(match_retry.group(0))
                if retry_after > 300:  # 5 mins
                    raise ValueError(
                        f"Rate limit retry-after time too long ({retry_after}s), exiting..."
                    )
                return {
                    "status": 429,
                    "message": "Rate limit exceeded",
                    "retry_after": retry_after + 1,  # +1 to avoid rounding errors
                }
            elif response.status_code == 404:
                return {"status": 404, "message": "Track not found"}
            else:
                raise ValueError(
                    f"Error fetching features for {recco_id}: {response.status_code}"
                )
        return response.json()
    except Exception as e:
        print(f"Exception for {recco_id}: {e}")
        raise e


# FETCH AUDIO FEATURES IN PARALLEL, WITH RATE LIMIT HANDLING
MAX_WORKERS = (
    2  # is probably enough for the recco rate limit of 120 requests per minute
)
SKIP_UNTIL = 0  # can be used to skip a specific number of tracks from a previous run

if SKIP_UNTIL > 0:
    tracks_for_features = tracks_for_features[SKIP_UNTIL:]
    start_from_track += SKIP_UNTIL + 1

features: List[Dict[str, Any]] = []
track_idx = 0
total = len(tracks_for_features)

# checkpoint setup (every 5%)
checkpoint_stride_features = max(1, math.ceil(total * 0.05)) if total > 0 else 1
last_checkpoint_at_features = 0

try:
    with tqdm(total=total, desc="Fetching audio features") as pbar:
        while track_idx < total:
            pbar.set_postfix_str("Fetching...")
            # Submit up to MAX_WORKERS tasks at a time, starting from track_idx
            batch = tracks_for_features[track_idx : track_idx + MAX_WORKERS]
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [
                    executor.submit(fetch_audio_feature, track) for track in batch
                ]
                got_rate_limit = False
                for idx, future in enumerate(futures):
                    result = future.result()
                    if isinstance(result, dict) and result.get("status") == 429:
                        retry_after = result.get("retry_after")
                        if retry_after is None:
                            raise ValueError(
                                f"No retry after info for track at index {track_idx+idx}"
                            )
                        pbar.set_postfix_str(
                            f"Rate limit at {track_idx+idx}th track, waiting {retry_after}s before retrying..."
                        )
                        got_rate_limit = True
                        time.sleep(int(retry_after))
                        break  # Break the for loop, do not advance track_idx
                    elif isinstance(result, dict) and result.get("status") == 404:
                        pbar.set_postfix_str(
                            f"{track_idx+idx}th track not found: {result}"
                        )
                        continue
                    elif result is not None:
                        features.append(result)
                        pbar.update(1)
                        # periodic checkpoint every 5% of features fetched
                        if (
                            len(features) - last_checkpoint_at_features
                            >= checkpoint_stride_features
                        ):
                            checkpoint_path = os.path.join(
                                TMP_DIR,
                                f"audio_features_checkpoint_{len(features)}_of_{total}.csv",
                            )
                            save_checkpoint_csv(features, checkpoint_path)
                            last_checkpoint_at_features = len(features)
                    else:
                        raise ValueError(
                            f"Unknown result for track at index {track_idx+idx}: {result}"
                        )
                if got_rate_limit:
                    # Dont advance track_idx, retry this batch / track
                    continue
                else:
                    # All batch processed, advance track_idx
                    track_idx += len(batch)
except NetworkOutageExceededError as e:
    error_suffix = sanitize_for_filename(type(e).__name__)
    err_path = (
        f"./data/audio-features/reccobeats/audio_features.error_{error_suffix}.csv"
    )
    save_checkpoint_csv(features, err_path)
    print(str(e))
    raise SystemExit(1)
except Exception as e:
    error_suffix = sanitize_for_filename(type(e).__name__)
    err_path = (
        f"./data/audio-features/reccobeats/audio_features.error_{error_suffix}.csv"
    )
    save_checkpoint_csv(features, err_path)
    print(f"Unhandled error; progress saved to {err_path}")
    raise SystemExit(1)

# Save to CSV
if not features:
    print("No audio features fetched.")
    exit()

print(f"Saving {len(features)} audio features to CSV")

file_name = f"./data/audio-features/reccobeats/audio_features_{start_from_track}_{start_from_track + len(features) - 1 }.csv"
df_features = pd.DataFrame(features)
df_features.to_csv(file_name, index=False)

print(f"Audio features saved to {file_name}")


# join all audio features csv files
audio_features_files = glob.glob(
    "./data/audio-features/reccobeats/audio_features_*_*.csv"
)
audio_features_files.sort()

df_recco_audio_features = pd.concat(
    [pd.read_csv(file) for file in audio_features_files]
)
length_before_drop = len(df_recco_audio_features)
df_recco_audio_features = df_recco_audio_features.drop_duplicates(subset=["id"])

print(
    f"Unique audio features: {len(df_recco_audio_features)} ({length_before_drop - len(df_recco_audio_features)} duplicates dropped)"
)

df_recco_audio_features.to_csv(
    "./data/audio-features/reccobeats/audio_features.csv", index=False
)


# join all track infos csv files
track_infos_files = glob.glob("./data/audio-features/reccobeats/track_infos_*_*.csv")
track_infos_files.sort()

df_recco_track_infos = pd.concat([pd.read_csv(file) for file in track_infos_files])
length_before_drop = len(df_recco_track_infos)
df_recco_track_infos = df_recco_track_infos.drop_duplicates(subset=["id"])
print(
    f"Unique track infos: {len(df_recco_track_infos)} ({length_before_drop - len(df_recco_track_infos)} duplicates dropped)"
)

df_recco_track_infos.to_csv(
    "./data/audio-features/reccobeats/track_infos.csv", index=False
)


# Join dataframes on 'id' using outer join to keep all records
df_recco_track_infos_final = pd.read_csv(
    "./data/audio-features/reccobeats/track_infos.csv"
)
df_recco_audio_features_final = pd.read_csv(
    "./data/audio-features/reccobeats/audio_features.csv"
)


df_recco_tracks_with_audio_features = pd.merge(
    df_recco_track_infos_final, df_recco_audio_features_final, on="id", how="outer"
)

# formatting
df_recco_tracks_with_audio_features = df_recco_tracks_with_audio_features.drop(
    columns=["trackTitle", "artists", "isrc", "ean", "upc", "availableCountries"]
)
df_recco_tracks_with_audio_features.rename(
    columns={"durationMs": "duration_ms"}, inplace=True
)


# Safely convert href to spotify_track_uri, handling missing or non-string values
def href_to_spotify_uri(x):
    if isinstance(x, str) and "/" in x:
        return f'spotify:track:{x.split("/")[-1]}'
    print(f"Invalid href: {x}")
    return pd.NA


df_recco_tracks_with_audio_features["spotify_track_uri"] = (
    df_recco_tracks_with_audio_features["href"].apply(href_to_spotify_uri)
)
df_recco_tracks_with_audio_features = df_recco_tracks_with_audio_features.drop(
    columns=["href"]
)

# save to csv
df_recco_tracks_with_audio_features.to_csv(
    "./data/audio-features/reccobeats/tracks_with_audio_features.csv", index=False
)
print(f"Saved {len(df_recco_tracks_with_audio_features)} tracks with audio features")
