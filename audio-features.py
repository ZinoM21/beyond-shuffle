import pandas as pd
import requests
import re
import glob
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import os

from data_import import load_streaming_data

streams = load_streaming_data()
streams = streams.reset_index()

print(f"Loaded {len(streams)} streams, {len(streams['spotify_track_uri'].unique())} unique tracks")
START_INDEX = 0
END_INDEX = 0
BATCH_SIZE = 40

# Extract all track URIs
uris = streams['spotify_track_uri']

# Count occurrences of each uri
uris_by_occurence = uris.value_counts()
print(f"Total unique tracks: {len(uris_by_occurence)}")

# Occuring more than once
uris_occuring_more_than_once = uris_by_occurence[uris_by_occurence >= 2]
print(f'Tracks occuring more than once: {len(uris_occuring_more_than_once)} / {len(uris_by_occurence)}')

# Prepare batches
spotify_ids = [uri.split(':')[-1] for uri in uris_occuring_more_than_once.index[START_INDEX:END_INDEX if END_INDEX > 1 else None]]
id_batches = [spotify_ids[i:i+BATCH_SIZE] for i in range(0, len(spotify_ids), BATCH_SIZE)]
print(f"{f'Limiting to {END_INDEX-START_INDEX} tracks. ' if END_INDEX > 1 else ''}Total batches of {BATCH_SIZE}: {len(id_batches)}")


# FETCH RECCOBEATS TRACK INFO FOR EACH BATCH OF IDS, WITH RATE LIMIT HANDLING AND TQDM PROGRESS BAR
RECCO_BASE_URL = 'https://api.reccobeats.com/v1'

recco_track_infos = []
total_batches = len(id_batches)
batch_idx = 0

# Use tqdm manual mode to update progress even if we retry a batch
with tqdm(total=total_batches, desc="Fetching ReccoBeats track info") as pbar:
    while batch_idx < total_batches:
        batch = id_batches[batch_idx]
        ids_param = ','.join(batch)
        url = f'{RECCO_BASE_URL}/track?ids={ids_param}'
        try:
            response = requests.get(url)
            pbar.set_postfix_str(f"Batch {batch_idx}, status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'content' in data and isinstance(data['content'], list):
                    recco_track_infos.extend(data['content'])
                batch_idx += 1
                pbar.update(1)
                time.sleep(0.05)
            elif response.status_code == 429:
                retry_after_header = response.headers.get('Retry-After')
                if retry_after_header is None:
                    raise ValueError(f"No Retry-After header for batch {batch_idx}")
                match = re.search(r'\d+', str(retry_after_header))
                if match is None:
                    raise ValueError(f"Could not extract number of seconds from Retry-After header for batch {batch_idx}. The header was: {retry_after_header}")
                retry_after = int(match.group(0))
                pbar.set_postfix_str(f"Rate limit for batch {batch_idx}: waiting {retry_after}s")
                time.sleep(retry_after)
                continue # Retry the same batch
            else:
                raise ValueError(f"Request failed for batch {batch_idx}: {response.status_code}")
        except Exception as e:
            print(f"Exception for batch {batch_idx}: {e}")
            batch_idx += 1
            pbar.update(1)
            time.sleep(0.05)

# Save track info to CSV
if recco_track_infos:
    print(f"Found {len(recco_track_infos)} ReccoBeats track infos out of {len(uris_occuring_more_than_once)} unique tracks occuring more than once")
    df_recco_track_infos = pd.DataFrame(recco_track_infos)
    dir_path = './data/recco-audio-features'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    suffix = f'_{START_INDEX}_{END_INDEX if END_INDEX > 1 else len(recco_track_infos)}'
    file_name = f'{dir_path}/track_infos{suffix}.csv'
    print(f"Saving ReccoBeats track info to {file_name} ...")
    df_recco_track_infos.to_csv(file_name, index=False)
    print("✅ Done")
else:
    print('No ReccoBeats track info fetched.') 



# LOAD AUDIO FEATURES FOR RECCOBEATS TRACKS FROM CSV FILES
CHOSEN_FILE = 1

recco_track_files = glob.glob('./data/recco-audio-features/track_infos_*_*.csv')
recco_track_files.sort()  # ensure consistent ordering

if CHOSEN_FILE < 1 or CHOSEN_FILE > len(recco_track_files):
    print(f"Invalid file index. Please select a number between 1 and {len(recco_track_files)}")
    exit()

selected_file = recco_track_files[CHOSEN_FILE - 1]

match = re.search(r'track_infos_(\d+)_(\d+)\.csv', selected_file)
if not match:
    print("Could not extract indices from filename")
    exit()

start_from_track = int(match.group(1))
end_to_track = int(match.group(2))
print(f"Processing file {CHOSEN_FILE}: {selected_file}")
print(f"Start index: {start_from_track}, End index: {end_to_track}")

# Load track info from selected CSV
df_tracks = pd.read_csv(selected_file)
recco_track_infos = df_tracks.to_dict('records')


# DEFINE FUNCTION TO FETCH AUDIO FEATURES
RECCO_BASE_URL = 'https://api.reccobeats.com/v1'
def fetch_audio_feature(track):
    recco_id = track.get('id')
    if not recco_id:
        return None
    url = f'{RECCO_BASE_URL}/track/{recco_id}/audio-features'
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            if response.status_code == 429:
                retry_after_header = response.headers.get('Retry-After', 10)
                retry_after = int(re.search(r'\d+', retry_after_header).group(0))
                return {"status": 429, "message": "Rate limit exceeded", "retry_after": retry_after}
            elif response.status_code == 404:
                return {"status": 404, "message": "Track not found"}
            else:
                raise ValueError(f"Error fetching features for {recco_id}: {response.status_code}")
        return response.json()
    except Exception as e:
        print(f"Exception for {recco_id}: {e}")
        raise e


# FETCH AUDIO FEATURES IN PARALLEL, WITH RATE LIMIT HANDLING
MAX_WORKERS = 12
SKIP_UNTIL = 0 # can be used to skip a specific number of tracks from a previous run

if SKIP_UNTIL > 0:
    recco_track_infos = recco_track_infos[SKIP_UNTIL:]
    start_from_track += SKIP_UNTIL + 1

features = []
track_idx = 0
total = len(recco_track_infos)

with tqdm(total=total, desc="Fetching audio features") as pbar:
    while track_idx < total:
        pbar.set_postfix_str("Fetching...")
        # Submit up to MAX_WORKERS tasks at a time, starting from track_idx
        batch = recco_track_infos[track_idx:track_idx+MAX_WORKERS]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(fetch_audio_feature, track) for track in batch]
            got_rate_limit = False
            for idx, future in enumerate(futures):
                result = future.result()
                if isinstance(result, dict) and result.get("status") == 429:
                    retry_after = result.get("retry_after")
                    if retry_after is None:
                        raise ValueError(f"No retry after info for track at index {track_idx+idx}")
                    if retry_after > 300:  # 5 mins
                        raise ValueError(f"Rate limit retry after time too long ({retry_after}s), stopping and saving current results...")
                    
                    retry_after += 1 # +1 to avoid rounding errors
                    pbar.set_postfix_str(f"Rate limit at {track_idx+idx}th track, waiting {retry_after}s before retrying...")
                    got_rate_limit = True
                    time.sleep(int(retry_after)) 
                    break  # Break the for loop, do not advance track_idx
                elif result is not None:
                    features.append(result)
                    pbar.update(1)
                else:
                    raise ValueError(f"Unknown result for track at index {track_idx+idx}: {result}")
            if got_rate_limit:
                # Dont advance track_idx, retry this batch / track
                continue
            else:
                # All batch processed, advance track_idx
                track_idx += len(batch)

# Save to CSV
if not features:
    print('No audio features fetched.')
    exit()

print(f"Saving {len(features)} audio features to CSV")

file_name = f'./data/recco-audio-features/audio_features_{start_from_track}_to_{start_from_track + len(features) - 1 }.csv'
df_features = pd.DataFrame(features)
df_features.to_csv(file_name, index=False)

print(f"Audio features saved to {file_name}")


# join all audio features csv files
audio_features_files = glob.glob('./data/recco-audio-features/audio_features_*_*.csv')
audio_features_files.sort()

recco_audio_features = pd.concat([pd.read_csv(file) for file in audio_features_files])
length_before_drop = len(recco_audio_features)
recco_audio_features = recco_audio_features.drop_duplicates(subset=['id'])

print(f"Unique audio features: {len(recco_audio_features)} ({length_before_drop - len(recco_audio_features)} duplicates dropped)")

recco_audio_features.to_csv('./data/recco-audio-features/audio_features.csv', index=False)


# join all track infos csv files
track_infos_files = glob.glob('./data/recco-audio-features/track_infos_*_*.csv')
track_infos_files.sort()

recco_track_infos = pd.concat([pd.read_csv(file) for file in track_infos_files])
length_before_drop = len(recco_track_infos)
recco_track_infos = recco_track_infos.drop_duplicates(subset=['id'])
print(f"Unique track infos: {len(recco_track_infos)} ({length_before_drop - len(recco_track_infos)} duplicates dropped)")

recco_track_infos.to_csv('./data/recco-audio-features/track_infos.csv', index=False)


# Join dataframes on 'id' using outer join to keep all records
recco_track_infos = pd.read_csv('./data/recco-audio-features/track_infos.csv')
recco_audio_features = pd.read_csv('./data/recco-audio-features/audio_features.csv')


recco_tracks_with_audio_features = pd.merge(
    recco_track_infos,
    recco_audio_features,
    on='id',
    how='outer'
)

# formatting
recco_tracks_with_audio_features = recco_tracks_with_audio_features.drop(columns=['trackTitle', 'artists', 'isrc', 'ean', 'upc', 'availableCountries'])
recco_tracks_with_audio_features.rename(columns={'durationMs': 'duration_ms'}, inplace=True)

# rename href to spotify_track_uri and format it into the correct uri format
recco_tracks_with_audio_features['spotify_track_uri'] = recco_tracks_with_audio_features['href'].apply(lambda x: f'spotify:track:{x.split("/")[-1]}')
recco_tracks_with_audio_features = recco_tracks_with_audio_features.drop(columns=['href'])

# save to csv
recco_tracks_with_audio_features.to_csv('./data/recco-audio-features/tracks_with_audio_features.csv', index=False)
print(f"Saved {len(recco_tracks_with_audio_features)} tracks with audio features")

