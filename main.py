import warnings
import pandas as pd
import argparse
import os
import sys

from data_import import load_streaming_data
from data_modelling import model_data
from generate_context_playlists import generate_context_playlists

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

DATA_PATH = './data/out/enriched_data.parquet'
AUDIO_FEATURES_PATH = './data/recco-audio-features/tracks_with_audio_features.csv'
EXCLUDE_DEVICES = ['iPhone 5', 'iPhone 7', 'iPhone XS', 'Samsung Galaxy A5', 'Android Tablet', 'Sony Smart TV']

# CLI arguments
parser = argparse.ArgumentParser(description='Spotify Streaming History Analysis')
parser.add_argument('--skip-import', action='store_true', help=f'Skip data import and modeling, load modeled data from parquet file in {DATA_PATH}')
parser.add_argument('--load-only', action='store_true', help='Only import & model data')
parser.add_argument('--playlists', nargs='+', default="Commute", help='A list of playlist types to generate (e.g., Commute, Workout, Study/Focus)')
parser.add_argument('--num-songs', type=int, default=20, help='Number of songs per playlist')
args = parser.parse_args()

# Main code
if args.skip_import:
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Cannot skip import.")
        sys.exit(1)
    print(f"Skipping import and loading data from {DATA_PATH}")
    data_df = pd.read_parquet(DATA_PATH)
else:
    raw_data_df = load_streaming_data()
    audio_features_df = pd.read_csv(AUDIO_FEATURES_PATH)
    data_df = model_data(raw_data_df, EXCLUDE_DEVICES, audio_features_df)
    data_df.to_parquet(DATA_PATH)
    print(f"Modeled data saved to {DATA_PATH}\n")

if args.load_only:
    print("Load only mode enabled. Exiting.")
    sys.exit(0)


playlists = generate_context_playlists(data_df, args.playlists, args.num_songs)
print("\n--- Generated Playlists ---")
for name, tracks in playlists.items():
    print(f"\n🎵 {name} Playlist ({len(tracks)} songs):")
    for i, (track, artist) in enumerate(tracks, 1):
        print(f"  {i}. {track} — {artist}")
print("\n---------------------------\n")

print("Done! 🎉 Check the ./data/out/ directory for the results.")
