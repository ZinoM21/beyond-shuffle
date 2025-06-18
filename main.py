import warnings
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import calplot
import imageio
import bar_chart_race as bcr
import wordcloud
import argparse
import os
import sys
import numpy as np
from PIL import Image as PILImage

from data_import import load_streaming_data
from data_modelling import model_data

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

MODELED_DATA_PATH = './out/modeled_data.parquet'
AUDIO_FEATURES_PATH = 'data/audio_features.csv'
EXCLUDE_DEVICES = ['iPhone 5', 'iPhone 7', 'iPhone XS', 'Samsung Galaxy A5', 'Android Tablet', 'Sony Smart TV']

parser = argparse.ArgumentParser(description='Spotify Streaming History Analysis')
parser.add_argument('--skip-import', action='store_true', help=f'Skip data import and modeling, load modeled data from parquet file in {MODELED_DATA_PATH}')
parser.add_argument('--import-only', action='store_true', help=f'Import data from {MODELED_DATA_PATH}')
args = parser.parse_args()

if args.skip_import:
    if not os.path.exists(MODELED_DATA_PATH):
        print(f"Error: {MODELED_DATA_PATH} not found. Cannot skip import.")
        sys.exit(1)
    modeled_data = pd.read_parquet(MODELED_DATA_PATH)
else:
    df = load_streaming_data()
    # This can be enabled if there are audio features available under the AUDIO_FEATURES_PATH
    # df_audio_features = pd.read_csv(AUDIO_FEATURES_PATH, sep=',', index_col='uri')
    df_audio_features = None
    modeled_data = model_data(df, EXCLUDE_DEVICES, df_audio_features)
    modeled_data.to_parquet(MODELED_DATA_PATH)
    print(f"Modeled data saved to {MODELED_DATA_PATH}")


if args.import_only:
    print("Import only mode enabled. Exiting.")
    sys.exit(0)

print("Done! 🎉 Check the ./out/ directory for the results.")