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

from plotters.HeatmapPlotter import HeatmapPlotter
from plotters.ScatterPlotter import ScatterPlotter
from plotters.WordcloudPlotter import WordcloudPlotter
from plotters.LinePlotter import LinePlotter
from plotters.BarPlotter import BarPlotter
from plotters.BarChartRacePlotter import BarChartRacePlotter
from plotters.Aggregator import Aggregator
from plotters.BoxPlotter import BoxPlotter

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

aggregator = Aggregator(modeled_data)

# Line plots
lineplotter = LinePlotter(modeled_data, plt, pd, go)
print("Creating line plots...")
lineplotter.lineplot_count('track')
lineplotter.lineplot_count('artist')
lineplotter.lineplot_count('album')
lineplotter.lineplot_count_log('track')
lineplotter.lineplot_count_log('artist')
lineplotter.lineplot_count_log('album')
lineplotter.lineplot_timeseries_plotly()
lineplotter.lineplot_timeseries_week()
lineplotter.lineplot_timeseries_audio_features_mean_month()
lineplotter.lineplot_timeseries_audio_features_mean_week()
lineplotter.lineplot_timeseries_bpm_month()
lineplotter.lineplot_timeseries_bpm_week()

# Box plots
boxplotter = BoxPlotter(modeled_data, plt, sns)
print("Creating box plots...")
boxplotter.boxplot_listen_time_start()
boxplotter.boxplot_listen_time_end()

# Heatmaps
heatmap_plotter = HeatmapPlotter(modeled_data, plt, sns, pd, calplot)
print("Creating heatmaps...")
heatmap_plotter.heatmap_correlation_matrix()

# Bar plots
barplotter = BarPlotter(modeled_data, plt, np, PILImage, aggregator, imageio)
print("Creating bar plots...")
barplotter.top_n_values_by_playing_time(10, 'track')
barplotter.top_n_values_by_playing_time(10, 'artist')
barplotter.top_n_values_by_playing_time(10, 'album')
barplotter.top_n_values_by_playing_time(10, 'podcast_show')
barplotter.top_n_values_by_playing_time(10, 'platform')
barplotter.top_n_values_by_count(10, 'track')
barplotter.top_n_values_by_count(10, 'artist')
barplotter.top_n_values_by_count(10, 'album')
barplotter.top_n_values_by_count(10, 'podcast_show')
barplotter.top_n_values_by_count(10, 'platform')
barplotter.avrg_listening_time_by('hour')
barplotter.avrg_listening_time_by('weekday')
barplotter.avrg_listening_time_by('month')
barplotter.avrg_listening_time_by('year')
barplotter.timeseries_count('track')
barplotter.timeseries_count('artist')
barplotter.timeseries_count('album')
barplotter.timeseries_count('podcast_episode')
barplotter.timeseries_playing_time()
barplotter.animation_monthly_top_n_by_aggregated_playing_time(10, 'track', 2017)
# barplotter.animation_monthly_top_n_by_aggregated_playing_time(10, 'album', 2017)
# barplotter.animation_monthly_top_n_by_aggregated_playing_time(10, 'artist', 2017)
# barplotter.animation_monthly_top_n_by_aggregated_playing_time(10, 'podcast_show', 2017)
# barplotter.animation_monthly_top_n_by_aggregated_playing_time(10, 'platform', 2017)
# barplotter.animation_monthly_top_n_by_aggregated_count(10, 'track', 2017)
# barplotter.animation_monthly_top_n_by_aggregated_count(10, 'album', 2017)
# barplotter.animation_monthly_top_n_by_aggregated_count(10, 'artist', 2017)
# barplotter.animation_monthly_top_n_by_aggregated_count(10, 'podcast_show', 2017)
# barplotter.animation_monthly_top_n_by_aggregated_count(10, 'platform', 2017)

# Bar chart race plots
barchart_race_plotter = BarChartRacePlotter(modeled_data, plt, pd,aggregator, bcr)
print("Creating bar chart race videos...")
barchart_race_plotter.bar_chart_race_top_n_by_aggregated_playing_time(10, 'track', 2017)
# barchart_race_plotter.bar_chart_race_top_n_by_aggregated_playing_time(10, 'album', 2017)
# barchart_race_plotter.bar_chart_race_top_n_by_aggregated_playing_time(10, 'artist', 2017)
# barchart_race_plotter.bar_chart_race_top_n_by_aggregated_playing_time(10, 'podcast_show', 2017)
# barchart_race_plotter.bar_chart_race_top_n_by_aggregated_playing_time(10, 'platform', 2017)
# barchart_race_plotter.bar_chart_race_top_n_by_aggregated_count(10, 'track', 2017)
# barchart_race_plotter.bar_chart_race_top_n_by_aggregated_count(10, 'album', 2017)
# barchart_race_plotter.bar_chart_race_top_n_by_aggregated_count(10, 'artist', 2017)
# barchart_race_plotter.bar_chart_race_top_n_by_aggregated_count(10, 'platform', 2017)

# Scatterplot
scatterplotter = ScatterPlotter(modeled_data, plt, sns, pd)
print("Creating scatterplots...")
scatterplotter.count_playing_time_mean('track')
scatterplotter.count_playing_time_median('track')
scatterplotter.count_playing_time_mean('album')
scatterplotter.count_playing_time_median('album')
scatterplotter.count_playing_time_mean('artist')
scatterplotter.count_playing_time_median('artist')

# Wordcloud
wordCloutPlotter = WordcloudPlotter(modeled_data, plt, wordcloud)
print("Creating wordclouds...")
wordCloutPlotter.plot('track')
wordCloutPlotter.plot('artist')
wordCloutPlotter.plot('album')
wordCloutPlotter.plot('podcast_show')

print("Done! 🎉 Check the ./out/ directory for the results.")