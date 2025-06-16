import os

from plotters.BasePlotter import BasePlotter
from plotters.Aggregator import Aggregator

class BarChartRacePlotter(BasePlotter):
    def __init__(self, df, plt, pd, aggregator: Aggregator, bcr, output_path="./out/bar_chart_race/"):
        super().__init__(df, plt, output_path=output_path)
        self.pd = pd
        self.aggregator = aggregator
        self.output_path = output_path
        self.bcr = bcr
        os.makedirs(self.output_path, exist_ok=True)

    def _create_monthly_dataframe(self, year, month, n, column, metric_type):
        """Create monthly dataframe for bar chart race"""
        try:
            if metric_type == 'playing_time':
                series = self.aggregator.monthly_top_n_by_aggregated_playing_time(n*5, column, year, month)
            else:  # count
                series = self.aggregator.monthly_top_n_by_aggregated_count(n*5, column, year, month)
                
            df_month = series.to_frame().T
            df_month['date'] = f'{year}-{month}'
            df_month.set_index('date', inplace=True)
            return df_month
        except Exception as e:
            print(f'Error creating monthly dataframe for {year}-{month}: {e}')
            return None

    def _concatenate_monthly_data(self, start_year, n, column, metric_type):
        """Concatenate monthly data into single dataframe"""
        df_concated = None
        for year in range(start_year, self.df.index.year.max() + 1):
            for month in range(1, 13):
                df_month = self._create_monthly_dataframe(year, month, n, column, metric_type)
                if df_month is not None:
                    if df_concated is None:
                        df_concated = df_month
                    else:
                        df_concated = self.pd.concat([df_concated, df_month])
        return df_concated

    def _create_bar_chart_race(self, df, n, column, metric_type):
        """Create and save bar chart race video"""
        if df is not None:
            metric_name = 'playing time in minutes' if metric_type == 'playing_time' else 'count'
            self.bcr.bar_chart_race(
                df=df,
                sort='desc',
                filename=f'{self.output_path}top_{n}_{column}_by_{metric_type}.mp4',
                title=f'Top {n} {column} by {metric_name} (aggregated)',
                filter_column_colors=True,
                n_bars=n,
            )

    def bar_chart_race_top_n_by_aggregated_playing_time(self, n, column, start_year):
        df_concated = self._concatenate_monthly_data(start_year, n, column, 'playing_time')
        self._create_bar_chart_race(df_concated, n, column, 'playing_time')

    def bar_chart_race_top_n_by_aggregated_count(self, n, column, start_year):
        df_concated = self._concatenate_monthly_data(start_year, n, column, 'count')
        self._create_bar_chart_race(df_concated, n, column, 'count')