

from plotters.BasePlotter import BasePlotter
from plotters.Aggregator import Aggregator

class BarPlotter(BasePlotter):
    def __init__(self, df, plt, np, pilimg,aggregator: Aggregator, imageio, output_path="./out/barplots/"):
        super().__init__(df, plt, output_path=output_path)
        self.np = np
        self.pilimg = pilimg
        self.aggregator = aggregator
        self.imageio = imageio

    def top_n_values_by_playing_time(self, n, column, color='blue'):
        self.plt.figure()
        df_top_tracks = self.df.groupby(column).sum(numeric_only=True)['listening_time_in_h'].nlargest(n, keep='all')
        plot = df_top_tracks.plot(
            color=color,
            kind='bar',
            ylabel='time in hours',
            title=f'Top {n} {column} of all time by playing time',
            legend=False,
        )
        self.plt.tight_layout()
        self._save_and_close_plot(plot, f"barplot_top_{n}_{column}_by_playing_time.png")

    def top_n_values_by_count(self, n, column, color='blue'):
        self.plt.figure()
        df_new = self.df.value_counts(column).nlargest(n, keep='all')
        plot = df_new.plot(
            color=color,
            kind='bar',
            xlabel=column,
            ylabel='times played',
            title=f'Top {n} {column} of all time by number of times played',
            legend=False
        )
        self.plt.tight_layout()
        self._save_and_close_plot(plot, f"barplot_top_{n}_{column}_by_count.png")
        
    def avrg_listening_time_by(self, by, color='blue'):
        df = self.df.copy()
        
        # Configure time period settings
        time_settings = {
            'hour': {
                'resample': 'H',
                'groupby': lambda x: x.hour,
                'ylabel': 'time in minutes',
                'xlabel': 'hour',
                'time_col': 'listening_time_in_min',
                'sorter': None
            },
            'weekday': {
                'resample': 'D',
                'groupby': lambda x: x.day_name(),
                'ylabel': 'time in minutes',
                'xlabel': 'day of week',
                'time_col': 'listening_time_in_min',
                'sorter': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            },
            'month': {
                'resample': 'M',
                'groupby': lambda x: x.month_name(),
                'ylabel': 'time in hours',
                'xlabel': 'month',
                'time_col': 'listening_time_in_h',
                'sorter': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            },
            'year': {
                'resample': 'Y',
                'groupby': lambda x: x.year,
                'ylabel': 'time in hours',
                'xlabel': 'year',
                'time_col': 'listening_time_in_h',
                'sorter': None
            }
        }
        
        settings = time_settings[by]
        df_resampled = df.resample(settings['resample']).sum(numeric_only=True)
        
        self.plt.figure()
        plot = df_resampled.groupby(settings['groupby'](df_resampled.index)).mean()
        
        if settings['sorter']:
            plot = plot.reindex(settings['sorter'])
            
        plot = plot.plot(
            color=color,
            kind='bar',
            y=settings['time_col'],
            ylabel=settings['ylabel'],
            xlabel=settings['xlabel'],
            title=f'Average listening time by {by}',
            legend=False,
        )
        
        self.plt.tight_layout()
        self._save_and_close_plot(plot, f"barplot_avrg_listening_time_by_{by}.png")

    def timeseries_count(self, column, color='blue'):
        df_group_year_month = self.df.groupby([self.df.index.year, self.df.index.month]).count()
        plot = df_group_year_month.plot(
            color=color,
            y=column,
            kind='bar',
            xlabel='Year and Month',
            ylabel=f'Number of {column}',
            title=f'Number of {column} Played per Month',
            legend=False,
        )
        self.plt.tight_layout()
        self._save_and_close_plot(plot, f"barplot_timeseries_count_{column}.png")

    def timeseries_playing_time(self, color='blue'):
        df_group_year_month = self.df.groupby([self.df.index.year, self.df.index.month]).sum(numeric_only=True)
        self.plt.figure()
        plot = df_group_year_month.plot(
            color=color,
            y='listening_time_in_h',
            kind='bar',
            xlabel='Year and Month',
            ylabel='Time in hours',
            title='Total playing time per Month',
            legend=False,
        )
        self.plt.tight_layout()
        self._save_and_close_plot(plot, "barplot_timeseries_playing_time.png")

    # --- Monthly/temporal top-N bar plots and animations ---
    def monthly_top_n_by_playing_time(self, n, column, year, month, color='blue'):
        df_top = self.aggregator.monthly_top_n_by_playing_time(n, column, year, month)
        self.plt.close()
        plot = df_top.plot(
            kind='bar',
            color=color,
            ylabel='time in minutes',
            title=f'Top {n} {column} by playing time in {month}-{year}',
            legend=False,
        )
        self._save_and_close_plot(plot, f"top_{n}_{column}_by_playing_time_{year}_{month}.png")

    def monthly_top_n_by_count(self, n, column, year, month, color='blue'):
        df_top = self.aggregator.monthly_top_n_by_count(n, column, year, month)
        self.plt.close()
        plot = df_top.plot(
            kind='bar',
            color=color,
            ylabel='count',
            title=f'Top {n} {column} by count in {month}-{year}',
            legend=False,
        )
        self._save_and_close_plot(plot, f"top_{n}_{column}_by_count_{year}_{month}.png")

    def monthly_top_n_by_aggregated_playing_time(self, n, column, year, month, color='blue'):
        df_top = self.aggregator.monthly_top_n_by_aggregated_playing_time(n, column, year, month)
        self.plt.close()
        plot = df_top.plot(
            kind='bar',
            color=color,
            ylabel='time in minutes',
            title=f'Top {n} {column} by playing time in {month}, {year} (aggregated)',
            legend=False,
        )
        self._save_and_close_plot(plot, f"top_{n}_{column}_by_aggregated_playing_time_{year}_{month}.png")

    def monthly_top_n_by_aggregated_count(self, n, column, year, month, color='blue'):
        df_top = self.aggregator.monthly_top_n_by_aggregated_count(n, column, year, month)
        self.plt.close()
        plot = df_top.plot(
            kind='bar',
            color=color,
            ylabel='count',
            title=f'Top {n} {column} by count in {month}, {year} (aggregated)',
            legend=False,
        )
        self._save_and_close_plot(plot, f"top_{n}_{column}_by_aggregated_count_{year}_{month}.png")

    def animation_monthly_top_n_playing_time(self, n, column, start_year, color='blue'):
        plots = []
        for year in range(start_year, self.df.index.year.max() + 1):
            for month in range(1, 13):
                try:
                    df_top = self.aggregator.monthly_top_n_by_playing_time(n, column, year, month)
                    self.plt.close()
                    plot = df_top.plot(
                        kind='bar',
                        color=color,
                        ylabel='time in minutes',
                        title=f'Top {n} {column} by playing time in {month}-{year}',
                        legend=False,
                    )
                    fname = f"{self.output_path}tmp_{year}_{month}.png"
                    self._save_and_close_plot(plot, fname)
                    plots.append(self.imageio.v2.imread(fname))
                except Exception as e:
                    print(f'Error in animation_monthly_top_n_playing_time: {year}-{month}: {e}')
                    pass
        self.imageio.mimsave(f"{self.output_path}animation_playing_time.gif", plots, fps=2, duration=0.5)

    def animation_monthly_top_n_count(self, n, column, start_year, color='blue'):
        plots = []
        for year in range(start_year, self.df.index.year.max() + 1):
            for month in range(1, 13):
                try:
                    df_top = self.aggregator.monthly_top_n_by_count(n, column, year, month)
                    self.plt.close()
                    plot = df_top.plot(
                        kind='bar',
                        color=color,
                        ylabel='count',
                        title=f'Top {n} {column} by count in {month}-{year}',
                        legend=False,
                    )
                    fname = f"{self.output_path}tmp_{year}_{month}.png"
                    self._save_and_close_plot(plot, fname)
                    plots.append(self.imageio.v2.imread(fname))
                except Exception as e:
                    print(f'Error in animation_monthly_top_n_count: {year}-{month}: {e}')
                    pass
        self.imageio.mimsave(f"{self.output_path}animation_count.gif", plots, fps=2, duration=0.5)

    def _create_tmp_dir(self):
        """Create temporary directory for animation frames"""
        import os
        tmp_dir = os.path.join(self.output_path, 'tmp_animation_frames')
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir

    def _cleanup_tmp_dir(self, tmp_dir):
        """Remove temporary directory and its contents"""
        import shutil
        shutil.rmtree(tmp_dir)

    def _create_animation_frame(self, df_top, year, month, n, column, metric, color, tmp_dir):
        """Create a single animation frame"""
        import os
        if df_top.empty:
            raise ValueError(f'No data for {year}-{month}')
            
        self.plt.close()
        plot = df_top.plot(
            kind='bar',
            color=color,
            ylabel=metric,
            title=f'Top {n} {column}s by {metric} in {month}, {year} (aggregated)',
            legend=False,
        )
        
        file_path = os.path.join(tmp_dir, f"tmp_{year}_{month}.png")
        self._save_and_close_plot(plot, file_path)
        img = self.imageio.v2.imread(file_path)
        return img

    def _resize_image(self, img, first_img_shape):
        """Resize image to match first image shape"""
        if first_img_shape is None:
            return img, img.shape
        else:
            img = self.pilimg.fromarray(img)
            img = img.resize((first_img_shape[1], first_img_shape[0]))
            return self.np.array(img), first_img_shape

    def _create_animation(self, plots: list, output_name: str):
        """Create and save animation from plots"""
        if not plots:
            print(f"No plots to create animation for {output_name}")
            return
            
        self.imageio.mimsave(f"{self.output_path}{output_name}.gif", plots, fps=2, duration=0.5)

    def animation_monthly_top_n_by_aggregated_playing_time(self, n, column, start_year, color='blue'):
        tmp_dir = self._create_tmp_dir()
        plots = []
        first_img_shape = None
        
        try:
            for year in range(start_year, self.df.index.year.max() + 1):
                for month in range(1, 13):
                    try:
                        df_top = self.aggregator.monthly_top_n_by_aggregated_playing_time(n, column, year, month)
                        img = self._create_animation_frame(df_top, year, month, n, column, 'time in minutes', color, tmp_dir)
                        
                        if img is not None:
                            img, first_img_shape = self._resize_image(img, first_img_shape)
                            plots.append(img)
                            
                    except Exception as e:
                        print(f'Error in animation_monthly_top_n_by_aggregated_playing_time: {year}-{month}: {e}')
                        continue
                        
            self._create_animation(plots, f"animation_aggregated_playing_time_{column}")
            
        finally:
            self._cleanup_tmp_dir(tmp_dir)

    def animation_monthly_top_n_by_aggregated_count(self, n, column, start_year, color='blue'):
        tmp_dir = self._create_tmp_dir()
        plots = []
        first_img_shape = None
        
        try:
            for year in range(start_year, self.df.index.year.max() + 1):
                for month in range(1, 13):
                    try:
                        df_top = self.aggregator.monthly_top_n_by_aggregated_count(n, column, year, month)
                        img = self._create_animation_frame(df_top, year, month, n, column, 'count', color, tmp_dir)
                        
                        if img is not None:
                            img, first_img_shape = self._resize_image(img, first_img_shape)
                            plots.append(img)
                            
                    except Exception as e:
                        print(f'Error in animation_monthly_top_n_by_aggregated_count: {year}-{month}: {e}')
                        continue
                        
            self._create_animation(plots, f"animation_aggregated_count_{column}")
            
        finally:
            self._cleanup_tmp_dir(tmp_dir)