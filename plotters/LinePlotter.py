from plotters.BasePlotter import BasePlotter

class LinePlotter(BasePlotter):
    def __init__(self, df, plt, pd, graph_objects, output_path="./out/lineplots/"):
        super().__init__(df, plt, output_path=output_path)
        self.pd = pd
        self.go = graph_objects

    def lineplot_count(self, column, color_palette='blue'):
        df_times_played = self.pd.DataFrame(self.df.value_counts(column).reset_index().rename(columns={0:'count'}))
        self.plt.figure()
        plot = df_times_played.plot(
            color=color_palette,
            kind='line', 
            x=column,
            xticks=[], 
            y='count', 
            ylabel='times played', 
            title=f'How many times a {column} has been played'
            )

        self.plt.title(f'Lineplot for count by {column}')
        self._save_and_close_plot(plot, f"lineplot_count_{column}.png")
        return plot

    def lineplot_count_log(self, column, color='blue'):
        df_times_played = self.pd.DataFrame(self.df.value_counts(column).reset_index().rename(columns={0:'count'}))
        self.plt.figure()
        plot = df_times_played.plot(
            color=color,
            kind='line', 
            x=column,
            xticks=[], 
            y='count', 
            ylabel='times played',
            logy=True, 
            title=f'How many times a {column} has been played'
        )
        self._save_and_close_plot(plot, f"lineplot_count_log_{column}.png")
        return plot

    def lineplot_count_loglog(self, column, color='blue'):
        df_times_played = self.pd.DataFrame(self.df.value_counts(column).reset_index().rename(columns={0:'count'}))
        self.plt.figure()
        plot = df_times_played.plot(
            color=color,
            kind='scatter', 
            x=column,
            xticks=[], 
            y='count', 
            ylabel='times played',
            loglog=True, 
            title=f'How many times a {column} has been played'
        )
        self._save_and_close_plot(plot, f"lineplot_count_loglog_{column}.png")
        return plot

    def lineplot_timeseries_week(self, color='blue'):
        df_week = self.df.resample('W').sum(numeric_only=True)
        self.plt.figure()
        plot = df_week.plot(
            color=color,
            y='listening_time_in_min',
            kind='line',
            xlabel='Week',
            ylabel='Time in minutes',
            title='Total playing time per Week',
            legend=False,
        )
        self._save_and_close_plot(plot, "lineplot_timeseries_week.png")
        return plot

    def lineplot_timeseries_bpm_week(self, color='blue'):
        df_group_year_week = self.df.groupby([self.df.index.year, self.df.index.isocalendar().week]).mean(numeric_only=True)
        self.plt.figure()
        plot = df_group_year_week.plot(
            color=color,
            kind='line', 
            y='tempo', 
            ylabel='BPM', 
            title='Timeseries of average BPM per week'
        )
        self._save_and_close_plot(plot, "lineplot_timeseries_bpm_week.png")
        return plot

    def lineplot_timeseries_bpm_month(self, color='blue'):
        df_group_year_month = self.df.groupby([self.df.index.year, self.df.index.month]).mean(numeric_only=True)
        self.plt.figure()
        plot = df_group_year_month.plot(
            color=color,
            kind='line', 
            y='tempo', 
            ylabel='BPM', 
            title='Timeseries of average BPM per month'
        )
        self._save_and_close_plot(plot, "lineplot_timeseries_bpm_month.png")
        return plot

    def lineplot_timeseries_audio_features_mean_month(self):
        df_af = self.df[['danceability','energy','mode','speechiness','instrumentalness','liveness','valence']]
        df_grouped = df_af.groupby([df_af.index.year, df_af.index.month]).mean()
        self.plt.figure()
        plot = df_grouped.plot(
            kind='line',
            title='Timeseries of average audio features per month',
            ylabel='average audio features',
        )
        self._save_and_close_plot(plot, "lineplot_timeseries_audio_features_month.png")
        return plot

    def lineplot_timeseries_audio_features_mean_week(self):
        df_af = self.df[['danceability','energy','mode','speechiness','instrumentalness','liveness','valence']]
        df_grouped = df_af.groupby([df_af.index.year, df_af.index.isocalendar().week]).mean()
        self.plt.figure()
        plot = df_grouped.plot(
            kind='line',
            title='Timeseries of average audio features per week',
            ylabel='average audio features',
        )
        self._save_and_close_plot(plot, "lineplot_timeseries_audio_features_week.png")
        return plot

    def lineplot_timeseries_plotly(self, color='blue'):
        df_hour = self.df.resample('H').sum(numeric_only=True)
        fig = self.go.Figure()
        fig.add_trace(
            self.go.Scatter(x=list(df_hour.index), y=list(df_hour.listening_time_in_h), marker = {'color' : color}))
        fig.update_layout(
            title_text="Time series of playing time for every hour"
        )
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                            label="1m",
                            step="month",
                            stepmode="backward"),
                        dict(count=6,
                            label="6m",
                            step="month",
                            stepmode="backward"),
                        dict(count=1,
                            label="YTD",
                            step="year",
                            stepmode="todate"),
                        dict(count=1,
                            label="1y",
                            step="year",
                            stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        fig.write_html(f"{self.output_path}lineplot_timeseries_plotly.html")
        self.plt.close('all')