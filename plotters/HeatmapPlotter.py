from plotters.BasePlotter import BasePlotter
import calplot

class HeatmapPlotter(BasePlotter):
    def __init__(self, df, plt, sns, pd, calplot: calplot, output_path="./out/heatmaps/"):
        super().__init__(df, plt, sns, output_path)
        self.pd = pd
        self.calplot = calplot

    def heatmap_correlation_matrix(self, colormap='Blues'):
        # Select only numeric columns for correlation
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        self.plt.figure(figsize=(20,10))
        plot = self.sns.heatmap(numeric_df.corr(), annot=True, cmap=colormap)
        self._save_and_close_plot(plot, "heatmap_correlation_matrix.png")

    # def heatmap_size_matrix(self, colormap='Blues'):
    #     # Ensure the index is datetime and sorted
    #     df_sorted = self.df.copy()
    #     df_sorted.index = self.pd.to_datetime(df_sorted.index)
    #     df_sorted = df_sorted.sort_index()

    #     # Aggregate to daily sums to remove duplicate dates
    #     daily_sum = df_sorted['listening_time_in_min'].groupby(df_sorted.index).sum()

    #     # Create a complete daily date range from min to max date
    #     full_range = self.pd.date_range(daily_sum.index.min(), daily_sum.index.max(), freq='D')
    #     # Reindex and fill missing days with 0
    #     listening_time = daily_sum.reindex(full_range, fill_value=0)
    #     listening_time.index.name = 'date'  # Optional: name the index

    #     # Create the calendar heatmap
    #     self.plt.figure(figsize=(20, 10))
    #     plot = self.calplot.calplot(
    #         data=listening_time,
    #         how='sum',
    #         cmap=colormap,
    #         suptitle="Total Listening Time by Month and Year",
    #         yearlabel_kws={'fontsize': 12}
    #     )
    #     self._save_and_close_plot(plot, "heatmap_size_matrix.png")
