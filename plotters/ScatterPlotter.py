from plotters.BasePlotter import BasePlotter

class ScatterPlotter(BasePlotter):
    def __init__(self, df, plt, sns, pandas, output_path = './out/scatterplots/'):
        """
        Initialize ScatterPlotter.

        Args:
            df: DataFrame containing the data.
            plt: Matplotlib pyplot module.
            pandas: Pandas module.
            output_path: Directory to save scatterplots.
        """
        super().__init__(df, plt, sns, output_path)
        self.pd = pandas

    def count_playing_time_mean(self, column, color = 'blue'):
        """
        Create a scatterplot of count over mean playing time for a given column.

        Args:
            column: The column to group by.
            color: Color of the scatter points.
        Returns:
            The matplotlib plot object.
        """
        self.plt.figure()
        try:
            s1 = self.df.value_counts(column).reset_index().rename(columns={0: 'count'})
            s2 = self.df.groupby(column).mean(numeric_only=True)['listening_time_in_min'].reset_index().rename(columns={0: column})
            df_new = self.pd.merge(s1, s2)
        except Exception as e:
            print(f"Error preparing data for mean scatterplot: {e}")
            raise RuntimeError(f"Error preparing data for mean scatterplot: {e}")

        plot = df_new.plot.scatter(
            color=color,
            x='listening_time_in_min',
            xlabel='mean playing time in minutes',
            y='count',
            ylabel='number of times played',
            title=f'Distribution of count over mean playing time for each {column}'
        )
        plot.axvline(df_new.listening_time_in_min.mean(), color='r', linestyle='-')
        plot.legend([column, 'overall mean playing time'])
        self.plt.tight_layout()
        self._save_and_close_plot(plot, f'scatterplot_count_playing_time_mean_{column}.png')
        return plot

    def count_playing_time_median(self, column, color = 'blue'):
        """
        Create a scatterplot of count over median playing time for a given column.

        Args:
            column: The column to group by.
            color: Color of the scatter points.
        Returns:
            The matplotlib plot object.
        """
        self.plt.figure()
        try:
            s1 = self.df.value_counts(column).reset_index().rename(columns={0: 'count'})
            s2 = self.df.groupby(column).median(numeric_only=True)['listening_time_in_min'].reset_index().rename(columns={0: column})
            df_new = self.pd.merge(s1, s2)
        except Exception as e:
            print(f"Error preparing data for median scatterplot: {e}")
            raise RuntimeError(f"Error preparing data for median scatterplot: {e}")

        plot = df_new.plot.scatter(
            color=color,
            x='listening_time_in_min',
            xlabel='median playing time in minutes',
            y='count',
            ylabel='number of times played',
            title=f'Distribution of count over median playing time for each {column}'
        )
        plot.axvline(df_new.listening_time_in_min.mean(), color='r', linestyle='-')
        plot.legend([column, 'overall mean playing time'])
        self.plt.tight_layout()
        self._save_and_close_plot(plot, f'scatterplot_count_playing_time_median_{column}.png')
        return plot

    def scatterplot_example(self, x_col, y_col, color_palette='Blues'):
        fig, ax = self.plt.subplots(figsize=(20,10))
        self.sns.scatterplot(x=x_col, y=y_col, data=self.df, palette=color_palette)
        self.plt.title(f'Scatterplot of {x_col} vs {y_col}')
        self._save_and_close_plot(fig, f"scatterplot_{x_col}_vs_{y_col}.png")
        return fig




