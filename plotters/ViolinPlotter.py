from plotters.BasePlotter import BasePlotter

class ViolinPlotter(BasePlotter):    
    def __init__(self, df, plt, sns, output_path="./out/violinplots/"):
        super().__init__(df, plt, sns, output_path)

    def violinplot_count(self, column, color_palette='Blues'):
        fig, ax = self.plt.subplots(figsize=(20,10))
        self.sns.violinplot(x=column, y='count', data=self.df, palette=color_palette)
        self.plt.title(f'Violinplot for count by {column}')
        self._save_and_close_plot(fig, f"violinplot_count_{column}.png")
        return fig