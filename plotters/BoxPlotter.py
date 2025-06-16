from plotters.BasePlotter import BasePlotter

class BoxPlotter(BasePlotter):
    def __init__(self, df, plt, sns, output_path="./out/boxplots/"):
        super().__init__(df, plt, sns, output_path)

    def boxplot_listen_time_start(self, color_palette='Blues'):
        fig, ax = self.plt.subplots(figsize=(20,10))
        df_reset = self.df.reset_index(drop=True)
        self.sns.boxplot(x="reason_start", y="relation_listening_lenght", data=df_reset, palette=color_palette)
        self.plt.title('Boxplot for the listening_time/song_lenght for the reason_start')
        self._save_and_close_plot(fig, "boxplot_listen_time_start.png")
        return fig

    ### Creating a boxplot for the relation_listening_lenght for the start_end
    def boxplot_listen_time_end(self, color_palette='Blues'):
        fig, ax = self.plt.subplots(figsize=(20,10))
        df_reset = self.df.reset_index(drop=True)
        self.sns.boxplot(x="reason_end", y="relation_listening_lenght", data=df_reset, palette=color_palette)
        self.plt.title('Boxplot for the listening_time/song_lenght for the reason_end')
        self._save_and_close_plot(fig, "boxplot_listen_time_end.png")
        return fig