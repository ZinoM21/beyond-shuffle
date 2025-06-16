class BasePlotter:
    def __init__(self, df, plt, sns=None, output_path=None):
        self.df = df
        self.plt = plt
        self.sns = sns
        self.output_path = output_path

        plt.figure()
        plt.rcParams.update({'font.size': 8, 'figure.autolayout': True})
        plt.rcParams["figure.figsize"] = (10,7)

    def _save_and_close_plot(self, plot, filename: str) -> None:
        """
        Save the plot to the specified filename and close the figure to free resources.

        Args:
            plot: The plot object to save (either a matplotlib Axes or Figure, or seaborn/matplotlib plot).
            filename: The filename to save the plot to (just the file name, not full path).
        """
        import os
        # Use self.output_path if set, else current directory
        output_dir = self.output_path if self.output_path else '.'
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if filename already contains the output path
        if self.output_path and filename.startswith(self.output_path):
            save_path = filename
        else:
            save_path = os.path.join(output_dir, filename)
            
        # Try to get the figure from the plot, else assume it's a figure
        fig = getattr(plot, 'get_figure', lambda: plot)()
        fig.savefig(save_path)
        self.plt.close(fig)