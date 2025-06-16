import wordcloud
from plotters.BasePlotter import BasePlotter
import matplotlib.pyplot as plt

class WordcloudPlotter(BasePlotter):
    def __init__(self, df, plt: plt, wc: wordcloud, output_path="./out/wordclouds/"):
        super().__init__(df, plt, output_path=output_path)
        self.wc = wc

    def plot(self, title):
        '''Creating a function to create a word cloud with the argument of the column'''

        text = self.df[title].str.cat(sep=' ')
        wordcloud = self.wc.WordCloud(stopwords=self.wc.STOPWORDS, background_color='white', width=3000, height=2500, min_font_size = 10).generate(text)
        fig = self.plt.figure()
        self.plt.imshow(wordcloud)
        self.plt.title(title)
        self.plt.axis("off")
        self.plt.tight_layout(pad = 0)
        filename = "wordcloud_" + title.lower().replace(' ', '_') + ".png"
        self._save_and_close_plot(fig, filename)


