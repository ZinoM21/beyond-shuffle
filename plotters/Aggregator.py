import pandas as pd

class Aggregator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def monthly_top_n_by_playing_time(self, n, column, year, month):
        """Get the top n values of a column by year and month"""
        df_grouped_year_month = self.df.groupby([self.df.index.year, self.df.index.month, self.df[column]]).sum()['duration_in_min']

        df_top_values_by_month_playing_time = df_grouped_year_month.loc[year, month].nlargest(n, keep='all')

        return df_top_values_by_month_playing_time

    def monthly_top_n_by_count(self, n, column, year, month):
        """Get the top n values of a column by year and month"""
        df_grouped_year_month = self.df.groupby([self.df.index.year, self.df.index.month, self.df[column]]).value_counts(column)

        df_top_values_by_month_count = df_grouped_year_month.loc[year, month].nlargest(n, keep='all')

        return df_top_values_by_month_count

    def monthly_top_n_by_aggregated_playing_time(self, n, column, year, month) -> pd.Series:
        """Get the top n values of a column by year and month"""

        # Create new df with the end date to sum to
        end = f'{year}-{month}-01 00:00:00+00:00'

        df_new = self.df.loc[:end]

        df_summed_year_month = df_new.groupby(df_new[column]).sum(numeric_only=True)['listening_time_in_min']

        # Get top n values
        df_top_values_by_month_listening_time = df_summed_year_month.nlargest(n, keep='all')

        return df_top_values_by_month_listening_time

    def monthly_top_n_by_aggregated_count(self, n, column, year, month):
        """Get the top n values of a column by year and month"""

        # Create new df with the end date to sum to
        end = f'{year}-{month}-01 00:00:00+00:00'

        df_new = self.df.loc[:end]
        
        s_counted_year_month = df_new.value_counts(column)

        # Get top n values
        s_top_values_by_month_count = s_counted_year_month.nlargest(n, keep='all')

        return s_top_values_by_month_count
