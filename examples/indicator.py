from pandas_composition import UserSeries
import pandas.io.data as pdd

df = pdd.get_data_yahoo('AAPL')

class Indicator(UserSeries):
    def __init__(self, *args, **kwargs):
        source = kwargs.pop('source')
        self.source = source

    def plot(self, source_col='close'):
        pass

def get_gaps(df, offset=0):
    gap_up = df.Open > (df.High.shift(1) + offset)
    gap_down = df.Open < (df.Low.shift(1) - offset)
    gaps = gap_up & gap_down
    return Indicator(gaps, source=df)
