from unittest import TestCase

import pandas as pd
import pandas.util.testing as tm
import numpy as np

import pandas_composition as composition
UserSeries = composition.UserSeries

class TestSeries(TestCase):

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)

    def runTest(self):
        pass

    def setUp(self):
        pass

    def test_timeseries_vs_series(self):
        """
        Due to the auto changing of Series to TimeSeries when 
        having a DatetimeIndex my _wrap check had a problem 
        with it's direct check. Technically, UserSeries has 
        two pandas types, pd.Series and pd.TimeSeries
        """
        class SubSeries(UserSeries):
            pass
        # check pd.Series
        s = SubSeries(range(10))
        bools = s > 0
        assert type(bools) is SubSeries

        # check TimeSeries
        ind = pd.date_range(start="2000", freq="D", periods=10)
        s = SubSeries(range(10), index=ind)
        bools = s > 0
        assert type(bools) is SubSeries

if __name__ == '__main__':                                                                                          
    import nose                                                                      
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],exit=False)   
