from unittest import TestCase
import cPickle as pickle

import pandas as pd
import pandas.util.testing as tm
import numpy as np

import pandas_composition as composition
UserSeries = composition.UserSeries

from trtools.util.tempdir import TemporaryDirectory

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

    def test_series_pickle(self):
        """
        Test that the UserSeries pickles correctly
        """
        s = UserSeries(range(10))
        s.frank = '123'
        with TemporaryDirectory() as td:
            fn = td + '/test.save'
            with open(fn, 'wb') as f:
                pickle.dump(s, f, protocol=0)

            with open(fn, 'rb') as f:
                test = pickle.load(f)
            tm.assert_almost_equal(s, test)
            assert isinstance(test, UserSeries)
            assert test.frank == '123'

if __name__ == '__main__':                                                                                          
    import nose                                                                      
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],exit=False)   
