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

    def test_init_args(self):
        """
        Support init params for things like `series + 1`. While metadata propogates, 
        currently (2013/07/01) wrapping fails because it calls the constructor instead
        of calling .view
        """
        class SubSeries(UserSeries):
            def __init__(self, *args, **kwargs):
                bob = kwargs.pop('bob')
                self.bob = bob
                super(SubSeries, self).__init__(*args, **kwargs)

        ss = SubSeries(range(10), bob=123)
        assert ss.bob == 123
        test = ss + 1 # currently errors
        assert test.bob == 123

    def test_init_args_set_meta_check(self):
        """
        Support init params for things like `series + 1`. While metadata propogates, 
        currently (2013/07/01) wrapping fails because it calls the constructor instead
        of calling .view
        """
        class SubSeries(UserSeries):
            def __init__(self, *args, **kwargs):
                bob = kwargs.pop('bob')
                self.bob = bob
                super(SubSeries, self).__init__(*args, **kwargs)

        ss = SubSeries(range(10), bob=123)
        assert ss.bob == 123

        # pandas constructors vars go to pandas object
        # this is due to the fact that pandas sets its init args as
        # member variables
        ss.copy = True
        assert 'copy' not in ss.meta
        assert 'copy' in ss.pobj.__dict__

        try:
            ss.set_meta('copy', False)
        except:
            pass
        else:
            assert False, 'copy should fail as it is a constructor arg'

    def test_monkeyed_pandas_object(self):
        """
        A monkey-patched method on base pandas object is callable
        but will pass in that base type instead of the subclass
        """
        return # not sure if this is error
        def type_method(self):
            return type(self)

        pd.Series.type_method = type_method

        class SubSeries(UserSeries):
            pass

        s = SubSeries(range(10))
        t = s.type_method()
        assert t is SubSeries

    def test_np_where(self):
        us = UserSeries(range(10))
        bools = us > 5
        tvals = np.repeat(1, len(us))
        fvals = np.repeat(0, len(us))
        wh = np.where(bools, tvals, fvals)

        assert isinstance(wh, UserSeries)

    def test_series_view(self):
        """
        """

us = UserSeries(range(10))
us.view('i8')


if __name__ == '__main__':                                                                                          
    import nose                                                                      
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],exit=False)   
