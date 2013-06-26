from unittest import TestCase

import pandas as pd
import pandas.util.testing as tm
import numpy as np

import pandas_composition.frame as pframe
import pandas_composition as composition
UserFrame = composition.UserFrame
UserSeries = composition.UserSeries


class TestUserFrame(TestCase):

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)

    def runTest(self):
        pass

    def setUp(self):
        pass

    def test_init_col_meta(self):
        """
        Test properly initializing the col meta from a constructor
        initializing.
        """
        class BobSeries(UserSeries):
            pass
        bob = BobSeries(range(10, 20))
        bob.bob = 'bob'

        dale = UserSeries(range(10))
        dale.whee = 'whee'

        df = UserFrame({'bob':bob, 'dale':dale})
        # add via setitem
        df['frank'] = UserSeries(np.random.randn(10))

        # check the proper class
        assert type(df.bob) is BobSeries
        assert type(df['dale']) is UserSeries
        assert type(df['frank']) is UserSeries

        # check meta
        assert df.bob.bob == 'bob'
        assert df.dale.whee == 'whee'

    def test_setitem_col_meta(self):
        """
        Test properly initializing the col meta from a constructor
        initializing.
        """
        class BobSeries(UserSeries):
            pass
        ind = pd.date_range(start="2000", freq="D", periods=10)
        bob = BobSeries(range(10, 20), index=ind)
        bob.bob = 'bob'

        dale = UserSeries(range(10), index=ind)
        dale.whee = 'whee'

        df = UserFrame(None, index=ind)
        # add via setitem
        df['bob'] = bob
        df['dale'] = dale
        df['frank'] = UserSeries(np.random.randn(10), index=ind)
        df['zero'] = 0
        df['ndarray'] = np.random.randn(10)

        # check the proper class
        assert type(df.bob) is BobSeries
        assert type(df['dale']) is UserSeries
        assert type(df['frank']) is UserSeries
        assert type(df.zero) is pd.TimeSeries
        assert type(df.ndarray) is pd.TimeSeries

        # check meta
        assert df.bob.bob == 'bob'
        assert df.dale.whee == 'whee'

if __name__ == '__main__':                                                                                          
    import nose                                                                      
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],exit=False)   
