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

    def test_infinite_column_loop(self):
        """
        https://github.com/dalejung/pandas-composition/issues/5
        Due to the DataSet changes and how __tr_getattr__ 
        we can run into an issue with subclasses the override
        __getitem__  that don't properly handle returning AttributeError
        for real attributes like ix.
        """
        class InfiniteFrame(UserFrame):
            counts = {} 
            log = []
            def __getitem__(self, name):
                InfiniteFrame.log.append(name)
                count = InfiniteFrame.counts.setdefault(name, 0)
                count += 1 
                InfiniteFrame.counts[name] = count
                # if not for this count check, we'd go into an infinite loop
                if count > 5:
                    raise AttributeError('error through exaustion')
                return self.ix[:2] 

        idf = InfiniteFrame(np.random.randn(10, 10))
        cols = idf.columns # trigger __tr_getattr__
        counts = InfiniteFrame.counts
        assert len(counts) == 0 # properly skip __getitem__ in __tr_getattr__

        idf['columns'] # call getitem
        # in the correct case, self.ix shouldn't find it's way back to __getitem__
        assert 'columns' in counts
        assert len(counts) == 1

    def test_userframe_override(self):
        """
        UserFrame.iteritems has a second sentinel value that does not
        return an iterator. It is not being called.
        """
        uf = UserFrame({'bob':range(5), 'frank':range(5)})
        correct = object.__getattribute__(uf, 'iteritems')(True)
        test = uf.iteritems(True)
        assert correct == test

class SubFrame(UserFrame):
    pass

sf = UserFrame(tm.makeDataFrame())

if __name__ == '__main__':                                                                                          
    import nose                                                                      
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],exit=False)   
