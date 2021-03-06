from unittest import TestCase
import os.path

import pandas as pd
import pandas.util.testing as tm
import numpy as np

from six import with_metaclass

import pandas_composition as composition
UserSeries = composition.UserSeries
UserFrame = composition.UserFrame

def curpath():
    pth, _ = os.path.split(os.path.abspath(__file__))
    return pth


class TestComposition(TestCase):

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)

    def runTest(self):
        pass

    def setUp(self):
        pass

    def test_user_series(self):
        s = pd.Series(range(1, 10), index=range(11, 20))
        us = UserSeries(s)
        tm.assert_series_equal(s, us)

        def assert_op(pobj, userobj, op):
            return
            cls = type(userobj)
            correct = op(pobj)
            test = op(userobj)
            assert isinstance(test, cls)
            if isinstance(correct, pd.Series):
                tm.assert_series_equal(correct, test)
            if isinstance(correct, pd.DataFrame):
                tm.assert_frame_equal(correct, test)

        assert_op(s, us, lambda s: s.pct_change())
        assert_op(s, us, lambda s: s + 19)
        assert_op(s, us, lambda s: s / 19)
        assert_op(s, us, lambda s: np.log(s))
        assert_op(s, us, lambda s: np.log(s))
        assert_op(s, us, lambda s: np.diff(s))

        bools = us > 5
        tvals = np.repeat(1, len(us))
        fvals = np.repeat(0, len(us))
        wh = np.where(bools, tvals, fvals)
        assert wh.pobj is not None
        assert wh.dtype == int
        tm.assert_series_equal(wh, bools.astype(int))

    def test_us_view(self):
        s = pd.Series(range(0, 10), index=range(10, 20))
        us = s.view(UserSeries)
        tm.assert_series_equal(s, us)

    def test_datetime_us_view(self):
        data = range(0, 10)
        ind = pd.date_range(start="1/1/2000", freq="D", periods=len(data))
        s = pd.Series(data, index=ind)
        us = s.view(UserSeries)
        tm.assert_series_equal(s, us)
        us.view(UserSeries)

    def test_subframe(self):
        """
        Assert that subclass of UserFrame/UserSeries work
        """
        class SubFrame(UserFrame):
            def resample(self):
                return 'test'

            def irow(self, row, parent=False):
                if parent:
                    return super(SubFrame, self).irow(row)
                return 'irow'

        df = pd.DataFrame(np.random.randn(10, 5))
        sub = SubFrame(df)

        # test overriden method
        assert sub.resample() == 'test'
        assert sub.irow(3) == 'irow'
        # this one calls super
        assert np.all(sub.irow(0, parent=True) == df.ix[0])

    def test_supermeta(self):
        """
        Test that supermeta metaclass acts like a super parent
        to both UserSeries and UserFrame
        """
        class CommonBase(composition.PandasSuperMeta):
            """
            Test common base
            """
            _bob = object()

            @property
            def bob(self):
                return self._bob

        class CommonSeries(with_metaclass(CommonBase, UserSeries)):
            pass

        class CommonFrame(with_metaclass(CommonBase, UserFrame)):
            pass

        bob = CommonBase._bob

        s = CommonSeries(range(10))
        assert s.ix[3] == 3
        tm.assert_almost_equal(s, range(10))
        assert s.bob is bob
        s._bob = 123
        assert s.bob == 123

        df = tm.makeDataFrame()
        fr = CommonFrame(df)
        tm.assert_frame_equal(fr, df)
        assert fr.bob is bob
        assert fr.tail().bob is bob

    def test___init__(self):
        """
        Test that supermeta metaclass acts like a super parent
        to both UserSeries and UserFrame
        """
        class InitSeries(UserSeries):
            def __init__(self, *args, **kwargs):
                # required
                bob = kwargs.pop('bob')
                self.bob = bob
                super(InitSeries, self).__init__(*args, **kwargs)

        class InitFrame(UserFrame):
            def __init__(self, *args, **kwargs):
                # required
                bob = kwargs.pop('bob')
                self.bob = bob
                super(InitFrame, self).__init__(*args, **kwargs)

        s = InitSeries(range(10), name='hello', bob=123)
        assert s.bob == 123

        df = tm.makeDataFrame()
        fr = InitFrame(df, bob='woot')
        assert fr.bob == 'woot'

    def test_subscriptable_callables(self):
        """
        Test that Indexers still work.
        """
        s = pd.Series(range(1, 10), index=range(11, 20))
        us = UserSeries(s)
        assert us.iloc[0] == 1
        assert us.iloc[5] == 6
        assert us.loc[11] == 1
        assert us.loc[19] == 9

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],exit=False)
