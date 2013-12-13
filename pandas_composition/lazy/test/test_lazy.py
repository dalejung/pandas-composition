from unittest import TestCase

import numpy as np
import pandas as pd
from pandas_composition.lazy import LazyFrame
import pandas.util.testing as tm

df = pd.DataFrame(np.random.randn(10000, 5))
lf = LazyFrame(df)

class TestLazy(TestCase):

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)

    def runTest(self):
        pass

    def setUp(self):
        pass

    def test_simple_eval(self):
        """
        Test simple binary oeprations
        """
        correct = df + 1
        test = lf + 1
        assert test.pobj.empty
        # following code should trigger eval
        tm.assert_frame_equal(correct, test)

    def test_simple_op(self):
        """
        Test simple binary oeprations
        """
        correct = df + 1
        test = lf + 1
        tm.assert_frame_equal(correct, test)

        correct = df - 1
        test = lf - 1
        tm.assert_frame_equal(correct, test)

        correct = df * 10.0
        test = lf * 10.0
        tm.assert_frame_equal(correct, test)

        correct = df / 10.0
        test = lf / 10.0
        tm.assert_frame_equal(correct, test)

        correct = df ** 10.0
        test = lf ** 10.0
        tm.assert_frame_equal(correct, test)

    def test_complex(self):
        """
        Test simple binary oeprations
        """
        correct = df ** 10.0 + 1 + df * df
        test = lf ** 10.0 + 1 + lf * lf
        tm.assert_almost_equal(correct.values, test.values)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],exit=False)
