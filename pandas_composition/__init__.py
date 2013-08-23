import pandas as pd
import numpy as np
import inspect

from pandas_composition.series import UserSeries
from pandas_composition.frame import UserFrame, _get_meta
from pandas_composition.metaclass import PandasSuperMeta, PandasMeta

# monkey patch
def view(self, dtype):
    if inspect.isclass(dtype) and issubclass(dtype, pd.Series):
        return dtype(self)

    return self._constructor(self.values.view(dtype), index=self.index, name=self.name)

pd.Series.view = view

def where(condition, *args):
    if len(args) > 0:
        res = np.core.multiarray.where(condition, *args)
    else: 
        res = np.core.multiarray.where(condition)

    if isinstance(condition, UserSeries):
        series = pd.Series(res, index=condition.index, name=condition.name)
        wrapped = condition.copy()
        wrapped.pobj = series
        return wrapped
    return res
np.where = where
