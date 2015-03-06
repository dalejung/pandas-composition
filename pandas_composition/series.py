import pandas as pd
import numpy as np
from numpy import ndarray

from six import with_metaclass

from pandas_composition.metaclass import PandasMeta

class UserSeries(with_metaclass(PandasMeta, pd.Series)):
    _pandas_type = pd.Series
    pobj = None
    def __new__(cls, *args, **kwargs):
        # since i am not calling npndarray.__new__, UserSeries.__array_finalize__ 
        # does not get called.
        # only pass the kwargs that pandas want
        panda_kwargs = {k:v for k, v in kwargs.items() if k in cls._init_args}
        pobj = cls._pandas_type(*args, **panda_kwargs)

        instance = object.__new__(cls)
        instance.pobj = pobj
        return instance

    # needed to trigger pickle to use UserSeries pickling methods
    __reduce_ex__ = object.__reduce_ex__

    def __getstate__(self):
        """ essentially wrap around pd.Series.__reduce__ and add out meta """
        data = {}
        meta = self._get('__dict__').copy()
        pobj = meta.pop('pobj') # remove pobj
        data['pobj'] = pobj
        data['meta'] = meta
        data['version'] = 1
        return data

    def __setstate__(self, state):
        """ Call normal pd.Series stuff and update with meta  """
        self.pobj = state['pobj']
        self._get('__dict__').update(state['meta'])

# IPYTHON
def install_ipython_completers():  # pragma: no cover
    # add the instance variable added within __init__
    from IPython.utils.generics import complete_object
    from pandas.util import py3compat
    import itertools

    @complete_object.when_type(UserSeries)
    def complete_user_series(obj, prev_completions):
        dicts = [obj._get('__dict__'), obj.__class__.__dict__]
        # add ability to define completers
        if hasattr(obj, '__completers__'):
            dicts.append(getattr(obj, '__completers__'))
        labels = itertools.chain(*dicts)
        return [c for c in labels
                    if isinstance(c, basestring) and py3compat.isidentifier(c)]                                          
# Importing IPython brings in about 200 modules, so we want to avoid it unless
# we're in IPython (when those modules are loaded anyway).
import sys
if "IPython" in sys.modules:  # pragma: no cover
    try: 
        install_ipython_completers()
    except Exception:
        pass 

us = UserSeries(range(10))
