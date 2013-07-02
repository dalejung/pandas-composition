import pandas as pd
import numpy as np
from numpy import ndarray

from pandas_composition.metaclass import PandasMeta

class UserSeries(pd.Series):
    _pandas_type = pd.Series
    pobj = None
    __metaclass__ = PandasMeta
    def __new__(cls, *args, **kwargs):
        # since i am not calling npndarray.__new__, UserSeries.__array_finalize__ 
        # does not get called.
        # only pass the kwargs that pandas want
        panda_kwargs = {k:v for k, v in kwargs.items() if k in cls._init_args}
        pobj = cls._pandas_type(*args, **panda_kwargs)
        instance = pobj.view(cls)
        return instance

    def __array_finalize__(self, obj):
        if isinstance(obj, UserSeries):
            # self.values will be correct, but we don't have the index
            # TODO go over this logic again. it works but uh
            # not too happy about it
            object.__setattr__(self, '_index', obj._index)
            self.pobj = self.view(pd.Series)
            return

        if isinstance(obj, pd.Series):
            self.pobj = obj
            return

        if isinstance(obj, np.ndarray):
            obj = pd.Series(obj)
            self.pobj = obj
            return

    # needed to trigger pickle to use UserSeries pickling methods
    __reduce_ex__ = object.__reduce_ex__

    def __reduce__(self):
        """ essentially wrap around pd.Series.__reduce__ and add out meta """
        object_state = list(super(UserSeries, self).__reduce__())
        meta = self._get('__dict__').copy()
        pobj = meta.pop('pobj') # remove pobj
        object_state[2] = (object_state[2], meta)
        return tuple(object_state)

    def __setstate__(self, state):
        """ Call normal pd.Series stuff and update with meta  """
        state, meta = state
        # hack to get around setstate error
        self._init_arg_check = False
        try:
            super(UserSeries, self).__setstate__(state)
        except:
            raise
        finally:
            self._init_arg_check = True
        self._get('__dict__').update(meta)

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
