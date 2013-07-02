import types
import collections

import pandas as pd
import numpy as np

from pandas_composition.metaclass import PandasMeta

def _get_meta(obj):
    # _get grabs from the obj itself and not it's pobj
    getter = getattr(obj, '_get', None)
    meta = {}
    if callable(getter):
        d = getter('__dict__')
        meta.update(d)
    meta.update(getattr(obj, '__dict__', {}))
    meta.pop('_index', None) # don't store index
    meta.pop('pobj', None) # don't store pobj
    return meta

class UserFrame(pd.DataFrame):
    _pandas_type = pd.DataFrame
    pobj = None
    __metaclass__ = PandasMeta
    def __new__(cls, *args, **kwargs):
        # only pass the kwargs that pandas want
        panda_kwargs = {k:v for k, v in kwargs.items() if k in cls._init_args}
        pobj = cls._pandas_type(*args, **panda_kwargs)

        instance = object.__new__(cls)
        instance.pobj = pobj
        return instance

    def __init__(self, data=None, *args, **kwargs):
        if isinstance(data, (dict, pd.DataFrame)):
            self._init_col_meta(data)

    def _init_col_meta(self, data):
        """ 
        Initialize the col meta. This is for times when we create
        a UserFrame with a block of data such as dict, pd.DataFrame.
        """
        for k in data:
            v = data[k]
            self._store_meta(k, v)

    _col_classes_ = None
    @property
    def _col_classes(self):
        if self._col_classes_ is None:
            self._col_classes_ = {}
        return self._col_classes_

    _col_meta_ = None 
    @property
    def _col_meta(self):
        if self._col_meta_ is None:
            self._col_meta_ = {}
        return self._col_meta_

    def _store_meta(self, key, val):
        """
        Store the metadata for a column
        """
        # just do isinstance(pd.Series) check?
        if hasattr(val, '__dict__'):
            d = _get_meta(val).copy()
            self._col_meta[key] = d
            self._col_classes[key] = type(val) 

    def __setitem__(self, key, val):
        # replicate DataFrame behavior and set name to 
        # dict key.
        if hasattr(val, 'name'):
            setattr(val, 'name', key)
        self._store_meta(key, val)
        super(UserFrame, self).__setitem__(key, val)

    def _wrap_series(self, key, val):
        """
        Wrap series data into correct class with metadata
        """
        if key in self._col_classes:
            val = val.view(self._col_classes[key])
            meta = self._col_meta[key]
            if hasattr(val, 'meta'):
                val.meta.update(meta)
            else:
                val.__dict__.update(meta)
        return val

    _default_boxer = None
    _default_boxer_func = None

    @property
    def default_boxer(self):
        """
        Caches and returns a default boxer.
        """
        if self._default_boxer_func is not None:
            return self._default_boxer_func

        boxer = self._default_boxer
        boxer = self._wrap_boxer(boxer)
        return boxer

    def _wrap_boxer(self, boxer):
        """
        Returns the proper callable for the various eligible boxer types. 

        Parameters
        ----------
        boxer : None, np.ndarray subclass, or callable

        None : returns an identity function
        np.ndarray : will call Series.view(boxer)
        Callable : Simple calls the callable
        """
        if boxer is None:
            return lambda x: x
        if isinstance(boxer, types.TypeType) and issubclass(boxer, np.ndarray):
            return lambda val: val.view(boxer)
        if isinstance(boxer, collections.Callable):
            return boxer
        raise Exception("_default_boxer must be a ndarray subclass or a callable")

    def __getitem__(self, key):
        if key in self.columns:
            val = super(UserFrame, self).__getitem__(key)
            # attempt wrap
            val = self._wrap_series(key, val)
            if type(val) in [pd.Series, pd.TimeSeries]:
                # if pandas object, try to wrap default
                val = self.default_boxer(val)
            return val
        raise AttributeError(key)

    def __tr_getattr__(self, key):
        """
        __tr_getattr__ runs before trying to grab from the
        pobj.

        We run the getattr for col name here so that we can box the 
        items with _wrap_series
        """
        # this is explicitly for columns. Make sure to error out quickly
        if key not in self.pobj.columns:
            raise AttributeError(key)
        res = self[key]
        return res
    #  For now just a dummy method to test subclasses overridding superclasses
    def iteritems(self, sentinel=False):
        if sentinel:
            return 10
        return ((k, self[k]) for k in self)

    # needed to trigger pickle to use UserFrame pickling methods
    __reduce_ex__ = object.__reduce_ex__

    def __getstate__(self):
        """
        Expicitly split up pobj and frame_meta.
        """
        data = {}
        fdict = self._get('__dict__').copy()
        pobj = fdict.pop('pobj')
        data['pobj'] = pobj
        data['frame_meta'] = fdict
        data['version'] = 1
        return data

    def __setstate__(self, state):
        version = state['version']
        if version == 1:
            self.pobj = state['pobj']
            self._get('__dict__').update(state['frame_meta'])

# IPYTHON
def install_ipython_completers():  # pragma: no cover
    # add the instance variable added within __init__
    from IPython.utils.generics import complete_object
    from pandas.util import py3compat
    import itertools

    @complete_object.when_type(UserFrame)
    def complete_user_frame(obj, prev_completions):
        dicts = [obj._get('__dict__'), obj.__class__.__dict__]
        # add ability to define completers
        if hasattr(obj, '__completers__'):
            dicts.append(getattr(obj, '__completers__'))
        column_names = obj.columns
        labels = itertools.chain(column_names, *dicts)
        completions = [c for c in labels 
                       if isinstance(c, basestring) and py3compat.isidentifier(c)]
        return completions

# Importing IPython brings in about 200 modules, so we want to avoid it unless
# we're in IPython (when those modules are loaded anyway).
import sys
if "IPython" in sys.modules:  # pragma: no cover
    try: 
        install_ipython_completers()
    except Exception:
        pass 
