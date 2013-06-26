import pandas as pd

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
        for k, v in data.iteritems():
            self._store_meta(k, v)

    _col_classes = {}
    _col_meta = {}

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
        self._store_meta(key, val)
        super(UserFrame, self).__setitem__(key, val)

    def _wrap_series(self, key, val):
        """
        Wrap series data into correct class with metadata
        """
        if key in self._col_classes:
            val = val.view(self._col_classes[key])
            meta = self._col_meta[key]
            val.__dict__.update(meta)
        return val

    def __getitem__(self, key):
        if key in self.columns:
            val = super(UserFrame, self).__getitem__(key)
            return self._wrap_series(key, val)
        raise AttributeError(key)

    def __tr_getattr__(self, key):
        """
        __tr_getattr__ runs before trying to grab from the
        pobj.

        We run the getattr for col name here so that we can box the 
        items with _wrap_series
        """
        return self[key]

# IPYTHON
def install_ipython_completers():  # pragma: no cover
    # add the instance variable added within __init__
    from IPython.utils.generics import complete_object
    from pandas.util import py3compat
    import itertools

    @complete_object.when_type(UserFrame)
    def complete_user_frame(obj, prev_completions):
        dicts = [obj._get('__dict__'), obj.__class__.__dict__]
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
