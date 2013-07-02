import collections
import inspect

from pandas_composition.base import UserPandasObject

class PandasMeta(type):
    def __new__(cls, name, bases, dct):
        new_attrs = dct
        # test for UserSeries/UserFrame by looking for _pandas_type
        if '_pandas_type' in dct:
            pandas_cls = dct['_pandas_type']
            new_attrs = get_methods(pandas_cls)
            new_attrs.update(dct)
            _init_args = init_args(pandas_cls)
            new_attrs['_init_args'] = _init_args
        else: # should be subclass of UserFrame/UserSeries
            pass

        return super(PandasMeta, cls).__new__(cls, name, bases, new_attrs)

class PandasSuperMeta(PandasMeta):
    """
    Currently, there's not a way to have a superclass that 
    both UserSeries and UserFrame inherit from. 

    So to share common methods and members, we this metaclass. 

    Define members and methods onto this class and it will move them
    to the class definition. 

    Note: currently doesn't support magic methods, ignores all '__' vars

    ```python
    class CommonBase(composition.PandasSuperMeta):
        _bob = 123

        @property
        def bob(self):
            return self._bob

    class CommonSeries(UserSeries):
        __metaclass__ = CommonBase

    class CommonFrame(UserFrame):
        __metaclass__ = CommonBase

    s = CommonSeries()
    fr = CommonFrame()
    s.bob == fr.bob # true
    ```
    """
    def __new__(meta, name, bases, attrs):
        # move all non double-underscore attrs from the 
        # metaclass to the class instance
        for k, attr in meta.__dict__.items():
            if k.startswith('__'):
                continue
            attrs[k] = attr
        klass = super(PandasSuperMeta, meta).__new__(meta, name, bases, attrs)
        return klass

def get_methods(pandas_cls):
    """
        Get a combination of PandasObject methods and wrapped DataFrame/Series magic
        methods to use in MetaClass
    """
    ignore_list = ['__class__', '__metaclass__']
    methods = {}
    user_methods = [(name, meth) for name, meth in UserPandasObject.__dict__.iteritems() \
                     if isinstance(meth, (collections.Callable, property)) and name not in ignore_list]

    for name, meth in user_methods:
        methods[name] = meth

    # Wrap the magic_methods which won't be called via __getattribute__
    # Things like __add__ won't run through __getattribute__. We grab them
    # and wrap below
    magic_methods = [(name, meth) for name, meth in pandas_cls.__dict__.iteritems() \
                     if name.startswith('_') and isinstance(meth, collections.Callable) \
                    and name not in ignore_list]

    for name, meth in magic_methods:
        if name not in methods: # don't override PandasObject methods
            methods[name] = _wrap_method(name)

    return methods

def _wrap_method(name):
    def _meth(self, *args, **kwargs):
        return self._delegate(name, *args, **kwargs)
    return _meth

def init_args(pandas_type):
    init_func = getattr(pandas_type, '__init__')
    argspec = inspect.getargspec(init_func)
    return argspec.args[1:] # skip self
