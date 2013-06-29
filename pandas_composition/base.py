from operator import attrgetter
import collections

import pandas as pd

def _is_user_class(obj):
    """ Check whether the obj is a UserFrame/UserSeries """
    type_dict = type(obj).__dict__
    is_user_class = '_pandas_type' in type_dict
    return is_user_class

def _wrap_callable(self, name):
    # delay delegation
    def _wrapped(*args, **kwargs):
        return self._delegate(name, *args, **kwargs)
    return _wrapped

# used by _wrap
# Done this way to allow other module to extend how to handle
# different types. Specifically for trtools.monkey.AttrNameSpace
WRAP_HANDLERS = []
WRAP_HANDLERS.append((collections.Callable, _wrap_callable))

class UserPandasObject(object):
    """
        Base methods of a quasi pandas subclass. 

        The general idea is that all methods from this class will
        wrap the output into the same class and transfer metadata
    """
    def __init__(self, *args, **kwargs):
        # do not call super. Superfluous since we have the .df
        pass

    def _get(self, name):
        """ Get base attribute. Not pandas object """
        return object.__getattribute__(self, name)
    
    def __getattribute__(self, name):
        """
            #NOTE The reason we use __getattribute__ is that we're
            subclassing pd.DataFrame. That means that our SubClass instance
            will have DataFrame methods that will be called on itself and 
            *not* the self.pobj. 

            This is confusing but in essense, UserFrame's self is an empty DataFrame.
            So calling its methods would operate on an empty DataFrame. We want
            to call the methods on pobj, which is where the data lives. 

            We will subclass the DataFrame to trick internal pandas machinery
            into thinking this class quacks like a duck.
        """
        # special attribute that need to go straight to this obj
        if name in ['pget', 'pobj', '_delegate', '_wrap', '_get', 
                    '__class__', '__array_finalize__', 'view', '__tr_getattr__']:
            return object.__getattribute__(self, name)

        try:
            return self.__tr_getattr__(name)
        except AttributeError:
            pass

        # Run through mro and use overridden values.
        mro = type(self).__mro__
        for kls in mro:
            # stop after pandas-composition class and before pandas classes
            if kls in [pd.DataFrame, pd.Series, pd.TimeSeries, pd.Panel]:
                break
            type_dict = kls.__dict__
            if name in type_dict:
                return object.__getattribute__(self, name) 

        if hasattr(self.pobj, name):
            return self._wrap(name) 
        
        return object.__getattribute__(self, name) 

    def __setattr__(self, name, value):
        if name in self._get('__dict__'):
            return object.__setattr__(self, name, value)
        if hasattr(self.pobj, name):
            return object.__setattr__(self.pobj, name, value)
        return object.__setattr__(self, name, value)

    def __getattr__(self, name):
        # unset the inherited logic here. 
        raise AttributeError(name)

    def __tr_getattr__(self, name):
        """
            Use this function to override getattr for subclasses

            This is necessary since we're subclassing pd.DataFrame as
            a hack. We can't use getattr since it'll return DataFrame attrs
            which we only use through the _wrap/delegate
        """
        raise AttributeError(name)

    def pget(self, name):
        """
            Shortcut to grab from pandas object
            Really just here to override on custom classes.
        """
        getter = attrgetter(name)
        attr = getter(self.pobj)
        return attr
    
    def _wrap(self, name):
        """
        Parameters
        ----------
        name : string
            name of attr found on self.pobj
        
        If the attr is callable, return a closure that calls the original
        method and wraps it's output..

        Otherwise immediately wrap it's output
        """
        attr = self.pget(name)
        for cls, handler in WRAP_HANDLERS:
            if isinstance(attr, cls):
                return handler(self, name)

        # immediately delegate to self.pboj
        return self._delegate(name)
        
    def _delegate(self, name, *args, **kwargs):
        """
        Parameters
        ----------
        name : string
            name of attr found on self.pobj
        *args, **kwargs 
            optional args that are passed along to method call

        Grab attr of self.pobj. If callable, call immeidately. 
        Then we box the output into the original type. 

        This is so things like

        >>> res = subclass_df.tail(10)
        >>> assert type(res) == type(subclass_df)
        
        are True. This is to address the big annoyance where you lose
        the class type when calling methods or attrs.

        Note:
            This has the side affect that if you monkey patch a method or property onto
            Series/DataFrame, this will autobox the results into the original class. 
            This is intended
        """
        attr = self.pget(name)
        res = attr
        if callable(attr):
            res = attr(*args, **kwargs) 

        # should just add pandas_types so UserSeries can have two panda types
        if isinstance(res, type(self)._pandas_type) and  \
           type(res) in [pd.DataFrame, pd.Series, pd.TimeSeries]:
            res = type(self)(res)
            # transfer metadata
            d = self._get('__dict__')
            new_dict = res._get('__dict__')
            for k in d.keys():
                # skip df
                if k == 'pobj':
                    continue
                new_dict[k] = d[k]
        return res
