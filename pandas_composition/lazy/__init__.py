from collections import deque

import numpy as np
import pandas as pd
import numexpr as ne

from pandas_composition import UserFrame

deferred_op = ['__add__', '__sub__', '__mul__', '__div__', '__pow__', 
               '__radd__', '__rsub__', '__rmul__', '__rdiv__', '__rpow__']
eval_op = ['__array__', '.values']
op_trans = {}
op_trans['__add__'] = '+'
op_trans['__radd__'] = '+'
op_trans['__sub__'] = '-'
op_trans['__rsub__'] = '-'
op_trans['__mul__'] = '*'
op_trans['__rmul__'] = '*'
op_trans['__div__'] = '/'
op_trans['__rdiv__'] = '/'
op_trans['__pow__'] = '**'
op_trans['__rpow__'] = '**'

class PandasExpression(object):
    def __init__(self, right, op):
        self.right = right
        self.op = op

def lazy(self):
    return LazyFrame(self)

pd.DataFrame.lazy = lazy

class LazyFrame(UserFrame):
    """
    DataFrame that defers doing operations until it has to. 

    An un-evaled LazyFrame will have an empty DataFrame for 
    `self.pobj`.

    Once a LazyFrame is evaled, it will act like an Ordinary
    DataFrame. Or more precisely, a UserFrame.
    """
    def __init__(self, *args, **kwargs):
        super(LazyFrame, self).__init__(*args, **kwargs)
        self.expressions = deque() 
        self.add_expr(self.pobj, None)

        evaled = kwargs.pop('evaled', False)
        self.evaled = evaled
        if not evaled:
            self.pobj = pd.DataFrame() # un evaluate

    def add_expr(self, right, op):
        expr = PandasExpression(right, op)
        self.expressions.append(expr)

    def eval(self, inplace=False):
        # already evaled
        if not self.pobj.empty:
            return self.pobj

        full, ns = self.gen_ne()
        res = ne.evaluate(full, local_dict=ns)
        if res.ndim == 1:
            pobj = pd.Series(res)
        if res.ndim == 2:
            pobj = pd.DataFrame(res)

        if inplace:
            self.pobj = pobj
            self.evaled = True
        return pobj

    def gen_ne(self):
        """
        Generate the values needed for numexpr
        Essentially a full string expression and a namespace
        """
        sub_count = 1
        ns = {}
        subs = []
        for expr in self.expressions:
            val = expr.right
            op = expr.op
            if np.isscalar(val):
                val = repr(val)
            else:
                placeholder = '_pobj'+str(sub_count)
                sub_count += 1
                assert placeholder not in ns
                ns[placeholder] = val
                val = placeholder

            if op is None:
                subs.append(val)
                continue
            else:
                op = op_trans[op]
                subs.append("{op} {val}".format(op=op, val=val))
        full = self._gen_full_string(subs)
        return full, ns

    def _gen_full_string(self, subs):
        if len(subs) == 1:
            full = subs[0]
            return full

        full = '({left} {right})'.format(left=subs[0], right=subs[1])
        for s in subs[2:]:
            full = '({left} {right})'.format(left=full, right=s)
        return full

    def _delegate(self, name, *args, **kwargs):
        if name in deferred_op:
            return self.defer_op(name, *args, **kwargs)
        # if not deferred or part of pass safe-list
        # we play it safe and eval
        if name not in ['pobj']:
            self.eval(inplace=True)
        return super(LazyFrame, self)._delegate(name, *args, **kwargs)

    def defer_op(self, name, *args, **kwargs):
        stack = LazyFrame()
        stack.expressions = deque(self.expressions)
        stack.add_expr(args[0], name)
        return stack

    def __repr__(self):
        if self.pobj:
            return repr(self.pobj)
        full, ns = self.gen_ne()
        return "LazyFrame: \n{full}".format(full=full)

    def _repr_html_(self):
        if self.pobj:
            return self.pobj._repr_html_()
        return repr(self)
