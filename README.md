pandas-composition
==================

pandas-composition enables you to subclass pd.DataFrame and pd.Series. It will propogate instance variables and class type through most types of operations.

```python
import numpy as np
import pandas as pd

from pandas_composition import UserFrame

class SubFrame(UserFrame):

    def __init__(self, *args, **kwargs):
        # note, always use kwargs.get for new args
        name = kwargs.get('name', None)
        self.name = name

    @property
    def positive(self):
        return self > 0

    def true(self):
        return self[self.astype(bool)]

data = np.random.randn(10, 5)
df = SubFrame(data, columns=list('abcde'), name='howdy')

assert df.name == 'howdy'

# class type is propogated through any @property or method call
assert type(df.positive) is SubFrame
assert type(df.tail(10)) is SubFrame

# instance variables persist as well
assert df.positive.name == 'howdy'
assert df.tail().name == 'howdy'

# even add-hoc variables
df.bob = 'bob'
pos = df.positive
assert pos.bob == 'bob'

# metadata is copied
df.bob = 'bye bye'
assert df.bob != pos.bob

sums = pd.rolling_sum(df, 5)
assert type(sums) is SubFrame
```

## Indicator Example

http://nbviewer.ipython.org/5864433

## Current Issues

* The indexers are currently not wrapped so `ix`, `iloc`, etc will still return a `pd.DataFrame` or `pd.Series` https://github.com/dalejung/pandas-composition/issues/1
