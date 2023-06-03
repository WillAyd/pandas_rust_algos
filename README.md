These are implementations of the pandas algorithms. You can find a working copy of pandas that uses these here:

https://github.com/WillAyd/pandas/tree/rust-algos

As of now about half of the groupby algorithms are implemented. If you would like to contribute any more to learn rust please do so!

You can also use this library directly. To do so run ``maturin develop --release`` from the project root and it will install the Python package into your environment

```python
import numpy as np
import pandas as pd
import pandas_rust_algos as pra

N = 10_000
np.random.seed(42)
values = np.random.randint(2 ** 8 -2, size=(N,), dtype="uint8")
np.random.seed(555)
indexer = np.random.randint(N-1, size=(N,), dtype="int64")
indexer[2] = -1
indexer[200] = -1
out1 = np.empty((N,), dtype="uint8")
out2 = np.empty((N,), dtype="uint8")

pd._libs.algos.take_1d_bool_bool(values, indexer, out1, 0)

# For now don't support fill-argument; will eventually, just need to
# figure out how to bind the out argument generic type T to the fill
# argument type
pra.take_1d(values, indexer, out2)
(out1 == out2).all()

# Feel free to try out other types
pra.take_1d(values.astype("int16"), indexer, out2.astype("int64"))
```

For median

```python
import numpy as np
import pandas._libs.groupby as libgroupby
import pandas_rust_algos as pra

N = 10_000
ngroups = 50
result1 = np.empty((ngroups, 1), dtype="float64")
result2 = np.empty((ngroups, 1), dtype="float64")
counts = np.zeros((ngroups,), dtype="int64")

np.random.seed(42)
values = np.random.rand(N, 1)
np.random.seed(42)
comp_ids = np.random.randint(ngroups, size=(N,))
min_count = -1
mask = None
result_mask = None

# might not be able to supply mask / result_mask depending on version of pandas
%timeit libgroupby.group_median_float64(result1, counts, values, comp_ids, min_count=min_count)
%timeit pra.group_median_float64(result2, counts, values, comp_ids, min_count, mask, result_mask)

assert (result1 == result2).all()
```


cumprod - this is currently slower than Cython by ~33%

```python
import numpy as np
import pandas._libs.groupby as libgroupby
import pandas_rust_algos as pra

N = 10_000
ngroups = 50
result1 = np.empty((N, 1), dtype="float64")
result2 = np.empty((N, 1), dtype="float64")

np.random.seed(42)
values = np.random.rand(N, 1)
np.random.seed(42)
comp_ids = np.random.randint(ngroups, size=(N,))
min_count = -1
mask = None
result_mask = None
    
# might not be able to supply mask / result_mask depending on version of pandas
%timeit libgroupby.group_cumprod_float64(result1, values, comp_ids, ngroups, False, False)
%timeit pra.group_cumprod(result2, values, comp_ids, ngroups, False, False, mask, result_mask)

assert (result1 == result2).all()
```

cumsum - can reuse above variables; performance of this seems to be slower after making generic - bottleneck may be passing arguments?

```python
%timeit libgroupby.group_cumsum(result1, values, comp_ids, ngroups, False, False)
%timeit pra.group_cumsum(result2, values, comp_ids, ngroups, False, False, mask, result_mask)

assert (result1 == result2).all()
```

group_shift_indexer - performance about the same

```python
result1 = np.empty((N,), dtype="int64")
result2 = np.empty((N,), dtype="int64")

%timeit libgroupby.group_shift_indexer(result1, comp_ids, ngroups, 1)
%timeit pra.group_shift_indexer(result2, comp_ids, ngroups, 1)

assert (result1 == result2).all()
```
