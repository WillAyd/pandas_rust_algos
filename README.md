If you would like to use this library run ``maturin develop --release`` from the project root and it will install the Python package into your environment

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
