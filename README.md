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
pra.take_1d(values, indexer, out2, 0)
(out1 == out2).all()

