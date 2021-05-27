import mymodule
import numpy as np

A = (np.random.random([100]) * 100).astype(np.uint64)
assert np.all(np.sort(A) == mymodule.foo(A))
