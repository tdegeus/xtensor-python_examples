import mymodule
import numpy as np

A = (np.random.random([100]) * 100).astype(np.uint64)
B = np.array(A, copy=True)
mymodule.foo(B)
assert np.all(A * 2 == B)
