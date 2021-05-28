import mymodule
import numpy as np

A = np.arange(10)
B = 5 * np.ones([3, 4], dtype=np.int64)
assert np.all(B == mymodule.foo(A))
assert np.all(B == mymodule.bar(A))
