# -*- coding: utf-8 -*-

import numpy as np

from sklearn import datasets
from sklearn.preprocessing import FunctionTransformer

import numpy as np


def _as_bits(x, nbits):
    s = '1' * x + '0' * (nbits - x)
    return np.array([int(c) for c in s])

def _unpack_bits(a, nbits):
    if len(a.shape) > 2:
        raise ValueError("_unpack_bits: input array cannot have more than 2 dimensions, got {}".format(len(a.shape)))

    a = np.clip(a, 0, nbits)
    a_ = np.empty_like(a, dtype=np.uint64)
    np.rint(a, out=a_, casting='unsafe')
    F = np.frompyfunc(_as_bits, 2, 1)
    rv = np.stack(F(a_.ravel(), nbits)).reshape(a.shape[0], -1)
    return rv


california_housing = datasets.fetch_california_housing()
X = california_housing.data
Y = california_housing.target

from sklearn.preprocessing import KBinsDiscretizer

kbd = KBinsDiscretizer(n_bins=11, encode='ordinal', strategy='quantile')

X_transformed = kbd.fit_transform(X).astype(int)

pre = FunctionTransformer(_unpack_bits, validate=False, kw_args={'nbits': 10})
X_transformed = pre.fit_transform(X_transformed)

np.savetxt("CaliforniaHousingData_X.txt", X_transformed, fmt="%d")
np.savetxt("CaliforniaHousingData_Y.txt", Y, fmt="%.3f")
