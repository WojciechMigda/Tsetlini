#!/usr/bin/python3


import numpy as np

from sklearn.datasets import load_digits

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedKFold

import numpy as np


def _as_bits(x, nbits):
    fmt = '{0:0' + str(nbits) + 'b}'
    return np.array([int(c) for c in fmt.format(x)][-nbits:])


def _unpack_bits(a, nbits):
    if len(a.shape) > 2:
        raise ValueError("_unpack_bits: input array cannot have more than 2 dimensions, got {}".format(len(a.shape)))

    a = np.clip(a, 0, 2 ** nbits - 1)
    if nbits == 8:
        a_ = np.empty_like(a, dtype=np.uint8)
        np.rint(a, out=a_, casting='unsafe')
        rv = np.unpackbits(a_, axis=1)
        return rv
    else:
        a_ = np.empty_like(a, dtype=np.uint64)
        np.rint(a, out=a_, casting='unsafe')
        F = np.frompyfunc(_as_bits, 2, 1)
        rv = np.stack(F(a_.ravel(), nbits)).reshape(a.shape[0], -1)
        return rv


class Preprocessor(Pipeline):
    def __init__(self, nbits):
        nbits = int(nbits)
        if not (1 <= nbits <= 64):
            raise ValueError("Preprocessor: nbits is out of a valid range, {} vs [1, 64]".format(nbits))
        self.nbits = nbits
        super(type(self), self).__init__(steps=[
            ('scaler', MinMaxScaler(feature_range=(0, 2 ** nbits - 1))),
            ('unpacker', FunctionTransformer(_unpack_bits, validate=False, kw_args={'nbits': nbits})),
        ])
    pass


def main():

    digits = load_digits()
    X, y = digits.data, digits.target

    pre = Preprocessor(nbits=3)
    X = pre.fit_transform(X)

    kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

    for ix, (_, test) in enumerate(kf.split(X, y)):
        X_test, y_test = X[test], y[test]
        df = np.hstack([y_test[:, None], X_test])
        np.savetxt('digits_{}.txt'.format(ix + 1), df, fmt='%u')

    pass


if __name__ == '__main__':
    main()
