# -*- coding: utf-8 -*-

import pytest

import numpy as np
from tsetlin_tk import TsetlinMachineClassifier


def _unpack_bits(a):
    a = np.clip(a, 0, 255)
    return np.unpackbits(a.astype(np.uint8), axis=1)


class XTsetlinMachineClassifier(TsetlinMachineClassifier):
    """Wrapped estimator

    Pipeline doesn't work well with check_estimator
    (https://github.com/scikit-learn/scikit-learn/issues/9768).
    This wrapper provides embedded input X transformation, also
    ensuring that all exceptions at the transformation step are caught
    so that they can be raised when the checks are run by the wrapped
    type.
    """
    def fit(self, X, y, n_iter=500):

        try:
            X = self._fit_transform(X)
        except:
            pass

        super(type(self), self).fit(X, y, n_iter)
        return self


    def partial_fit(self, X, y, classes=None, n_iter=500):

        try:
            X = self._fit_transform(X)
        except:
            pass

        super(type(self), self).partial_fit(X, y, classes=classes, n_iter=n_iter)
        return self


    def predict(self, X):

        try:
            X = self.xformer_.transform(X)
        except:
            pass

        return super(type(self), self).predict(X)


    def predict_proba(self, X):

        try:
            X = self.xformer_.transform(X)
        except:
            pass

        return super(type(self), self).predict_proba(X)


    def _fit_transform(self, X):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import FunctionTransformer

        xformer = Pipeline(steps=[
            ('scaler', MinMaxScaler(feature_range=(0, 255))),
            ('unpacker', FunctionTransformer(_unpack_bits)),
        ])

        try:
            X = xformer.fit_transform(X)
            self.xformer_ = xformer
        except:
            pass

        return X


def test_classifier_passes_check_estimator():
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(XTsetlinMachineClassifier)
