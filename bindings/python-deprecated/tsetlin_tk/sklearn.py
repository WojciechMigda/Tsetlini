# coding: utf-8
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (check_X_y, check_array, check_is_fitted,
    column_or_1d)
from sklearn.utils.multiclass import unique_labels, check_classification_targets
from sklearn.preprocessing import LabelEncoder

from .base import (
    _validate_params, _classifier_fit, _classifier_partial_fit,
    _classifier_predict, _classifier_predict_proba)


class TsetlinMachineClassifier(BaseEstimator, ClassifierMixin):
    """Tsetlin Machine Multiclass classifier.

    This estimator implements Tsetlin Machine multiclass classifier
    following the example code from https://github.com/cair/TsetlinMachineCython
    with several speed and logic improvements.

    Parameters
    ----------
    number_of_pos_neg_clauses_per_label : int, default=5
        Number of positive / negative clauses per class. E.g. for N classes this
        will lead to the model having
        2 * number_of_pos_neg_clauses_per_label * N clauses.
    number_of_states: int, default=100
        Number of integral states associated with a single Tsetlin automaton.
    s: float, default=2.0
        TODO.
    threshold: int, default=15
        Threshold value for Tsetlin automata.
    boost_true_positive_feedback: int, default=0
        TODO.
    n_jobs: int, default=-1
        The number of CPUs to use for computation.
        ``-1`` means using all processors.
    verbose: bool, default=False
        Flag to disable/enable verbose output
    random_state: int, default=None
        The seed of the pseudo random number generator to use.
        If None, the random number generator is the RandomState
        instance used by `np.random`.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self,
                 number_of_pos_neg_clauses_per_label=5,
                 number_of_states=100,
                 s=2.0,
                 threshold=15,
                 boost_true_positive_feedback=0,
                 counting_type='auto',
                 clause_output_tile_size=16,
                 n_jobs=-1,
                 verbose=False,
                 random_state=None):
        self.number_of_pos_neg_clauses_per_label = number_of_pos_neg_clauses_per_label
        self.number_of_states = number_of_states
        self.s = s
        self.threshold = threshold
        self.boost_true_positive_feedback = boost_true_positive_feedback
        self.counting_type = counting_type
        self.clause_output_tile_size = clause_output_tile_size
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state


    def fit(self, X, y, n_iter=500):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        checked_y = column_or_1d(y, warn=True)
        check_classification_targets(checked_y)

        return self._fit(X, checked_y, classes=checked_y, n_iter=n_iter)


    def partial_fit(self, X, y, classes=None, n_iter=500):
        """Fit using existing state of the classifier for online-learning.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of the training data
        y : numpy array, shape (n_samples,)
            Subset of the target values

        Returns
        -------
        self : returns an instance of self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        checked_y = column_or_1d(y, warn=True)
        check_classification_targets(checked_y)

        if classes is None:
            classes = checked_y

        # if not fitted:
        if not hasattr(self, 'model_'):
            self._fit(X, y, classes=classes, n_iter=n_iter)
        else:
            if X.shape[1] != self.n_features_:
                raise ValueError("Number of features in X and"
                                 " fitted array does not match."
                                 " X: {}, fitted: {}".format(
                                 X.shape[1], self.n_features_))
            self._partial_fit(X, y, classes=classes, n_iter=n_iter)

        return self


    def _fit(self, X, y, classes, n_iter):
        n_iter = int(n_iter)
        if n_iter <= 0:
            raise ValueError("Number of iterations must be a positive"
                             " integer but fit was called with"
                             " n_iter: {}".format(n_iter))

        encoder = LabelEncoder().fit(classes)
        y = encoder.transform(y)

        if len(encoder.classes_) < 2:
            raise ValueError("This estimator needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: {}".format(encoder.classes_[0]))

        self.set_params(**_validate_params(self.get_params()))

        self.model_ = _classifier_fit(
            X, y, self.get_params(), len(encoder.classes_), n_iter)

        self.encoder_ = encoder
        self.classes_ = encoder.classes_
        self.n_features_ = X.shape[1]

        return self


    def _partial_fit(self, X, y, classes, n_iter):
        n_iter = int(n_iter)
        if n_iter <= 0:
            raise ValueError("Number of iterations must be a positive"
                             " integer but fit was called with"
                             " n_iter: {}".format(n_iter))
        y = self.encoder_.transform(y)

        self.model_ = _classifier_partial_fit(
            X, y, self.model_, n_iter)

        return self


    def predict(self, X):
        """A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        X = self._validate_for_predict(X)

        y_hat_raw = _classifier_predict(X, self.model_)

        y_hat = self.encoder_.inverse_transform(y_hat_raw)

        return y_hat


    def predict_proba(self, X):
        X = self._validate_for_predict(X)
        probas = _classifier_predict_proba(X, self.model_, self.threshold)
        return probas


    def _validate_for_predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['model_'])

        # Input validation
        X = check_array(X)

        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] should be {0:d}, not {1:d}.".format(
                self.n_features_, X.shape[1]))
        return X
