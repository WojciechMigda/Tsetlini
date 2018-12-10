# -*- coding: utf-8 -*-

import pytest
from tsetlin_tk import TsetlinMachineClassifier


def test_classifier_can_be_created():
    clf = TsetlinMachineClassifier()


def test_classifier_can_be_created_with_named_params():
    params = dict(
        s=7.5,
        number_of_states=256,
        threshold=27,
        number_of_pos_neg_clauses_per_label=9,
        boost_true_positive_feedback=1,
        n_jobs=2,
        verbose=True,
        random_state=42
    )
    clf = TsetlinMachineClassifier(**params)


def test_classifier_throws_when_constructed_with_unknown_param():
    params = dict(
        this_should_throw=True
    )

    with pytest.raises(TypeError):
        clf = TsetlinMachineClassifier(**params)


def test_classifier_passes_check_estimator():
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(TsetlinMachineClassifier)
