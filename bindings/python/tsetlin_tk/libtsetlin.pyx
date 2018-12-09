# -*- coding: utf-8 -*-
# distutils: language = c++
#
# cython: wraparound  = False
# cython: boundscheck = False
# cython: cdivision   = True

import  numpy as np
cimport numpy as np
cimport libtsetlin

#from either cimport Either

from libcpp.memory cimport unique_ptr
#from libcpp.string cimport string

from tsetlin_status_code cimport status_message_t
from tsetlin_classifier_state cimport ClassifierState
from tsetlin_classifier_state_maker cimport make_classifier_state_ptr
from move_unique cimport move_unique


cdef unique_ptr[ClassifierState] right_fn(unique_ptr[ClassifierState] && state_p):
    return move_unique(state_p)

cdef status_message_t left_fn(status_message_t && msg):
    return msg


def fit_classifier(js_params):

    make_classifier_state_ptr(js_params)#.leftMap(left_fn).rightMap(right_fn)

    return
