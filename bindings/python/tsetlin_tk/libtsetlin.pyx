# -*- coding: utf-8 -*-
# distutils: language = c++
#
# cython: wraparound  = False
# cython: boundscheck = False
# cython: cdivision   = True

import  numpy as np
cimport numpy as np
cimport tsetlin_tk.libtsetlin

#from either cimport Either

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from tsetlin_tk.tsetlin_status_code cimport status_message_t
from tsetlin_tk.tsetlin_classifier_state cimport ClassifierState
from tsetlin_tk.tsetlin_classifier_state_maker cimport make_classifier_state_ptr
from tsetlin_tk.move_unique cimport move_unique


cdef unique_ptr[ClassifierState] classifier_fn_right(unique_ptr[ClassifierState] && state_p):
    return move_unique(state_p)


cdef status_message_t classifier_fn_left(status_message_t && msg):
    #raise ValueError(<bytes>msg.second)
    return msg


def fit_classifier(js_params):

    cdef string jss_params = js_params

    make_classifier_state_ptr(jss_params).leftMap(classifier_fn_left).rightMap(classifier_fn_right)

    return
