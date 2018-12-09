# -*- coding: utf-8 -*-
# distutils: language = c++


from either cimport Either

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from tsetlin_status_code cimport status_message_t
from tsetlin_classifier_state cimport ClassifierState


cdef extern from "tsetlin_classifier_state_maker.hpp" namespace "Tsetlin" nogil:
    cdef Either[status_message_t, unique_ptr[ClassifierState]] make_classifier_state_ptr(string json_params)
