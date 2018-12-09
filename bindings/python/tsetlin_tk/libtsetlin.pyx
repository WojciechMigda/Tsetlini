# distutils: language = c++

import  numpy as np
cimport numpy as np
cimport libtsetlin

cimport either
cimport tsetlin_status_code as Tsetlin

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string


cdef extern from "tsetlin_state.hpp" namespace "Tsetlin":
    cdef cppclass ClassifierState


cdef extern from "tsetlin_classifier_state_maker.hpp" namespace "Tsetlin":
    cdef either.Either[Tsetlin.status_message_t, unique_ptr[ClassifierState]] make_classifier_state_ptr(string json_params)


def classifier_fit(js_params):

    make_classifier_state_ptr(js_params)

    return
