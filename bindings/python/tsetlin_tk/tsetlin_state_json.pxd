# -*- coding: utf-8 -*-
# distutils: language = c++

from tsetlin_tk.tsetlin_classifier_state cimport ClassifierState

from libcpp.string cimport string


cdef extern from "tsetlin_state_json.hpp" namespace "Tsetlin" nogil:
    cdef string to_json_string(ClassifierState state)
