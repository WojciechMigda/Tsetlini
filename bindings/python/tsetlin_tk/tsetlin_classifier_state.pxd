# -*- coding: utf-8 -*-
# distutils: language = c++


cdef extern from "tsetlin_state.hpp" namespace "Tsetlin":
    cdef cppclass ClassifierState
