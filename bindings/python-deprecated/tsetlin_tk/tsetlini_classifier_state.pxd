# -*- coding: utf-8 -*-
# distutils: language = c++


cdef extern from "tsetlini_state.hpp" namespace "Tsetlini":
    cdef cppclass ClassifierState
