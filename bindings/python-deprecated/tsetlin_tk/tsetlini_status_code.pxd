# -*- coding: utf-8 -*-
# distutils: language = c++


from libcpp.pair cimport pair
from libcpp.string cimport string


cdef extern from "tsetlini_status_code.hpp" namespace "Tsetlini":
    cdef enum StatusCode:
        S_OK = 0
        S_BAD_JSON
        S_BAD_LABELS
    ctypedef pair[StatusCode, string] status_message_t
