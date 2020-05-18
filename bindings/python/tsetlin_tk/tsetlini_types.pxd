# -*- coding: utf-8 -*-
# distutils: language = c++


from libcpp.vector cimport vector


cdef extern from "tsetlini_types.hpp" namespace "Tsetlini":
    ctypedef vector[int] aligned_vector_int
    ctypedef vector[char] aligned_vector_char
    ctypedef vector[float] aligned_vector_float

    ctypedef int label_type
    ctypedef vector[label_type] label_vector_type
