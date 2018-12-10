# -*- coding: utf-8 -*-
# distutils: language = c++

from libcpp cimport bool

cdef extern from "neither/either.hpp" namespace "neither":
    cdef cppclass Either[L, R]:
        union u:
            L leftValue
            R rightValue
        const bool isLeft
        Either[L, R] rightMap[FR](const FR & fn)
        Either[L, R] leftMap[FL](const FL & fn)
