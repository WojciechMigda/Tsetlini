# -*- coding: utf-8 -*-
# distutils: language = c++

from libcpp cimport bool


cdef extern from "neither/either.hpp" namespace "neither":
    cdef cppclass Either[L, R]:
        union u:
            L leftValue
            R rightValue
        const bool isLeft
        #Either[L, V] rightMap[F, V](const F & fn)
        #Either[V, R] leftMap[F, V](const F & fn)
