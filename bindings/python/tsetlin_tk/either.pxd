# -*- coding: utf-8 -*-
# distutils: language = c++

from libcpp cimport bool
from cython cimport typeof

cdef extern from "neither/either.hpp" namespace "neither":
    cdef cppclass Either[L, R, LL=*, RR=*]:
        union u:
            L leftValue
            R rightValue
        const bool isLeft
        Either[L, RR] rightMap[F](const F & fn)
        Either[LL, R] leftMap[F](const F & fn)
