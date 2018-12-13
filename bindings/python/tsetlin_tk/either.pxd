# -*- coding: utf-8 -*-
# distutils: language = c++

from libcpp cimport bool

cdef extern from "neither/either.hpp" namespace "neither" nogil:
    cdef cppclass Either[L, R]:
        union u:
            L leftValue
            R rightValue
        const bool isLeft

        Either(Either &)

        Either[L, R] rightMap[FR](const FR & fn)
        Either[L, R] leftMap[FL](const FL & fn)

        Either[L, R] rightFlatMap[FFR](FFR)
        Either[L, R] leftFlatMap[FFL](FFL)

        T _join "join"[T]()

        @staticmethod
        Either[L, R] rightOf(R)

        @staticmethod
        Either[L, R] leftOf(L)
