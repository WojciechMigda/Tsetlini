# distutils: language = c++

from libcpp cimport bool


cdef extern from "neither/either.hpp" namespace "neither":
    cdef cppclass Either[L, R]:
        union u:
            L leftValue
            R rightValue
        const bool isLeft
