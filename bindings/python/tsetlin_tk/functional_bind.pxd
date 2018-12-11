# -*- coding: utf-8 -*-
# distutils: language = c++

cdef extern from "<functional>" namespace "std" nogil:
    cdef cppclass _bind_expression:
        pass
    _bind_expression bind "std::bind"(...)

cdef extern from "<functional>" namespace "std::placeholders" nogil:
    cdef int _1
    cdef int _2
    cdef int _3
    cdef int _4
    cdef int _5
    cdef int _6
    cdef int _7
    cdef int _8
    cdef int _9
    cdef int _10
