# -*- coding: utf-8 -*-
# distutils: language = c++


from tsetlin_tk.tsetlin_classifier_state cimport ClassifierState

from libcpp.memory cimport unique_ptr


cdef extern from "<utility>" namespace "std" nogil:
    cdef unique_ptr[ClassifierState] move_unique "std::move"(unique_ptr[ClassifierState])
