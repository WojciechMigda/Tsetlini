#pragma once

#ifndef LIB_INCLUDE_ASSUME_ALIGNED_HPP_
#define LIB_INCLUDE_ASSUME_ALIGNED_HPP_


template<int A, typename T>
inline
T * assume_aligned(T * p)
{
    return (T *)__builtin_assume_aligned(p, A);
}


#endif /* LIB_INCLUDE_ASSUME_ALIGNED__HPP_ */
