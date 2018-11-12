#pragma once

template<int A, typename T>
inline
T * assume_aligned(T * p)
{
    return (T *)__builtin_assume_aligned(p, A);
}
