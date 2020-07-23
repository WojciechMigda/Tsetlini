#pragma once

#ifndef LIB_INCLUDE_BASIC_BIT_VECTOR_COMPANION_HPP_
#define LIB_INCLUDE_BASIC_BIT_VECTOR_COMPANION_HPP_


#include "basic_bit_vector.hpp"
#include "aligned_allocator.hpp"

#include <iterator>
#include <climits>
#include <vector>


namespace basic_bit_vectors
{


template<typename BlockType, typename Iter, unsigned int Alignment = 64>
basic_bit_vector<BlockType, Alignment> from_range(Iter begin, Iter end)
{
    using block_type = BlockType;
    using size_type = decltype (sizeof (0));
    using storage_vector_type = std::vector<block_type, aligned_allocator<block_type, Alignment>>;

    constexpr size_type block_bits = std::numeric_limits<block_type>::digits;
    size_type const n_elem = std::distance(begin, end);
    size_type const n_blocks = (n_elem + block_bits - 1) / block_bits;

    basic_bit_vector<BlockType, Alignment> rv(n_elem, storage_vector_type(n_blocks));
    size_type pos = 0;

    for (Iter it = begin; it != end; ++it, ++pos)
    {
        if (*it)
        {
            rv.set(pos);
        }
    }

    return rv;
}


}  // namespace basic_bit_vectors


#endif /* LIB_INCLUDE_BASIC_BIT_VECTOR_COMPANION_HPP_ */
