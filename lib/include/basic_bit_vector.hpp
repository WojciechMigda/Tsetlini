#pragma once

#ifndef LIB_INCLUDE_BASIC_BIT_VECTOR_HPP_
#define LIB_INCLUDE_BASIC_BIT_VECTOR_HPP_

#include "aligned_allocator.hpp"
#include "is_power_of_two.hpp"

#include <vector>
#include <cstdint>
#include <utility>
#include <climits>
#include <algorithm>


template<typename BlockType, unsigned int Alignment = 64>
struct basic_bit_vector
{
    using block_type = BlockType;
    using size_type = decltype (sizeof (0));
    using uint = unsigned int;
    using aligned_vector = std::vector<block_type, aligned_allocator<block_type, Alignment>>;

    static constexpr size_type block_bits = std::numeric_limits<block_type>::digits;

    static constexpr uint alignment = std::max<uint>(Alignment, alignof (block_type));

    static_assert(is_power_of_two(alignment), "Alignment must be a power of 2");


    basic_bit_vector()
        : m_bit_length(0)
        , m_vector()
    {
    }

    basic_bit_vector(size_type bitlen, aligned_vector && v) :
        m_bit_length(bitlen),
        m_vector(std::move(v))
    {
        m_bit_length = std::min(m_bit_length, m_vector.size() * block_bits);
    }

    explicit basic_bit_vector(size_type bitlen) :
        m_bit_length(bitlen),
        m_vector((m_bit_length + block_bits - 1) / block_bits)
    {
    }


    inline
    size_type size() const
    {
        return m_bit_length;
    }

    inline
    size_type capacity_blocks() const
    {
        return m_vector.size();
    }

    inline
    size_type content_blocks() const
    {
        return (m_bit_length + block_bits - 1) / block_bits;
    }

    inline
    block_type set_mask(size_type pos) const
    {
        return (block_type)1 << (pos % block_bits);
    }

    inline
    block_type clear_mask(size_type pos) const
    {
        return ~set_mask(pos);
    }

    inline
    block_type xor_mask(size_type pos, bool val) const
    {
        return (block_type)val << (pos % block_bits);
    }

    inline
    uint operator[](size_type pos) const
    {
        auto rv = !!(m_vector[pos / block_bits] & set_mask(pos));
        return rv;
    }

    inline
    void set(size_type pos)
    {
        m_vector[pos / block_bits] |= set_mask(pos);
    }

    inline
    void clear(size_type pos)
    {
        m_vector[pos / block_bits] &= clear_mask(pos);
    }

    inline
    void flip(size_type pos)
    {
        m_vector[pos / block_bits] ^= set_mask(pos);
    }

    inline
    void assign(size_type pos, bool val)
    {
        if (val)
        {
            set(pos);
        }
        else
        {
            clear(pos);
        }
    }


    inline
    void XOR(size_type pos, bool val)
    {
        m_vector[pos / block_bits] ^= xor_mask(pos, val);
    }


    inline
    block_type const * data() const
    {
        return m_vector.data();
    }


    inline
    size_type tail_bits() const
    {
        return m_bit_length % block_bits;
    }

    size_type m_bit_length;
    aligned_vector m_vector;
};

#endif /* LIB_INCLUDE_BASIC_BIT_VECTOR_HPP_ */
