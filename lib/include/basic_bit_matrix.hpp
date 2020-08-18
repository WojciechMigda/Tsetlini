#pragma once

#ifndef LIB_INCLUDE_BASIC_BIT_MATRIX_HPP_
#define LIB_INCLUDE_BASIC_BIT_MATRIX_HPP_

#include "aligned_allocator.hpp"
#include "is_power_of_two.hpp"

#include <vector>
#include <limits>
#include <algorithm>
#include <utility>

#include <limits.h>


template<typename BlockType, unsigned int Alignment = 64>
struct basic_bit_matrix
{
    using block_type = BlockType;
    using size_type = decltype (sizeof (0));
    using uint = unsigned int;
    using aligned_vector = std::vector<block_type, aligned_allocator<block_type, Alignment>>;

    static constexpr uint block_size = sizeof (block_type);
    static constexpr uint block_bits = std::numeric_limits<block_type>::digits;
    static constexpr uint alignment = std::max<uint>(Alignment, alignof (block_type));

    static_assert(is_power_of_two(alignment), "Alignment must be a power of 2");


    struct bit_view
    {
        using block_type = BlockType;
        using size_type = decltype (sizeof (0));
        using uint = unsigned int;

        static constexpr uint block_bits = std::numeric_limits<block_type>::digits;

        inline
        bit_view(uint nbits, block_type * data)
            : m_nbits(nbits)
            , m_p(data)
        {
        }

        inline
        size_type content_blocks() const
        {
            return (m_nbits + block_bits - 1) / block_bits;
        }

        inline
        size_type tail_bits() const
        {
            return m_nbits % block_bits;
        }

        inline
        void set(size_type pos)
        {
            m_p[pos / block_bits] |= set_mask(pos);
        }

        inline
        void clear(size_type pos)
        {
            m_p[pos / block_bits] &= clear_mask(pos);
        }

        inline
        void flip(size_type pos)
        {
            m_p[pos / block_bits] ^= set_mask(pos);
        }

        inline
        uint test(size_type pos) const
        {
            return !!(m_p[pos / block_bits] & set_mask(pos));
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
        uint operator[](size_type pos) const
        {
            return test(pos);
        }

        inline
        block_type * data()
        {
            return m_p;
        }

        uint m_nbits;
        block_type * m_p;
    };


    inline
    basic_bit_matrix()
        : m_nrows(0)
        , m_ncols(0)
        , m_row_blocks(0)
        , m_v()
    {
    }

    inline
    basic_bit_matrix(uint const nrows, uint const ncols)
        : m_nrows(nrows)
        , m_ncols(ncols)
        , m_row_blocks((m_ncols + alignment * CHAR_BIT - 1) / (alignment * CHAR_BIT) * (alignment / block_size))
        , m_v(row_blocks() * nrows)
    {
    }

    inline
    basic_bit_matrix(basic_bit_matrix && rhs)
        : m_nrows(rhs.m_nrows)
        , m_ncols(rhs.m_ncols)
        , m_row_blocks(rhs.m_row_blocks)
        , m_v(std::move(rhs.m_v))
    {
        rhs.m_nrows = 0;
        rhs.m_ncols = 0;
    }

    inline
    basic_bit_matrix(basic_bit_matrix const & rhs)
        : m_nrows(rhs.m_nrows)
        , m_ncols(rhs.m_ncols)
        , m_row_blocks(rhs.m_row_blocks)
        , m_v(rhs.m_v)
    {
    }

    inline
    basic_bit_matrix & operator=(basic_bit_matrix && rhs)
    {
        this->m_nrows = rhs.m_nrows;
        rhs.m_nrows = 0;

        this->m_ncols = rhs.m_ncols;
        rhs.m_ncols = 0;

        this->m_row_blocks = rhs.m_row_blocks;
        rhs.m_row_blocks = 0;

        this->m_v = std::move(rhs.m_v);

        return *this;
    }


    inline
    basic_bit_matrix & operator=(basic_bit_matrix const & rhs)
    {
        this->m_nrows = rhs.m_nrows;
        this->m_ncols = rhs.m_ncols;
        this->m_row_blocks = rhs.m_row_blocks;
        this->m_v = rhs.m_v;

        return *this;
    }


    inline
    uint rows() const
    {
        return m_nrows;
    }

    inline
    uint cols() const
    {
        return m_ncols;
    }

    inline
    std::pair<uint, uint> shape() const
    {
        return {m_nrows, m_ncols};
    }

    inline
    uint row_blocks() const
    {
        return m_row_blocks;
    }

    inline
    size_type row_offset(size_type nr) const
    {
        return nr * row_blocks();
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
    uint test(size_type nr, size_type nc) const
    {
        auto const data = &m_v.data()[row_offset(nr)];

        auto const rv = !!(data[nc / block_bits] & set_mask(nc));
        return rv;
    }

    inline
    void clear(size_type nr, size_type nc)
    {
        auto const data = &m_v.data()[row_offset(nr)];

        data[nc / block_bits] &= clear_mask(nc);
    }

    inline
    void set(size_type nr, size_type nc)
    {
        auto const data = &m_v.data()[row_offset(nr)];

        data[nc / block_bits] |= set_mask(nc);
    }

    inline
    void flip(size_type nr, size_type nc)
    {
        auto const data = &m_v.data()[row_offset(nr)];

        data[nc / block_bits] ^= set_mask(nc);
    }

    inline
    uint operator[](std::pair<size_type, size_type> pos) const
    {
        return test(pos.first, pos.second);
    }

    inline
    block_type const * row_data(size_type nr) const
    {
        auto data_p = &m_v.data()[row_offset(nr)];
        return data_p;
    }

    inline
    bit_view row(size_type nr)// const
    {
        auto data = &m_v.data()[row_offset(nr)];
        return bit_view(m_ncols, data);
    }

    inline
    bit_view operator[](size_type nr) //const
    {
        return row(nr);
    }

    inline
    block_type const * data() const
    {
        return m_v.data();
    }


    uint m_nrows;
    uint m_ncols;
    uint m_row_blocks;
    aligned_vector m_v;
};

template<typename BlockType, unsigned int Alignment>
inline
bool operator==(basic_bit_matrix<BlockType, Alignment> const & lhs, basic_bit_matrix<BlockType, Alignment> const & rhs)
{
    return
        (lhs.m_nrows == rhs.m_nrows) and
        (lhs.m_ncols == rhs.m_ncols) and
        // no need to compare m_row_blocks
        (lhs.m_v == rhs.m_v);
}

#endif /* LIB_INCLUDE_BASIC_BIT_MATRIX_HPP_ */
