#pragma once

#include "aligned_allocator.hpp"
#include "is_power_of_two.hpp"

#include <vector>
#include <limits>
#include <algorithm>
#include <utility>

#include <limits.h>


template<typename BlockType, unsigned int Alignment = 64>
struct bit_matrix
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

        bit_view(uint nbits, block_type * data)
            : m_nbits(nbits)
            , m_p(data)
        {
        }


        void set(size_type pos)
        {
            m_p[pos / block_bits] |= set_mask(pos);
        }

        void clear(size_type pos)
        {
            m_p[pos / block_bits] &= clear_mask(pos);
        }

        void flip(size_type pos)
        {
            m_p[pos / block_bits] ^= set_mask(pos);
        }

        uint test(size_type pos) const
        {
            return !!(m_p[pos / block_bits] & set_mask(pos));
        }


        block_type set_mask(size_type pos) const
        {
            return (block_type)1 << (pos % block_bits);
        }

        block_type clear_mask(size_type pos) const
        {
            return ~set_mask(pos);
        }

        uint operator[](size_type pos) const
        {
            return test(pos);
        }

        uint m_nbits;
        block_type * m_p;
    };


    bit_matrix()
        : m_nrows(0)
        , m_ncols(0)
        , m_v()
    {
    }

    bit_matrix(uint const nrows, uint const ncols)
        : m_nrows(nrows)
        , m_ncols(ncols)
        , m_v(row_blocks() * nrows)
    {
    }

    bit_matrix(bit_matrix && rhs)
        : m_nrows(rhs.m_nrows)
        , m_ncols(rhs.m_ncols)
        , m_v(std::move(rhs.m_v))
    {
        rhs.m_nrows = 0;
        rhs.m_ncols = 0;
    }

    bit_matrix(bit_matrix const & rhs)
        : m_nrows(rhs.m_nrows)
        , m_ncols(rhs.m_ncols)
        , m_v(rhs.m_v)
    {
    }

    bit_matrix & operator=(bit_matrix && rhs)
    {
        this->m_nrows = rhs.m_nrows;
        rhs.m_nrows = 0;

        this->m_ncols = rhs.m_ncols;
        rhs.m_ncols = 0;

        this->m_v = std::move(rhs.m_v);

        return *this;
    }


    bit_matrix & operator=(bit_matrix const & rhs)
    {
        this->m_nrows = rhs.m_nrows;
        this->m_ncols = rhs.m_ncols;
        this->m_v = rhs.m_v;

        return *this;
    }


    uint rows() const
    {
        return m_nrows;
    }

    uint cols() const
    {
        return m_ncols;
    }

    std::pair<uint, uint> shape() const
    {
        return {m_nrows, m_ncols};
    }

    uint row_blocks() const
    {
        auto const rv = (m_ncols + alignment * CHAR_BIT - 1) / (alignment * CHAR_BIT) * (alignment / block_size);
        return rv;
    }

    size_type row_offset(size_type nr) const
    {
        return nr * row_blocks();
    }

    block_type set_mask(size_type pos) const
    {
        return (block_type)1 << (pos % block_bits);
    }

    block_type clear_mask(size_type pos) const
    {
        return ~set_mask(pos);
    }


    uint test(size_type nr, size_type nc) const
    {
        auto const data = &m_v.data()[row_offset(nr)];

        auto const rv = !!(data[nc / block_bits] & set_mask(nc));
        return rv;
    }


    void clear(size_type nr, size_type nc)
    {
        auto const data = &m_v.data()[row_offset(nr)];

        data[nc / block_bits] &= clear_mask(nc);
    }


    void set(size_type nr, size_type nc)
    {
        auto const data = &m_v.data()[row_offset(nr)];

        data[nc / block_bits] |= set_mask(nc);
    }


    void flip(size_type nr, size_type nc)
    {
        auto const data = &m_v.data()[row_offset(nr)];

        data[nc / block_bits] ^= set_mask(nc);
    }


    uint operator[](std::pair<size_type, size_type> pos) const
    {
        return test(pos.first, pos.second);
    }

    bit_view row(size_type nr)// const
    {
        auto data = &m_v.data()[row_offset(nr)];
        return bit_view(m_ncols, data);
    }

    bit_view operator[](size_type nr) //const
    {
        return row(nr);
    }

    block_type const * data() const
    {
        return m_v.data();
    }


    uint m_nrows;
    uint m_ncols;
    aligned_vector m_v;
};
