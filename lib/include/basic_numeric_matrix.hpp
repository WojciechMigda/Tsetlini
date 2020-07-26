#pragma once

#ifndef LIB_INCLUDE_BASIC_NUMERIC_MATRIX_HPP_
#define LIB_INCLUDE_BASIC_NUMERIC_MATRIX_HPP_

#include "aligned_allocator.hpp"
#include "is_power_of_two.hpp"

#include <vector>
#include <limits>
#include <algorithm>
#include <utility>

#include <limits.h>


template<typename ValueType, unsigned int Alignment = 64>
struct basic_numeric_matrix
{
    using value_type = ValueType;
    using size_type = decltype (sizeof (0));
    using uint = unsigned int;
    using aligned_vector = std::vector<value_type, aligned_allocator<value_type, Alignment>>;

    static constexpr uint item_size = sizeof (value_type);
    static constexpr uint alignment = std::max<uint>(Alignment, alignof (value_type));

    static_assert(is_power_of_two(alignment), "Alignment must be a power of 2");


    basic_numeric_matrix()
        : m_nrows(0)
        , m_ncols(0)
        , m_v()
    {
    }

    basic_numeric_matrix(uint const nrows, uint const ncols)
        : m_nrows(nrows)
        , m_ncols(ncols)
        , m_v(row_items() * nrows)
    {
    }

    basic_numeric_matrix(basic_numeric_matrix && rhs)
        : m_nrows(rhs.m_nrows)
        , m_ncols(rhs.m_ncols)
        , m_v(std::move(rhs.m_v))
    {
        rhs.m_nrows = 0;
        rhs.m_ncols = 0;
    }

    basic_numeric_matrix(basic_numeric_matrix const & rhs)
        : m_nrows(rhs.m_nrows)
        , m_ncols(rhs.m_ncols)
        , m_v(rhs.m_v)
    {
    }

    basic_numeric_matrix & operator=(basic_numeric_matrix && rhs)
    {
        this->m_nrows = rhs.m_nrows;
        rhs.m_nrows = 0;

        this->m_ncols = rhs.m_ncols;
        rhs.m_ncols = 0;

        this->m_v = std::move(rhs.m_v);

        return *this;
    }


    basic_numeric_matrix & operator=(basic_numeric_matrix const & rhs)
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

    uint row_items() const
    {
        auto const rv = (m_ncols * item_size + alignment - 1) / alignment * (alignment / item_size);

        return rv;
    }

    size_type row_offset(size_type nr) const
    {
        return nr * row_items();
    }

    value_type operator[](std::pair<size_type, size_type> pos) const
    {
        auto const [nr, nc] = pos;
        auto const data = &m_v.data()[row_offset(nr)];

        auto const rv = data[nc];

        return rv;
    }


    value_type const * data() const
    {
        return m_v.data();
    }

    value_type * data()
    {
        return m_v.data();
    }


    value_type const * row_data(size_type nr) const
    {
        return &m_v.data()[row_offset(nr)];
    }

    value_type * row_data(size_type nr)
    {
        return &m_v.data()[row_offset(nr)];
    }

    bool operator==(basic_numeric_matrix const & other) const
    {
        return
            m_nrows == other.m_nrows
            and m_ncols == other.m_ncols
            and m_v == other.m_v;
    }

    uint m_nrows;
    uint m_ncols;
    aligned_vector m_v;
};

#endif /* LIB_INCLUDE_BASIC_NUMERIC_MATRIX_HPP_ */
