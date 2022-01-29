#pragma once

#ifndef LIB_INCLUDE_ALIGNED_ARRAY_HPP_
#define LIB_INCLUDE_ALIGNED_ARRAY_HPP_

#include "assume_aligned.hpp"
#include "is_power_of_two.hpp"

#include <cstddef>
#include <algorithm>
#include <type_traits>


template <typename Tp, std::size_t Nelem, std::size_t Alignment>
struct AlignedArray
{
    static_assert(std::alignment_of<Tp>::value <= Alignment, "Alignment cannot be smaller than alignment of Tp");
    static_assert(is_power_of_two(Alignment), "Alignment must be a power of 2");

    using value_type = Tp;
    using value_ptr = value_type *;
    using const_value_ptr = value_type const *;
    using reference = value_type &;
    using const_reference = value_type const &;
    using size_type = std::size_t;

    static constexpr size_type alignment = Alignment;
    static constexpr size_type nelem = Nelem;

    AlignedArray() :
        offset(((reinterpret_cast<size_type>(&a[0]) + Alignment - 1) & ~(Alignment - 1)) - reinterpret_cast<size_type>(&a[0]))
    {
    }

    AlignedArray(AlignedArray const & other) :
        offset(((reinterpret_cast<size_type>(&a[0]) + Alignment - 1) & ~(Alignment - 1)) - reinterpret_cast<size_type>(&a[0]))
    {
        if (this != &other)
        {
            *this = other;
        }
    }

    AlignedArray & operator=(AlignedArray const & other)
    {
        std::copy_n(other.data(), nelem, data());
        return *this;
    }

    constexpr value_ptr data() noexcept
    {
        return assume_aligned<alignment>(reinterpret_cast<value_ptr>(&a[offset]));
    }

    constexpr const_value_ptr data() const noexcept
    {
        return assume_aligned<alignment>(reinterpret_cast<const_value_ptr>(&a[offset]));
    }

    constexpr reference operator[](size_type pos)
    {
        return data()[pos];
    }

    constexpr const_reference operator[](size_type pos) const
    {
        return data()[pos];
    }

    value_ptr begin()
    {
        return data();
    }

    value_ptr end()
    {
        return data() + nelem;
    }

    const_value_ptr cbegin() const
    {
        return data();
    }

    const_value_ptr cend() const
    {
        return data() + nelem;
    }

private:
    size_type offset;
    unsigned char a[Alignment - 1 + nelem * sizeof (value_type)];
};


template <typename Tp, std::size_t Nelem, std::size_t Alignment>
inline
bool operator==(AlignedArray<Tp, Nelem, Alignment> const & lhs, AlignedArray<Tp, Nelem, Alignment> const & rhs)
{
    return std::equal(lhs.data(), lhs.data() + Nelem, rhs.data());
}


template <typename Tp, std::size_t Nelem, std::size_t Alignment>
inline
bool operator!=(AlignedArray<Tp, Nelem, Alignment> const & lhs, AlignedArray<Tp, Nelem, Alignment> const & rhs)
{
    return not (lhs == rhs);
}


#endif /* LIB_INCLUDE_ALIGNED_ARRAY_HPP_ */
