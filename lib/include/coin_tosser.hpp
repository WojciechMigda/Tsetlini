#pragma once

#ifndef LIB_INCLUDE_NEITHER_COIN_TOSSER_HPP_
#define LIB_INCLUDE_NEITHER_COIN_TOSSER_HPP_

#include "tsetlini_types.hpp"

#include <cmath>
#include <cstring>


namespace Tsetlini
{


struct CoinTosser
{
    unsigned int m_base_size;
    aligned_vector_char m_cache;

    CoinTosser(unsigned int base_size, unsigned int extra_size);

    template<typename PRNG>
    void populate(real_type const s, PRNG & prng);

    template<typename PRNG>
    char const * tosses(PRNG & prng);

    template<typename PRNG>
    void mutate(PRNG & prng);
};


CoinTosser::CoinTosser(unsigned int base_size, unsigned int extra_size)
    : m_base_size(base_size)
    , m_cache(base_size + extra_size)
{

}


template<typename PRNG>
void CoinTosser::populate(real_type const s, PRNG & prng)
{
    auto const vsz = m_cache.size();
    unsigned int one_count = std::round(vsz / s);

    //memset(m_cache.data(), 0, vsz * sizeof (decltype (m_cache)::value_type));
    std::fill(m_cache.begin(), m_cache.end(), 0);

    for (auto it = 0u; it < one_count; /* nop */)
    {
        auto const ix = prng() % vsz;

        it += (m_cache[ix] == 0);
        m_cache[ix] = 1;
    }
}


template<typename PRNG>
void CoinTosser::mutate(PRNG & prng)
{
    auto const vsz = m_cache.size();

    // mutate tosses, swap two randomly selected items
    std::swap(m_cache[prng() % vsz], m_cache[prng() % vsz]);
}

template<typename PRNG>
char const * CoinTosser::tosses(PRNG & prng)
{
    auto const vsz = m_cache.size();

    mutate(prng);

    auto begin_offset = prng() % (vsz - m_base_size);
    auto constexpr A = decltype (m_cache)::allocator_type::alignment;
    auto constexpr mask = ~(A - 1);

    auto aligned_begin_offset = begin_offset & mask;

    return &m_cache[aligned_begin_offset];
}


}  // namespace Tsetlini


#endif /* LIB_INCLUDE_NEITHER_COIN_TOSSER_HPP_ */
