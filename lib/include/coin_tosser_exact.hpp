#pragma once

#ifndef LIB_INCLUDE_COIN_TOSSER_EXACT_HPP_
#define LIB_INCLUDE_COIN_TOSSER_EXACT_HPP_

#include "tsetlini_types.hpp"

#include <cmath>
#include <cstring>


namespace Tsetlini
{


struct CoinTosserExact
{
    real_type m_s_inv;
    unsigned int m_hits;
    aligned_vector_char m_cache1;
    aligned_vector_char m_cache2;

    inline
    CoinTosserExact();
    inline
    CoinTosserExact(real_type s_inv, unsigned int size);

    template<typename PRNG>
    void populate(real_type const s_inv, PRNG & prng);

    template<typename PRNG>
    char const * tosses1(PRNG & prng);

    template<typename PRNG>
    char const * tosses2(PRNG & prng);

    inline
    void fill(char val);

    inline
    unsigned int hits() const;

private:
    template<typename PRNG>
    char const * tosses_(aligned_vector_char & cache, PRNG & prng);
};

CoinTosserExact::CoinTosserExact()
    : m_s_inv(1)
    , m_hits(0)
{

}

CoinTosserExact::CoinTosserExact(real_type s_inv, unsigned int size)
    : m_s_inv(s_inv)
    , m_hits(std::round(size * s_inv))
    , m_cache1(size)
    , m_cache2(size)
{

}


template<typename PRNG>
void CoinTosserExact::populate(real_type const s, PRNG & prng)
{
}


template<typename PRNG>
char const * CoinTosserExact::tosses1(PRNG & prng)
{
    return tosses_(m_cache1, prng);
}


template<typename PRNG>
char const * CoinTosserExact::tosses2(PRNG & prng)
{
    return tosses_(m_cache2, prng);
}


void CoinTosserExact::fill(char val)
{
    std::fill(m_cache1.begin(), m_cache1.end(), val);
    std::fill(m_cache2.begin(), m_cache2.end(), val);
}


unsigned int CoinTosserExact::hits() const
{
    return m_hits;
}


template<typename PRNG>
char const * CoinTosserExact::tosses_(aligned_vector_char & cache, PRNG & prng)
{
    auto const vsz = cache.size();

    //memset(m_cache.data(), 0, vsz * sizeof (decltype (m_cache)::value_type));
    std::fill(cache.begin(), cache.end(), 0);

    for (auto it = 0u; it < m_hits; /* nop */)
    {
        auto const ix = prng() % vsz;

        it += (cache[ix] == 0);
        cache[ix] = 1;
    }

    return cache.data();
}


}  // namespace Tsetlini


#endif /* LIB_INCLUDE_COIN_TOSSER_EXACT_HPP_ */
