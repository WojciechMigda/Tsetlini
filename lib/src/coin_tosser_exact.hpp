#pragma once

#ifndef LIB_INCLUDE_COIN_TOSSER_EXACT_HPP_
#define LIB_INCLUDE_COIN_TOSSER_EXACT_HPP_

#include "tsetlini_types.hpp"
#include "tsetlini_strong_params.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdint>


namespace Tsetlini
{


struct CoinTosserExact
{
    real_type m_s_inv;
    unsigned int m_hits_floor;
    /*
     * This 32-bit integer threshold will be used to stochastically decide
     * whether m_hits should be set to `m_hits_floor` or `m_hits_floor + 1`.
     * We need this to avoid rounding bias when S_inv is multiplied by
     * number of features.
     */
    std::uint32_t m_hits_ceil_threshold;

    aligned_vector_char m_cache1;
    aligned_vector_char m_cache2;

    inline
    CoinTosserExact();
    inline
    CoinTosserExact(real_type s_inv, unsigned int size);

    template<typename PRNG>
    void populate(specificity_t const s, PRNG & prng);

    template<typename PRNG>
    char const * tosses1(PRNG & prng);

    template<typename PRNG>
    char const * tosses2(PRNG & prng);

    inline
    void fill(char val);

    template<typename PRNG>
    unsigned int estimate_hits(PRNG & prng) const;

private:
    template<typename PRNG>
    char const * tosses_(aligned_vector_char & cache, PRNG & prng);
};

CoinTosserExact::CoinTosserExact()
    : m_s_inv(1)
    , m_hits_floor(0)
    , m_hits_ceil_threshold(0)
{

}

CoinTosserExact::CoinTosserExact(real_type s_inv, unsigned int size)
    : m_s_inv(s_inv)
    , m_hits_floor(std::floor(size * s_inv))
    , m_hits_ceil_threshold(
        std::round(static_cast<double>(s_inv * size - std::floor(s_inv * size)) * std::numeric_limits<std::uint32_t>::max()))
    , m_cache1(size)
    , m_cache2(size)
{

}


template<typename PRNG>
void CoinTosserExact::populate(specificity_t const s, PRNG & prng)
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


template<typename PRNG>
unsigned int CoinTosserExact::estimate_hits(PRNG & prng) const
{
    unsigned int hits = m_hits_floor + (prng() < m_hits_ceil_threshold);
    return hits;
}


template<typename PRNG>
char const * CoinTosserExact::tosses_(aligned_vector_char & cache, PRNG & prng)
{
    static_assert(sizeof (typename PRNG::result_type) == sizeof (m_hits_ceil_threshold));

    auto const vsz = cache.size();

    //memset(m_cache.data(), 0, vsz * sizeof (decltype (m_cache)::value_type));
    std::fill(cache.begin(), cache.end(), 0);

    /*
     * Estimate number of hits.
     * It will be either floor() or ceil() of S_inv * cache.size()
     */
    auto const hits = estimate_hits(prng);

    for (auto it = 0u; it < hits; /* nop */)
    {
        auto const ix = prng() % vsz;

        it += (cache[ix] == 0);
        cache[ix] = 1;
    }

    return cache.data();
}


}  // namespace Tsetlini


#endif /* LIB_INCLUDE_COIN_TOSSER_EXACT_HPP_ */
