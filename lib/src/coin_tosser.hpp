#pragma once

#ifndef LIB_INCLUDE_COIN_TOSSER_HPP_
#define LIB_INCLUDE_COIN_TOSSER_HPP_

#include "tsetlini_types.hpp"

#include <cmath>
#include <cstring>


namespace Tsetlini
{


struct CoinTosser
{
    unsigned int m_base_size;
    aligned_vector_char m_cache;

    inline
    CoinTosser();
    inline
    CoinTosser(unsigned int base_size, unsigned int extra_size);

    template<typename PRNG>
    void populate(real_type const s, PRNG & prng);

    template<typename PRNG>
    char const * tosses(PRNG & prng);

private:
    template<typename PRNG>
    void mutate(PRNG & prng);
};

CoinTosser::CoinTosser()
    : m_base_size(0)
{

}

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


////////////////////////////////////////////////////////////////////////////////


struct CoinTosserBitwise
{
    unsigned int m_base_size;
    bit_vector_uint64 m_cache;

    static constexpr auto alignment = decltype(m_cache)::alignment;
    static constexpr auto block_bits = decltype(m_cache)::block_bits;

    inline
    CoinTosserBitwise();
    inline
    CoinTosserBitwise(unsigned int base_size, unsigned int extra_size);

    template<typename PRNG>
    void populate(real_type const s, PRNG & prng);

    template<typename PRNG>
    bit_vector_uint64::block_type const * tosses(PRNG & prng);

private:
    template<typename PRNG>
    void mutate(PRNG & prng);

    /*
     * ceil integer v to the nearest multiple of A
     */
    inline static constexpr
    unsigned int ceil2(unsigned int const v, unsigned int const A);
};


CoinTosserBitwise::CoinTosserBitwise()
    : m_base_size(0)
    , m_cache(0)
{

}

CoinTosserBitwise::CoinTosserBitwise(unsigned int base_size, unsigned int extra_size)
    // round up arguments to multiples of block bits
    : m_base_size(ceil2(base_size, block_bits))
    , m_cache(ceil2(base_size + extra_size, block_bits))
{

}


template<typename PRNG>
void CoinTosserBitwise::populate(real_type const s, PRNG & prng)
{
    auto const bit_sz = m_cache.size();
    unsigned int ones_count = std::round(bit_sz / s);

    std::fill(m_cache.m_vector.begin(), m_cache.m_vector.end(), 0);

    for (auto it = 0u; it < ones_count; /* nop */)
    {
        auto const ix = prng() % bit_sz;

        it += (m_cache[ix] == 0);
        m_cache.set(ix);
    }
}


template<typename PRNG>
void CoinTosserBitwise::mutate(PRNG & prng)
{
    auto const bit_sz = m_cache.size();

    // mutate tosses, swap two randomly selected items
    auto const ix1 = prng() % bit_sz;
    auto const ix2 = prng() % bit_sz;

    auto const t1 = m_cache[ix1];
    auto const t2 = m_cache[ix2];

    if (t1 ^ t2)
    {
        m_cache.flip(ix1);
        m_cache.flip(ix2);
    }
}


constexpr
unsigned int CoinTosserBitwise::ceil2(unsigned int const v, unsigned int const A)
{
    unsigned int const nblx = (v + A - 1) / A;

    return nblx * A;
}


template<typename PRNG>
bit_vector_uint64::block_type const * CoinTosserBitwise::tosses(PRNG & prng)
{
    auto const nblx = m_cache.content_blocks();
    auto const nblx_base = m_base_size / block_bits;

    mutate(prng);

    auto begin_blk = prng() % (nblx - nblx_base);

    return &m_cache.m_vector[begin_blk];
}


}  // namespace Tsetlini


#endif /* LIB_INCLUDE_COIN_TOSSER_HPP_ */
