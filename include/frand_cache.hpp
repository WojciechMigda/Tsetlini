#pragma once

#include "tsetlini_types.hpp"
#include "aligned_allocator.hpp"
#include "assume_aligned.hpp"

#include <vector>

namespace Tsetlini
{

template<typename RNG, int alignment=32, typename real_type=float>
struct frand_cache
{
    explicit frand_cache(int sz, seed_type seed) :
        m_pos(0),
        m_fcache(sz),
        m_rng(seed)
    {
    }

    inline
    void refill()
    {
        real_type * fcache_p = assume_aligned<alignment>(m_fcache.data());

        for (auto it = 0; it < m_pos; ++it)
        {
            fcache_p[it] = m_rng.next();
        }
        m_pos = 0;
    }

    inline
    real_type next()
    {
        real_type * fcache_p = assume_aligned<alignment>(m_fcache.data());
        return fcache_p[m_pos++];
    }

    inline
    real_type operator()()
    {
        return next();
    }

    int m_pos;
    aligned_vector_float m_fcache;
    RNG m_rng;
};


} // namespace Tsetlini
