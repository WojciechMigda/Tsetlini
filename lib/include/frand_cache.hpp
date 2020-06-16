#pragma once

#include "tsetlini_types.hpp"
#include "aligned_allocator.hpp"
#include "assume_aligned.hpp"

#include <vector>
#include <algorithm>

namespace Tsetlini
{

template<int alignment=32, typename real_type=float>
struct frand_cache
{
    template<typename TFRNG>
    explicit frand_cache(TFRNG & frng, int sz, seed_type seed) :
        m_pos(sz),
        m_fcache(sz)
    {
    }

    frand_cache()
        : m_pos(0)
        , m_fcache(0)
    {
    }

    template<typename TFRNG>
    inline
    void refill(TFRNG & frng)
    {
        real_type * fcache_p = assume_aligned<alignment>(m_fcache.data());

        for (auto it = 0u; it < std::min<unsigned int>(m_pos, m_fcache.size()); ++it)
        {
            fcache_p[it] = frng();
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

    unsigned int m_pos;
    aligned_vector_float m_fcache;
};


} // namespace Tsetlini
