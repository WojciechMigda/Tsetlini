#pragma once

#ifndef LIB_INCLUDE_FRAND_CACHE_HPP_
#define LIB_INCLUDE_FRAND_CACHE_HPP_

#include "tsetlini_types.hpp"
#include "aligned_allocator.hpp"
#include "assume_aligned.hpp"

#include <vector>
#include <algorithm>

namespace Tsetlini
{

template<int alignment=alignment, typename ValueType=float>
struct frand_cache
{
    using value_type = ValueType;

    explicit frand_cache(int sz) :
        m_pos(sz),
        m_fcache(sz)
    {
    }

    frand_cache()
        : m_pos(0)
        , m_fcache(0)
    {
    }

    template<typename TPRNG>
    inline
    void refill(TPRNG & frng)
    {
        value_type * fcache_p = assume_aligned<alignment>(m_fcache.data());

        for (auto it = 0u; it < std::min<unsigned int>(m_pos, m_fcache.size()); ++it)
        {
            fcache_p[it] = frng();
        }
        m_pos = 0;
    }

    inline
    value_type next()
    {
        value_type * fcache_p = assume_aligned<alignment>(m_fcache.data());
        return fcache_p[m_pos++];
    }

    inline
    value_type operator()()
    {
        return next();
    }

    unsigned int m_pos;
    aligned_vector<value_type> m_fcache;
};


} // namespace Tsetlini


#endif /* LIB_INCLUDE_FRAND_CACHE_HPP_ */
