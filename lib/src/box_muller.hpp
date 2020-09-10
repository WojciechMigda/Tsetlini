#pragma once

#ifndef LIB_SRC_BOX_MULLER_HPP_
#define LIB_SRC_BOX_MULLER_HPP_


#include <cmath>
#include <algorithm>


namespace box_muller
{


template<typename PRNG>
static float u1(PRNG & frng)
{
    float rv = frng();

    while (rv == 0)
    {
        rv = frng();
    }

    return rv;
}


template<typename PRNG>
static float u2(PRNG & frng)
{
    float rv = frng();

    return rv;
}


template<typename PRNG>
static float normal(float mean, float variance, PRNG & frng)
{
    auto const n1 = std::sqrt(-2.f * std::log(u1(frng))) * std::sin(2 * M_PI * u2(frng));

    return mean + std::sqrt(variance) * n1;
}


}  // namespace box_muller


template<typename PRNG>
unsigned int binomial(unsigned int n, float p, PRNG & frng)
{
    unsigned int const rv = std::round(box_muller::normal(n * p, n * p * (1.f - p), frng));

    return std::clamp(rv, 0u, n);
}


#endif /* LIB_SRC_BOX_MULLER_HPP_ */
