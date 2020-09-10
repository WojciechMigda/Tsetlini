#include "mt.hpp"

#include "box_muller_approx.hpp"

#include <cmath>
#include <algorithm>
#include <array>
#include <utility>


namespace box_muller
{


/// https://stackoverflow.com/a/19023500/2003487
template<class Function, std::size_t... Indices>
constexpr auto make_array_helper(Function f, std::index_sequence<Indices...>)
-> std::array<typename std::result_of<Function(std::size_t)>::type, sizeof...(Indices)>
{
    return {{ f(Indices)... }};
}

template<int N, class Function>
constexpr auto make_array(Function f)
-> std::array<typename std::result_of<Function(std::size_t)>::type, N>
{
    return make_array_helper(f, std::make_index_sequence<N>{});
}
///

static
constexpr float sin_u2(std::uint8_t x)
{
    return std::cos(2 * M_PI * (1u + x) / 257.f);
}

static
constexpr auto sin_u2_arr = make_array<256>(sin_u2);


static
constexpr float sqrt_log_u1(std::uint8_t x)
{
    return std::sqrt(-2.f * std::log((1u + x) / 257.f));
}


static
constexpr auto sqrt_log_u1_arr = make_array<256>(sqrt_log_u1);


template<typename PRNG>
static float normal(float mean, float variance, PRNG & irng)
{
    std::uint32_t const r = irng();
    std::uint8_t const u1 = r;
    std::uint8_t const u2 = r >> 8;

    auto const n1 = sqrt_log_u1_arr[u1] * sin_u2_arr[u2];

    return mean + std::sqrt(variance) * n1;
}


}  // namespace box_muller



template<typename PRNG>
unsigned int binomial(unsigned int n, float p, PRNG & irng)
{
    unsigned int const rv = std::round(box_muller::normal(n * p, n * p * (1.f - p), irng));

    return std::clamp(rv, 0u, n);
}


template unsigned int binomial<IRNG>(unsigned int n, float p, IRNG & irng);
