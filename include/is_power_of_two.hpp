#pragma once

template<typename T>
constexpr bool is_power_of_two(T x)
{
    return x && !(x & (x - 1));
}
