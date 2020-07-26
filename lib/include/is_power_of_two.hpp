#pragma once

#ifndef LIB_INCLUDE_IS_POWER_OF_TWO_HPP_
#define LIB_INCLUDE_IS_POWER_OF_TWO_HPP_


template<typename T>
constexpr bool is_power_of_two(T x)
{
    return x && !(x & (x - 1));
}


#endif /* LIB_INCLUDE_IS_POWER_OF_TWO_HPP_ */
