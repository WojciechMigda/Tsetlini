#pragma once

#ifndef LIB_INCLUDE_TA_STATE_HPP_
#define LIB_INCLUDE_TA_STATE_HPP_

#include "tsetlini_types.hpp"

#include <variant>
#include <type_traits>


namespace Tsetlini
{


template<typename, typename = std::void_t<>>
struct is_TA_State
    : std::false_type
{
};


template<typename T>
struct is_TA_State<T, std::void_t<
        typename T::value_type
    >>
    : std::true_type
{
};


struct TAState
{
    using value_type = std::variant<
        numeric_matrix_int32
        , numeric_matrix_int16
        , numeric_matrix_int8
    >;
};


struct TAStateWithSignum
{
    struct value_type
    {
        std::variant<
                numeric_matrix_int32
                , numeric_matrix_int16
                , numeric_matrix_int8
            > values;
        bit_matrix_uint64 signum;
    };
};



}  // namespace Tsetlini



#endif /* LIB_INCLUDE_TA_STATE_HPP_ */
