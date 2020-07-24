#pragma once

#ifndef LIB_INCLUDE_TA_STATE_HPP_
#define LIB_INCLUDE_TA_STATE_HPP_

#include "tsetlini_types.hpp"
#include "mt.hpp"

#include <variant>
#include <type_traits>


namespace Tsetlini
{


template<typename, typename = std::void_t<>>
struct is_TA_state
    : std::false_type
{
};


template<typename T>
struct is_TA_state<T, std::void_t<
        typename T::value_type
    >>
    : std::true_type
{
};


std::variant<
    numeric_matrix_int32
    , numeric_matrix_int16
    , numeric_matrix_int8
>
make_ta_state_matrix(std::string const & counting_type, int number_of_clauses, int number_of_features)
{
    if (counting_type == "int8")
    {
        return numeric_matrix_int8(number_of_clauses, number_of_features);
    }
    else if (counting_type == "int16")
    {
        return numeric_matrix_int16(number_of_clauses, number_of_features);
    }
    else
    {
        return numeric_matrix_int32(number_of_clauses, number_of_features);
    }
}


struct TAStateBase
{
    using matrix_variant_type = std::variant<
        numeric_matrix_int32
        , numeric_matrix_int16
        , numeric_matrix_int8
    >;
};


struct TAState : public TAStateBase
{
    using value_type = matrix_variant_type;

    static void initialize(
        value_type & state,
        std::string const & counting_type,
        int number_of_clauses,
        int number_of_features,
        IRNG & igen)
    {
        state = make_ta_state_matrix(counting_type, number_of_clauses, number_of_features);

        auto state_gen = [&igen](auto & matrix)
        {
            for (auto rit = 0u; rit < matrix.rows(); ++rit)
            {
                auto row_data = matrix.row_data(rit);

                for (auto cit = 0u; cit < matrix.cols(); ++cit)
                {
                    row_data[cit] = igen.next(-1, 0);
                }
            }
        };

        std::visit(state_gen, state);
    }
};


struct TAStateWithSignum : public TAStateBase
{
    struct value_type
    {
        matrix_variant_type matrix;
        bit_matrix_uint64 signum;
    };

    static void initialize(
        value_type & state,
        std::string const & counting_type,
        int number_of_clauses,
        int number_of_features,
        IRNG & igen)
    {
        state.matrix = make_ta_state_matrix(counting_type, number_of_clauses, number_of_features);

        auto & signum = state.signum;

        auto state_gen = [&igen, &signum](auto & matrix)
        {
            for (auto rit = 0u; rit < matrix.rows(); ++rit)
            {
                auto row_data = matrix.row_data(rit);
                auto row_signum = signum.row(rit);

                for (auto cit = 0u; cit < matrix.cols(); ++cit)
                {
                    row_data[cit] = igen.next(-1, 0);

                    if (row_data[cit] >= 0)
                    {
                        row_signum.set(rit);
                    }
                    else
                    {
                        row_signum.clear(rit);
                    }
                }
            }
        };

        std::visit(state_gen, state.matrix);
    }
};


}  // namespace Tsetlini


#endif /* LIB_INCLUDE_TA_STATE_HPP_ */
