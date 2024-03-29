#pragma once

#ifndef LIB_SRC_TA_STATE_HPP_
#define LIB_SRC_TA_STATE_HPP_

#include "tsetlini_types.hpp"
#include "tsetlini_strong_params.hpp"
#include "tsetlini_strong_params_private.hpp"
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
        typename T::value_type,
        decltype(T::initialize(
            std::declval<typename T::value_type &>(),
            std::declval<counting_type_t const &>(),
            std::declval<number_of_physical_estimator_clauses_t>(),
            std::declval<number_of_estimator_clause_outputs_t>(),
            std::declval<number_of_features_t>(),
            std::declval<weighted_flag_t const>(),
            std::declval<IRNG &>()))
    >>
    : std::true_type
{
};


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
    struct value_type
    {
        using matrix_variant_type = TAStateBase::matrix_variant_type;

        matrix_variant_type matrix;
        w_vector_type weights;

        bool operator==(struct value_type const & other) const
        {
            return
                this->matrix == other.matrix and
                this->weights == other.weights;
        }
    };

    static void initialize(
        value_type & state,
        counting_type_t const & counting_type,
        number_of_physical_estimator_clauses_t number_of_clauses,
        number_of_estimator_clause_outputs_t number_of_clause_outputs,
        number_of_features_t number_of_features,
        weighted_flag_t const weighted,
        IRNG & igen);
};


struct TAStateWithPolarity : public TAStateBase
{
    struct value_type
    {
        using matrix_variant_type = TAStateBase::matrix_variant_type;

        matrix_variant_type matrix;
        bit_matrix_uint64 polarity;
        w_vector_type weights;

        bool operator==(struct value_type const & other) const
        {
            return
                this->matrix == other.matrix and
                this->weights == other.weights and
                this->polarity == other.polarity;
        }
    };

    static void initialize(
        value_type & state,
        counting_type_t const & counting_type,
        number_of_physical_estimator_clauses_t number_of_clauses,
        number_of_estimator_clause_outputs_t number_of_clause_outputs,
        number_of_features_t number_of_features,
        weighted_flag_t const weighted,
        IRNG & igen);
};


}  // namespace Tsetlini


#endif /* LIB_SRC_TA_STATE_HPP_ */
