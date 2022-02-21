#include "ta_state.hpp"
#include "tsetlini_strong_params.hpp"
#include "mt.hpp"

#include "strong_type/strong_type.hpp"

#include <string>
#include <variant>


namespace Tsetlini
{


static inline
std::variant<
    numeric_matrix_int32
    , numeric_matrix_int16
    , numeric_matrix_int8
>
make_ta_state_matrix(counting_type_t const & counting_type,
    number_of_physical_estimator_clauses_t number_of_clauses,
    number_of_features_t number_of_features)
{
    if (counting_type == "int8")
    {
        return numeric_matrix_int8(value_of(number_of_clauses), value_of(number_of_features));
    }
    else if (counting_type == "int16")
    {
        return numeric_matrix_int16(value_of(number_of_clauses), value_of(number_of_features));
    }
    else
    {
        return numeric_matrix_int32(value_of(number_of_clauses), value_of(number_of_features));
    }
}


void
TAState::initialize(
    value_type & state,
    counting_type_t const & counting_type,
    number_of_physical_estimator_clauses_t number_of_clauses,
    number_of_estimator_clause_outputs_t number_of_clause_outputs,
    number_of_features_t number_of_features,
    weighted_flag_t const weighted,
    IRNG & igen)
{
    state.matrix = make_ta_state_matrix(counting_type, number_of_clauses, number_of_features);

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

    std::visit(state_gen, state.matrix);

    if (weighted == true)
    {
        state.weights.resize(value_of(number_of_clause_outputs));
    }
}


void
TAStateWithPolarity::initialize(
    value_type & state,
    counting_type_t const & counting_type,
    number_of_physical_estimator_clauses_t number_of_clauses,
    number_of_estimator_clause_outputs_t number_of_clause_outputs,
    number_of_features_t number_of_features,
    weighted_flag_t const weighted,
    IRNG & igen)
{
    state.matrix = make_ta_state_matrix(counting_type, number_of_clauses, number_of_features);
    state.polarity = bit_matrix_uint64(value_of(number_of_clauses), value_of(number_of_features));

    auto & polarity = state.polarity;

    auto state_gen = [&igen, &polarity](auto & matrix)
    {
        for (auto rit = 0u; rit < matrix.rows(); ++rit)
        {
            auto row_data = matrix.row_data(rit);
            auto row_polarity = polarity.row(rit);

            for (auto cit = 0u; cit < matrix.cols(); ++cit)
            {
                row_data[cit] = igen.next(-1, 0);

                if (row_data[cit] >= 0)
                {
                    row_polarity.set(cit);
                }
                else
                {
                    row_polarity.clear(cit);
                }
            }
        }
    };

    std::visit(state_gen, state.matrix);

    if (weighted == true)
    {
        state.weights.resize(value_of(number_of_clause_outputs));
    }
}


}  // namespace Tsetlini
