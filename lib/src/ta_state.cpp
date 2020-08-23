#include "ta_state.hpp"
#include "mt.hpp"

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


void
TAState::initialize(
    value_type & state,
    std::string const & counting_type,
    int number_of_clauses,
    int number_of_features,
    bool const weighted,
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

    if (weighted)
    {
        state.weights.resize(number_of_clauses / 2);
    }
}


void
TAStateWithSignum::initialize(
    value_type & state,
    std::string const & counting_type,
    int number_of_clauses,
    int number_of_features,
    bool const weighted,
    IRNG & igen)
{
    state.matrix = make_ta_state_matrix(counting_type, number_of_clauses, number_of_features);
    state.signum = bit_matrix_uint64(number_of_clauses, number_of_features);

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
                    row_signum.set(cit);
                }
                else
                {
                    row_signum.clear(cit);
                }
            }
        }
    };

    std::visit(state_gen, state.matrix);

    if (weighted)
    {
        state.weights.resize(number_of_clauses / 2);
    }
}


}  // namespace Tsetlini
