#include "tsetlini_algo_common.hpp"
#include "tsetlini_strong_params.hpp"
#include "tsetlini_strong_params_private.hpp"
#include "tsetlini_types.hpp"

#include "strong_type/strong_type.hpp"
#include "rapidcheck.h"
#include "boost/ut.hpp"

#include <cstdlib>
#include <algorithm>


using namespace boost::ut;


auto constexpr MAX_NUM_OF_FEATURES = 2048;
auto constexpr MAX_NUM_OF_CLAUSE_OUTPUTS = 16;


////////////////////////////////////////////////////////////////////////////////


/*
 * Reference algorithm for clause output calculation
 *
 * Source:
 * "The Regression Tsetlin Machine: A Tsetlin Machine for Continuous Output Problems"
 * https://arxiv.org/abs/1905.04206
 * Table 1.
 */
template<typename feature_value_type, typename state_type>
void reference(
    Tsetlini::aligned_vector<feature_value_type> const & X,
    Tsetlini::aligned_vector_char & clause_output,
    Tsetlini::numeric_matrix<state_type> const & ta_state_matrix
)
{
    for (auto oix = 0u; oix < clause_output.size(); ++oix)
    {
        bool output = true;

        for (auto fix = 0u; fix < X.size(); ++fix)
        {
            if (((ta_state_matrix[{0 + 2 * oix, fix}] >= 0) and (X[fix] == 0)) or
                ((ta_state_matrix[{1 + 2 * oix, fix}] >= 0) and (X[fix] != 0)))
            {
                output = false;
                break;
            }
        }

        clause_output[oix] = output;
    }
}


/*
 * Pruning in only being mentioned in the original Tsetlin Machine paper:
 * "The Tsetlin Machine - A Game Theoretic Bandit Driven Approach to Optimal
 * Pattern Recognition with Propositional Logic"
 * https://arxiv.org/abs/1804.01508, Algorithm 1, pp 12.
 *
 * The rest of details about it can be inferred from associated CAIR implementations.
 */
template<typename feature_value_type, typename state_type>
void reference_with_pruning(
    Tsetlini::aligned_vector<feature_value_type> const & X,
    Tsetlini::aligned_vector_char & clause_output,
    Tsetlini::numeric_matrix<state_type> const & ta_state_matrix
)
{
    for (auto oix = 0u; oix < clause_output.size(); ++oix)
    {
        if (std::all_of(ta_state_matrix.row_data(0 + 2 * oix), ta_state_matrix.row_data(0 + 2 * oix) + X.size(), [](auto x){ return x < 0;}) and
            std::all_of(ta_state_matrix.row_data(1 + 2 * oix), ta_state_matrix.row_data(1 + 2 * oix) + X.size(), [](auto x){ return x < 0;})
        )
        {
            // pruning
            clause_output[oix] = false;
            continue;
        }

        bool output = true;

        for (auto fix = 0u; fix < X.size(); ++fix)
        {
            if (((ta_state_matrix[{0 + 2 * oix, fix}] >= 0) and (X[fix] == 0)) or
                ((ta_state_matrix[{1 + 2 * oix, fix}] >= 0) and (X[fix] != 0)))
            {
                output = false;
                break;
            }
        }

        clause_output[oix] = output;
    }
}


////////////////////////////////////////////////////////////////////////////////


suite BytewiseCalculateClauseOutput = []
{


using matrix_type = Tsetlini::numeric_matrix_int16;


auto make_ta_state_matrix = [](
    Tsetlini::number_of_estimator_clause_outputs_t number_of_clause_outputs,
    Tsetlini::number_of_features_t number_of_features)
{
    matrix_type ta_state_matrix(2 * value_of(number_of_clause_outputs), value_of(number_of_features));

    // fill entire matrix storage space, regardless of alignment and padding
    ta_state_matrix.m_v = *rc::gen::container<matrix_type::aligned_vector>(ta_state_matrix.m_v.size(), rc::gen::inRange<matrix_type::value_type>(-1, 1));

    return ta_state_matrix;
};


"Bytewise calculate_clause_output replicates paper formula"_test = [&]
{
    auto ok = rc::check(
        [&]()
        {
            auto const number_of_features = Tsetlini::number_of_features_t{*rc::gen::inRange(1, MAX_NUM_OF_FEATURES + 1)};
            Tsetlini::number_of_estimator_clause_outputs_t number_of_clause_outputs{2 * *rc::gen::inRange(1, MAX_NUM_OF_CLAUSE_OUTPUTS / 2 + 1)};

            auto const X = *rc::gen::container<Tsetlini::aligned_vector_char>(value_of(number_of_features), rc::gen::arbitrary<bool>());
            auto const ta_state_matrix = make_ta_state_matrix(number_of_clause_outputs, number_of_features);

            Tsetlini::aligned_vector_char ground_truth(value_of(number_of_clause_outputs));
            reference(X, ground_truth, ta_state_matrix);

            Tsetlini::aligned_vector_char clause_output(value_of(number_of_clause_outputs));
            Tsetlini::TAState::value_type const ta_state{ta_state_matrix};

            Tsetlini::calculate_clause_output(X, clause_output, 0, value_of(number_of_clause_outputs), ta_state,
                Tsetlini::number_of_jobs_t{1}, Tsetlini::clause_output_tile_size_t{16});

            RC_ASSERT(clause_output == ground_truth);
        }
    );

    expect(that % true == ok);
};


"Bytewise calculate_clause_output_with_pruning replicates paper formula"_test = [&]
{
    auto ok = rc::check(
        [&]()
        {
            auto const number_of_features = Tsetlini::number_of_features_t{*rc::gen::inRange(1, MAX_NUM_OF_FEATURES + 1)};
            Tsetlini::number_of_estimator_clause_outputs_t number_of_clause_outputs{2 * *rc::gen::inRange(1, MAX_NUM_OF_CLAUSE_OUTPUTS / 2 + 1)};

            auto const X = *rc::gen::container<Tsetlini::aligned_vector_char>(value_of(number_of_features), rc::gen::arbitrary<bool>());
            auto const ta_state_matrix = make_ta_state_matrix(number_of_clause_outputs, number_of_features);

            Tsetlini::aligned_vector_char ground_truth(value_of(number_of_clause_outputs));
            reference_with_pruning(X, ground_truth, ta_state_matrix);

            Tsetlini::aligned_vector_char clause_output(value_of(number_of_clause_outputs));
            Tsetlini::TAState::value_type const ta_state{ta_state_matrix};

            Tsetlini::calculate_clause_output_with_pruning(X, clause_output, number_of_clause_outputs, ta_state,
                Tsetlini::number_of_jobs_t{1}, Tsetlini::clause_output_tile_size_t{16});

            RC_ASSERT(clause_output == ground_truth);
        }
    );

    expect(that % true == ok);
};


}; // suite BytewiseCalculateClauseOutput


////////////////////////////////////////////////////////////////////////////////


/*
 * Source:
 * "The Regression Tsetlin Machine: A Tsetlin Machine for Continuous Output Problems"
 * https://arxiv.org/abs/1905.04206
 * Table 1.
 */
void reference(
    Tsetlini::bit_vector_uint64 const & X,
    Tsetlini::aligned_vector_char & clause_output,
    Tsetlini::bit_matrix_uint64 const & polarity
)
{
    for (auto oix = 0u; oix < clause_output.size(); ++oix)
    {
        bool output = true;

        for (auto fix = 0u; fix < X.size(); ++fix)
        {
            /*
             * polarity == 1 means 'include'
             *
             * See TAStateWithPolarity::initialize() :
             *
             *  if (row_data[cit] >= 0)
             *  {
             *      row_polarity.set(cit);
             *  }
             *  else
             *  {
             *      row_polarity.clear(cit);
             *  }
             */
            if (((polarity[{0 + 2 * oix, fix}] == 1) and (X[fix] == 0)) or
                ((polarity[{1 + 2 * oix, fix}] == 1) and (X[fix] != 0)))
            {
                output = false;
                break;
            }
        }

        clause_output[oix] = output;
    }
}


void reference_with_pruning(
    Tsetlini::bit_vector_uint64 const & X,
    Tsetlini::aligned_vector_char & clause_output,
    Tsetlini::bit_matrix_uint64 const & polarity
)
{
    for (auto oix = 0u; oix < clause_output.size(); ++oix)
    {
        // this will work because all padding space is zeroed.
        if (std::all_of(polarity.row_data(0 + 2 * oix), polarity.row_data(0 + 2 * oix) + polarity.row_blocks(), [](auto x){ return x == 0;}) and
            std::all_of(polarity.row_data(1 + 2 * oix), polarity.row_data(1 + 2 * oix) + polarity.row_blocks(), [](auto x){ return x == 0;})
        )
        {
            // pruning
            clause_output[oix] = false;
            continue;
        }

        bool output = true;

        for (auto fix = 0u; fix < X.size(); ++fix)
        {
            if (((polarity[{0 + 2 * oix, fix}] == 1) and (X[fix] == 0)) or
                ((polarity[{1 + 2 * oix, fix}] == 1) and (X[fix] != 0)))
            {
                output = false;
                break;
            }
        }

        clause_output[oix] = output;
    }
}


suite BitwiseCalculateClauseOutput = []
{


using matrix_type = Tsetlini::bit_matrix_uint64;


auto make_X = [](Tsetlini::number_of_features_t number_of_features)
{
    Tsetlini::bit_vector_uint64 X(value_of(number_of_features));

    X.m_vector = *rc::gen::container<Tsetlini::bit_vector_uint64::aligned_vector>(X.m_vector.size(), rc::gen::arbitrary<Tsetlini::bit_vector_uint64::block_type>());

    // clear any random bits beyond valid index range
    auto const total_bits = X.m_vector.size() * X.block_bits;
    for (Tsetlini::size_type ix = value_of(number_of_features); ix < total_bits; ++ix)
    {
        X.clear(ix);
    }

    return X;
};


auto make_polarity_matrix = [](
    Tsetlini::number_of_estimator_clause_outputs_t number_of_clause_outputs,
    Tsetlini::number_of_features_t number_of_features)
{
    matrix_type polarity(2 * value_of(number_of_clause_outputs), value_of(number_of_features));

    // fill entire matrix storage space, regardless of alignment and padding
    polarity.m_v = *rc::gen::container<matrix_type::aligned_vector>(polarity.m_v.size(), rc::gen::arbitrary<matrix_type::block_type>());

    // clear any random bits beyond valid column number range
    auto const total_row_bits = polarity.row_blocks() * polarity.block_bits;
    for (Tsetlini::size_type row = 0; row < polarity.rows(); ++row)
    {
        for (Tsetlini::size_type col = value_of(number_of_features); col < total_row_bits; ++col)
        {
            polarity.clear(row, col);
        }
    }

    return polarity;
};


"Bitwise calculate_clause_output replicates paper formula"_test = [&]
{
    auto ok = rc::check(
        [&]()
        {
            auto const number_of_features = Tsetlini::number_of_features_t{*rc::gen::inRange(1, MAX_NUM_OF_FEATURES + 1)};
            Tsetlini::number_of_estimator_clause_outputs_t number_of_clause_outputs{2 * *rc::gen::inRange(1, MAX_NUM_OF_CLAUSE_OUTPUTS / 2 + 1)};

            auto const X = make_X(number_of_features);
            auto const ta_state_polarity = make_polarity_matrix(number_of_clause_outputs, number_of_features);

            Tsetlini::aligned_vector_char ground_truth(value_of(number_of_clause_outputs));
            reference(X, ground_truth, ta_state_polarity);

            Tsetlini::aligned_vector_char clause_output(value_of(number_of_clause_outputs));
            Tsetlini::TAStateWithPolarity::value_type ta_state;
            ta_state.polarity = ta_state_polarity;

            Tsetlini::calculate_clause_output(X, clause_output, 0, value_of(number_of_clause_outputs), ta_state,
                Tsetlini::number_of_jobs_t{1}, Tsetlini::clause_output_tile_size_t{16});

            RC_ASSERT(clause_output == ground_truth);
        }
    );

    expect(that % true == ok);
};


"Bitwise calculate_clause_output_with_pruning replicates paper formula"_test = [&]
{
    auto ok = rc::check(
        [&]()
        {
            auto const number_of_features = Tsetlini::number_of_features_t{*rc::gen::inRange(1, MAX_NUM_OF_FEATURES + 1)};
            Tsetlini::number_of_estimator_clause_outputs_t number_of_clause_outputs{2 * *rc::gen::inRange(1, MAX_NUM_OF_CLAUSE_OUTPUTS / 2 + 1)};

            auto const X = make_X(number_of_features);
            auto const ta_state_polarity = make_polarity_matrix(number_of_clause_outputs, number_of_features);

            Tsetlini::aligned_vector_char ground_truth(value_of(number_of_clause_outputs));
            reference_with_pruning(X, ground_truth, ta_state_polarity);

            Tsetlini::aligned_vector_char clause_output(value_of(number_of_clause_outputs));
            Tsetlini::TAStateWithPolarity::value_type ta_state;
            ta_state.polarity = ta_state_polarity;

            Tsetlini::calculate_clause_output_with_pruning(X, clause_output, number_of_clause_outputs, ta_state,
                Tsetlini::number_of_jobs_t{1}, Tsetlini::clause_output_tile_size_t{16});

            RC_ASSERT(clause_output == ground_truth);
        }
    );

    expect(that % true == ok);
};


}; // BitwiseCalculateClauseOutput


int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
