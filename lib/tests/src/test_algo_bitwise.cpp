#include "ta_state.hpp"
#include "basic_bit_vector_companion.hpp"
#include "tsetlini_types.hpp"
#include "tsetlini_algo_bitwise.hpp"
#include "tsetlini_algo_classic.hpp"
#include "tsetlini_algo_common.hpp"
#include "mt.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>


namespace
{


template<typename state_type, typename signum_type>
void
signum_from_ta_state(Tsetlini::numeric_matrix<state_type> const & ta_state, Tsetlini::bit_matrix<signum_type> & signum_matrix)
{
    auto const [nrows, ncols] = ta_state.shape();

    for (auto rix = 0u; rix < nrows; ++rix)
    {
        for (auto cix = 0u; cix < ncols; ++cix)
        {
            // x >= 0  --> 1
            // x < 0   --> 0
            auto const negative = ta_state[{rix, cix}] < 0;

            if (negative)
            {
                signum_matrix.clear(rix, cix);
            }
            else
            {
                signum_matrix.set(rix, cix);
            }
        }
    }
}


TEST(BitwiseCalculateClauseOutput, replicates_result_of_classic_code)
{
    IRNG irng(2345);
    int const NJOBS = 1;
    int const TILE_SZ = 16;

    for (auto it = 0u; it < 1000; /* nop */)
    {
        int const number_of_features = irng.next(2, 256);
        int const number_of_clauses = irng.next(1, 10) * 2; // must be even

        Tsetlini::aligned_vector_char X(number_of_features);

        std::generate(X.begin(), X.end(), [&irng](){ return irng.next(0, 1); });

        Tsetlini::numeric_matrix_int8 ta_state_values(2 * number_of_clauses, number_of_features);

        auto ta_state_gen = [&irng](auto & ta_state)
        {
            for (auto rit = 0u; rit < ta_state.rows(); ++rit)
            {
                auto row_data = ta_state.row_data(rit);

                for (auto cit = 0u; cit < ta_state.cols(); ++cit)
                {
                    row_data[cit] = irng.next(-1, 0);
                }
            }
        };

        ta_state_gen(ta_state_values);

        Tsetlini::aligned_vector_char clause_output_classic(number_of_clauses);
        Tsetlini::calculate_clause_output(X, clause_output_classic, 0, number_of_clauses, ta_state_values, NJOBS, TILE_SZ);

        if (0 != std::accumulate(clause_output_classic.cbegin(), clause_output_classic.cend(), 0u))
        {
            ++it;
        }

        auto const bitwise_X = basic_bit_vectors::from_range<std::uint64_t>(X.cbegin(), X.cend());
        Tsetlini::bit_matrix_uint64 ta_state_signum(2 * number_of_clauses, number_of_features);

        signum_from_ta_state(ta_state_values, ta_state_signum);

        Tsetlini::TAStateWithSignum::value_type ta_state;
        ta_state.signum = ta_state_signum;

        Tsetlini::aligned_vector_char clause_output_bitwise(number_of_clauses);
        Tsetlini::calculate_clause_output(bitwise_X, clause_output_bitwise, 0, number_of_clauses, ta_state, NJOBS, TILE_SZ);

        EXPECT_TRUE(clause_output_bitwise == clause_output_classic);
    }
}


TEST(BitwiseCalculateClauseOutput, replicates_result_of_classic_code_with_imbalanced_ta_state)
{
    IRNG irng(2345);
    int const NJOBS = 1;
    int const TILE_SZ = 8;

    for (auto it = 0u; it < 1000; /* nop */)
    {
        int const number_of_features = irng.next(15, 1280);
        int const number_of_clauses = irng.next(1, 10) * 2; // must be even

        Tsetlini::aligned_vector_char X(number_of_features);

        std::generate(X.begin(), X.end(), [&irng](){ return irng.next(0, 1); });

        Tsetlini::numeric_matrix_int8 ta_state_values(2 * number_of_clauses, number_of_features);

        auto ta_state_gen = [&irng](auto & ta_state)
        {
            // we will be initializing odd or even rows with all-1s
            auto const odd_even = irng.next(0, 1);

            for (auto rit = 0u; rit < ta_state.rows(); ++rit)
            {
                auto row_data = ta_state.row_data(rit);

                if (odd_even == (rit % 2))
                {
                    for (auto cit = 0u; cit < ta_state.cols(); ++cit)
                    {
                        // and for the other rows we will assign -1 63x more frequently
                        // This is to ensure more frequent non-zero clause output
                        row_data[cit] = irng.next(64) == 0 ? 0 : -1;
                    }
                }
                else
                {
                    std::fill_n(row_data, ta_state.cols(), -1);
                }
            }
        };

        ta_state_gen(ta_state_values);

        Tsetlini::aligned_vector_char clause_output_classic(number_of_clauses);
        Tsetlini::calculate_clause_output(X, clause_output_classic, 0, number_of_clauses, ta_state_values, NJOBS, TILE_SZ);

        if (0 != std::accumulate(clause_output_classic.cbegin(), clause_output_classic.cend(), 0u))
        {
            ++it;
        }

        auto const bitwise_X = basic_bit_vectors::from_range<std::uint64_t>(X.cbegin(), X.cend());
        Tsetlini::bit_matrix_uint64 ta_state_signum(2 * number_of_clauses, number_of_features);
        signum_from_ta_state(ta_state_values, ta_state_signum);

        Tsetlini::TAStateWithSignum::value_type ta_state;
        ta_state.signum = ta_state_signum;

        Tsetlini::aligned_vector_char clause_output_bitwise(number_of_clauses);
        Tsetlini::calculate_clause_output(bitwise_X, clause_output_bitwise, 0, number_of_clauses, ta_state, NJOBS, TILE_SZ);

        EXPECT_TRUE(clause_output_bitwise == clause_output_classic);
    }
}


TEST(BitwiseCalculateClauseOutputForPredict, replicates_result_of_classic_code)
{
    IRNG irng(2345);
    int const NJOBS = 1;
    int const TILE_SZ = 16;

    for (auto it = 0u; it < 1000; /* nop */)
    {
        int const number_of_features = irng.next(2, 256);
        int const number_of_clauses = irng.next(1, 10) * 2; // must be even

        Tsetlini::aligned_vector_char X(number_of_features);

        std::generate(X.begin(), X.end(), [&irng](){ return irng.next(0, 1); });

        Tsetlini::numeric_matrix_int8 ta_state_values(2 * number_of_clauses, number_of_features);

        auto ta_state_gen = [&irng](auto & ta_state)
        {
            for (auto rit = 0u; rit < ta_state.rows(); ++rit)
            {
                auto row_data = ta_state.row_data(rit);

                for (auto cit = 0u; cit < ta_state.cols(); ++cit)
                {
                    row_data[cit] = irng.next(-1, 0);
                }
            }
        };

        ta_state_gen(ta_state_values);

        Tsetlini::aligned_vector_char clause_output_classic(number_of_clauses);
        Tsetlini::calculate_clause_output_for_predict(X, clause_output_classic, number_of_clauses, ta_state_values, NJOBS, TILE_SZ);

        if (0 != std::accumulate(clause_output_classic.cbegin(), clause_output_classic.cend(), 0u))
        {
            ++it;
        }

        auto const bitwise_X = basic_bit_vectors::from_range<std::uint64_t>(X.cbegin(), X.cend());
        Tsetlini::bit_matrix_uint64 ta_state_signum(2 * number_of_clauses, number_of_features);

        signum_from_ta_state(ta_state_values, ta_state_signum);

        Tsetlini::TAStateWithSignum::value_type ta_state;
        ta_state.signum = ta_state_signum;

        Tsetlini::aligned_vector_char clause_output_bitwise(number_of_clauses);

        Tsetlini::calculate_clause_output_for_predict(bitwise_X, clause_output_bitwise, number_of_clauses, ta_state, NJOBS, TILE_SZ);

        EXPECT_TRUE(clause_output_bitwise == clause_output_classic);
    }
}


TEST(BitwiseCalculateClauseOutputForPredict, replicates_result_of_classic_code_with_imbalanced_ta_state)
{
    IRNG irng(2345);
    int const NJOBS = 1;
    int const TILE_SZ = 8;

    for (auto it = 0u; it < 1000; /* nop */)
    {
        int const number_of_features = irng.next(15, 1280);
        int const number_of_clauses = irng.next(1, 10) * 2; // must be even

        Tsetlini::aligned_vector_char X(number_of_features);

        std::generate(X.begin(), X.end(), [&irng](){ return irng.next(0, 1); });

        Tsetlini::numeric_matrix_int8 ta_state_values(2 * number_of_clauses, number_of_features);

        auto ta_state_gen = [&irng](auto & ta_state)
        {
            // we will be initializing odd or even rows with all-1s
            auto const odd_even = irng.next(0, 1);

            for (auto rit = 0u; rit < ta_state.rows(); ++rit)
            {
                auto row_data = ta_state.row_data(rit);

                if (odd_even == (rit % 2))
                {
                    for (auto cit = 0u; cit < ta_state.cols(); ++cit)
                    {
                        // and for the other rows we will assign -1 63x more frequently
                        // This is to ensure more frequent non-zero clause output
                        row_data[cit] = irng.next(64) == 0 ? 0 : -1;
                    }
                }
                else
                {
                    std::fill_n(row_data, ta_state.cols(), -1);
                }
            }
        };

        ta_state_gen(ta_state_values);

        Tsetlini::aligned_vector_char clause_output_classic(number_of_clauses);
        Tsetlini::calculate_clause_output_for_predict(X, clause_output_classic, number_of_clauses, ta_state_values, NJOBS, TILE_SZ);

        if (0 != std::accumulate(clause_output_classic.cbegin(), clause_output_classic.cend(), 0u))
        {
            ++it;
        }

        auto const bitwise_X = basic_bit_vectors::from_range<std::uint64_t>(X.cbegin(), X.cend());
        Tsetlini::bit_matrix_uint64 ta_state_signum(2 * number_of_clauses, number_of_features);

        signum_from_ta_state(ta_state_values, ta_state_signum);

        Tsetlini::TAStateWithSignum::value_type ta_state;
        ta_state.signum = ta_state_signum;

        Tsetlini::aligned_vector_char clause_output_bitwise(number_of_clauses);

        Tsetlini::calculate_clause_output_for_predict(bitwise_X, clause_output_bitwise, number_of_clauses, ta_state, NJOBS, TILE_SZ);

        EXPECT_TRUE(clause_output_bitwise == clause_output_classic);
    }
}


TEST(BitwiseTrainClassifierAutomata, replicates_result_of_classic_code)
{
    IRNG    irng(1234);
    FRNG    fgen(4567);
    IRNG    prng(4567);
    IRNG    prng_classic(4567);

    for (auto it = 0u; it < 1000; ++it)
    {
        int const number_of_features = irng.next(1, 200);
        int const number_of_clauses = irng.next(1, 50) * 2; // must be even
        int const number_of_states = irng.next(2, 127);

        Tsetlini::aligned_vector_char X(number_of_features);

        std::generate(X.begin(), X.end(), [&irng](){ return irng.next(0, 1); });

        Tsetlini::numeric_matrix_int8 ta_state_values(2 * number_of_clauses, number_of_features);

        auto ta_state_gen = [number_of_states, &irng](auto & ta_state)
        {
            for (auto rit = 0u; rit < ta_state.rows(); ++rit)
            {
                auto row_data = ta_state.row_data(rit);

                for (auto cit = 0u; cit < ta_state.cols(); ++cit)
                {
                    row_data[cit] = irng.next(-number_of_states, number_of_states - 1);
                }
            }
        };

        ta_state_gen(ta_state_values);

        Tsetlini::numeric_matrix_int8 ta_state_classic = ta_state_values;

        Tsetlini::feedback_vector_type feedback_to_clauses(number_of_clauses);
        std::generate(feedback_to_clauses.begin(), feedback_to_clauses.end(), [&irng](){ return irng.next(-1, +1); });

        Tsetlini::aligned_vector_char clause_output(number_of_clauses);
        std::generate(clause_output.begin(), clause_output.end(), [&irng](){ return irng.next(0, 1); });

        bool const boost_true_positive_feedback = irng.next(0, 1) != 0;
        /*
         * Setting S_inv to either 0.0 or 1.0 removes stochasticity from testing
         */
        char const ct_val = irng.next(0, 1);

        Tsetlini::EstimatorStateCacheBase::coin_tosser_type ct(ct_val, number_of_features);
        ct.fill(ct_val);
        Tsetlini::EstimatorStateCacheBase::coin_tosser_type ct_classic = ct;

        Tsetlini::train_classifier_automata(
            ta_state_classic, 0, number_of_clauses, feedback_to_clauses.data(), clause_output.data(),
            number_of_states, X, boost_true_positive_feedback, prng_classic, ct_classic);


        auto const bitwise_X = basic_bit_vectors::from_range<std::uint64_t>(X.cbegin(), X.cend());

        Tsetlini::bit_matrix_uint64 ta_state_signum(2 * number_of_clauses, number_of_features);
        signum_from_ta_state(ta_state_values, ta_state_signum);

        // this will be fed to train_classifier_automata
        Tsetlini::TAStateWithSignum::value_type ta_state;
        ta_state.signum = ta_state_signum;
        ta_state.matrix = ta_state_values;

        // mock prng which returns duplicated running integers modulo number of features
        // 0, 0, 1, 1, 2, 2, ...
        auto iota_counter = 0u;
        auto iota_prng = [&iota_counter, number_of_features]()
        {
            auto rv = iota_counter % (2 * number_of_features);
            ++iota_counter;
            return rv / 2;
        };

        Tsetlini::train_classifier_automata(
            ta_state, 0, number_of_clauses, feedback_to_clauses.data(), clause_output.data(),
            number_of_states, bitwise_X, boost_true_positive_feedback, iota_prng, ct);

        // retrieve TA State values from ta_state variant for verifiation
        ta_state_values = std::get<Tsetlini::numeric_matrix_int8>(ta_state.matrix);
        EXPECT_TRUE(ta_state_values == ta_state_classic);

        // assert whether signum was synchronized
        Tsetlini::bit_matrix_uint64 ta_state_signum_post(2 * number_of_clauses, number_of_features);
        signum_from_ta_state(ta_state_values, ta_state_signum_post);

        EXPECT_TRUE(ta_state.signum == ta_state_signum_post);
    }
}


} // anonymous namespace
