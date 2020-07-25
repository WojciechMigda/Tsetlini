#include "basic_bit_vector_companion.hpp"
#include "tsetlini_types.hpp"
#include "tsetlini_algo.hpp"
#include "tsetlini_algo_bitwise.hpp"
#include "mt.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

namespace
{


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

        Tsetlini::numeric_matrix_int8 ta_state(2 * number_of_clauses, number_of_features);

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

        ta_state_gen(ta_state);

        Tsetlini::aligned_vector_char clause_output_classic(number_of_clauses);
        Tsetlini::calculate_clause_output(X, clause_output_classic, 0, number_of_clauses, ta_state, NJOBS, TILE_SZ);

        if (0 != std::accumulate(clause_output_classic.cbegin(), clause_output_classic.cend(), 0u))
        {
            ++it;
        }

        auto const bitwise_X = basic_bit_vectors::from_range<std::uint32_t>(X.cbegin(), X.cend());
        Tsetlini::bit_matrix_uint32 ta_state_signum(2 * number_of_clauses, number_of_features);
        Tsetlini::signum_from_ta_state(ta_state, ta_state_signum);

        Tsetlini::aligned_vector_char clause_output_bitwise(number_of_clauses);
        Tsetlini::calculate_clause_output(bitwise_X, clause_output_bitwise, 0, number_of_clauses, ta_state_signum, NJOBS, TILE_SZ);

        EXPECT_TRUE(clause_output_bitwise == clause_output_classic);
    }
}


TEST(BitwiseCalculateClauseOutput, replicates_result_of_classic_code_with_imbalanced_ta_state)
{
    IRNG irng(2345);
    int const NJOBS = 1;
    int const TILE_SZ = 16;

    for (auto it = 0u; it < 1000; /* nop */)
    {
        int const number_of_features = irng.next(15, 500);
        int const number_of_clauses = irng.next(1, 10) * 2; // must be even

        Tsetlini::aligned_vector_char X(number_of_features);

        std::generate(X.begin(), X.end(), [&irng](){ return irng.next(0, 1); });

        Tsetlini::numeric_matrix_int8 ta_state(2 * number_of_clauses, number_of_features);

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

        ta_state_gen(ta_state);

        Tsetlini::aligned_vector_char clause_output_classic(number_of_clauses);
        Tsetlini::calculate_clause_output(X, clause_output_classic, 0, number_of_clauses, ta_state, NJOBS, TILE_SZ);

        if (0 != std::accumulate(clause_output_classic.cbegin(), clause_output_classic.cend(), 0u))
        {
            ++it;
        }

        auto const bitwise_X = basic_bit_vectors::from_range<std::uint32_t>(X.cbegin(), X.cend());
        Tsetlini::bit_matrix_uint32 ta_state_signum(2 * number_of_clauses, number_of_features);
        Tsetlini::signum_from_ta_state(ta_state, ta_state_signum);

        Tsetlini::aligned_vector_char clause_output_bitwise(number_of_clauses);
        Tsetlini::calculate_clause_output(bitwise_X, clause_output_bitwise, 0, number_of_clauses, ta_state_signum, NJOBS, TILE_SZ);

        EXPECT_TRUE(clause_output_bitwise == clause_output_classic);
    }
}


TEST(BitwiseTrainAutomata, replicates_result_of_classic_code)
{
    IRNG    irng(1234);
    FRNG    fgen(4567);
    FRNG    frng(4567);
    FRNG    frng_classic(4567);

    for (auto it = 0u; it < 1000; ++it)
    {
        int const number_of_features = irng.next(1, 200);
        int const number_of_clauses = irng.next(1, 50) * 2; // must be even
        int const number_of_states = irng.next(2, 127);

        Tsetlini::aligned_vector_char X(number_of_features);

        std::generate(X.begin(), X.end(), [&irng](){ return irng.next(0, 1); });

        Tsetlini::numeric_matrix_int8 ta_state(2 * number_of_clauses, number_of_features);

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

        ta_state_gen(ta_state);

        Tsetlini::numeric_matrix_int8 ta_state_classic = ta_state;

        Tsetlini::feedback_vector_type feedback_to_clauses(number_of_clauses);
        std::generate(feedback_to_clauses.begin(), feedback_to_clauses.end(), [&irng](){ return irng.next(-1, +1); });

        Tsetlini::aligned_vector_char clause_output(number_of_clauses);
        std::generate(clause_output.begin(), clause_output.end(), [&irng](){ return irng.next(0, 1); });

        bool const boost_true_positive_feedback = irng.next(0, 1) != 0;
        /*
         * Setting S_inv to either 0.0 or 1.0 removes stochasticity from testing
         */
        Tsetlini::real_type const S_inv = irng.next(0, 1);

        Tsetlini::ClassifierState::cache_type::frand_cache_type fcache_classic(fgen, 2 * number_of_features, 0);
        Tsetlini::ClassifierState::cache_type::frand_cache_type fcache = fcache_classic;

        Tsetlini::train_classifier_automata(
            ta_state_classic, 0, number_of_clauses, feedback_to_clauses.data(), clause_output.data(),
            number_of_states, S_inv, X, boost_true_positive_feedback, frng_classic, fcache_classic);


        auto const bitwise_X = basic_bit_vectors::from_range<std::uint32_t>(X.cbegin(), X.cend());
        Tsetlini::bit_matrix_uint32 ta_state_signum(2 * number_of_clauses, number_of_features);
        Tsetlini::signum_from_ta_state(ta_state, ta_state_signum);

        Tsetlini::train_classifier_automata(
            ta_state, ta_state_signum, 0, number_of_clauses, feedback_to_clauses.data(), clause_output.data(),
            number_of_states, S_inv, bitwise_X, boost_true_positive_feedback, frng, fcache);

        EXPECT_TRUE(ta_state == ta_state_classic);

        // assert whether signum was synchronized
        Tsetlini::bit_matrix_uint32 ta_state_signum_post(2 * number_of_clauses, number_of_features);
        Tsetlini::signum_from_ta_state(ta_state, ta_state_signum_post);

        EXPECT_TRUE(ta_state_signum == ta_state_signum_post);
    }
}


} // anonymous namespace
