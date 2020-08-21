#include "cair_algos.hpp"

#include "estimator_state_cache.hpp"
#include "tsetlini_types.hpp"
#include "tsetlini_algo_classic.hpp"
#include "tsetlini_algo_common.hpp"
#include "mt.hpp"
#include "assume_aligned.hpp"

#include <gtest/gtest.h>


namespace
{


TEST(ClassicCalculateClauseOutput, replicates_result_of_CAIR_code)
{
    IRNG    irng(1234);

    for (auto it = 0u; it < 1000; /* nop */)
    {
        int const number_of_features = irng.next(1, 200);
        int const number_of_clauses = irng.next(1, 10) * 2; // must be even

        Tsetlini::aligned_vector_char X(number_of_features);

        std::generate(X.begin(), X.end(), [&irng](){ return irng.next(0, 1); });


        Tsetlini::numeric_matrix_int8 ta_state_matrix(2 * number_of_clauses, number_of_features);

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

        ta_state_gen(ta_state_matrix);

        Tsetlini::TAState::value_type ta_state;
        ta_state.matrix = ta_state_matrix;

        Tsetlini::aligned_vector_char clause_output_CAIR(number_of_clauses);
        Tsetlini::aligned_vector_char clause_output(number_of_clauses);

        CAIR::calculate_clause_output(X, clause_output_CAIR, number_of_clauses, number_of_features, ta_state_matrix, false);
        Tsetlini::calculate_clause_output(X, clause_output, 0, number_of_clauses, ta_state, 1, 16);

        if (0 != std::accumulate(clause_output_CAIR.cbegin(), clause_output_CAIR.cend(), 0u))
        {
            ++it;
        }

        EXPECT_TRUE(clause_output_CAIR == clause_output);
    }
}


TEST(ClassicCalculateClauseOutputForPredict, replicates_result_of_CAIR_code)
{
    IRNG    irng(1234);

    for (auto it = 0u; it < 1000; /* nop */)
    {
        int const number_of_features = irng.next(1, 200);
        int const number_of_clauses = irng.next(1, 10) * 2; // must be even

        Tsetlini::aligned_vector_char X(number_of_features);

        std::generate(X.begin(), X.end(), [&irng](){ return irng.next(0, 1); });

        Tsetlini::numeric_matrix_int8 ta_state_matrix(2 * number_of_clauses, number_of_features);

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

        ta_state_gen(ta_state_matrix);

        Tsetlini::TAState::value_type ta_state;
        ta_state.matrix = ta_state_matrix;

        Tsetlini::aligned_vector_char clause_output_CAIR(number_of_clauses);
        Tsetlini::aligned_vector_char clause_output(number_of_clauses);

        CAIR::calculate_clause_output(X, clause_output_CAIR, number_of_clauses, number_of_features, ta_state_matrix, true);
        Tsetlini::calculate_clause_output_for_predict(X, clause_output, number_of_clauses,
            ta_state, 1, 16);

        if (0 != std::accumulate(clause_output_CAIR.cbegin(), clause_output_CAIR.cend(), 0u))
        {
            ++it;
        }

        EXPECT_TRUE(clause_output_CAIR == clause_output);
    }
}


TEST(SumUpAllLabelVotes, replicates_result_of_CAIR_code)
{
    IRNG    irng(1234);

    for (auto it = 0u; it < 1000; ++it)
    {
        int const number_of_pos_neg_clauses = irng.next(1, 10) * 2; // must be even
        int const number_of_labels = irng.next(2, 12);
        int const threshold = irng.next(1, 127);

        Tsetlini::aligned_vector_char clause_output(number_of_pos_neg_clauses * number_of_labels);
        std::generate(clause_output.begin(), clause_output.end(), [&irng](){ return irng.next(0, 1); });

        Tsetlini::aligned_vector_int label_sum(number_of_labels);
        Tsetlini::aligned_vector_int label_sum_CAIR(number_of_labels);

        CAIR::sum_up_class_votes(clause_output, label_sum_CAIR, number_of_labels, number_of_pos_neg_clauses, threshold);
        Tsetlini::sum_up_all_label_votes(clause_output, label_sum, number_of_labels, number_of_pos_neg_clauses, threshold);

        EXPECT_TRUE(label_sum_CAIR == label_sum);
    }
}


TEST(CalculateClassifierFeedbackToClauses, replicates_result_of_CAIR_code)
{
    IRNG    irng(1234);
    FRNG    fgen(4567);
    FRNG    fgen_CAIR(4567);

    for (auto it = 0u; it < 1000; ++it)
    {
        int const number_of_labels = irng.next(2, 12);
        Tsetlini::label_type const target_label = irng.next(0, number_of_labels - 1);
        Tsetlini::label_type const opposite_label = (target_label + 1 + irng() % (number_of_labels - 1)) % number_of_labels;

        int const threshold = irng.next(1, 127);
        int const target_label_votes = irng.next(-threshold, threshold);
        int const opposite_label_votes = irng.next(-threshold, threshold);

        int const number_of_pos_neg_clauses_per_label = irng.next(1, 10) * 2; // must be even

        Tsetlini::feedback_vector_type feedback_to_clauses(number_of_pos_neg_clauses_per_label * number_of_labels);
        Tsetlini::feedback_vector_type feedback_to_clauses_CAIR(number_of_pos_neg_clauses_per_label * number_of_labels);

        CAIR::calculate_feedback_to_clauses(
            feedback_to_clauses_CAIR,
            target_label,
            opposite_label,
            target_label_votes,
            opposite_label_votes,
            number_of_pos_neg_clauses_per_label,
            number_of_labels,
            threshold,
            fgen_CAIR);

        Tsetlini::calculate_classifier_feedback_to_clauses(
            feedback_to_clauses,
            target_label,
            opposite_label,
            target_label_votes,
            opposite_label_votes,
            number_of_pos_neg_clauses_per_label,
            threshold,
            fgen);

        EXPECT_TRUE(feedback_to_clauses_CAIR == feedback_to_clauses);
    }
}


TEST(ClassicTrainClassifierAutomata, replicates_result_of_CAIR_code)
{
    IRNG    irng(1234);
    FRNG    fgen(4567);
    IRNG    prng(4567);
    FRNG    prng_CAIR(4567);

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

        Tsetlini::numeric_matrix_int8 ta_state_CAIR = ta_state;

        Tsetlini::feedback_vector_type feedback_to_clauses(number_of_clauses);
        std::generate(feedback_to_clauses.begin(), feedback_to_clauses.end(), [&irng](){ return irng.next(-1, +1); });

        Tsetlini::aligned_vector_char clause_output(number_of_clauses);
        std::generate(clause_output.begin(), clause_output.end(), [&irng](){ return irng.next(0, 1); });

        bool const boost_true_positive_feedback = irng.next(0, 1) != 0;
        /*
         * Setting S_inv to either 0.0 or 1.0 removes stochasticity from testing
         */
        char const ct_val = irng.next(0, 1);
        Tsetlini::real_type const S_inv = ct_val;
        Tsetlini::EstimatorStateCacheBase::coin_tosser_type ct(S_inv, number_of_features);


        CAIR::train_classifier_automata(
            ta_state_CAIR, 0, number_of_clauses, feedback_to_clauses.data(), clause_output.data(),
            number_of_features, number_of_states, S_inv, X.data(), boost_true_positive_feedback, prng_CAIR);

        Tsetlini::train_classifier_automata(
            ta_state, 0, number_of_clauses, feedback_to_clauses.data(), clause_output.data(),
            number_of_states, X, boost_true_positive_feedback, prng, ct);

        EXPECT_TRUE(ta_state == ta_state_CAIR);
    }
}


} // anonymous namespace
