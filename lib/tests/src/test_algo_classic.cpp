#include "cair_algos.hpp"

#include "estimator_state_cache.hpp"
#include "tsetlini_types.hpp"
#include "tsetlini_algo_classic.hpp"
#include "tsetlini_algo_common.hpp"
#include "tsetlini_strong_params.hpp"

#include "mt.hpp"
#include "assume_aligned.hpp"

#include "strong_type/strong_type.hpp"

#include <gtest/gtest.h>


namespace
{


TEST(ClassicTrainClassifierAutomata, replicates_result_of_CAIR_code)
{
    IRNG    irng(1234);
    FRNG    fgen(4567);
    IRNG    prng(4567);
    FRNG    prng_CAIR(4567);
    Tsetlini::max_weight_t constexpr MAX_WEIGHT{10000000};

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
        Tsetlini::w_vector_type weights;

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
            ta_state, weights, 0, number_of_clauses, feedback_to_clauses.data(), clause_output.data(),
            Tsetlini::number_of_states_t{number_of_states}, X, MAX_WEIGHT,
            Tsetlini::boost_tpf_t{boost_true_positive_feedback}, prng, ct);

        EXPECT_TRUE(ta_state == ta_state_CAIR);
    }
}


} // anonymous namespace
