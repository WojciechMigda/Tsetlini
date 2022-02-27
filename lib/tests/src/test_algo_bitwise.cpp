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


template<typename state_type, typename polarity_type>
void
polarity_from_ta_state(Tsetlini::numeric_matrix<state_type> const & ta_state, Tsetlini::bit_matrix<polarity_type> & polarity_matrix)
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
                polarity_matrix.clear(rix, cix);
            }
            else
            {
                polarity_matrix.set(rix, cix);
            }
        }
    }
}


TEST(BitwiseTrainClassifierAutomata, replicates_result_of_classic_code)
{
    IRNG    irng(1234);
    FRNG    fgen(4567);
    IRNG    prng(4567);
    IRNG    prng_classic(4567);
    Tsetlini::max_weight_t constexpr MAX_WEIGHT{10000000};

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

        Tsetlini::EstimatorStateCacheBase::coin_tosser_type ct(ct_val, number_of_features);
        ct.fill(ct_val);
        Tsetlini::EstimatorStateCacheBase::coin_tosser_type ct_classic = ct;

        Tsetlini::train_classifier_automata(
            ta_state_classic, weights, 0, number_of_clauses, feedback_to_clauses.data(), clause_output.data(),
            Tsetlini::number_of_states_t{number_of_states}, X, MAX_WEIGHT,
            Tsetlini::boost_tpf_t{boost_true_positive_feedback}, prng_classic, ct_classic);


        auto const bitwise_X = basic_bit_vectors::from_range<std::uint64_t>(X.cbegin(), X.cend());

        Tsetlini::bit_matrix_uint64 ta_state_polarity(2 * number_of_clauses, number_of_features);
        polarity_from_ta_state(ta_state_values, ta_state_polarity);

        // this will be fed to train_classifier_automata
        Tsetlini::TAStateWithPolarity::value_type ta_state;
        ta_state.polarity = ta_state_polarity;
        ta_state.matrix = ta_state_values;

        // mock prng which returns duplicated running integers modulo number of features
        // 0, 0, 1, 1, 2, 2, ...
        struct IotaPrng
        {
            using result_type = unsigned int;
            result_type iota_counter = 0u;
            int number_of_features;
            IotaPrng(int nfeat) : number_of_features(nfeat){}
            auto operator()()
            {
                result_type rv = iota_counter % (2 * number_of_features);
                ++iota_counter;
                return rv / 2;
            }
            result_type max() const { return std::numeric_limits<result_type>::max(); }
        } iota_prng(number_of_features);

        Tsetlini::train_classifier_automata(
            ta_state, 0, number_of_clauses, feedback_to_clauses.data(), clause_output.data(),
            Tsetlini::number_of_states_t{number_of_states}, bitwise_X, MAX_WEIGHT,
            Tsetlini::boost_tpf_t{boost_true_positive_feedback}, iota_prng, ct);

        // retrieve TA State values from ta_state variant for verifiation
        ta_state_values = std::get<Tsetlini::numeric_matrix_int8>(ta_state.matrix);
        EXPECT_TRUE(ta_state_values == ta_state_classic);

        // assert whether polarity was synchronized
        Tsetlini::bit_matrix_uint64 ta_state_polarity_post(2 * number_of_clauses, number_of_features);
        polarity_from_ta_state(ta_state_values, ta_state_polarity_post);

        EXPECT_TRUE(ta_state.polarity == ta_state_polarity_post);
    }
}


} // anonymous namespace
