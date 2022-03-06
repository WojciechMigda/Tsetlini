#include "tsetlini_algo_bitwise.hpp"
#include "tsetlini_algo_classic.hpp" // TODO: move Feedback enum out of this header
#include "tsetlini_strong_params.hpp"
#include "tsetlini_types.hpp"

#include "strong_type/strong_type.hpp"
#include "rapidcheck.h"
#include "boost/ut.hpp"

#include <cstdlib>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cmath>


using namespace boost::ut;


auto constexpr MAX_NUM_OF_FEATURES = 800;
auto constexpr MAX_NUM_OF_CLAUSE_OUTPUTS = 16;
auto constexpr MAX_NUM_OF_STATES = 1000;

/*
 * `inRange` is exclusive on upper bound, so this is OK as max value for filling
 * weight test vectors.
 *
 *      weight = [0, MAX_WEIGHT)
 *
 * In real life scenario weight will never equal MAX_WEIGHT, because for
 * incrementation it is compared against `max_weight` after adding +1 to it.
 */
std::uint32_t constexpr MAX_WEIGHT = std::numeric_limits<strong::underlying_type<Tsetlini::max_weight_t>::type>::max();
std::uint32_t constexpr MIN_WEIGHT = 0;


/*
 * generate random integer in closed range [lo, hi]
 */
template<typename T, typename Gen>
T random_int(Gen & gen, T lo, T hi)
{
    return std::uniform_int_distribution<T>(lo, hi)(gen);
};


auto gen_number_of_features(int max_num_of_features = MAX_NUM_OF_FEATURES) -> Tsetlini::number_of_features_t
{
    return Tsetlini::number_of_features_t{*rc::gen::inRange(1, max_num_of_features + 1)};
}

auto gen_number_of_clause_outputs(int max_num_of_clause_outputs = MAX_NUM_OF_CLAUSE_OUTPUTS) -> Tsetlini::number_of_estimator_clause_outputs_t
{
    return Tsetlini::number_of_estimator_clause_outputs_t{2 * *rc::gen::inRange(1, max_num_of_clause_outputs / 2 + 1)};
}

auto gen_number_of_states() -> Tsetlini::number_of_states_t
{
    return Tsetlini::number_of_states_t{*rc::gen::inRange(1, MAX_NUM_OF_STATES + 1)};
}

auto gen_boost_tpf()
{
    return Tsetlini::boost_tpf_t{*rc::gen::arbitrary<bool>()};
}

auto gen_S_inv() -> Tsetlini::real_type
{
    return *rc::gen::map(rc::gen::arbitrary<std::uint32_t>(), [](auto x){ return (x + 0.5f) * (1.0f / 4294967296.0f); });
}

auto gen_arbitrary_clause_output(Tsetlini::number_of_estimator_clause_outputs_t number_of_clause_outputs)
{
    return *rc::gen::container<Tsetlini::aligned_vector_char>(value_of(number_of_clause_outputs), rc::gen::arbitrary<bool>());
}

auto gen_arbitrary_X(Tsetlini::number_of_features_t number_of_features)
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
}

using matrix_type = Tsetlini::numeric_matrix_int16;
using polarity_matrix_type = Tsetlini::bit_matrix_uint64;

auto gen_ta_state_matrix(
    Tsetlini::number_of_estimator_clause_outputs_t number_of_clause_outputs,
    Tsetlini::number_of_features_t number_of_features,
    int const lo_closed,
    int const hi_open) -> matrix_type
{
    matrix_type ta_state_matrix(2 * value_of(number_of_clause_outputs), value_of(number_of_features));

    // fill entire matrix storage space, regardless of alignment and padding
    ta_state_matrix.m_v =
        *rc::gen::container<matrix_type::aligned_vector>(ta_state_matrix.m_v.size(), rc::gen::inRange<matrix_type::value_type>(lo_closed, hi_open));

    return ta_state_matrix;
}

auto make_polarity_matrix_from(matrix_type const & ta_state_matrix)
{
    auto const [nrows, ncols] = ta_state_matrix.shape();

    polarity_matrix_type polarity(nrows, ncols);

    for (auto rix = 0u; rix < nrows; ++rix)
    {
        for (auto cix = 0u; cix < ncols; ++cix)
        {
            // x >= 0  --> 1
            // x < 0   --> 0
            auto const negative = ta_state_matrix[{rix, cix}] < 0;

            if (negative)
            {
                polarity.clear(rix, cix);
            }
            else
            {
                polarity.set(rix, cix);
            }
        }
    }

    return polarity;
};


////////////////////////////////////////////////////////////////////////////////


suite TrainClassifierAutomata = []
{


/*
 * Feedback: None
 * Clause outputs: n/a
 * X: n/a
 */

"Bitwise non-weighted train_classifier_automata"
" does not modify TA state"
" when all feedback is None"_test = [&]
{
    auto ok = rc::check(
        [&]
        {
            IRNG prng(*rc::gen::arbitrary<int>());

            auto const number_of_features = gen_number_of_features();
            auto const number_of_clause_outputs = gen_number_of_clause_outputs();

            auto const number_of_states = gen_number_of_states();
            auto const boost_tpf = gen_boost_tpf();
            auto const S_inv = gen_S_inv();
            auto const max_weight = Tsetlini::max_weight_t{0};

            auto const ta_state_reference = gen_ta_state_matrix(number_of_clause_outputs, number_of_features, -value_of(number_of_states), value_of(number_of_states));
            auto const polarity_reference = make_polarity_matrix_from(ta_state_reference);
            auto const clause_output = gen_arbitrary_clause_output(number_of_clause_outputs);
            auto const X = gen_arbitrary_X(number_of_features);
            Tsetlini::w_vector_type empty_weights;

            Tsetlini::ClassifierStateCache::coin_tosser_type ct(S_inv, value_of(number_of_features));

            Tsetlini::feedback_vector_type const feedback_to_clauses(value_of(number_of_clause_outputs), Tsetlini::No_Feedback);

            matrix_type ta_state = ta_state_reference;
            polarity_matrix_type polarity = polarity_reference;

            Tsetlini::train_classifier_automata(
                ta_state, polarity,
                empty_weights,
                0, value_of(number_of_clause_outputs),
                feedback_to_clauses.data(),
                clause_output.data(),
                number_of_states, X, max_weight,
                boost_tpf, prng, ct);

            RC_ASSERT(ta_state.m_v == ta_state_reference.m_v);
            RC_ASSERT(polarity.m_v == polarity_reference.m_v);
        }
    );

    expect(that % true == ok);
};


"Bitwise weighted train_classifier_automata"
" does not modify TA state nor weights"
" when all feedback is None"_test = [&]
{
    auto ok = rc::check(
        [&]
        {
            IRNG prng(*rc::gen::arbitrary<int>());

            auto const number_of_features = gen_number_of_features();
            auto const number_of_clause_outputs = gen_number_of_clause_outputs();

            auto const number_of_states = gen_number_of_states();
            auto const boost_tpf = gen_boost_tpf();
            auto const S_inv = gen_S_inv();

            auto const ta_state_reference = gen_ta_state_matrix(number_of_clause_outputs, number_of_features, -value_of(number_of_states), value_of(number_of_states));
            auto const polarity_reference = make_polarity_matrix_from(ta_state_reference);
            auto const clause_output = gen_arbitrary_clause_output(number_of_clause_outputs);
            auto const X = gen_arbitrary_X(number_of_features);
            auto const weights_reference = *rc::gen::container<Tsetlini::w_vector_type>(value_of(number_of_clause_outputs),
                rc::gen::inRange(MIN_WEIGHT, MAX_WEIGHT));

            Tsetlini::ClassifierStateCache::coin_tosser_type ct(S_inv, value_of(number_of_features));

            Tsetlini::feedback_vector_type const feedback_to_clauses(value_of(number_of_clause_outputs), Tsetlini::No_Feedback);

            matrix_type ta_state = ta_state_reference;
            polarity_matrix_type polarity = polarity_reference;
            Tsetlini::w_vector_type weights = weights_reference;

            Tsetlini::train_classifier_automata(
                ta_state, polarity,
                weights,
                0, value_of(number_of_clause_outputs),
                feedback_to_clauses.data(),
                clause_output.data(),
                number_of_states, X,
                Tsetlini::max_weight_t{MAX_WEIGHT},
                boost_tpf, prng, ct);

            RC_ASSERT(ta_state.m_v == ta_state_reference.m_v);
            RC_ASSERT(polarity.m_v == polarity_reference.m_v);
            RC_ASSERT(weights == weights_reference);
        }
    );

    expect(that % true == ok);
};


/*
 * Feedback: Type II
 * Clause outputs: 0
 * X: n/a
 */

"Bitwise non-weighted train_classifier_automata"
" does not modify TA state"
" when all feedback is Type II"
" and clause outputs are 0"_test = [&]
{
    auto ok = rc::check(
        [&]
        {
            IRNG prng(*rc::gen::arbitrary<int>());

            auto const number_of_features = gen_number_of_features();
            auto const number_of_clause_outputs = gen_number_of_clause_outputs();

            auto const number_of_states = gen_number_of_states();
            auto const boost_tpf = gen_boost_tpf();
            auto const S_inv = gen_S_inv();
            auto const max_weight = Tsetlini::max_weight_t{0};

            auto const ta_state_reference = gen_ta_state_matrix(number_of_clause_outputs, number_of_features, -value_of(number_of_states), value_of(number_of_states));
            auto const polarity_reference = make_polarity_matrix_from(ta_state_reference);
            auto const X = gen_arbitrary_X(number_of_features);
            Tsetlini::w_vector_type empty_weights;

            Tsetlini::ClassifierStateCache::coin_tosser_type ct(S_inv, value_of(number_of_features));

            Tsetlini::aligned_vector_char const clause_output(value_of(number_of_clause_outputs), 0);
            Tsetlini::feedback_vector_type const feedback_to_clauses(value_of(number_of_clause_outputs), Tsetlini::Type_II_Feedback);

            matrix_type ta_state = ta_state_reference;
            polarity_matrix_type polarity = polarity_reference;

            Tsetlini::train_classifier_automata(
                ta_state, polarity,
                empty_weights,
                0, value_of(number_of_clause_outputs),
                feedback_to_clauses.data(),
                clause_output.data(),
                number_of_states, X, max_weight,
                boost_tpf, prng, ct);

            RC_ASSERT(ta_state.m_v == ta_state_reference.m_v);
            RC_ASSERT(polarity.m_v == polarity_reference.m_v);
        }
    );

    expect(that % true == ok);
};


"Bitwise weighted train_classifier_automata"
" does not modify TA state nor weights"
" when all feedback is Type II"
" and clause outputs are 0"_test = [&]
{
    auto ok = rc::check(
        [&]
        {
            IRNG prng(*rc::gen::arbitrary<int>());

            auto const number_of_features = gen_number_of_features();
            auto const number_of_clause_outputs = gen_number_of_clause_outputs();

            auto const number_of_states = gen_number_of_states();
            auto const boost_tpf = gen_boost_tpf();
            auto const S_inv = gen_S_inv();

            auto const ta_state_reference = gen_ta_state_matrix(number_of_clause_outputs, number_of_features, -value_of(number_of_states), value_of(number_of_states));
            auto const polarity_reference = make_polarity_matrix_from(ta_state_reference);
            auto const X = gen_arbitrary_X(number_of_features);
            auto const weights_reference = *rc::gen::container<Tsetlini::w_vector_type>(value_of(number_of_clause_outputs),
                rc::gen::inRange(MIN_WEIGHT, MAX_WEIGHT));

            Tsetlini::ClassifierStateCache::coin_tosser_type ct(S_inv, value_of(number_of_features));

            Tsetlini::aligned_vector_char const clause_output(value_of(number_of_clause_outputs), 0);
            Tsetlini::feedback_vector_type const feedback_to_clauses(value_of(number_of_clause_outputs), Tsetlini::Type_II_Feedback);

            matrix_type ta_state = ta_state_reference;
            polarity_matrix_type polarity = polarity_reference;
            Tsetlini::w_vector_type weights = weights_reference;

            Tsetlini::train_classifier_automata(
                ta_state, polarity,
                weights,
                0, value_of(number_of_clause_outputs),
                feedback_to_clauses.data(),
                clause_output.data(),
                number_of_states, X,
                Tsetlini::max_weight_t{MAX_WEIGHT},
                boost_tpf, prng, ct);

            RC_ASSERT(ta_state.m_v == ta_state_reference.m_v);
            RC_ASSERT(polarity.m_v == polarity_reference.m_v);
            RC_ASSERT(weights == weights_reference);
        }
    );

    expect(that % true == ok);
};


"Bitwise weighted train_classifier_automata"
" decrements weights"
" when all feedback is Type II"
" and clause outputs are 1"_test = [&]
{
    auto ok = rc::check(
        [&]
        {
            IRNG prng(*rc::gen::arbitrary<int>());

            auto const number_of_features = gen_number_of_features();
            auto const number_of_clause_outputs = gen_number_of_clause_outputs();

            auto const number_of_states = gen_number_of_states();
            auto const boost_tpf = gen_boost_tpf();
            auto const S_inv = gen_S_inv();

            auto const ta_state_reference = gen_ta_state_matrix(number_of_clause_outputs, number_of_features, -value_of(number_of_states), value_of(number_of_states));
            auto const polarity_reference = make_polarity_matrix_from(ta_state_reference);
            auto const X = gen_arbitrary_X(number_of_features);
            auto const weights_reference = *rc::gen::container<Tsetlini::w_vector_type>(value_of(number_of_clause_outputs),
                rc::gen::inRange(MIN_WEIGHT + 1, MAX_WEIGHT));

            Tsetlini::ClassifierStateCache::coin_tosser_type ct(S_inv, value_of(number_of_features));

            Tsetlini::aligned_vector_char const clause_output(value_of(number_of_clause_outputs), 1);
            Tsetlini::feedback_vector_type const feedback_to_clauses(value_of(number_of_clause_outputs), Tsetlini::Type_II_Feedback);

            matrix_type ta_state = ta_state_reference;
            polarity_matrix_type polarity = polarity_reference;
            Tsetlini::w_vector_type weights = weights_reference;

            Tsetlini::train_classifier_automata(
                ta_state, polarity,
                weights,
                0, value_of(number_of_clause_outputs),
                feedback_to_clauses.data(),
                clause_output.data(),
                number_of_states, X,
                Tsetlini::max_weight_t{MAX_WEIGHT},
                boost_tpf, prng, ct);

            /* increment weights so that they can be compared against reference */
            std::for_each(weights.begin(), weights.end(), [](auto & x){ x += 1; });
            RC_ASSERT(weights == weights_reference);
        }
    );

    expect(that % true == ok);
};


"Bitwise weighted train_classifier_automata"
" does not decrement zero weights"
" when all feedback is Type II"
" and clause outputs are 1"_test = [&]
{
    auto ok = rc::check(
        [&]
        {
            IRNG prng(*rc::gen::arbitrary<int>());

            auto const number_of_features = gen_number_of_features();
            auto const number_of_clause_outputs = gen_number_of_clause_outputs();

            auto const number_of_states = gen_number_of_states();
            auto const boost_tpf = gen_boost_tpf();
            auto const S_inv = gen_S_inv();

            auto const ta_state_reference = gen_ta_state_matrix(number_of_clause_outputs, number_of_features, -value_of(number_of_states), value_of(number_of_states));
            auto const polarity_reference = make_polarity_matrix_from(ta_state_reference);
            auto const X = gen_arbitrary_X(number_of_features);
            Tsetlini::w_vector_type zero_weights(value_of(number_of_clause_outputs), MIN_WEIGHT);

            Tsetlini::ClassifierStateCache::coin_tosser_type ct(S_inv, value_of(number_of_features));

            Tsetlini::aligned_vector_char const clause_output(value_of(number_of_clause_outputs), 1);
            Tsetlini::feedback_vector_type const feedback_to_clauses(value_of(number_of_clause_outputs), Tsetlini::Type_II_Feedback);

            matrix_type ta_state = ta_state_reference;
            polarity_matrix_type polarity = polarity_reference;

            Tsetlini::train_classifier_automata(
                ta_state, polarity,
                zero_weights,
                0, value_of(number_of_clause_outputs),
                feedback_to_clauses.data(),
                clause_output.data(),
                number_of_states, X,
                Tsetlini::max_weight_t{MAX_WEIGHT},
                boost_tpf, prng, ct);

            RC_ASSERT(std::all_of(zero_weights.cbegin(), zero_weights.cend(), [](auto x){ return x == MIN_WEIGHT; }));
        }
    );

    expect(that % true == ok);
};


}; // suite TrainClassifierAutomata


int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
