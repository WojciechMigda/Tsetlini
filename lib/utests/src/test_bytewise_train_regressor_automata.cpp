#include "tsetlini_algo_classic.hpp"
#include "tsetlini_strong_params.hpp"
#include "tsetlini_strong_params_private.hpp"
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
auto constexpr MAX_THRESHOLD = 1024;

/*
 * `inRange` is exclusive on upper bound, so this is OK as max value for filling
 * weight test vectors.
 *
 *      weight = [0, MAX_WEIGHT)
 *
 * In real life scenario weight will never equal MAX_WEIGHT, because for
 * incrementation it is compared against `max_weight` after adding +1 to it.
 */
std::uint32_t constexpr MAX_WEIGHT = value_of(Tsetlini::MAX_WEIGHT_DEFAULT);
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

auto gen_threshold() -> Tsetlini::threshold_t
{
    return Tsetlini::threshold_t{*rc::gen::inRange(1, MAX_THRESHOLD + 1)};
}

auto gen_boost_tpf()
{
    return Tsetlini::boost_tpf_t{*rc::gen::arbitrary<bool>()};
}

auto gen_box_muller_flag()
{
    return Tsetlini::box_muller_flag_t{*rc::gen::arbitrary<bool>()};
}

auto gen_positive_response_error() -> Tsetlini::response_error_t
{
    using underlying_type = strong::underlying_type_t<Tsetlini::response_error_t>;
    using gen_type = long;

    static_assert(sizeof (underlying_type) < sizeof (gen_type));

    static auto constexpr MAX = gen_type{std::numeric_limits<underlying_type>::max()};

    return Tsetlini::response_error_t{*rc::gen::inRange<gen_type>(1, MAX + 1)};
}

auto gen_negative_response_error() -> Tsetlini::response_error_t
{
    using underlying_type = strong::underlying_type_t<Tsetlini::response_error_t>;
    using gen_type = long;

    static_assert(sizeof (underlying_type) < sizeof (gen_type));

    static auto constexpr MIN = gen_type{std::numeric_limits<underlying_type>::min()};

    return Tsetlini::response_error_t{*rc::gen::inRange<gen_type>(MIN - 1, -1)};
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
    return *rc::gen::container<Tsetlini::aligned_vector_char>(value_of(number_of_features), rc::gen::arbitrary<bool>());
}

auto gen_random_loss_fn = []()
{
    return [](float x)
        {
            return *rc::gen::arbitrary<float>();
        };
};


using matrix_type = Tsetlini::numeric_matrix_int16;


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


////////////////////////////////////////////////////////////////////////////////


suite TrainRegressorAutomata = []
{


/*
 * Response: 0
 * Clause outputs: n/a
 * X: n/a
 */

"Bytewise non-weighted train_regressor_automata"
" does not modify TA state"
" when response error is 0"_test = [&]
{
    auto ok = rc::check(
        [&]
        {
            IRNG prng(*rc::gen::arbitrary<int>());

            auto const number_of_features = gen_number_of_features();
            auto const number_of_clause_outputs = gen_number_of_clause_outputs();

            auto const number_of_states = gen_number_of_states();
            auto const boost_tpf = gen_boost_tpf();
            auto const threshold = gen_threshold();
            auto const box_muller_flag = gen_box_muller_flag();
            auto const S_inv = gen_S_inv();
            auto const max_weight = Tsetlini::max_weight_t{0};

            auto const ta_state_reference = gen_ta_state_matrix(number_of_clause_outputs, number_of_features, -value_of(number_of_states), value_of(number_of_states));
            auto const clause_output = gen_arbitrary_clause_output(number_of_clause_outputs);
            auto const X = gen_arbitrary_X(number_of_features);
            auto const loss_fn = gen_random_loss_fn();
            Tsetlini::w_vector_type empty_weights;

            Tsetlini::ClassifierStateCache::coin_tosser_type ct(S_inv, value_of(number_of_features));

            Tsetlini::response_error_t zero_response_error{0};

            matrix_type ta_state = ta_state_reference;

            Tsetlini::train_regressor_automata(
                ta_state,
                empty_weights,
                0, value_of(number_of_clause_outputs),
                clause_output.data(),
                number_of_states,
                zero_response_error,
                X,
                max_weight,
                loss_fn,
                box_muller_flag,
                boost_tpf, prng, threshold, ct);

            RC_ASSERT(ta_state.m_v == ta_state_reference.m_v);
        }
    );

    expect(that % true == ok);
};


"Bytewise weighted train_regressor_automata"
" does not modify TA state"
" when response error is 0"_test = [&]
{
    auto ok = rc::check(
        [&]
        {
            IRNG prng(*rc::gen::arbitrary<int>());

            auto const number_of_features = gen_number_of_features();
            auto const number_of_clause_outputs = gen_number_of_clause_outputs();

            auto const number_of_states = gen_number_of_states();
            auto const boost_tpf = gen_boost_tpf();
            auto const threshold = gen_threshold();
            auto const box_muller_flag = gen_box_muller_flag();
            auto const S_inv = gen_S_inv();

            auto const ta_state_reference = gen_ta_state_matrix(number_of_clause_outputs, number_of_features, -value_of(number_of_states), value_of(number_of_states));
            auto const clause_output = gen_arbitrary_clause_output(number_of_clause_outputs);
            auto const X = gen_arbitrary_X(number_of_features);
            auto const loss_fn = gen_random_loss_fn();
            auto const weights_reference = *rc::gen::container<Tsetlini::w_vector_type>(value_of(number_of_clause_outputs),
                rc::gen::inRange(MIN_WEIGHT, MAX_WEIGHT));

            Tsetlini::ClassifierStateCache::coin_tosser_type ct(S_inv, value_of(number_of_features));

            Tsetlini::response_error_t zero_response_error{0};

            matrix_type ta_state = ta_state_reference;
            Tsetlini::w_vector_type weights = weights_reference;

            Tsetlini::train_regressor_automata(
                ta_state,
                weights,
                0, value_of(number_of_clause_outputs),
                clause_output.data(),
                number_of_states,
                zero_response_error,
                X,
                Tsetlini::max_weight_t{MAX_WEIGHT},
                loss_fn,
                box_muller_flag,
                boost_tpf, prng, threshold, ct);

            RC_ASSERT(ta_state.m_v == ta_state_reference.m_v);
            RC_ASSERT(weights == weights_reference);
        }
    );

    expect(that % true == ok);
};


/*
 * Response error: Positive
 * Clause outputs: 0
 * X: n/a
 */

"Bytewise non-weighted train_regressor_automata"
" does not modify TA state"
" when response error is positive"
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
            auto const threshold = gen_threshold();
            auto const box_muller_flag = gen_box_muller_flag();
            auto const S_inv = gen_S_inv();
            auto const max_weight = Tsetlini::max_weight_t{0};

            auto const ta_state_reference = gen_ta_state_matrix(number_of_clause_outputs, number_of_features, -value_of(number_of_states), value_of(number_of_states));
            auto const X = gen_arbitrary_X(number_of_features);
            auto const loss_fn = gen_random_loss_fn();
            Tsetlini::w_vector_type empty_weights;

            Tsetlini::ClassifierStateCache::coin_tosser_type ct(S_inv, value_of(number_of_features));

            Tsetlini::aligned_vector_char const clause_output(value_of(number_of_clause_outputs), 0);

            auto const response_error = gen_positive_response_error();

            matrix_type ta_state = ta_state_reference;

            Tsetlini::train_regressor_automata(
                ta_state,
                empty_weights,
                0, value_of(number_of_clause_outputs),
                clause_output.data(),
                number_of_states,
                response_error,
                X,
                max_weight,
                loss_fn,
                box_muller_flag,
                boost_tpf, prng, threshold, ct);

            RC_ASSERT(ta_state.m_v == ta_state_reference.m_v);
        }
    );

    expect(that % true == ok);
};


"Bytewise weighted train_regressor_automata"
" does not modify TA state nor weights"
" when response error is positive"
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
            auto const threshold = gen_threshold();
            auto const box_muller_flag = gen_box_muller_flag();
            auto const S_inv = gen_S_inv();

            auto const ta_state_reference = gen_ta_state_matrix(number_of_clause_outputs, number_of_features, -value_of(number_of_states), value_of(number_of_states));
            auto const X = gen_arbitrary_X(number_of_features);
            auto const loss_fn = gen_random_loss_fn();
            auto const weights_reference = *rc::gen::container<Tsetlini::w_vector_type>(value_of(number_of_clause_outputs),
                rc::gen::inRange(MIN_WEIGHT, MAX_WEIGHT));

            Tsetlini::ClassifierStateCache::coin_tosser_type ct(S_inv, value_of(number_of_features));

            Tsetlini::aligned_vector_char const clause_output(value_of(number_of_clause_outputs), 0);

            auto const response_error = gen_positive_response_error();

            matrix_type ta_state = ta_state_reference;
            Tsetlini::w_vector_type weights = weights_reference;

            Tsetlini::train_regressor_automata(
                ta_state,
                weights,
                0, value_of(number_of_clause_outputs),
                clause_output.data(),
                number_of_states,
                response_error,
                X,
                Tsetlini::max_weight_t{MAX_WEIGHT},
                loss_fn,
                box_muller_flag,
                boost_tpf, prng, threshold, ct);

            RC_ASSERT(ta_state.m_v == ta_state_reference.m_v);
            RC_ASSERT(weights == weights_reference);
        }
    );

    expect(that % true == ok);
};


}; // suite TrainRegressorAutomata


int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
