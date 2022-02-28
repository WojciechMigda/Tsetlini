#include "tsetlini_algo_classic.hpp"
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
 * In real life scenario weight will never MAX_WEIGHT, because for
 * incrementation it is compared against `max_weight` after adding +1 to it.
 */
std::uint64_t constexpr MAX_WEIGHT = std::numeric_limits<Tsetlini::w_vector_type::value_type>::max();


/*
 * generate random integer in closed range [lo, hi]
 */
template<typename T, typename Gen>
T random_int(Gen & gen, T lo, T hi)
{
    return std::uniform_int_distribution<T>(lo, hi)(gen);
};


////////////////////////////////////////////////////////////////////////////////


suite TrainClassifierAutomata = []
{


using matrix_type = Tsetlini::numeric_matrix_int16;


auto gen_ta_state_matrix = [](
    Tsetlini::number_of_estimator_clause_outputs_t number_of_clause_outputs,
    Tsetlini::number_of_features_t number_of_features)
{
    matrix_type ta_state_matrix(2 * value_of(number_of_clause_outputs), value_of(number_of_features));

    // fill entire matrix storage space, regardless of alignment and padding
    ta_state_matrix.m_v =
        *rc::gen::container<matrix_type::aligned_vector>(ta_state_matrix.m_v.size(), rc::gen::inRange<matrix_type::value_type>(-MAX_NUM_OF_STATES, MAX_NUM_OF_STATES));

    return ta_state_matrix;
};


auto aggregate_diff = [](
    matrix_type const & ta_state_matrix,
    matrix_type const & reference_matrix,
    auto & diff)
{
    for (auto rix = 0u; rix < ta_state_matrix.rows(); ++rix)
    {
        for (auto cix = 0u; cix < ta_state_matrix.cols(); ++cix)
        {
            diff.row_data(rix)[cix] += (ta_state_matrix[{rix, cix}] - reference_matrix[{rix, cix}]);
        }
    }
};


auto make_ta_state_matrix = [](
    auto && gen,
    Tsetlini::number_of_estimator_clause_outputs_t number_of_clause_outputs,
    Tsetlini::number_of_features_t number_of_features)
{
    matrix_type ta_state_matrix(2 * value_of(number_of_clause_outputs), value_of(number_of_features));

    // fill entire matrix storage space, regardless of alignment and padding
    std::generate(ta_state_matrix.m_v.begin(), ta_state_matrix.m_v.end(), gen);

    return ta_state_matrix;
};


"Bytewise non-weighted train_classifier_automata does not modify TA state when all feedback is None"_test = [&]
{
    auto ok = rc::check(
        [&]
        {
            IRNG prng(*rc::gen::arbitrary<int>());

            auto const number_of_features = Tsetlini::number_of_features_t{*rc::gen::inRange(1, MAX_NUM_OF_FEATURES + 1)};
            auto const number_of_clause_outputs = Tsetlini::number_of_estimator_clause_outputs_t{2 * *rc::gen::inRange(1, MAX_NUM_OF_CLAUSE_OUTPUTS / 2 + 1)};

            auto const number_of_states = Tsetlini::number_of_states_t{*rc::gen::inRange(1, MAX_NUM_OF_STATES + 1)};
            auto const boost_tpf = Tsetlini::boost_tpf_t{*rc::gen::arbitrary<bool>()};
            auto const S_inv = *rc::gen::suchThat(rc::gen::arbitrary<Tsetlini::real_type>(), [](auto x){ return x < 1.; });
            auto const max_weight = Tsetlini::max_weight_t{0};

            auto ta_state_reference = gen_ta_state_matrix(number_of_clause_outputs, number_of_features);
            auto const clause_output = *rc::gen::container<Tsetlini::aligned_vector_char>(value_of(number_of_clause_outputs), rc::gen::arbitrary<bool>());
            auto const X = *rc::gen::container<Tsetlini::aligned_vector_char>(value_of(number_of_features), rc::gen::arbitrary<bool>());
            Tsetlini::w_vector_type empty_weights;

            Tsetlini::ClassifierStateCache::coin_tosser_type ct(S_inv, value_of(number_of_features));

            Tsetlini::feedback_vector_type const feedback_to_clauses(value_of(number_of_clause_outputs), Tsetlini::No_Feedback);

            matrix_type ta_state = ta_state_reference;

            Tsetlini::train_classifier_automata(
                ta_state,
                empty_weights,
                0, value_of(number_of_clause_outputs),
                feedback_to_clauses.data(),
                clause_output.data(),
                number_of_states, X, max_weight,
                boost_tpf, prng, ct);

            RC_ASSERT(ta_state.m_v == ta_state_reference.m_v);
        }
    );

    expect(that % true == ok);
};


"Bytewise weighted train_classifier_automata does not modify TA state when all feedback is None"_test = [&]
{
    auto ok = rc::check(
        [&]
        {
            IRNG prng(*rc::gen::arbitrary<int>());

            auto const number_of_features = Tsetlini::number_of_features_t{*rc::gen::inRange(1, MAX_NUM_OF_FEATURES + 1)};
            auto const number_of_clause_outputs = Tsetlini::number_of_estimator_clause_outputs_t{2 * *rc::gen::inRange(1, MAX_NUM_OF_CLAUSE_OUTPUTS / 2 + 1)};

            auto const number_of_states = Tsetlini::number_of_states_t{*rc::gen::inRange(1, MAX_NUM_OF_STATES + 1)};
            auto const boost_tpf = Tsetlini::boost_tpf_t{*rc::gen::arbitrary<bool>()};
            auto const S_inv = *rc::gen::suchThat(rc::gen::arbitrary<Tsetlini::real_type>(), [](auto x){ return x < 1.; });

            auto ta_state_reference = gen_ta_state_matrix(number_of_clause_outputs, number_of_features);
            auto const clause_output = *rc::gen::container<Tsetlini::aligned_vector_char>(value_of(number_of_clause_outputs), rc::gen::arbitrary<bool>());
            auto const X = *rc::gen::container<Tsetlini::aligned_vector_char>(value_of(number_of_features), rc::gen::arbitrary<bool>());
            auto weights = *rc::gen::container<Tsetlini::w_vector_type>(value_of(number_of_clause_outputs), rc::gen::inRange<std::uint64_t>(0, MAX_WEIGHT));

            Tsetlini::ClassifierStateCache::coin_tosser_type ct(S_inv, value_of(number_of_features));

            Tsetlini::feedback_vector_type const feedback_to_clauses(value_of(number_of_clause_outputs), Tsetlini::No_Feedback);

            matrix_type ta_state = ta_state_reference;

            Tsetlini::train_classifier_automata(
                ta_state,
                weights,
                0, value_of(number_of_clause_outputs),
                feedback_to_clauses.data(),
                clause_output.data(),
                number_of_states, X,
                Tsetlini::max_weight_t{MAX_WEIGHT},
                boost_tpf, prng, ct);

            RC_ASSERT(ta_state.m_v == ta_state_reference.m_v);
        }
    );

    expect(that % true == ok);
};


"Bytewise non-weighted train_classifier_automata does not modify TA state"
" when all feedback is Type II and clause outputs are 0"_test = [&]
{
    auto ok = rc::check(
        [&]
        {
            IRNG prng(*rc::gen::arbitrary<int>());

            auto const number_of_features = Tsetlini::number_of_features_t{*rc::gen::inRange(1, MAX_NUM_OF_FEATURES + 1)};
            auto const number_of_clause_outputs = Tsetlini::number_of_estimator_clause_outputs_t{2 * *rc::gen::inRange(1, MAX_NUM_OF_CLAUSE_OUTPUTS / 2 + 1)};

            auto const number_of_states = Tsetlini::number_of_states_t{*rc::gen::inRange(1, MAX_NUM_OF_STATES + 1)};
            auto const boost_tpf = Tsetlini::boost_tpf_t{*rc::gen::arbitrary<bool>()};
            auto const S_inv = *rc::gen::suchThat(rc::gen::arbitrary<Tsetlini::real_type>(), [](auto x){ return x < 1.; });
            auto const max_weight = Tsetlini::max_weight_t{0};

            auto ta_state_reference = gen_ta_state_matrix(number_of_clause_outputs, number_of_features);
            auto const X = *rc::gen::container<Tsetlini::aligned_vector_char>(value_of(number_of_features), rc::gen::arbitrary<bool>());
            Tsetlini::w_vector_type empty_weights;

            Tsetlini::ClassifierStateCache::coin_tosser_type ct(S_inv, value_of(number_of_features));

            Tsetlini::aligned_vector_char const clause_output(value_of(number_of_clause_outputs), 0);
            Tsetlini::feedback_vector_type const feedback_to_clauses(value_of(number_of_clause_outputs), Tsetlini::Type_II_Feedback);

            matrix_type ta_state = ta_state_reference;

            Tsetlini::train_classifier_automata(
                ta_state,
                empty_weights,
                0, value_of(number_of_clause_outputs),
                feedback_to_clauses.data(),
                clause_output.data(),
                number_of_states, X, max_weight,
                boost_tpf, prng, ct);

            RC_ASSERT(ta_state.m_v == ta_state_reference.m_v);
        }
    );

    expect(that % true == ok);
};


"Bytewise weighted train_classifier_automata does not modify TA state when"
" all feedback is Type II and clause outputs are 0"_test = [&]
{
    auto ok = rc::check(
        [&]
        {
            IRNG prng(*rc::gen::arbitrary<int>());

            auto const number_of_features = Tsetlini::number_of_features_t{*rc::gen::inRange(1, MAX_NUM_OF_FEATURES + 1)};
            auto const number_of_clause_outputs = Tsetlini::number_of_estimator_clause_outputs_t{2 * *rc::gen::inRange(1, MAX_NUM_OF_CLAUSE_OUTPUTS / 2 + 1)};

            auto const number_of_states = Tsetlini::number_of_states_t{*rc::gen::inRange(1, MAX_NUM_OF_STATES + 1)};
            auto const boost_tpf = Tsetlini::boost_tpf_t{*rc::gen::arbitrary<bool>()};
            auto const S_inv = *rc::gen::suchThat(rc::gen::arbitrary<Tsetlini::real_type>(), [](auto x){ return x < 1.; });

            auto ta_state_reference = gen_ta_state_matrix(number_of_clause_outputs, number_of_features);
            auto const X = *rc::gen::container<Tsetlini::aligned_vector_char>(value_of(number_of_features), rc::gen::arbitrary<bool>());
            auto weights = *rc::gen::container<Tsetlini::w_vector_type>(value_of(number_of_clause_outputs), rc::gen::inRange<std::uint64_t>(0, MAX_WEIGHT));

            Tsetlini::ClassifierStateCache::coin_tosser_type ct(S_inv, value_of(number_of_features));

            Tsetlini::aligned_vector_char const clause_output(value_of(number_of_clause_outputs), 0);
            Tsetlini::feedback_vector_type const feedback_to_clauses(value_of(number_of_clause_outputs), Tsetlini::Type_II_Feedback);

            matrix_type ta_state = ta_state_reference;

            Tsetlini::train_classifier_automata(
                ta_state,
                weights,
                0, value_of(number_of_clause_outputs),
                feedback_to_clauses.data(),
                clause_output.data(),
                number_of_states, X,
                Tsetlini::max_weight_t{MAX_WEIGHT},
                boost_tpf, prng, ct);

            RC_ASSERT(ta_state.m_v == ta_state_reference.m_v);
        }
    );

    expect(that % true == ok);
};


"Bytewise non-weighted train_classifier_automata adjusts TA states with 1/s probability when"
" feedback is Type I and clause outputs are 0"_test = [&]
{
    /*
     * override few limits for faster execution
     */
    auto constexpr MAX_NUM_OF_FEATURES = 400;
    auto constexpr MAX_NUM_OF_CLAUSE_OUTPUTS = 8;

    /*
     * Begin with a PRNG section
     */
    std::random_device rd;
    std::mt19937 gen(rd());

    auto const seed = rd();
    IRNG prng(seed);

    /*
     * Initialize few random constants for the algorithm
     */
    auto const number_of_features = Tsetlini::number_of_features_t{random_int(gen, 1, MAX_NUM_OF_FEATURES)};
    auto const number_of_clause_outputs = Tsetlini::number_of_estimator_clause_outputs_t{2 * random_int(gen, 1, MAX_NUM_OF_CLAUSE_OUTPUTS / 2)};

    auto const number_of_states = Tsetlini::number_of_states_t{random_int(gen, 1, MAX_NUM_OF_STATES)};
    auto const boost_tpf = Tsetlini::boost_tpf_t{random_int(gen, 0, 1)};
    auto const S_inv = std::uniform_real_distribution<>(0.f, 1.f)(gen);

    Tsetlini::aligned_vector_char const X(value_of(number_of_features));
    Tsetlini::w_vector_type empty_weights;

    Tsetlini::ClassifierStateCache::coin_tosser_type ct(S_inv, value_of(number_of_features));

    Tsetlini::aligned_vector_char const clause_output(value_of(number_of_clause_outputs), 0);
    Tsetlini::feedback_vector_type const feedback_to_clauses(value_of(number_of_clause_outputs), Tsetlini::Type_I_Feedback);
    auto const ta_state_reference = make_ta_state_matrix(
        /*
         * training is expected to decrement state, so we need to fill TA state
         * with random numbers in range open on number_of_states:
         *
         *      (-number_of_states, number_of_states]
         */
        [&](){ return random_int(gen, -value_of(number_of_states) + 1, value_of(number_of_states)); },
        number_of_clause_outputs, number_of_features);

    /*
     * Here we will aggregate differences between ta_state and its base reference
     */
    Tsetlini::numeric_matrix_int32 diff(2 * value_of(number_of_clause_outputs), value_of(number_of_features));

    /*
     * Repeatedly call the algorithm and aggregate differences to the state
     */
    auto N_REPEAT = 16'000u * (value_of(number_of_clause_outputs) + 1);

    for (auto it = 0u; it < N_REPEAT; ++it)
    {
        Tsetlini::numeric_matrix_int16 ta_state = ta_state_reference;

        Tsetlini::train_classifier_automata(
            ta_state,
            empty_weights,
            0, value_of(number_of_clause_outputs),
            feedback_to_clauses.data(),
            clause_output.data(),
            number_of_states, X,
            Tsetlini::max_weight_t{MAX_WEIGHT},
            boost_tpf, prng, ct);

        aggregate_diff(ta_state, ta_state_reference, diff);
    }

    /*
     * This is the target average value given TA state would be decreased by
     */
    int target = -std::round(N_REPEAT * S_inv);

    /*
     * Check that no TA state element deviates from that target by more than
     * a margin of N_REPEAT / 100.
     */
    bool all_ok = true;
    for (auto rix = 0u; rix < diff.rows(); ++rix)
    {
        auto within_margin = [target, margin = std::round(N_REPEAT / 100)](auto x){ return (target - margin) <= x and x <= (target + margin); };

        auto begin = diff.row_data(rix);
        auto end = diff.row_data(rix) + diff.cols();
        auto where_failed = std::find_if_not(begin, end, within_margin);

        if (where_failed != end)
        {
            if (all_ok)
            {
                // log this only on first failure
                boost::ut::log << "Number of rows: " << diff.rows();
                boost::ut::log << "Number of columns: " << diff.cols();
                boost::ut::log << "1 / s: " << S_inv;
                boost::ut::log << "Target decrease: " << target;
            }
            boost::ut::log << "Failed element row/col: " << *where_failed << " @ [" << rix << ", " << (where_failed - begin) << ']';
        }

        all_ok = all_ok and (where_failed == end);
    }
    expect(that % true == all_ok);
};


"Bytewise weighted train_classifier_automata adjusts TA states with 1/s probability"
" and leaves weights unchanged when feedback is Type I and clause outputs are 0"_test = [&]
{
    /*
     * override few limits for faster execution
     */
    auto constexpr MAX_NUM_OF_FEATURES = 400;
    auto constexpr MAX_NUM_OF_CLAUSE_OUTPUTS = 8;

    /*
     * Begin with a PRNG section
     */
    std::random_device rd;
    std::mt19937 gen(rd());

    auto const seed = rd();
    IRNG prng(seed);

    /*
     * Initialize few random constants for the algorithm
     */
    auto const number_of_features = Tsetlini::number_of_features_t{random_int(gen, 1, MAX_NUM_OF_FEATURES)};
    auto const number_of_clause_outputs = Tsetlini::number_of_estimator_clause_outputs_t{2 * random_int(gen, 1, MAX_NUM_OF_CLAUSE_OUTPUTS / 2)};

    auto const number_of_states = Tsetlini::number_of_states_t{random_int(gen, 1, MAX_NUM_OF_STATES)};
    auto const boost_tpf = Tsetlini::boost_tpf_t{random_int(gen, 0, 1)};
    auto const S_inv = std::uniform_real_distribution<>(0.f, 1.f)(gen);

    Tsetlini::aligned_vector_char const X(value_of(number_of_features));

    Tsetlini::ClassifierStateCache::coin_tosser_type ct(S_inv, value_of(number_of_features));

    Tsetlini::aligned_vector_char const clause_output(value_of(number_of_clause_outputs), 0);
    Tsetlini::feedback_vector_type const feedback_to_clauses(value_of(number_of_clause_outputs), Tsetlini::Type_I_Feedback);
    Tsetlini::w_vector_type const weights_reference(value_of(number_of_clause_outputs), random_int(gen, 0u, std::uint32_t(MAX_WEIGHT - 1)));
    auto const ta_state_reference = make_ta_state_matrix(
        /*
         * training is expected to decrement state, so we need to fill TA state
         * with random numbers in range open on number_of_states:
         *
         *      (-number_of_states, number_of_states]
         */
        [&](){ return random_int(gen, -value_of(number_of_states) + 1, value_of(number_of_states)); },
        number_of_clause_outputs, number_of_features);

    /*
     * Here we will aggregate differences between ta_state and its base reference
     */
    Tsetlini::numeric_matrix_int32 diff(2 * value_of(number_of_clause_outputs), value_of(number_of_features));
    bool all_weights_unchanged = true;

    /*
     * Repeatedly call the algorithm and aggregate differences to the state
     */
    auto N_REPEAT = 16'000u * (value_of(number_of_clause_outputs) + 1);

    for (auto it = 0u; it < N_REPEAT; ++it)
    {
        Tsetlini::numeric_matrix_int16 ta_state = ta_state_reference;
        Tsetlini::w_vector_type weights = weights_reference;

        Tsetlini::train_classifier_automata(
            ta_state,
            weights,
            0, value_of(number_of_clause_outputs),
            feedback_to_clauses.data(),
            clause_output.data(),
            number_of_states, X,
            Tsetlini::max_weight_t{MAX_WEIGHT},
            boost_tpf, prng, ct);

        aggregate_diff(ta_state, ta_state_reference, diff);

        all_weights_unchanged = all_weights_unchanged and (weights == weights_reference);
    }

    expect(that % true == all_weights_unchanged);

    /*
     * This is the target average value given TA state would be decreased by
     */
    int target = -std::round(N_REPEAT * S_inv);

    /*
     * Check that no TA state element deviates from that target by more than
     * a margin of N_REPEAT / 100.
     */
    bool all_within_margin = true;
    for (auto rix = 0u; rix < diff.rows(); ++rix)
    {
        auto within_margin = [target, margin = std::round(N_REPEAT / 100)](auto x){ return (target - margin) <= x and x <= (target + margin); };

        auto begin = diff.row_data(rix);
        auto end = diff.row_data(rix) + diff.cols();
        auto where_failed = std::find_if_not(begin, end, within_margin);

        if (where_failed != end)
        {
            if (all_within_margin)
            {
                // log this only on first failure
                boost::ut::log << "Number of rows: " << diff.rows();
                boost::ut::log << "Number of columns: " << diff.cols();
                boost::ut::log << "1 / s: " << S_inv;
                boost::ut::log << "Target decrease: " << target;
            }
            boost::ut::log << "Failed element row/col: " << *where_failed << " @ [" << rix << ", " << (where_failed - begin) << ']';
        }

        all_within_margin = all_within_margin and (where_failed == end);
    }
    expect(that % true == all_within_margin);
};


}; // suite TrainClassifierAutomata


int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
