#include "tsetlini_algo_classic.hpp"
#include "tsetlini_strong_params.hpp"
#include "tsetlini_types.hpp"

#include "strong_type/strong_type.hpp"
#include "rapidcheck.h"
#include "boost/ut.hpp"

#include <cstdlib>
#include <cstdint>


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


suite TrainClassifierAutomata = []
{


using matrix_type = Tsetlini::numeric_matrix_int16;


auto make_ta_state_matrix = [](
    Tsetlini::number_of_estimator_clause_outputs_t number_of_clause_outputs,
    Tsetlini::number_of_features_t number_of_features)
{
    matrix_type ta_state_matrix(2 * value_of(number_of_clause_outputs), value_of(number_of_features));

    // fill entire matrix storage space, regardless of alignment and padding
    ta_state_matrix.m_v =
        *rc::gen::container<matrix_type::aligned_vector>(ta_state_matrix.m_v.size(), rc::gen::inRange<matrix_type::value_type>(-MAX_NUM_OF_STATES, MAX_NUM_OF_STATES));

    return ta_state_matrix;
};


"Bytewise non-weighted train_classifier_automata does not modify state when all feedback is None"_test = [&]
{
    auto ok = rc::check(
        [&]()
        {
            IRNG prng(*rc::gen::arbitrary<int>());

            auto const number_of_features = Tsetlini::number_of_features_t{*rc::gen::inRange(1, MAX_NUM_OF_FEATURES + 1)};
            auto const number_of_clause_outputs = Tsetlini::number_of_estimator_clause_outputs_t{2 * *rc::gen::inRange(1, MAX_NUM_OF_CLAUSE_OUTPUTS / 2 + 1)};

            auto const number_of_states = Tsetlini::number_of_states_t{*rc::gen::inRange(1, MAX_NUM_OF_STATES + 1)};
            auto const boost_tpf = Tsetlini::boost_tpf_t{*rc::gen::arbitrary<bool>()};
            auto const S_inv = *rc::gen::suchThat(rc::gen::arbitrary<Tsetlini::real_type>(), [](auto x){ return x < 1.; });
            auto const max_weight = Tsetlini::max_weight_t{0};

            auto ta_state_reference = make_ta_state_matrix(number_of_clause_outputs, number_of_features);
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


"Bytewise weighted train_classifier_automata does not modify state when all feedback is None"_test = [&]
{
    auto ok = rc::check(
        [&]()
        {
            IRNG prng(*rc::gen::arbitrary<int>());

            auto const number_of_features = Tsetlini::number_of_features_t{*rc::gen::inRange(1, MAX_NUM_OF_FEATURES + 1)};
            auto const number_of_clause_outputs = Tsetlini::number_of_estimator_clause_outputs_t{2 * *rc::gen::inRange(1, MAX_NUM_OF_CLAUSE_OUTPUTS / 2 + 1)};

            auto const number_of_states = Tsetlini::number_of_states_t{*rc::gen::inRange(1, MAX_NUM_OF_STATES + 1)};
            auto const boost_tpf = Tsetlini::boost_tpf_t{*rc::gen::arbitrary<bool>()};
            auto const S_inv = *rc::gen::suchThat(rc::gen::arbitrary<Tsetlini::real_type>(), [](auto x){ return x < 1.; });

            auto ta_state_reference = make_ta_state_matrix(number_of_clause_outputs, number_of_features);
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


"Bytewise non-weighted train_classifier_automata does not modify state when all feedback is Type II and clause outputs are 0"_test = [&]
{
    auto ok = rc::check(
        [&]()
        {
            IRNG prng(*rc::gen::arbitrary<int>());

            auto const number_of_features = Tsetlini::number_of_features_t{*rc::gen::inRange(1, MAX_NUM_OF_FEATURES + 1)};
            auto const number_of_clause_outputs = Tsetlini::number_of_estimator_clause_outputs_t{2 * *rc::gen::inRange(1, MAX_NUM_OF_CLAUSE_OUTPUTS / 2 + 1)};

            auto const number_of_states = Tsetlini::number_of_states_t{*rc::gen::inRange(1, MAX_NUM_OF_STATES + 1)};
            auto const boost_tpf = Tsetlini::boost_tpf_t{*rc::gen::arbitrary<bool>()};
            auto const S_inv = *rc::gen::suchThat(rc::gen::arbitrary<Tsetlini::real_type>(), [](auto x){ return x < 1.; });
            auto const max_weight = Tsetlini::max_weight_t{0};

            auto ta_state_reference = make_ta_state_matrix(number_of_clause_outputs, number_of_features);
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


"Bytewise weighted train_classifier_automata does not modify state when all feedback is Type II and clause outputs are 0"_test = [&]
{
    auto ok = rc::check(
        [&]()
        {
            IRNG prng(*rc::gen::arbitrary<int>());

            auto const number_of_features = Tsetlini::number_of_features_t{*rc::gen::inRange(1, MAX_NUM_OF_FEATURES + 1)};
            auto const number_of_clause_outputs = Tsetlini::number_of_estimator_clause_outputs_t{2 * *rc::gen::inRange(1, MAX_NUM_OF_CLAUSE_OUTPUTS / 2 + 1)};

            auto const number_of_states = Tsetlini::number_of_states_t{*rc::gen::inRange(1, MAX_NUM_OF_STATES + 1)};
            auto const boost_tpf = Tsetlini::boost_tpf_t{*rc::gen::arbitrary<bool>()};
            auto const S_inv = *rc::gen::suchThat(rc::gen::arbitrary<Tsetlini::real_type>(), [](auto x){ return x < 1.; });

            auto ta_state_reference = make_ta_state_matrix(number_of_clause_outputs, number_of_features);
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


}; // suite TrainClassifierAutomata


int main()
{
    auto failed = cfg<>.run({.report_errors = true});

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
