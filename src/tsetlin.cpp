#define LOG_MODULE "tsetlin"
#include "logger.hpp"

#include "tsetlin_params.hpp"
#include "tsetlin.hpp"
#include "mt.hpp"
#include "assume_aligned.hpp"
#include "params_companion.hpp"
#include "tsetlin_types.hpp"
#include "tsetlin_state.hpp"
#include "tsetlin_status_code.hpp"
#include "tsetlin_classifier_state_private.hpp"

#include "neither/either.hpp"

#include <utility>
#include <algorithm>
#include <cstddef>
#include <unordered_set>
#include <string>
#include <numeric>


using namespace neither;
using namespace std::string_literals;


namespace Tsetlin
{

namespace
{


inline
int pos_feat_index(int k)
{
    return k;
}


inline
int neg_feat_index(int k, int number_of_features)
{
    return k + number_of_features;
}


inline
bool action(int state, int number_of_states)
{
    return state > number_of_states;
}


inline
int pos_clause_index(int target_label, int j, int number_of_pos_neg_clauses_per_label)
{
    return 2 * target_label * number_of_pos_neg_clauses_per_label + j;
}


inline
int neg_clause_index(int target_label, int j, int number_of_pos_neg_clauses_per_label)
{
    return pos_clause_index(target_label, j, number_of_pos_neg_clauses_per_label) + number_of_pos_neg_clauses_per_label;
}


inline
void sum_up_label_votes(
    aligned_vector_char const & clause_output,
    aligned_vector_int & label_sum,
    int target_label,

    int const number_of_pos_neg_clauses_per_label,
    int const threshold)
{
    label_sum[target_label] = 0;

    for (int j = 0; j < number_of_pos_neg_clauses_per_label; ++j)
    {
        label_sum[target_label] += clause_output[pos_clause_index(target_label, j, number_of_pos_neg_clauses_per_label)];
    }

    for (int j = 0; j < number_of_pos_neg_clauses_per_label; ++j)
    {
        label_sum[target_label] -= clause_output[neg_clause_index(target_label, j, number_of_pos_neg_clauses_per_label)];
    }
    label_sum[target_label] = std::clamp(label_sum[target_label], -threshold, threshold);
}


inline
void sum_up_all_class_votes(
    aligned_vector_char const & clause_output,
    aligned_vector_int & label_sum,

    int const number_of_labels,
    int const number_of_pos_neg_clauses_per_label,
    int const threshold)
{
    for (int target_label = 0; target_label < number_of_labels; ++target_label)
    {
        sum_up_label_votes(clause_output, label_sum, target_label, number_of_pos_neg_clauses_per_label, threshold);
    }
}


inline
void calculate_clause_output(
    aligned_vector_char const & X,
    aligned_vector_char & clause_output,
    bool predict,
    int const number_of_clauses,
    int const number_of_features,
    int const number_of_states,
    std::vector<aligned_vector_int> const & ta_state)
{
    char const * X_p = assume_aligned<alignment>(X.data());
//    char const * clause_output_p = assume_aligned<alignment>(clause_output.data());

    for (int j = 0; j < number_of_clauses; ++j)
    {
        bool output = true;
        bool all_exclude = true;

        int const * ta_state_j = assume_aligned<alignment>(ta_state[j].data());

        for (int k = 0; k < number_of_features and output == true; ++k)
        {
            bool const action_include = action(ta_state_j[pos_feat_index(k)], number_of_states);
            bool const action_include_negated = action(ta_state_j[neg_feat_index(k, number_of_features)], number_of_states);

            all_exclude = (action_include == true or action_include_negated == true) ? false : all_exclude;

            output = ((action_include == true and X_p[k] == 0) or (action_include_negated == true and X_p[k] != 0)) ? false : output;
        }

        output = (predict == true and all_exclude == true) ? false : output;

        clause_output[j] = output;
    }
}


// Feedback Type I, negative
int block1(
    int const number_of_features,
    float const S_inv,
    int * __restrict ta_state_j,
    float const * __restrict fcache,
    int fcache_pos
)
{
    fcache = assume_aligned<alignment>(fcache);
    ta_state_j = assume_aligned<alignment>(ta_state_j);

    for (int k = 0; k < number_of_features; ++k)
    {
        {
            auto cond = fcache[fcache_pos++] <= S_inv;
            auto tix = pos_feat_index(k);
// slower
//            if (cond)
//            {
//                ta_state_j[tix] = (ta_state_j[tix] > 1 ? ta_state_j[tix] - 1 : ta_state_j[tix]);
//            }

            ta_state_j[tix] = cond ? (ta_state_j[tix] > 1 ? ta_state_j[tix] - 1 : ta_state_j[tix]) : ta_state_j[tix];
        }

        {
            auto cond = fcache[fcache_pos++] <= S_inv;
            auto tix = neg_feat_index(k, number_of_features);
// slower
//            if (cond)
//            {
//                ta_state_j[tix] = (ta_state_j[tix] > 1 ? ta_state_j[tix] - 1 : ta_state_j[tix]);
//            }
            ta_state_j[tix] = cond ? (ta_state_j[tix] > 1 ? ta_state_j[tix] - 1 : ta_state_j[tix]) : ta_state_j[tix];
        }
    }
    return fcache_pos;
}


// Feedback Type I, positive
template<bool boost_true_positive_feedback>
int block2(
    int const number_of_features,
    int const number_of_states,
    float const S_inv,
    int * __restrict ta_state_j,
    char const * __restrict X,
    float const * __restrict fcache,
    int fcache_pos
)
{
    constexpr float ONE = 1.0f;
    fcache = assume_aligned<alignment>(fcache);
    ta_state_j = assume_aligned<alignment>(ta_state_j);
//    X = assume_aligned(X);

    for (int k = 0; k < number_of_features; ++k)
    {
        auto cond1 = boost_true_positive_feedback == true or (fcache[fcache_pos++] <= (ONE - S_inv));
        auto cond2 = fcache[fcache_pos++] <= S_inv;

        if (X[k] != 0)
        {
            if (cond1)
            {
                if (ta_state_j[pos_feat_index(k)] < number_of_states * 2)
                {
                    ta_state_j[pos_feat_index(k)]++;
                }
            }
            if (cond2)
            {
                if (ta_state_j[neg_feat_index(k, number_of_features)] > 1)
                {
                    ta_state_j[neg_feat_index(k, number_of_features)]--;
                }
            }
        }
        else // if (X[k] == 0)
        {
            if (cond1)
            {
                if (ta_state_j[neg_feat_index(k, number_of_features)] < number_of_states * 2)
                {
                    ta_state_j[neg_feat_index(k, number_of_features)]++;
                }
            }

            if (cond2)
            {
                if (ta_state_j[pos_feat_index(k)] > 1)
                {
                    ta_state_j[pos_feat_index(k)]--;
                }
            }
        }
    }

    return fcache_pos;
}


// Feedback Type II
void block3(
    int const number_of_features,
    int const number_of_states,
    int * __restrict ta_state_j,
    char const * __restrict X
)
{
    ta_state_j = assume_aligned<alignment>(ta_state_j);
    X = assume_aligned<alignment>(X);

    for (int k = 0; k < number_of_features; ++k)
    {
        if (X[k] == 0)
        {
            auto tix = pos_feat_index(k);
            auto action_include = (ta_state_j[tix]) > number_of_states;
            if (action_include == false)
            {
                ta_state_j[tix]++;
            }
        }
        else //if(X[k] == 1)
        {
            auto tix = neg_feat_index(k, number_of_features);
            auto action_include_negated = (ta_state_j[tix]) > number_of_states;
            if (action_include_negated == false)
            {
                ta_state_j[tix]++;
            }
        }
    }
#if 0
    for (int k = 0; k < number_of_features; ++k)
    {
        bool const action_include = action(ta_state[j][pos_feat_index(k)]);
        bool const action_include_negated = action(ta_state[j][neg_feat_index(k)]);

        if (X[k] == 0)
        {
            if (action_include == false and ta_state[j][pos_feat_index(k)] < number_of_states * 2)
            {
                ta_state[j][pos_feat_index(k)]++;
            }
        }
        else if(X[k] == 1)
        {
            if (action_include_negated == false and ta_state[j][neg_feat_index(k)] < number_of_states * 2)
            {
                ta_state[j][neg_feat_index(k)]++;
            }
        }
    }
#endif
}



void train_automata_batch(
    aligned_vector_int * __restrict ta_state,
    int const begin,
    int const end,
    feedback_vector_type::value_type const * __restrict feedback_to_clauses,
    char const * __restrict clause_output,
    int const number_of_features,
    int const number_of_states,
    float const S_inv,
    char const * __restrict X,
    bool const boost_true_positive_feedback,
    ClassifierState::frand_cache_type & fcache
    )
{
    float const * fcache_ = assume_aligned<alignment>(fcache.m_fcache.data());

    for (int j = begin; j < end; ++j)
    {
        int * ta_state_j = ::assume_aligned<alignment>(ta_state[j].data());

        if (feedback_to_clauses[j] > 0)
        {
            if (clause_output[j] == 0)
            {
                fcache.refill();

                fcache.m_pos = block1(number_of_features, S_inv, ta_state_j, fcache_, fcache.m_pos);
            }
            else if (clause_output[j] == 1)
            {
                fcache.refill();

                if (boost_true_positive_feedback)
                    fcache.m_pos = block2<true>(number_of_features, number_of_states, S_inv, ta_state_j, X, fcache_, fcache.m_pos);
                else
                    fcache.m_pos = block2<false>(number_of_features, number_of_states, S_inv, ta_state_j, X, fcache_, fcache.m_pos);
            }
        }
        else if (feedback_to_clauses[j] < 0)
        {
            if (clause_output[j] == 1)
            {
                block3(number_of_features, number_of_states, ta_state_j, X);
            }
        }
    }
}


void update_impl(
    aligned_vector_char const & X,
    label_type target_label,

    int const number_of_labels,
    int const number_of_pos_neg_clauses_per_label,
    int const threshold,
    int const number_of_clauses,
    int const number_of_features,
    int const number_of_states,
    real_type s,
    int const boost_true_positive_feedback,

    IRNG & igen,
    FRNG & fgen,
    std::vector<aligned_vector_int> & ta_state,
    ClassifierState::Cache & cache
    )
{
    // Randomly pick one of the other classes, for pairwise learning of class output
    int negative_target_label = igen.next(0, number_of_labels - 1);
    while (negative_target_label == target_label)
    {
        negative_target_label = igen.next(0, number_of_labels - 1);
    }

    calculate_clause_output(
        X,
        cache.clause_output,
        false,
        number_of_clauses,
        number_of_features,
        number_of_states,
        ta_state
    );

    sum_up_label_votes(
        cache.clause_output,
        cache.label_sum,
        target_label,
        number_of_pos_neg_clauses_per_label,
        threshold);

    sum_up_label_votes(
        cache.clause_output,
        cache.label_sum,
        negative_target_label,
        number_of_pos_neg_clauses_per_label,
        threshold);


    std::fill(cache.feedback_to_clauses.begin(), cache.feedback_to_clauses.end(), 0);


    const auto S_inv = ONE / s;

    const auto THR2_inv = (ONE / (threshold * 2));
    const auto THR_pos = THR2_inv * (threshold - cache.label_sum[target_label]);
    const auto THR_neg = THR2_inv * (threshold + cache.label_sum[negative_target_label]);

    for (int j = 0; j < number_of_pos_neg_clauses_per_label; ++j)
    {
        if (fgen.next() > THR_pos)
        {
            continue;
        }

        // Type I Feedback
        cache.feedback_to_clauses[pos_clause_index(target_label, j, number_of_pos_neg_clauses_per_label)]++;
    }
    for (int j = 0; j < number_of_pos_neg_clauses_per_label; ++j)
    {
        if (fgen.next() > THR_pos)
        {
            continue;
        }

        // Type II Feedback
        cache.feedback_to_clauses[neg_clause_index(target_label, j, number_of_pos_neg_clauses_per_label)]--;
    }

    for (int j = 0; j < number_of_pos_neg_clauses_per_label; ++j)
    {
        if (fgen.next() > THR_neg)
        {
            continue;
        }

        cache.feedback_to_clauses[pos_clause_index(negative_target_label, j, number_of_pos_neg_clauses_per_label)]--;
    }
    for (int j = 0; j < number_of_pos_neg_clauses_per_label; ++j)
    {
        if (fgen.next() > THR_neg)
        {
            continue;
        }

        cache.feedback_to_clauses[neg_clause_index(negative_target_label, j, number_of_pos_neg_clauses_per_label)]++;
    }


    train_automata_batch(
        ta_state.data(),
        0,
        number_of_clauses,
        cache.feedback_to_clauses.data(),
        cache.clause_output.data(),
        number_of_features,
        number_of_states,
        S_inv,
        X.data(),
        boost_true_positive_feedback,
        cache.fcache[0]
    );
}


Either<status_message_t, std::unordered_set<label_type>>
unique_labels(label_vector_type const & y)
{
    if (y.size() == 0u)
    {
        return Either<status_message_t, std::unordered_set<label_type>>::leftOf(
            {S_BAD_LABELS, "Labels are empty"s});
    }

    std::unordered_set<label_type> uniq(y.cbegin(), y.cend());

    auto const [lo, hi] = std::minmax_element(uniq.cbegin(), uniq.cend());

    if (*lo != 0)
    {
        return Either<status_message_t, std::unordered_set<label_type>>::leftOf(
            {S_BAD_LABELS, "Smallest label is not zero: "s + std::to_string(*lo)});
    }
    else if (*hi >= ssize_t(uniq.size()))
    {
        return Either<status_message_t, std::unordered_set<label_type>>::leftOf(
            {S_BAD_LABELS, "Unique labels are not a contiguous set"});
    }
    else if (uniq.size() == 1u)
    {
        return Either<status_message_t, std::unordered_set<label_type>>::leftOf(
            {S_BAD_LABELS, "One label is too few"});
    }
    else
    {
        return Either<status_message_t, std::unordered_set<label_type>>::rightOf(uniq);
    }
}


Either<status_message_t, real_type>
evaluate_impl(
    ClassifierState const & state,
    std::vector<aligned_vector_char> const & X,
    label_vector_type const & y)
{
    // let it crash - no state validation for now

    auto const number_of_examples = X.size();

    auto const & params = state.m_params;

    auto const number_of_labels = Params::number_of_labels(params);
    auto const number_of_pos_neg_clauses_per_label = Params::number_of_pos_neg_clauses_per_label(params);
    auto const threshold = Params::threshold(params);
    auto const number_of_clauses = Params::number_of_clauses(params);
    auto const number_of_features = Params::number_of_features(params);
    auto const number_of_states = Params::number_of_states(params);

    int errors = 0;

    for (auto it = 0u; it < number_of_examples; ++it)
    {
        calculate_clause_output(
            X[it],
            state.cache.clause_output,
            true,
            number_of_clauses,
            number_of_features,
            number_of_states,
            state.ta_state
        );

        sum_up_all_class_votes(
            state.cache.clause_output,
            state.cache.label_sum,
            number_of_labels,
            number_of_pos_neg_clauses_per_label,
            threshold);


        const int max_class = std::distance(
            state.cache.label_sum.cbegin(),
            std::max_element(state.cache.label_sum.cbegin(), state.cache.label_sum.cend()));

        errors += (max_class != y[it]);
    }

    real_type const rv = ONE - ONE * errors / number_of_examples;

    return Either<status_message_t, real_type>::rightOf(rv);
}


Either<status_message_t, label_type>
predict_impl(ClassifierState const & state, aligned_vector_char const & sample)
{
    calculate_clause_output(
        sample,
        state.cache.clause_output,
        true,
        Params::number_of_clauses(state.m_params),
        Params::number_of_features(state.m_params),
        Params::number_of_states(state.m_params),
        state.ta_state
    );

    sum_up_all_class_votes(
        state.cache.clause_output,
        state.cache.label_sum,
        Params::number_of_labels(state.m_params),
        Params::number_of_pos_neg_clauses_per_label(state.m_params),
        Params::threshold(state.m_params));

    label_type rv = std::distance(
        state.cache.label_sum.cbegin(),
        std::max_element(state.cache.label_sum.cbegin(), state.cache.label_sum.cend()));

    return Either<status_message_t, label_type>::rightOf(rv);
}


} // anonymous


Either<status_message_t, label_vector_type>
predict_impl(ClassifierState const & state, std::vector<aligned_vector_char> const & X)
{
    // let it crash - no state validation for now

    auto const number_of_examples = X.size();

    auto const & params = state.m_params;

    auto const number_of_labels = Params::number_of_labels(params);
    auto const number_of_pos_neg_clauses_per_label = Params::number_of_pos_neg_clauses_per_label(params);
    auto const threshold = Params::threshold(params);
    auto const number_of_clauses = Params::number_of_clauses(params);
    auto const number_of_features = Params::number_of_features(params);
    auto const number_of_states = Params::number_of_states(params);

    label_vector_type rv(number_of_examples);

    for (auto it = 0u; it < number_of_examples; ++it)
    {
        calculate_clause_output(
            X[it],
            state.cache.clause_output,
            true,
            number_of_clauses,
            number_of_features,
            number_of_states,
            state.ta_state
        );

        sum_up_all_class_votes(
            state.cache.clause_output,
            state.cache.label_sum,
            number_of_labels,
            number_of_pos_neg_clauses_per_label,
            threshold);


        const int max_class = std::distance(
            state.cache.label_sum.cbegin(),
            std::max_element(state.cache.label_sum.cbegin(), state.cache.label_sum.cend()));

        rv[it] = max_class;
    }

    return Either<status_message_t, label_vector_type>::rightOf(rv);
}


Either<status_message_t, aligned_vector_int>
predict_raw_impl(ClassifierState const & state, aligned_vector_char const & sample)
{
    // let it crash - no state validation for now

    auto const & params = state.m_params;

    auto const number_of_labels = Params::number_of_labels(params);
    auto const number_of_pos_neg_clauses_per_label = Params::number_of_pos_neg_clauses_per_label(params);
    auto const threshold = Params::threshold(params);
    auto const number_of_clauses = Params::number_of_clauses(params);
    auto const number_of_features = Params::number_of_features(params);
    auto const number_of_states = Params::number_of_states(params);

    calculate_clause_output(
        sample,
        state.cache.clause_output,
        true,
        number_of_clauses,
        number_of_features,
        number_of_states,
        state.ta_state
    );

    sum_up_all_class_votes(
        state.cache.clause_output,
        state.cache.label_sum,
        number_of_labels,
        number_of_pos_neg_clauses_per_label,
        threshold);

    return Either<status_message_t, aligned_vector_int>::rightOf(state.cache.label_sum);
}


Either<status_message_t, std::vector<aligned_vector_int>>
predict_raw_impl(ClassifierState const & state, std::vector<aligned_vector_char> const & X)
{
    // let it crash - no state validation for now

    auto const number_of_examples = X.size();

    auto const & params = state.m_params;

    auto const number_of_labels = Params::number_of_labels(params);
    auto const number_of_pos_neg_clauses_per_label = Params::number_of_pos_neg_clauses_per_label(params);
    auto const threshold = Params::threshold(params);
    auto const number_of_clauses = Params::number_of_clauses(params);
    auto const number_of_features = Params::number_of_features(params);
    auto const number_of_states = Params::number_of_states(params);

    std::vector<aligned_vector_int> rv(number_of_examples);

    for (auto it = 0u; it < number_of_examples; ++it)
    {
        calculate_clause_output(
            X[it],
            state.cache.clause_output,
            true,
            number_of_clauses,
            number_of_features,
            number_of_states,
            state.ta_state
        );

        sum_up_all_class_votes(
            state.cache.clause_output,
            state.cache.label_sum,
            number_of_labels,
            number_of_pos_neg_clauses_per_label,
            threshold);

        rv[it] = state.cache.label_sum;
    }

    return Either<status_message_t, std::vector<aligned_vector_int>>::rightOf(rv);
}


status_message_t
fit_online_impl(
    ClassifierState & state,
    std::vector<aligned_vector_char> const & X,
    label_vector_type const & y,
    unsigned int epochs)
{
    auto const number_of_examples = X.size();

    std::vector<int> ix(number_of_examples);
    std::iota(ix.begin(), ix.end(), 0);

    auto const & params = state.m_params;

    auto const number_of_labels = Params::number_of_labels(params);
    auto const number_of_pos_neg_clauses_per_label = Params::number_of_pos_neg_clauses_per_label(params);
    auto const threshold = Params::threshold(params);
    auto const number_of_clauses = Params::number_of_clauses(params);
    auto const number_of_features = Params::number_of_features(params);
    auto const number_of_states = Params::number_of_states(params);
    auto const s = Params::s(params);
    auto const boost_true_positive_feedback = Params::boost_true_positive_feedback(params);

    std::mt19937 gen(state.igen());

    for (unsigned int epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(ix.begin(), ix.end(), gen);

        for (auto i = 0u; i < number_of_examples; ++i)
        {
            update_impl(
                X[ix[i]],
                y[ix[i]],

                number_of_labels,
                number_of_pos_neg_clauses_per_label,
                threshold,
                number_of_clauses,
                number_of_features,
                number_of_states,
                s,
                boost_true_positive_feedback,

                state.igen,
                state.fgen,
                state.ta_state,
                state.cache
            );
        }
    }

    return {S_OK, ""};
}


status_message_t
partial_fit_impl(
    ClassifierState & state,
    std::vector<aligned_vector_char> const & X,
    label_vector_type const & y,
    unsigned int epochs)
{
    // TODO do verification whether we've fit anything before and either
    // do fit_online_impl or fit_impl
    return fit_online_impl(state, X, y, epochs);
}


status_message_t
fit_impl(
    ClassifierState & state,
    std::vector<aligned_vector_char> const & X,
    label_vector_type const & y,
    int max_number_of_labels,
    unsigned int epochs)
{
    (void)unique_labels;
//    auto const labels = unique_labels(y);
//    auto const number_of_labels = labels.rightMap([](auto const & labels) -> int { return labels.size(); });
//
//    validate_params();
//
//    initialize_state();


    // let it crash - no input validation for now
    {
        int const number_of_labels = std::max(*std::max_element(y.cbegin(), y.cend()) + 1, max_number_of_labels);
        state.m_params["number_of_labels"] = param_value_t(number_of_labels);

        int const number_of_features = X.front().size();
        state.m_params["number_of_features"] = param_value_t(number_of_features);

        initialize_state(state);
    }

    return fit_online_impl(state, X, y, epochs);
}


Classifier::Classifier(params_t const & params) :
    m_state(params)
{
}


Classifier::Classifier(params_t && params) :
    m_state(params)
{
}


Either<status_message_t, label_type>
Classifier::predict(aligned_vector_char const & sample) const
{
    return predict_impl(m_state, sample);
}


Either<status_message_t, label_vector_type>
Classifier::predict(std::vector<aligned_vector_char> const & X) const
{
    return predict_impl(m_state, X);
}


Either<status_message_t, aligned_vector_int>
Classifier::predict_raw(aligned_vector_char const & sample) const
{
    return predict_raw_impl(m_state, sample);
}


Either<status_message_t, std::vector<aligned_vector_int>>
Classifier::predict_raw(std::vector<aligned_vector_char> const & X) const
{
    return predict_raw_impl(m_state, X);
}


#if 0
aligned_vector_int Classifier::predict_raw(aligned_vector_char const & sample) const
{
    calculate_clause_output(
        sample,
        state.cache.clause_output,
        true,
        Config::number_of_clauses(state.config),
        Config::number_of_features(state.config),
        Config::number_of_states(state.config),
        state.ta_state
    );

    sum_up_all_class_votes(
        state.cache.clause_output,
        state.cache.label_sum,
        Config::number_of_labels(state.config),
        Config::number_of_pos_neg_clauses_per_label(state.config),
        Config::threshold(state.config));

    return state.cache.label_sum;
}


void Classifier::predict_raw(aligned_vector_char const & sample, int * out_p) const
{
    if (out_p == nullptr)
    {
        return;
    }

    calculate_clause_output(
        sample,
        state.cache.clause_output,
        true,
        Config::number_of_clauses(state.config),
        Config::number_of_features(state.config),
        Config::number_of_states(state.config),
        state.ta_state
    );

    sum_up_all_class_votes(
        state.cache.clause_output,
        state.cache.label_sum,
        Config::number_of_labels(state.config),
        Config::number_of_pos_neg_clauses_per_label(state.config),
        Config::threshold(state.config));

    std::copy(state.cache.label_sum.cbegin(), state.cache.label_sum.cend(), out_p);
}


void Classifier::update(aligned_vector_char const & X, label_type target_label)
{
    auto const & config = state.config;

    update_impl(
        X,
        target_label,

        Config::number_of_labels(config),
        Config::number_of_pos_neg_clauses_per_label(config),
        Config::threshold(config),
        Config::number_of_clauses(config),
        Config::number_of_features(config),
        Config::number_of_states(config),
        Config::s(config),
        Config::boost_true_positive_feedback(config),

        state.igen,
        state.fgen,
        state.ta_state,
        state.cache
    );
}


void Classifier::fit_batch(std::vector<aligned_vector_char> const & X, label_vector_type const & y)
{
    auto const & config = state.config;

    for (auto i = 0u; i < std::min(X.size(), y.size()); ++i)
    {
        update_impl(
            X[i],
            y[i],

            Config::number_of_labels(config),
            Config::number_of_pos_neg_clauses_per_label(config),
            Config::threshold(config),
            Config::number_of_clauses(config),
            Config::number_of_features(config),
            Config::number_of_states(config),
            Config::s(config),
            Config::boost_true_positive_feedback(config),

            state.igen,
            state.fgen,
            state.ta_state,
            state.cache
        );
    }
}


void Classifier::fit(std::vector<aligned_vector_char> const & X, label_vector_type const & y, std::size_t number_of_examples, int epochs)
{
    std::vector<int> ix(X.size());
    std::iota(ix.begin(), ix.end(), 0);

    auto const & config = state.config;

    auto const number_of_labels = Config::number_of_labels(config);
    auto const number_of_pos_neg_clauses_per_label = Config::number_of_pos_neg_clauses_per_label(config);
    auto const threshold = Config::threshold(config);
    auto const number_of_clauses = Config::number_of_clauses(config);
    auto const number_of_features = Config::number_of_features(config);
    auto const number_of_states = Config::number_of_states(config);
    auto const s = Config::s(config);
    auto const boost_true_positive_feedback = Config::boost_true_positive_feedback(config);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(ix.begin(), ix.end(), std::mt19937(state.gen));

        for (auto i = 0u; i < number_of_examples; ++i)
        {
            update_impl(
                X[ix[i]],
                y[ix[i]],

                number_of_labels,
                number_of_pos_neg_clauses_per_label,
                threshold,
                number_of_clauses,
                number_of_features,
                number_of_states,
                s,
                boost_true_positive_feedback,

                state.igen,
                state.fgen,
                state.ta_state,
                state.cache
            );
        }
    }

}


#endif


Either<status_message_t, real_type>
Classifier::evaluate(std::vector<aligned_vector_char> const & X, label_vector_type const & y) const
{
    return evaluate_impl(m_state, X, y);
}


status_message_t
Classifier::partial_fit(std::vector<aligned_vector_char> const & X, label_vector_type const & y, int epochs)
{
    return partial_fit_impl(m_state, X, y, epochs);
}


status_message_t
Classifier::fit(std::vector<aligned_vector_char> const & X, label_vector_type const & y, int max_number_of_labels, unsigned int epochs)
{
    return fit_impl(m_state, X, y, max_number_of_labels, epochs);
}


params_t Classifier::read_params() const
{
    return m_state.m_params;
}


ClassifierState Classifier::read_state() const
{
    return m_state;
}


Either<status_message_t, Classifier>
make_classifier(std::string const & json_params)
{
    auto rv =
        make_params_from_json(json_params)
        .rightMap([](params_t && params){ return Classifier(params); })
        ;

    return rv;
}


} // namespace Tsetlin
