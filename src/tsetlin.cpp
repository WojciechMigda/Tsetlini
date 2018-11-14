#define LOG_MODULE "tsetlin"
#include "logger.hpp"

#include "tsetlin.hpp"
#include "assume_aligned.hpp"
#include "config_companion.hpp"
#include "tsetlin_types.hpp"
#include "tsetlin_state.hpp"

#include <utility>
#include <algorithm>
#include <cstddef>


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
int pos_clause_index(int target_class, int j, int number_of_pos_neg_clauses_per_class)
{
    return 2 * target_class * number_of_pos_neg_clauses_per_class + j;
}


inline
int neg_clause_index(int target_class, int j, int number_of_pos_neg_clauses_per_class)
{
    return pos_clause_index(target_class, j, number_of_pos_neg_clauses_per_class) + number_of_pos_neg_clauses_per_class;
}


inline
void sum_up_class_votes(
    aligned_vector_char const & clause_output,
    aligned_vector_int & class_sum,
    int target_class,

    int const number_of_pos_neg_clauses_per_class,
    int const threshold)
{
    class_sum[target_class] = 0;

    for (int j = 0; j < number_of_pos_neg_clauses_per_class; ++j)
    {
        class_sum[target_class] += clause_output[pos_clause_index(target_class, j, number_of_pos_neg_clauses_per_class)];
    }

    for (int j = 0; j < number_of_pos_neg_clauses_per_class; ++j)
    {
        class_sum[target_class] -= clause_output[neg_clause_index(target_class, j, number_of_pos_neg_clauses_per_class)];
    }
    class_sum[target_class] = std::clamp(class_sum[target_class], -threshold, threshold);
}


inline
void sum_up_all_class_votes(
    aligned_vector_char const & clause_output,
    aligned_vector_int & class_sum,

    int const number_of_classes,
    int const number_of_pos_neg_clauses_per_class,
    int const threshold)
{
    for (int target_class = 0; target_class < number_of_classes; ++target_class)
    {
        sum_up_class_votes(clause_output, class_sum, target_class, number_of_pos_neg_clauses_per_class, threshold);
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


} // anonymous




Classifier::Classifier(ClassifierState const & state) :
    state(state)
{

}


Classifier::Classifier(ClassifierState && state) :
    state(std::move(state))
{

}


int Classifier::predict(aligned_vector_char const & sample) const
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
        state.cache.class_sum,
        Config::number_of_classes(state.config),
        Config::number_of_pos_neg_clauses_per_class(state.config),
        Config::threshold(state.config));

    int rv = std::distance(
        state.cache.class_sum.cbegin(),
        std::max_element(state.cache.class_sum.cbegin(), state.cache.class_sum.cend()));

    return rv;
}


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
        state.cache.class_sum,
        Config::number_of_classes(state.config),
        Config::number_of_pos_neg_clauses_per_class(state.config),
        Config::threshold(state.config));

    return state.cache.class_sum;
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
        state.cache.class_sum,
        Config::number_of_classes(state.config),
        Config::number_of_pos_neg_clauses_per_class(state.config),
        Config::threshold(state.config));

    std::copy(state.cache.class_sum.cbegin(), state.cache.class_sum.cend(), out_p);
}


void Classifier::update(aligned_vector_char const & X, y_vector_type::value_type target_class)
{
#if 0
    // Randomly pick one of the other classes, for pairwise learning of class output
    int negative_target_class = igen_.next(0, number_of_classes - 1);
    while (negative_target_class == target_class)
    {
        negative_target_class = igen_.next(0, number_of_classes - 1);
    }

    calculate_clause_output(X, cache.clause_output, false);

    sum_up_class_votes(cache.clause_output, cache.class_sum, target_class);
    sum_up_class_votes(cache.clause_output, cache.class_sum, negative_target_class);

    std::fill(cache.feedback_to_clauses.begin(), cache.feedback_to_clauses.end(), 0);

#endif

    const auto S_inv = ONE / Config::s(state.config);

#if 0
    const auto THR2_inv = (ONE / (Config::threshold(state.config) * 2));
    const auto THR_pos = THR2_inv * (threshold - state.cache.class_sum[target_class]);
    const auto THR_neg = THR2_inv * (threshold + state,cache.class_sum[negative_target_class]);

    for (int j = 0; j < number_of_pos_neg_clauses_per_class; ++j)
    {
        if (frand() > THR_pos)
        {
            continue;
        }

        // Type I Feedback
        cache.feedback_to_clauses[pos_clause_index(target_class, j)]++;
    }
    for (int j = 0; j < number_of_pos_neg_clauses_per_class; ++j)
    {
        if (frand() > THR_pos)
        {
            continue;
        }

        // Type II Feedback
        cache.feedback_to_clauses[neg_clause_index(target_class, j)]--;
    }

    for (int j = 0; j < number_of_pos_neg_clauses_per_class; ++j)
    {
        if (frand() > THR_neg)
        {
            continue;
        }

        cache.feedback_to_clauses[pos_clause_index(negative_target_class, j)]--;
    }
    for (int j = 0; j < number_of_pos_neg_clauses_per_class; ++j)
    {
        if (frand() > THR_neg)
        {
            continue;
        }

        cache.feedback_to_clauses[neg_clause_index(negative_target_class, j)]++;
    }

#endif

    train_automata_batch(
        state.ta_state.data(),
        0,
        Config::number_of_clauses(state.config),
        state.cache.feedback_to_clauses.data(),
        state.cache.clause_output.data(),
        Config::number_of_features(state.config),
        Config::number_of_states(state.config),
        S_inv,
        X.data(),
        Config::boost_true_positive_feedback(state.config),
        state.cache.fcache[0]
    );
}


void Classifier::fit_batch(std::vector<aligned_vector_char> const & X, y_vector_type const & y)
{
    for (auto i = 0u; i < std::min(X.size(), y.size()); ++i)
    {
        update(X[i], y[i]);
    }
}


void Classifier::fit(std::vector<aligned_vector_char> const & X, y_vector_type const & y, std::size_t number_of_examples, int epochs)
{
    std::vector<int> ix(X.size());
    std::iota(ix.begin(), ix.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(ix.begin(), ix.end(), std::mt19937(state.gen));

        for (auto i = 0u; i < number_of_examples; ++i)
        {
            update(X[ix[i]], y[ix[i]]);
        }
    }

}


real_type Classifier::evaluate(std::vector<aligned_vector_char> const & X, y_vector_type const & y, int number_of_examples)
{
    int errors = 0;

    for (int l = 0; l < number_of_examples; ++l)
    {
        calculate_clause_output(
            X[l],
            state.cache.clause_output,
            true,
            Config::number_of_clauses(state.config),
            Config::number_of_features(state.config),
            Config::number_of_states(state.config),
            state.ta_state
        );

        sum_up_all_class_votes(
            state.cache.clause_output,
            state.cache.class_sum,
            Config::number_of_classes(state.config),
            Config::number_of_pos_neg_clauses_per_class(state.config),
            Config::threshold(state.config));


        const int max_class = std::distance(
            state.cache.class_sum.cbegin(),
            std::max_element(state.cache.class_sum.cbegin(), state.cache.class_sum.cend()));

        errors += (max_class != y[l]);
    }

    return ONE - ONE * errors / number_of_examples;

    return 0.;
}


} // namespace Tsetlin
