#pragma once

#ifndef LIB_SRC_TSETLINI_ALGO_CLASSIC_HPP_
#define LIB_SRC_TSETLINI_ALGO_CLASSIC_HPP_

#include "estimator_state.hpp"
#include "estimator_state_cache.hpp"
#include "tsetlini_types.hpp"
#include "loss_fn.hpp"
#include "box_muller_approx.hpp"
#include "tsetlini_strong_params.hpp"

#include "strong_type/strong_type.hpp"

#include <cstdint>


#ifndef TSETLINI_USE_OMP
#define TSETLINI_USE_OMP 1
#endif


namespace Tsetlini
{


namespace
{


enum
{
    Type_II_Feedback = -1,
    No_Feedback = 0,
    Type_I_Feedback = +1
};


template<typename state_type>
inline
bool action(state_type state)
{
    return state >= 0;
}


/*
 * for use with clause output and clause feedback
 */
inline
auto clause_outputs_range_for_label(
    int label, number_of_classifier_clause_outputs_per_label_t number_of_clause_outputs_per_label) -> std::pair<int, int>
{
    auto const begin = label * value_of(number_of_clause_outputs_per_label);

    return std::make_pair(begin, begin + value_of(number_of_clause_outputs_per_label));
}


inline
void sum_up_label_votes(
    aligned_vector_char const & clause_output,
    w_vector_type const & weights,
    aligned_vector_int & label_sum,
    int target_label,

    number_of_classifier_clause_outputs_per_label_t const number_of_clause_outputs_per_label,
    threshold_t const threshold)
{
    using sum_type = std::int64_t;
    static_assert(sizeof (sum_type) > sizeof (w_vector_type::value_type), "sum_type must be wider than weight value type");

    sum_type rv = 0;

    auto const [output_begin_ix, output_end_ix] = clause_outputs_range_for_label(target_label, number_of_clause_outputs_per_label);

    if (weights.size() != 0)
    {
        for (int oidx = output_begin_ix; oidx < output_end_ix; ++oidx)
        {
            auto const val = clause_output[oidx];
            rv += oidx % 2 == 0
                ? val * (sum_type{1} + weights[oidx])
                : -val * (sum_type{1} + weights[oidx]);
        }
    }
    else
    {
        for (int oidx = output_begin_ix; oidx < output_end_ix; ++oidx)
        {
            auto const val = clause_output[oidx];
            rv += oidx % 2 == 0 ? val : -val;
        }
    }

    label_sum[target_label] = std::clamp<sum_type>(rv, -value_of(threshold), value_of(threshold));
}


/**
 * @param clause_output
 *      Calculated output of clauses, vector of 0s and 1s, with size equal
 *      to @c number_of_labels * @c number_of_clause_outputs_per_label .
 *
 * @param label_sum
 *      Output vector of integers of @c number_of_labels length where
 *      calculated vote scores will be placed.
 *
 * @param number_of_labels
 *      Integer count of labels the model was trained for.
 *
 * @param number_of_clause_outputs_per_label
 *      Count of clause outputs per label used for training.
 *
 * @param threshold
 *      Integer threshold to count votes against.
 */
inline
void sum_up_all_label_votes(
    aligned_vector_char const & clause_output,
    w_vector_type const & weights,
    aligned_vector_int & label_sum,

    number_of_labels_t const number_of_labels,
    number_of_classifier_clause_outputs_per_label_t const number_of_clause_outputs_per_label,
    threshold_t const threshold)
{
    for (int target_label = 0; target_label < number_of_labels; ++target_label)
    {
        sum_up_label_votes(clause_output, weights, label_sum, target_label, number_of_clause_outputs_per_label, threshold);
    }
}

/*
 * https://godbolt.org/z/bxh1rY
 */
template<unsigned int BATCH_SZ, typename state_type>
inline
void calculate_clause_output_with_pruning_T(
    aligned_vector_char const & X,
    aligned_vector_char & clause_output,
    number_of_estimator_clause_outputs_t const number_of_clause_outputs,
    numeric_matrix<state_type> const & ta_state,
    number_of_jobs_t const n_jobs)
{
    auto const number_of_features = number_of_features_t{X.size()};
    char const * X_p = assume_aligned<alignment>(X.data());

    auto const openmp_number_of_clause_outputs = value_of(number_of_clause_outputs);

    if (number_of_features < (int)BATCH_SZ)
    {
#if TSETLINI_USE_OMP == 1
#pragma omp parallel for if (n_jobs > 1) num_threads(value_of(n_jobs))
#endif
        // NOTE: OpenMP cannot directly work with number_of_estimator_clause_outputs_t
        // because the standard requires 'Canonical Loop Form'
        // Also, clang refuses use of in-place `value_of()`.
        for (int oidx = 0; oidx < openmp_number_of_clause_outputs; ++oidx)
        {
            bool output = true;
            bool all_exclude = true;

            state_type const * ta_state_pos_j = assume_aligned<alignment>(ta_state.row_data(2 * oidx + 0));
            state_type const * ta_state_neg_j = assume_aligned<alignment>(ta_state.row_data(2 * oidx + 1));

            for (int fidx = 0; fidx < number_of_features and output == true; ++fidx)
            {
                bool const action_include = action(ta_state_pos_j[fidx]);
                bool const action_include_negated = action(ta_state_neg_j[fidx]);

                all_exclude = (action_include == true or action_include_negated == true) ? false : all_exclude;

                output = ((action_include == true and X_p[fidx] == 0) or (action_include_negated == true and X_p[fidx] != 0)) ? false : output;
            }

            output = (all_exclude == true) ? false : output;

            clause_output[oidx] = output;
        }
    }
    else
    {
#if TSETLINI_USE_OMP == 1
#pragma omp parallel for if (n_jobs > 1) num_threads(value_of(n_jobs))
#endif
        for (int oidx = 0; oidx < openmp_number_of_clause_outputs; ++oidx)
        {
            char toggle_output = 0;
            char neg_all_exclude = 0;

            state_type const * ta_state_pos_j = assume_aligned<alignment>(ta_state.row_data(2 * oidx + 0));
            state_type const * ta_state_neg_j = assume_aligned<alignment>(ta_state.row_data(2 * oidx + 1));

            unsigned int kk = 0;
            for (; kk < value_of(number_of_features) - (BATCH_SZ - 1); kk += BATCH_SZ)
            {
                for (auto fidx = kk; fidx < BATCH_SZ + kk; ++fidx)
                {
                    bool const action_include = action(ta_state_pos_j[fidx]);
                    bool const action_include_negated = action(ta_state_neg_j[fidx]);

                    char flag = ((X_p[fidx] | !action_include) ^ 1) | (((!action_include_negated) | (X_p[fidx] ^ 1)) ^ 1);
                    toggle_output = flag > toggle_output ? flag : toggle_output;

                    char xflag = action_include + action_include_negated;
                    neg_all_exclude = xflag > neg_all_exclude ? xflag : neg_all_exclude;
                }
                if (toggle_output != 0)
                {
                    break;
                }
            }
            for (int fidx = kk; fidx < value_of(number_of_features) and toggle_output == false; ++fidx)
            {
                bool const action_include = action(ta_state_pos_j[fidx]);
                bool const action_include_negated = action(ta_state_neg_j[fidx]);

                char flag = ((X_p[fidx] | !action_include) ^ 1) | (((!action_include_negated) | (X_p[fidx] ^ 1)) ^ 1);
                toggle_output = flag > toggle_output ? flag : toggle_output;

                char xflag = action_include + action_include_negated;
                neg_all_exclude = xflag > neg_all_exclude ? xflag : neg_all_exclude;
            }

            clause_output[oidx] = neg_all_exclude == 0 ? 0 : !toggle_output;
        }
    }
}


template<unsigned int BATCH_SZ>
inline
void calculate_clause_output_with_pruning_T(
    aligned_vector_char const & X,
    aligned_vector_char & clause_output,
    number_of_estimator_clause_outputs_t const number_of_clause_outputs,
    TAState::value_type const & ta_state,
    number_of_jobs_t const n_jobs)
{
    std::visit(
        [&](auto & ta_state_values)
        {
            calculate_clause_output_with_pruning_T<BATCH_SZ>(
                X,
                clause_output,
                number_of_clause_outputs,
                ta_state_values,
                n_jobs
            );
        },
        ta_state.matrix);
}

/*
 * https://godbolt.org/z/5xhafK
 */
template<unsigned int BATCH_SZ, typename state_type>
inline
void calculate_clause_output_T(
    aligned_vector_char const & X,
    aligned_vector_char & clause_output,
    int const output_begin_ix,
    int const output_end_ix,
    numeric_matrix<state_type> const & ta_state,
    number_of_jobs_t const n_jobs)
{
    auto const number_of_features = number_of_features_t{X.size()};
    char const * X_p = assume_aligned<alignment>(X.data());

    if (number_of_features < (int)BATCH_SZ)
    {
#if TSETLINI_USE_OMP == 1
#pragma omp parallel for if (n_jobs > 1) num_threads(value_of(n_jobs))
#endif
        for (int oidx = output_begin_ix; oidx < output_end_ix; ++oidx)
        {
            bool output = true;

            state_type const * ta_state_pos_j = assume_aligned<alignment>(ta_state.row_data(2 * oidx + 0));
            state_type const * ta_state_neg_j = assume_aligned<alignment>(ta_state.row_data(2 * oidx + 1));

            for (int fidx = 0; fidx < number_of_features and output == true; ++fidx)
            {
                bool const action_include = action(ta_state_pos_j[fidx]);
                bool const action_include_negated = action(ta_state_neg_j[fidx]);

                output = ((action_include == true and X_p[fidx] == 0) or (action_include_negated == true and X_p[fidx] != 0)) ? false : output;
            }

            clause_output[oidx] = output;
        }
    }
    else
    {
#if TSETLINI_USE_OMP == 1
#pragma omp parallel for if (n_jobs > 1) num_threads(value_of(n_jobs))
#endif
        for (int oidx = output_begin_ix; oidx < output_end_ix; ++oidx)
        {
            char toggle_output = 0;

            state_type const * ta_state_pos_j = assume_aligned<alignment>(ta_state.row_data(2 * oidx + 0));
            state_type const * ta_state_neg_j = assume_aligned<alignment>(ta_state.row_data(2 * oidx + 1));

            unsigned int kk = 0;
            for (; kk < value_of(number_of_features) - (BATCH_SZ - 1); kk += BATCH_SZ)
            {
                for (auto fidx = kk; fidx < BATCH_SZ + kk; ++fidx)
                {
                    bool const action_include = action(ta_state_pos_j[fidx]);
                    bool const action_include_negated = action(ta_state_neg_j[fidx]);

                    char flag = ((X_p[fidx] | !action_include) ^ 1) | (((!action_include_negated) | (X_p[fidx] ^ 1)) ^ 1);
                    toggle_output = flag > toggle_output ? flag : toggle_output;
                }
                if (toggle_output != 0)
                {
                    break;
                }
            }
            for (int fidx = kk; fidx < number_of_features and toggle_output == false; ++fidx)
            {
                bool const action_include = action(ta_state_pos_j[fidx]);
                bool const action_include_negated = action(ta_state_neg_j[fidx]);

                char flag = ((X_p[fidx] | !action_include) ^ 1) | (((!action_include_negated) | (X_p[fidx] ^ 1)) ^ 1);
                toggle_output = flag > toggle_output ? flag : toggle_output;
            }

            clause_output[oidx] = !toggle_output;
        }
    }
}


template<unsigned int BATCH_SZ>
inline
void calculate_clause_output_T(
    aligned_vector_char const & X,
    aligned_vector_char & clause_output,
    int const output_begin_ix,
    int const output_end_ix,
    TAState::value_type const & ta_state,
    number_of_jobs_t const n_jobs)
{
    std::visit(
        [&](auto & ta_state_values)
        {
            calculate_clause_output_T<BATCH_SZ>(
                X,
                clause_output,
                output_begin_ix,
                output_end_ix,
                ta_state_values,
                n_jobs
            );
        },
        ta_state.matrix);
}


/*
 * Feedback Type I, negative
 *
 * https://godbolt.org/z/GdjvPG
 */
template<typename state_type>
void block1(
    number_of_features_t const number_of_features,
    number_of_states_t const number_of_states,
    state_type * __restrict ta_state_pos_j,
    state_type * __restrict ta_state_neg_j,
    char const * __restrict ct_pos, // __restrict does not quite hold here with CoinTosser
    char const * __restrict ct_neg
)
{
    ta_state_pos_j = assume_aligned<alignment>(ta_state_pos_j);
    ta_state_neg_j = assume_aligned<alignment>(ta_state_neg_j);

    ct_pos = assume_aligned<alignment>(ct_pos);
    ct_neg = assume_aligned<alignment>(ct_neg);

    for (int fidx = 0; fidx < number_of_features; ++fidx)
    {
        // This is nicely vectorized by gcc
        {
            auto cond = ct_pos[fidx];

            ta_state_pos_j[fidx] = UNLIKELY(cond) ? (ta_state_pos_j[fidx] > -number_of_states ? ta_state_pos_j[fidx] - 1 : ta_state_pos_j[fidx]) : ta_state_pos_j[fidx];
        }

        {
            auto cond = ct_neg[fidx];

            ta_state_neg_j[fidx] = UNLIKELY(cond) ? (ta_state_neg_j[fidx] > -number_of_states ? ta_state_neg_j[fidx] - 1 : ta_state_neg_j[fidx]) : ta_state_neg_j[fidx];
        }
    }
}


/*
 * Feedback Type I, positive
 *
 * https://godbolt.org/z/97r1h7
 */
template<bool boost_true_positive_feedback, typename state_type>
void block2(
    number_of_features_t const number_of_features,
    number_of_states_t const number_of_states,
    state_type * __restrict ta_state_pos_j,
    state_type * __restrict ta_state_neg_j,
    char const * __restrict X,
    char const * __restrict ct_pos,
    char const * __restrict ct_neg
)
{
    ta_state_pos_j = assume_aligned<alignment>(ta_state_pos_j);
    ta_state_neg_j = assume_aligned<alignment>(ta_state_neg_j);

    ct_pos = assume_aligned<alignment>(ct_pos);
    ct_neg = assume_aligned<alignment>(ct_neg);

    X = assume_aligned<alignment>(X);

    for (int fidx = 0; fidx < number_of_features; ++fidx)
    {
        auto cond1_pos = boost_true_positive_feedback == true or not ct_pos[fidx];
        auto cond2_pos = ct_pos[fidx];

        auto cond1_neg = boost_true_positive_feedback == true or not ct_neg[fidx];
        auto cond2_neg = ct_neg[fidx];

#if 0
        auto cond = X[fidx];
        ta_state_pos_j[fidx] = cond
            ? (cond1_pos
                ? (ta_state_pos_j[fidx] < number_of_states - 1
                    ? ta_state_pos_j[fidx] + 1
                    : ta_state_pos_j[fidx])
                : ta_state_pos_j[fidx])
            : (cond2_pos
                ? (ta_state_pos_j[fidx] > -number_of_states
                    ? ta_state_pos_j[fidx] - 1
                    : ta_state_pos_j[fidx])
                : ta_state_pos_j[fidx])
            ;

        ta_state_neg_j[fidx] = cond
            ? (cond2_neg
                ? (ta_state_neg_j[fidx] > -number_of_states
                    ? ta_state_neg_j[fidx] - 1
                    : ta_state_neg_j[fidx])
                : ta_state_neg_j[fidx])
            : (cond1_neg ?
                (ta_state_neg_j[fidx] < number_of_states - 1
                    ? ta_state_neg_j[fidx] + 1
                    : ta_state_neg_j[fidx])
                : ta_state_neg_j[fidx])
            ;
#endif

        // This is nicely vectorized by gcc
        auto Xcond_1 = X[fidx];
        auto Xcond_0 = !X[fidx];

        auto cond_pos_inc = ta_state_pos_j[fidx] < number_of_states - 1;
        auto cond_pos_dec = ta_state_pos_j[fidx] > -number_of_states;

        auto cond_neg_inc = ta_state_neg_j[fidx] < number_of_states - 1;
        auto cond_neg_dec = ta_state_neg_j[fidx] > -number_of_states;

        ta_state_pos_j[fidx] = (Xcond_1 & cond1_pos & cond_pos_inc) ? ta_state_pos_j[fidx] + 1 : ta_state_pos_j[fidx];
        ta_state_neg_j[fidx] = (Xcond_1 & cond2_neg & cond_neg_dec) ? ta_state_neg_j[fidx] - 1 : ta_state_neg_j[fidx];
        ta_state_pos_j[fidx] = (Xcond_0 & cond2_pos & cond_pos_dec) ? ta_state_pos_j[fidx] - 1 : ta_state_pos_j[fidx];
        ta_state_neg_j[fidx] = (Xcond_0 & cond1_neg & cond_neg_inc) ? ta_state_neg_j[fidx] + 1 : ta_state_neg_j[fidx];

#if 0
        if (X[fidx] != 0)
        {
            if (LIKELY(cond1_pos))
            {
                if (ta_state_pos_j[fidx] < number_of_states - 1)
                {
                    ta_state_pos_j[fidx]++;
                }
            }
            if (UNLIKELY(cond2_neg))
            {
                if (ta_state_neg_j[fidx] > -number_of_states)
                {
                    ta_state_neg_j[fidx]--;
                }
            }
        }
        else // if (X[k] == 0)
        {
            if (LIKELY(cond1_neg))
            {
                if (ta_state_neg_j[fidx] < number_of_states - 1)
                {
                    ta_state_neg_j[fidx]++;
                }
            }

            if (UNLIKELY(cond2_pos))
            {
                if (ta_state_pos_j[fidx] > -number_of_states)
                {
                    ta_state_pos_j[fidx]--;
                }
            }
        }
#endif
    }
}


/*
 * Feedback Type II
 *
 * https://godbolt.org/z/WTa4vK
 */
template<typename state_type>
void block3(
    number_of_features_t const number_of_features,
    state_type * __restrict ta_state_pos_j,
    state_type * __restrict ta_state_neg_j,
    char const * __restrict X
)
{
    ta_state_pos_j = assume_aligned<alignment>(ta_state_pos_j);
    ta_state_neg_j = assume_aligned<alignment>(ta_state_neg_j);
    X = assume_aligned<alignment>(X);

    // this is nicely vectorized by gcc
    for (int fidx = 0; fidx < number_of_features; ++fidx)
    {
        //auto X_cond_0 = !X[fidx]; // gcc 7.5 vectorizes with this construct,
                                    // newer gcc needs construct below
        auto X_cond_0 = X[fidx] ^ 1;
        auto X_cond_1 = X[fidx];
        auto X_pos_inc = ta_state_pos_j[fidx] < 0;
        auto X_neg_inc = ta_state_neg_j[fidx] < 0;

        ta_state_pos_j[fidx] = (X_cond_0 & X_pos_inc) ? ta_state_pos_j[fidx] + 1 : ta_state_pos_j[fidx];
        ta_state_neg_j[fidx] = (X_cond_1 & X_neg_inc) ? ta_state_neg_j[fidx] + 1 : ta_state_neg_j[fidx];
    }

#if 0
    for (int fidx = 0; fidx < number_of_features; ++fidx)
    {
        if (X[fidx] == 0)
        {
            auto action_include = (ta_state_pos_j[fidx] >= 0);
            if (action_include == false)
            {
                ta_state_pos_j[fidx]++;
            }
        }
        else //if(X[k] == 1)
        {
            auto action_include_negated = (ta_state_neg_j[fidx] >= 0);
            if (action_include_negated == false)
            {
                ta_state_neg_j[fidx]++;
            }
        }
    }
#endif
}


template<typename state_type>
void train_classifier_automata(
    numeric_matrix<state_type> & ta_state,
    w_vector_type & weights,
    int const input_begin_ix,
    int const input_end_ix,
    feedback_vector_type::value_type const * __restrict feedback_to_clauses,
    char const * __restrict clause_output,
    number_of_states_t const number_of_states,
    aligned_vector_char const & X,
    max_weight_t const max_weight,
    boost_tpf_t const boost_true_positive_feedback,
    IRNG & prng,
    EstimatorStateCacheBase::coin_tosser_type & ct
    )
{
    auto const number_of_features = number_of_features_t{X.size()};

    for (int iidx = input_begin_ix; iidx < input_end_ix; ++iidx)
    {
        state_type * ta_state_pos_j = ::assume_aligned<alignment>(ta_state.row_data(2 * iidx + 0));
        state_type * ta_state_neg_j = ::assume_aligned<alignment>(ta_state.row_data(2 * iidx + 1));

        if (feedback_to_clauses[iidx] >= Type_I_Feedback)
        {
            if (clause_output[iidx] == 0)
            {
                block1(number_of_features, number_of_states, ta_state_pos_j, ta_state_neg_j, ct.tosses1(prng), ct.tosses2(prng));
            }
            else // if (clause_output[iidx] == 1)
            {
                if (boost_true_positive_feedback == true)
                {
                    block2<true>(number_of_features, number_of_states, ta_state_pos_j, ta_state_neg_j, X.data(), ct.tosses1(prng), ct.tosses2(prng));
                }
                else
                {
                    block2<false>(number_of_features, number_of_states, ta_state_pos_j, ta_state_neg_j, X.data(), ct.tosses1(prng), ct.tosses2(prng));
                }

                if (weights.size() != 0)
                {
                    // plus 1, because weights are offset by -1, haha
                    weights[iidx] += ((weights[iidx] + 1) < max_weight);
                }
            }
        }
        else if (feedback_to_clauses[iidx] <= Type_II_Feedback)
        {
            if (clause_output[iidx] == 1)
            {
                block3(number_of_features, ta_state_pos_j, ta_state_neg_j, X.data());
            }

            if (weights.size() != 0)
            {
                weights[iidx] -= (weights[iidx] != 0);
            }
        }
    }
}


inline
void train_classifier_automata(
    TAState::value_type & ta_state,
    int const input_begin_ix,
    int const input_end_ix,
    feedback_vector_type::value_type const * __restrict feedback_to_clauses,
    char const * __restrict clause_output,
    number_of_states_t const number_of_states,
    aligned_vector_char const & X,
    max_weight_t const max_weight,
    boost_tpf_t const boost_true_positive_feedback,
    IRNG & prng,
    EstimatorStateCacheBase::coin_tosser_type & ct
    )
{
    std::visit(
        [&](auto & ta_state_values)
        {
            train_classifier_automata(
                ta_state_values,
                ta_state.weights,
                input_begin_ix,
                input_end_ix,
                feedback_to_clauses,
                clause_output,
                number_of_states,
                X,
                max_weight,
                boost_true_positive_feedback,
                prng,
                ct
            );
        },
        ta_state.matrix
    );
}


template<typename TFRNG>
inline
void calculate_classifier_feedback_to_clauses(
    feedback_vector_type & feedback_to_clauses,
    label_type const target_label,
    label_type const opposite_label,
    int const target_label_votes,
    int const opposite_label_votes,
    number_of_classifier_clause_outputs_per_label_t const number_of_clause_outputs_per_label,
    threshold_t const threshold,
    TFRNG & fgen)
{
    const auto THR2_inv = (ONE / (value_of(threshold) * 2));
    const auto THR_pos = THR2_inv * (value_of(threshold) - target_label_votes);
    const auto THR_neg = THR2_inv * (value_of(threshold) + opposite_label_votes);

    std::fill(feedback_to_clauses.begin(), feedback_to_clauses.end(), No_Feedback);

    {
        auto const [feedback_begin_ix, feedback_end_ix] = clause_outputs_range_for_label(target_label, number_of_clause_outputs_per_label);

        for (int fidx = feedback_begin_ix; fidx < feedback_end_ix; ++fidx)
        {
            if (fgen.next() > THR_pos)
            {
                continue;
            }

            // Type I and II Feedback
            feedback_to_clauses[fidx] = fidx % 2 == 0 ? Type_I_Feedback : Type_II_Feedback;
        }
    }

    {
        auto const [feedback_begin_ix, feedback_end_ix] = clause_outputs_range_for_label(opposite_label, number_of_clause_outputs_per_label);

        for (int fidx = feedback_begin_ix; fidx < feedback_end_ix; ++fidx)
        {
            if (fgen.next() > THR_neg)
            {
                continue;
            }

            // Type I and II Feedback
            feedback_to_clauses[fidx] = fidx % 2 == 0 ? Type_II_Feedback : Type_I_Feedback;
        }
    }
}


inline
response_type sum_up_regressor_votes(
    aligned_vector_char const & clause_output,
    threshold_t const threshold,
    w_vector_type const & weights)
{
    using sum_type = std::int64_t;
    static_assert(sizeof (sum_type) > sizeof (w_vector_type::value_type), "sum_type must be wider than weight value type");

    auto accumulate_weighted = [](auto const & clause_output, auto const & weights)
    {
        sum_type acc = 0;

        for (auto ix = 0u; ix < clause_output.size(); ++ix)
        {
            acc += clause_output[ix] * (sum_type{1} + weights[ix]);
        }

        return acc;
    };

    auto const sum = weights.size() == 0 ?
        std::accumulate(clause_output.cbegin(), clause_output.cend(), sum_type{0})
        :
        accumulate_weighted(clause_output, weights);

    return std::clamp<sum_type>(sum, 0, value_of(threshold));
}


template<typename state_type>
void train_regressor_automata(
    numeric_matrix<state_type> & ta_state,
    w_vector_type & weights,
    int const input_begin_ix,
    int const input_end_ix,
    char const * __restrict clause_output,
    number_of_states_t const number_of_states,
    int const response_error,
    aligned_vector_char const & X,
    max_weight_t const max_weight,
    loss_fn_type const & loss_fn,
    box_muller_flag_t const box_muller,
    boost_tpf_t const boost_true_positive_feedback,
    IRNG & prng,
    threshold_t const threshold,
    EstimatorStateCacheBase::coin_tosser_type & ct
    )
{
    auto const number_of_features = number_of_features_t{X.size()};

    unsigned int const N = input_end_ix - input_begin_ix;
    real_type const P = loss_fn(static_cast<real_type>(response_error) / value_of(threshold));
    /*
     * For sparse feedback if N * P >= 0.5 we will just round the number of hits,
     * else we will pick either 0 or 1 with probability proportional to P.
     */
    unsigned int const feedback_hits = box_muller == true
        ? binomial(N, P, prng)
        :
        std::clamp<unsigned int>(
            N * P >= 0.5 ? std::round(N * P) : prng() < N * P * static_cast<double>(IRNG::max()), // 0xffff'ffff when converted to float changes to 0x1'0000'0000
            0, N
        );

    for (unsigned int idx = 0; idx < feedback_hits; ++idx)    {
        // randomly pick index that corresponds to non-zero feedback
        auto const iidx = prng() % N + input_begin_ix;

        state_type * ta_state_pos_j = ::assume_aligned<alignment>(ta_state.row_data(2 * iidx + 0));
        state_type * ta_state_neg_j = ::assume_aligned<alignment>(ta_state.row_data(2 * iidx + 1));

        if (response_error < 0)
        {
            if (clause_output[iidx] == 0)
            {
                block1(number_of_features, number_of_states, ta_state_pos_j, ta_state_neg_j, ct.tosses1(prng), ct.tosses2(prng));
            }
            else // if (clause_output[iidx] == 1)
            {
                if (boost_true_positive_feedback == true)
                {
                    block2<true>(number_of_features, number_of_states, ta_state_pos_j, ta_state_neg_j, X.data(), ct.tosses1(prng), ct.tosses2(prng));
                }
                else
                {
                    block2<false>(number_of_features, number_of_states, ta_state_pos_j, ta_state_neg_j, X.data(), ct.tosses1(prng), ct.tosses2(prng));
                }

                if (weights.size() != 0)
                {
                    // plus 1, because weights are offset by -1, haha
                    weights[iidx] += ((weights[iidx] + 1) < max_weight);
                }
            }
        }
        else if (response_error > 0)
        {
            if (clause_output[iidx] != 0)
            {
                block3(number_of_features, ta_state_pos_j, ta_state_neg_j, X.data());

                if (weights.size() != 0)
                {
                    weights[iidx] -= (weights[iidx] != 0);
                }
            }
        }
    }
}


inline
void train_regressor_automata(
    TAState::value_type & ta_state,
    int const input_begin_ix,
    int const input_end_ix,
    char const * __restrict clause_output,
    number_of_states_t const number_of_states,
    int const response_error,
    aligned_vector_char const & X,
    max_weight_t const max_weight,
    loss_fn_type const & loss_fn,
    box_muller_flag_t const box_muller,
    boost_tpf_t const boost_true_positive_feedback,
    IRNG & prng,
    threshold_t const threshold,
    EstimatorStateCacheBase::coin_tosser_type & ct
    )
{
    std::visit(
        [&](auto & ta_state_values)
        {
            train_regressor_automata(
                ta_state_values,
                ta_state.weights,
                input_begin_ix,
                input_end_ix,
                clause_output,
                number_of_states,
                response_error,
                X,
                max_weight,
                loss_fn,
                box_muller,
                boost_true_positive_feedback,
                prng,
                threshold,
                ct
            );
        },
        ta_state.matrix
    );
}


} // anonymous namespace


} // namespace Tsetlini

#endif /* LIB_SRC_TSETLINI_ALGO_CLASSIC_HPP_ */
