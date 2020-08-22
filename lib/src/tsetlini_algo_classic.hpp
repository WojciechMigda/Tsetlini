#pragma once

#ifndef LIB_SRC_TSETLINI_ALGO_CLASSIC_HPP_
#define LIB_SRC_TSETLINI_ALGO_CLASSIC_HPP_

#include "estimator_state.hpp"
#include "estimator_state_cache.hpp"
#include "tsetlini_types.hpp"


#ifndef TSETLINI_USE_OMP
#define TSETLINI_USE_OMP 1
#endif


namespace Tsetlini
{


namespace
{


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
auto clause_range_for_label(int label, int number_of_pos_neg_clauses_per_label) -> std::pair<int, int>
{
    // in contrary to pos_clause_index we do not double, because there is no
    // distinction into positive and negative entries for clause output
    // and feedback
    auto const begin = label * number_of_pos_neg_clauses_per_label;

    return std::make_pair(begin, begin + number_of_pos_neg_clauses_per_label);
}


inline
void sum_up_label_votes(
    aligned_vector_char const & clause_output,
    aligned_vector_int & label_sum,
    int target_label,

    int const number_of_pos_neg_clauses_per_label,
    int const threshold)
{
    int rv = 0;

    auto const [output_begin_ix, output_end_ix] = clause_range_for_label(target_label, number_of_pos_neg_clauses_per_label);

    for (int oidx = output_begin_ix; oidx < output_end_ix; ++oidx)
    {
        auto const val = clause_output[oidx];
        rv += oidx % 2 == 0 ? val : -val;
    }

    label_sum[target_label] = std::clamp(rv, -threshold, threshold);
}


/**
 * @param clause_output
 *      Calculated output of clauses, vector of 0s and 1s, with size equal
 *      to 2 * @c number_of_labels * @c number_of_pos_neg_clauses_per_label .
 *
 * @param label_sum
 *      Output vector of integers of @c @c number_of_labels length where
 *      calculated vote scores will be placed.
 *
 * @param number_of_labels
 *      Integer count of labels the model was trained for.
 *
 * @param number_of_pos_neg_clauses_per_label
 *      Integer count of wither positive or negative clauses used for training.
 *
 * @param threshold
 *      Integer threshold to count votes against.
 */
inline
void sum_up_all_label_votes(
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

/*
 * https://godbolt.org/z/bxh1rY
 */
template<unsigned int BATCH_SZ, typename state_type>
inline
void calculate_clause_output_for_predict_T(
    aligned_vector_char const & X,
    aligned_vector_char & clause_output,
    int const number_of_clauses,
    numeric_matrix<state_type> const & ta_state,
    int const n_jobs)
{
    int const number_of_features = X.size();
    char const * X_p = assume_aligned<alignment>(X.data());

    if (number_of_features < (int)BATCH_SZ)
    {
#if TSETLINI_USE_OMP == 1
#pragma omp parallel for if (n_jobs > 1) num_threads(n_jobs)
#endif
        for (int oidx = 0; oidx < number_of_clauses; ++oidx)
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
#pragma omp parallel for if (n_jobs > 1) num_threads(n_jobs)
#endif
        for (int oidx = 0; oidx < number_of_clauses; ++oidx)
        {
            char toggle_output = 0;
            char neg_all_exclude = 0;

            state_type const * ta_state_pos_j = assume_aligned<alignment>(ta_state.row_data(2 * oidx + 0));
            state_type const * ta_state_neg_j = assume_aligned<alignment>(ta_state.row_data(2 * oidx + 1));

            unsigned int kk = 0;
            for (; kk < number_of_features - (BATCH_SZ - 1); kk += BATCH_SZ)
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
            for (int fidx = kk; fidx < number_of_features and toggle_output == false; ++fidx)
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
void calculate_clause_output_for_predict_T(
    aligned_vector_char const & X,
    aligned_vector_char & clause_output,
    int const number_of_clauses,
    TAState::value_type const & ta_state,
    int const n_jobs)
{
    std::visit(
        [&](auto & ta_state_values)
        {
            calculate_clause_output_for_predict_T<BATCH_SZ>(
                X,
                clause_output,
                number_of_clauses,
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
    int const n_jobs)
{
    int const number_of_features = X.size();
    char const * X_p = assume_aligned<alignment>(X.data());

    if (number_of_features < (int)BATCH_SZ)
    {
#if TSETLINI_USE_OMP == 1
#pragma omp parallel for if (n_jobs > 1) num_threads(n_jobs)
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
#pragma omp parallel for if (n_jobs > 1) num_threads(n_jobs)
#endif
        for (int oidx = output_begin_ix; oidx < output_end_ix; ++oidx)
        {
            char toggle_output = 0;

            state_type const * ta_state_pos_j = assume_aligned<alignment>(ta_state.row_data(2 * oidx + 0));
            state_type const * ta_state_neg_j = assume_aligned<alignment>(ta_state.row_data(2 * oidx + 1));

            unsigned int kk = 0;
            for (; kk < number_of_features - (BATCH_SZ - 1); kk += BATCH_SZ)
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
    int const n_jobs)
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
    int const number_of_features,
    int const number_of_states,
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
    int const number_of_features,
    int const number_of_states,
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

        ta_state_pos_j[fidx] = Xcond_1 & cond1_pos & cond_pos_inc ? ta_state_pos_j[fidx] + 1 : ta_state_pos_j[fidx];
        ta_state_neg_j[fidx] = Xcond_1 & cond2_neg & cond_neg_dec ? ta_state_neg_j[fidx] - 1 : ta_state_neg_j[fidx];
        ta_state_pos_j[fidx] = Xcond_0 & cond2_pos & cond_pos_dec ? ta_state_pos_j[fidx] - 1 : ta_state_pos_j[fidx];
        ta_state_neg_j[fidx] = Xcond_0 & cond1_neg & cond_neg_inc ? ta_state_neg_j[fidx] + 1 : ta_state_neg_j[fidx];

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
    int const number_of_features,
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

        ta_state_pos_j[fidx] = X_cond_0 & X_pos_inc ? ta_state_pos_j[fidx] + 1 : ta_state_pos_j[fidx];
        ta_state_neg_j[fidx] = X_cond_1 & X_neg_inc ? ta_state_neg_j[fidx] + 1 : ta_state_neg_j[fidx];
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
    int const input_begin_ix,
    int const input_end_ix,
    feedback_vector_type::value_type const * __restrict feedback_to_clauses,
    char const * __restrict clause_output,
    int const number_of_states,
    aligned_vector_char const & X,
    bool const boost_true_positive_feedback,
    IRNG & prng,
    EstimatorStateCacheBase::coin_tosser_type & ct
    )
{
    int const number_of_features = X.size();

    for (int iidx = input_begin_ix; iidx < input_end_ix; ++iidx)
    {
        state_type * ta_state_pos_j = ::assume_aligned<alignment>(ta_state.row_data(2 * iidx + 0));
        state_type * ta_state_neg_j = ::assume_aligned<alignment>(ta_state.row_data(2 * iidx + 1));

        if (feedback_to_clauses[iidx] > 0)
        {
            if (clause_output[iidx] == 0)
            {
                block1(number_of_features, number_of_states, ta_state_pos_j, ta_state_neg_j, ct.tosses1(prng), ct.tosses2(prng));
            }
            else // if (clause_output[iidx] == 1)
            {
                if (boost_true_positive_feedback)
                {
                    block2<true>(number_of_features, number_of_states, ta_state_pos_j, ta_state_neg_j, X.data(), ct.tosses1(prng), ct.tosses2(prng));
                }
                else
                {
                    block2<false>(number_of_features, number_of_states, ta_state_pos_j, ta_state_neg_j, X.data(), ct.tosses1(prng), ct.tosses2(prng));
                }
            }
        }
        else if (feedback_to_clauses[iidx] < 0)
        {
            if (clause_output[iidx] == 1)
            {
                block3(number_of_features, ta_state_pos_j, ta_state_neg_j, X.data());
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
    int const number_of_states,
    aligned_vector_char const & X,
    bool const boost_true_positive_feedback,
    IRNG & prng,
    EstimatorStateCacheBase::coin_tosser_type & ct
    )
{
    std::visit(
        [&](auto & ta_state_values)
        {
            train_classifier_automata(
                ta_state_values,
                input_begin_ix,
                input_end_ix,
                feedback_to_clauses,
                clause_output,
                number_of_states,
                X,
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
    int const number_of_pos_neg_clauses_per_label,
    int const threshold,
    TFRNG & fgen)
{
    const auto THR2_inv = (ONE / (threshold * 2));
    const auto THR_pos = THR2_inv * (threshold - target_label_votes);
    const auto THR_neg = THR2_inv * (threshold + opposite_label_votes);

    std::fill(feedback_to_clauses.begin(), feedback_to_clauses.end(), 0);

    {
        auto const [feedback_begin_ix, feedback_end_ix] = clause_range_for_label(target_label, number_of_pos_neg_clauses_per_label);

        for (int fidx = feedback_begin_ix; fidx < feedback_end_ix; ++fidx)
        {
            if (fgen.next() > THR_pos)
            {
                continue;
            }

            // Type I and II Feedback
            feedback_to_clauses[fidx] = fidx % 2 == 0 ? 1 : -1;
        }
    }

    {
        auto const [feedback_begin_ix, feedback_end_ix] = clause_range_for_label(opposite_label, number_of_pos_neg_clauses_per_label);

        for (int fidx = feedback_begin_ix; fidx < feedback_end_ix; ++fidx)
        {
            if (fgen.next() > THR_neg)
            {
                continue;
            }

            // Type I and II Feedback
            feedback_to_clauses[fidx] = fidx % 2 == 0 ? -1 : 1;
        }
    }
}


inline
response_type sum_up_regressor_votes(
    aligned_vector_char const & clause_output,
    int const threshold,
    w_vector_type const & weights)
{
    auto accumulate_weighted = [](auto const & clause_output, auto const & weights)
    {
        int acc = 0;

        for (auto ix = 0u; ix < clause_output.size(); ++ix)
        {
            acc += clause_output[ix] * (weights[ix] + 1);
        }

        return acc;
    };

    auto const sum = weights.size() == 0 ?
        std::accumulate(clause_output.cbegin(), clause_output.cend(), 0)
        :
        accumulate_weighted(clause_output, weights);

    return std::clamp(sum, 0, threshold);
}


template<typename TFRNG>
inline
void calculate_regressor_feedback_to_clauses(
    feedback_vector_type & feedback_to_clauses,
    int const response_error,
    int const threshold,
    TFRNG & fgen)
{
    real_type const R2 = static_cast<real_type>(response_error) * response_error / (threshold * threshold);

    std::generate(feedback_to_clauses.begin(), feedback_to_clauses.end(), [&fgen, R2](){ return fgen.next() < R2; });
}


template<typename state_type>
void train_regressor_automata(
    numeric_matrix<state_type> & ta_state,
    w_vector_type & weights,
    int const input_begin_ix,
    int const input_end_ix,
    char const * __restrict clause_output,
    int const number_of_states,
    int const response_error,
    aligned_vector_char const & X,
    bool const boost_true_positive_feedback,
    IRNG & prng,
    unsigned int const threshold,
    EstimatorStateCacheBase::coin_tosser_type & ct
    )
{
    int const number_of_features = X.size();
    int const feedback_hits = std::round((input_end_ix - input_begin_ix) *
        static_cast<real_type>(response_error) * response_error / (threshold * threshold));

    for (int idx = 0; idx < feedback_hits; ++idx)
    {
        // randomly pick index that corresponds to non-zero feedback
        auto const iidx = prng() % (input_end_ix - input_begin_ix) + input_begin_ix;

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
                if (boost_true_positive_feedback)
                {
                    block2<true>(number_of_features, number_of_states, ta_state_pos_j, ta_state_neg_j, X.data(), ct.tosses1(prng), ct.tosses2(prng));
                }
                else
                {
                    block2<false>(number_of_features, number_of_states, ta_state_pos_j, ta_state_neg_j, X.data(), ct.tosses1(prng), ct.tosses2(prng));
                }

                if (weights.size() != 0)
                {
                    weights[iidx]++;
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
    int const number_of_states,
    int const response_error,
    aligned_vector_char const & X,
    bool const boost_true_positive_feedback,
    IRNG & prng,
    unsigned int const threshold,
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
