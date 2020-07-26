#pragma once

#ifndef LIB_SRC_TSETLINI_ALGO_CLASSIC_HPP_
#define LIB_SRC_TSETLINI_ALGO_CLASSIC_HPP_

#include "estimator_state.hpp"
#include "estimator_state_cache.hpp"
#include "tsetlini_types.hpp"

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
#pragma omp parallel for if (n_jobs > 1) num_threads(n_jobs)
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
#pragma omp parallel for if (n_jobs > 1) num_threads(n_jobs)
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


// Feedback Type I, negative
template<typename state_type>
int block1(
    int const number_of_features,
    int const number_of_states,
    float const S_inv,
    state_type * __restrict ta_state_pos_j,
    state_type * __restrict ta_state_neg_j,
    float const * __restrict fcache,
    int fcache_pos
)
{
    fcache = assume_aligned<alignment>(fcache);
    ta_state_pos_j = assume_aligned<alignment>(ta_state_pos_j);
    ta_state_neg_j = assume_aligned<alignment>(ta_state_neg_j);

    for (int fidx = 0; fidx < number_of_features; ++fidx)
    {
        {
            auto cond = fcache[fcache_pos++] <= S_inv;

            ta_state_pos_j[fidx] = cond ? (ta_state_pos_j[fidx] > -number_of_states ? ta_state_pos_j[fidx] - 1 : ta_state_pos_j[fidx]) : ta_state_pos_j[fidx];
        }

        {
            auto cond = fcache[fcache_pos++] <= S_inv;

            ta_state_neg_j[fidx] = cond ? (ta_state_neg_j[fidx] > -number_of_states ? ta_state_neg_j[fidx] - 1 : ta_state_neg_j[fidx]) : ta_state_neg_j[fidx];
        }
    }
    return fcache_pos;
}


// Feedback Type I, positive
template<bool boost_true_positive_feedback, typename state_type>
int block2(
    int const number_of_features,
    int const number_of_states,
    float const S_inv,
    state_type * __restrict ta_state_pos_j,
    state_type * __restrict ta_state_neg_j,
    char const * __restrict X,
    float const * __restrict fcache,
    int fcache_pos
)
{
    constexpr float ONE = 1.0f;
    fcache = assume_aligned<alignment>(fcache);
    ta_state_pos_j = assume_aligned<alignment>(ta_state_pos_j);
    ta_state_neg_j = assume_aligned<alignment>(ta_state_neg_j);
//    X = assume_aligned(X);

    for (int fidx = 0; fidx < number_of_features; ++fidx)
    {
        auto cond1 = boost_true_positive_feedback == true or (fcache[fcache_pos++] <= (ONE - S_inv));
        auto cond2 = fcache[fcache_pos++] <= S_inv;

        if (X[fidx] != 0)
        {
            if (cond1)
            {
                if (ta_state_pos_j[fidx] < number_of_states - 1)
                {
                    ta_state_pos_j[fidx]++;
                }
            }
            if (cond2)
            {
                if (ta_state_neg_j[fidx] > -number_of_states)
                {
                    ta_state_neg_j[fidx]--;
                }
            }
        }
        else // if (X[k] == 0)
        {
            if (cond1)
            {
                if (ta_state_neg_j[fidx] < number_of_states - 1)
                {
                    ta_state_neg_j[fidx]++;
                }
            }

            if (cond2)
            {
                if (ta_state_pos_j[fidx] > -number_of_states)
                {
                    ta_state_pos_j[fidx]--;
                }
            }
        }
    }

    return fcache_pos;
}


// Feedback Type II
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
}


template<typename state_type>
void train_classifier_automata(
    numeric_matrix<state_type> & ta_state,
    int const input_begin_ix,
    int const input_end_ix,
    feedback_vector_type::value_type const * __restrict feedback_to_clauses,
    char const * __restrict clause_output,
    int const number_of_states,
    float const S_inv,
    aligned_vector_char const & X,
    bool const boost_true_positive_feedback,
    FRNG & frng,
    EstimatorStateCacheBase::frand_cache_type & fcache
    )
{
    int const number_of_features = X.size();
    float const * fcache_ = assume_aligned<alignment>(fcache.m_fcache.data());

    for (int iidx = input_begin_ix; iidx < input_end_ix; ++iidx)
    {
        state_type * ta_state_pos_j = ::assume_aligned<alignment>(ta_state.row_data(2 * iidx + 0));
        state_type * ta_state_neg_j = ::assume_aligned<alignment>(ta_state.row_data(2 * iidx + 1));

        if (feedback_to_clauses[iidx] > 0)
        {
            if (clause_output[iidx] == 0)
            {
                fcache.refill(frng);

                fcache.m_pos = block1(number_of_features, number_of_states, S_inv, ta_state_pos_j, ta_state_neg_j, fcache_, fcache.m_pos);
            }
            else // if (clause_output[iidx] == 1)
            {
                fcache.refill(frng);

                if (boost_true_positive_feedback)
                {
                    fcache.m_pos = block2<true>(number_of_features, number_of_states, S_inv, ta_state_pos_j, ta_state_neg_j, X.data(), fcache_, fcache.m_pos);
                }
                else
                {
                    fcache.m_pos = block2<false>(number_of_features, number_of_states, S_inv, ta_state_pos_j, ta_state_neg_j, X.data(), fcache_, fcache.m_pos);
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
    int const threshold)
{
    auto const sum = std::accumulate(clause_output.cbegin(), clause_output.cend(), 0);

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
    int const input_begin_ix,
    int const input_end_ix,
    feedback_vector_type::value_type const * __restrict feedback_to_clauses,
    char const * __restrict clause_output,
    int const number_of_states,
    float const S_inv,
    int const response_error,
    aligned_vector_char const & X,
    bool const boost_true_positive_feedback,
    FRNG & frng,
    EstimatorStateCacheBase::frand_cache_type & fcache
    )
{
    int const number_of_features = X.size();
    float const * fcache_ = assume_aligned<alignment>(fcache.m_fcache.data());

    for (int iidx = input_begin_ix; iidx < input_end_ix; ++iidx)
    {
        state_type * ta_state_pos_j = ::assume_aligned<alignment>(ta_state.row_data(2 * iidx + 0));
        state_type * ta_state_neg_j = ::assume_aligned<alignment>(ta_state.row_data(2 * iidx + 1));

        if (feedback_to_clauses[iidx] == 0)
        {
            continue;
        }

        if (response_error < 0)
        {
            if (clause_output[iidx] == 0)
            {
                fcache.refill(frng);

                fcache.m_pos = block1(number_of_features, number_of_states, S_inv, ta_state_pos_j, ta_state_neg_j, fcache_, fcache.m_pos);
            }
            else // if (clause_output[iidx] == 1)
            {
                fcache.refill(frng);

                if (boost_true_positive_feedback)
                    fcache.m_pos = block2<true>(number_of_features, number_of_states, S_inv, ta_state_pos_j, ta_state_neg_j, X.data(), fcache_, fcache.m_pos);
                else
                    fcache.m_pos = block2<false>(number_of_features, number_of_states, S_inv, ta_state_pos_j, ta_state_neg_j, X.data(), fcache_, fcache.m_pos);
            }
        }
        else if (response_error > 0)
        {
            if (clause_output[iidx] == 1)
            {
                block3(number_of_features, ta_state_pos_j, ta_state_neg_j, X.data());
            }
        }
    }
}


} // anonymous namespace


} // namespace Tsetlini

#endif /* LIB_SRC_TSETLINI_ALGO_CLASSIC_HPP_ */
