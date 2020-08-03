#pragma once

#ifndef LIB_SRC_TSETLINI_ALGO_BITWISE_HPP_
#define LIB_SRC_TSETLINI_ALGO_BITWISE_HPP_

#include "ta_state.hpp"
#include "estimator_state_cache.hpp"
#include "tsetlini_types.hpp"
#include "basic_bit_vector.hpp"
#include "assume_aligned.hpp"

namespace Tsetlini
{


namespace
{


/*
 * https://godbolt.org/z/WEhrnc
 */
template<unsigned int BATCH_SZ, typename bit_block_type>
inline
void calculate_clause_output_T(
    bit_vector<bit_block_type> const & X,
    aligned_vector_char & clause_output,
    int const output_begin_ix,
    int const output_end_ix,
    TAStateWithSignum::value_type const & ta_state,
    int const n_jobs)
{
    auto const & ta_state_signum = ta_state.signum;
    bit_block_type const * X_p = assume_aligned<alignment>(X.data());
    int const feature_blocks = X.content_blocks();

    if (feature_blocks < (int)BATCH_SZ)
    {
        for (int oidx = output_begin_ix; oidx < output_end_ix; ++oidx)
        {
            bool output = true;

            bit_block_type const * ta_sign_pos_j = assume_aligned<alignment>(ta_state_signum.row_data(2 * oidx + 0));
            bit_block_type const * ta_sign_neg_j = assume_aligned<alignment>(ta_state_signum.row_data(2 * oidx + 1));

            for (int fidx = 0; fidx < feature_blocks and output == true; ++fidx)
            {
                bit_block_type const action_include = ta_sign_pos_j[fidx];
                bit_block_type const action_include_negated = ta_sign_neg_j[fidx];
                bit_block_type const features = X_p[fidx];

                bit_block_type const eval = (action_include & ~features) | (action_include_negated & features);
                output = (eval == 0);
            }

            clause_output[oidx] = output;
        }
    }
    else
    {
#pragma omp parallel for if (n_jobs > 1) num_threads(n_jobs)
        for (int oidx = output_begin_ix; oidx < output_end_ix; ++oidx)
        {
            bit_block_type toggle_output = 0;

            bit_block_type const * ta_sign_pos_j = assume_aligned<alignment>(ta_state_signum.row_data(2 * oidx + 0));
            bit_block_type const * ta_sign_neg_j = assume_aligned<alignment>(ta_state_signum.row_data(2 * oidx + 1));

            unsigned int kk = 0;
            for (; kk < feature_blocks - (BATCH_SZ - 1); kk += BATCH_SZ)
            {
                for (auto fidx = kk; fidx < BATCH_SZ + kk; ++fidx)
                {
                    bit_block_type const action_include = ta_sign_pos_j[fidx];
                    bit_block_type const action_include_negated = ta_sign_neg_j[fidx];
                    bit_block_type const features = X_p[fidx];

                    bit_block_type const eval = (action_include & ~features) | (action_include_negated & features);

                    toggle_output = eval > toggle_output ? eval : toggle_output;
                }
                if (toggle_output != 0)
                {
                    break;
                }
            }
            for (int fidx = kk; fidx < feature_blocks and toggle_output == 0; ++fidx)
            {
                bit_block_type const action_include = ta_sign_pos_j[fidx];
                bit_block_type const action_include_negated = ta_sign_neg_j[fidx];
                bit_block_type const features = X_p[fidx];

                bit_block_type const eval = (action_include & ~features) | (action_include_negated & features);

                toggle_output = eval > toggle_output ? eval : toggle_output;
            }

            clause_output[oidx] = !toggle_output;
        }
    }
}


/*
 * https://godbolt.org/z/1bKs9b
 */
template<unsigned int BATCH_SZ, typename bit_block_type>
inline
void calculate_clause_output_for_predict_T(
    bit_vector<bit_block_type> const & X,
    aligned_vector_char & clause_output,
    int const number_of_clauses,
    TAStateWithSignum::value_type const & ta_state,
    int const n_jobs)
{
    auto const & ta_state_signum = ta_state.signum;
    bit_block_type const * X_p = assume_aligned<alignment>(X.data());
    int const feature_blocks = X.content_blocks();

    if (feature_blocks < (int)BATCH_SZ)
    {
        for (int oidx = 0; oidx < number_of_clauses; ++oidx)
        {
            bool output = true;

            bit_block_type const * ta_sign_pos_j = assume_aligned<alignment>(ta_state_signum.row_data(2 * oidx + 0));
            bit_block_type const * ta_sign_neg_j = assume_aligned<alignment>(ta_state_signum.row_data(2 * oidx + 1));

            bit_block_type any_inclusions = 0;

            for (int fidx = 0; fidx < feature_blocks and output == true; ++fidx)
            {
                bit_block_type const action_include = ta_sign_pos_j[fidx];
                bit_block_type const action_include_negated = ta_sign_neg_j[fidx];
                bit_block_type const features = X_p[fidx];
                any_inclusions = (action_include | action_include_negated) > any_inclusions ? (action_include | action_include_negated) : any_inclusions;

                bit_block_type const eval = (action_include & ~features) | (action_include_negated & features);
                output = (eval == 0);
            }

            output = any_inclusions > 0 ? output : false;

            clause_output[oidx] = output;
        }
    }
    else
    {
#pragma omp parallel for if (n_jobs > 1) num_threads(n_jobs)
        for (int oidx = 0; oidx < number_of_clauses; ++oidx)
        {
            bit_block_type toggle_output = 0;
            bit_block_type any_inclusions = 0;

            bit_block_type const * ta_sign_pos_j = assume_aligned<alignment>(ta_state_signum.row_data(2 * oidx + 0));
            bit_block_type const * ta_sign_neg_j = assume_aligned<alignment>(ta_state_signum.row_data(2 * oidx + 1));

            unsigned int kk = 0;
            for (; kk < feature_blocks - (BATCH_SZ - 1); kk += BATCH_SZ)
            {
                for (auto fidx = kk; fidx < BATCH_SZ + kk; ++fidx)
                {
                    bit_block_type const action_include = ta_sign_pos_j[fidx];
                    bit_block_type const action_include_negated = ta_sign_neg_j[fidx];
                    bit_block_type const features = X_p[fidx];
                    any_inclusions = (action_include | action_include_negated) > any_inclusions ? (action_include | action_include_negated) : any_inclusions;

                    bit_block_type const eval = (action_include & ~features) | (action_include_negated & features);

                    toggle_output = eval > toggle_output ? eval : toggle_output;
                }
                if (toggle_output != 0)
                {
                    break;
                }
            }
            for (int fidx = kk; fidx < feature_blocks and toggle_output == 0; ++fidx)
            {
                bit_block_type const action_include = ta_sign_pos_j[fidx];
                bit_block_type const action_include_negated = ta_sign_neg_j[fidx];
                bit_block_type const features = X_p[fidx];
                any_inclusions = (action_include | action_include_negated) > any_inclusions ? (action_include | action_include_negated) : any_inclusions;

                bit_block_type const eval = (action_include & ~features) | (action_include_negated & features);

                toggle_output = eval > toggle_output ? eval : toggle_output;
            }

            toggle_output = any_inclusions > 0 ? toggle_output : 1;

            clause_output[oidx] = !toggle_output;
        }
    }
}


// Feedback Type I, negative
template<typename state_type, typename bit_block_type>
void block1(
    int const number_of_features,
    int const number_of_states,
    state_type * __restrict ta_state_pos_j,
    state_type * __restrict ta_state_neg_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_pos_signum_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_neg_signum_j,
    char const * __restrict ct_pos,
    char const * __restrict ct_neg
)
{
    ta_state_pos_j = assume_aligned<alignment>(ta_state_pos_j);
    ta_state_neg_j = assume_aligned<alignment>(ta_state_neg_j);

    ct_pos = assume_aligned<alignment>(ct_pos);
    ct_neg = assume_aligned<alignment>(ct_neg);

    // TODO: check vectorization
    for (int fidx = 0; fidx < number_of_features; ++fidx)
    {
        {
            auto cond = ct_pos[fidx];

            if (ta_state_pos_j[fidx] == 0 and cond)
            {
                ta_state_pos_signum_j.flip(fidx); // flip positive clause bit in signum matrix
            }

            ta_state_pos_j[fidx] = cond ? (ta_state_pos_j[fidx] > -number_of_states ? ta_state_pos_j[fidx] - 1 : ta_state_pos_j[fidx]) : ta_state_pos_j[fidx];
        }

        {
            auto cond = ct_neg[fidx];

            if (ta_state_neg_j[fidx] == 0 and cond)
            {
                ta_state_neg_signum_j.flip(fidx); // flip negative clause bit in signum matrix
            }

            ta_state_neg_j[fidx] = cond ? (ta_state_neg_j[fidx] > -number_of_states ? ta_state_neg_j[fidx] - 1 : ta_state_neg_j[fidx]) : ta_state_neg_j[fidx];
        }
    }
}


// Feedback Type I, positive
template<bool boost_true_positive_feedback, typename state_type, typename bit_block_type>
void block2(
    int const number_of_states,
    state_type * __restrict ta_state_pos_j,
    state_type * __restrict ta_state_neg_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_pos_signum_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_neg_signum_j,
    bit_vector<bit_block_type> const & X,
    char const * __restrict ct_pos,
    char const * __restrict ct_neg
)
{
    int const number_of_features = X.size();

    ta_state_pos_j = assume_aligned<alignment>(ta_state_pos_j);
    ta_state_neg_j = assume_aligned<alignment>(ta_state_neg_j);

    ct_pos = assume_aligned<alignment>(ct_pos);
    ct_neg = assume_aligned<alignment>(ct_neg);

    for (int fidx = 0; fidx < number_of_features; ++fidx)
    {
        auto cond1_pos = boost_true_positive_feedback == true or not ct_pos[fidx];
        auto cond2_pos = ct_pos[fidx];

        auto cond1_neg = boost_true_positive_feedback == true or not ct_neg[fidx];
        auto cond2_neg = ct_neg[fidx];

        if (X[fidx] != 0)
        {
            if (cond1_pos)
            {
                if (ta_state_pos_j[fidx] < number_of_states - 1)
                {
                    ta_state_pos_j[fidx]++;
                }

                if (ta_state_pos_j[fidx] == 0)
                {
                    ta_state_pos_signum_j.flip(fidx); // flip positive clause bit in signum matrix
                }

            }
            if (cond2_neg)
            {
                if (ta_state_neg_j[fidx] == 0)
                {
                    ta_state_neg_signum_j.flip(fidx); // flip negative clause bit in signum matrix
                }

                if (ta_state_neg_j[fidx] > -number_of_states)
                {
                    ta_state_neg_j[fidx]--;
                }
            }
        }
        else // if (X[k] == 0)
        {
            if (cond1_neg)
            {
                if (ta_state_neg_j[fidx] < number_of_states - 1)
                {
                    ta_state_neg_j[fidx]++;

                    if (ta_state_neg_j[fidx] == 0)
                    {
                        ta_state_neg_signum_j.flip(fidx); // flip negative clause bit in signum matrix
                    }
                }
            }

            if (cond2_pos)
            {
                if (ta_state_pos_j[fidx] == 0)
                {
                    ta_state_pos_signum_j.flip(fidx); // flip positive clause bit in signum matrix
                }

                if (ta_state_pos_j[fidx] > -number_of_states)
                {
                    ta_state_pos_j[fidx]--;
                }
            }
        }
    }
}


// Feedback Type II
template<typename state_type, typename bit_block_type>
void block3(
    int const number_of_features,
    state_type * __restrict ta_state_pos_j,
    state_type * __restrict ta_state_neg_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_pos_signum_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_neg_signum_j,
    bit_vector<bit_block_type> const & X
)
{
    ta_state_pos_j = assume_aligned<alignment>(ta_state_pos_j);
    ta_state_neg_j = assume_aligned<alignment>(ta_state_neg_j);

    // TODO: check vectorization
    for (int fidx = 0; fidx < number_of_features; ++fidx)
    {
        if (X[fidx] == 0)
        {
            auto action_include = (ta_state_pos_j[fidx] >= 0);

            if (action_include == false)
            {
                ta_state_pos_j[fidx]++;

                if (ta_state_pos_j[fidx] == 0)
                {
                    ta_state_pos_signum_j.flip(fidx); // flip positive clause bit in signum matrix
                }

            }
        }
        else //if(X[k] == 1)
        {
            auto action_include_negated = (ta_state_neg_j[fidx] >= 0);

            if (action_include_negated == false)
            {
                ta_state_neg_j[fidx]++;

                if (ta_state_neg_j[fidx] == 0)
                {
                    ta_state_neg_signum_j.flip(fidx); // flip negative clause bit in signum matrix
                }

            }
        }
    }
}


template<typename state_type, typename bit_block_type>
void train_classifier_automata_T(
    numeric_matrix<state_type> & ta_state_matrix,
    bit_matrix<bit_block_type> & ta_state_signum,
    int const input_begin_ix,
    int const input_end_ix,
    feedback_vector_type::value_type const * __restrict feedback_to_clauses,
    char const * __restrict clause_output,
    int const number_of_states,
    bit_vector<bit_block_type> const & X,
    bool const boost_true_positive_feedback,
    IRNG & prng,
    EstimatorStateCacheBase::coin_tosser_type & ct
    )
{
    int const number_of_features = X.size();

    for (int iidx = input_begin_ix; iidx < input_end_ix; ++iidx)
    {
        state_type * ta_state_pos_j = ::assume_aligned<alignment>(ta_state_matrix.row_data(2 * iidx + 0));
        state_type * ta_state_neg_j = ::assume_aligned<alignment>(ta_state_matrix.row_data(2 * iidx + 1));

        if (feedback_to_clauses[iidx] > 0)
        {
            if (clause_output[iidx] == 0)
            {
                block1<state_type, bit_block_type>(number_of_features, number_of_states,
                    ta_state_pos_j,
                    ta_state_neg_j,
                    ta_state_signum.row(2 * iidx + 0),
                    ta_state_signum.row(2 * iidx + 1),
                    ct.tosses(prng), ct.tosses(prng));
            }
            else // if (clause_output[iidx] == 1)
            {
                if (boost_true_positive_feedback)
                {
                    block2<true>(number_of_states,
                        ta_state_pos_j,
                        ta_state_neg_j,
                        ta_state_signum.row(2 * iidx + 0),
                        ta_state_signum.row(2 * iidx + 1),
                        X, ct.tosses(prng), ct.tosses(prng));
                }
                else
                {
                    block2<false>(number_of_states,
                        ta_state_pos_j,
                        ta_state_neg_j,
                        ta_state_signum.row(2 * iidx + 0),
                        ta_state_signum.row(2 * iidx + 1),
                        X, ct.tosses(prng), ct.tosses(prng));
                }
            }
        }
        else if (feedback_to_clauses[iidx] < 0)
        {
            if (clause_output[iidx] == 1)
            {
                block3<state_type, bit_block_type>(number_of_features,
                    ta_state_pos_j,
                    ta_state_neg_j,
                    ta_state_signum.row(2 * iidx + 0),
                    ta_state_signum.row(2 * iidx + 1),
                    X);
            }
        }
    }
}


template<typename bit_block_type>
void train_classifier_automata(
    TAStateWithSignum::value_type & ta_state,
    int const input_begin_ix,
    int const input_end_ix,
    feedback_vector_type::value_type const * __restrict feedback_to_clauses,
    char const * __restrict clause_output,
    int const number_of_states,
    bit_vector<bit_block_type> const & X,
    bool const boost_true_positive_feedback,
    IRNG & prng,
    EstimatorStateCacheBase::coin_tosser_type & ct
    )
{
    auto & ta_state_variant = ta_state.matrix;
    auto & ta_state_signum = ta_state.signum;

    std::visit(
        [&](auto & ta_state_matrix)
        {
            train_classifier_automata_T(
                ta_state_matrix,
                ta_state_signum,
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
        ta_state_variant);
}


template<typename state_type, typename bit_block_type>
void train_regressor_automata(
    numeric_matrix<state_type> & ta_state_matrix,
    bit_matrix<bit_block_type> & ta_state_signum,
    int const input_begin_ix,
    int const input_end_ix,
    feedback_vector_type::value_type const * __restrict feedback_to_clauses,
    char const * __restrict clause_output,
    int const number_of_states,
    int const response_error,
    bit_vector<bit_block_type> const & X,
    bool const boost_true_positive_feedback,
    IRNG & prng,
    EstimatorStateCacheBase::coin_tosser_type & ct
    )
{
    int const number_of_features = X.size();

    for (int iidx = input_begin_ix; iidx < input_end_ix; ++iidx)
    {
        state_type * ta_state_pos_j = ::assume_aligned<alignment>(ta_state_matrix.row_data(2 * iidx + 0));
        state_type * ta_state_neg_j = ::assume_aligned<alignment>(ta_state_matrix.row_data(2 * iidx + 1));

        if (feedback_to_clauses[iidx] == 0)
        {
            continue;
        }

        if (response_error < 0)
        {
            if (clause_output[iidx] == 0)
            {
                block1<state_type, bit_block_type>(number_of_features, number_of_states,
                    ta_state_pos_j,
                    ta_state_neg_j,
                    ta_state_signum.row(2 * iidx + 0),
                    ta_state_signum.row(2 * iidx + 1),
                    ct.tosses(prng), ct.tosses(prng));
            }
            else // if (clause_output[iidx] == 1)
            {
                if (boost_true_positive_feedback)
                {
                    block2<true>(number_of_states,
                        ta_state_pos_j,
                        ta_state_neg_j,
                        ta_state_signum.row(2 * iidx + 0),
                        ta_state_signum.row(2 * iidx + 1),
                        X, ct.tosses(prng), ct.tosses(prng));
                }
                else
                {
                    block2<false>(number_of_states,
                        ta_state_pos_j,
                        ta_state_neg_j,
                        ta_state_signum.row(2 * iidx + 0),
                        ta_state_signum.row(2 * iidx + 1),
                        X, ct.tosses(prng), ct.tosses(prng));
                }
            }
        }
        else if (response_error > 0)
        {
            if (clause_output[iidx] == 1)
            {
                block3<state_type, bit_block_type>(number_of_features,
                    ta_state_pos_j,
                    ta_state_neg_j,
                    ta_state_signum.row(2 * iidx + 0),
                    ta_state_signum.row(2 * iidx + 1),
                    X);
            }
        }
    }
}


template<typename bit_block_type>
void train_regressor_automata(
    TAStateWithSignum::value_type & ta_state,
    int const input_begin_ix,
    int const input_end_ix,
    feedback_vector_type::value_type const * __restrict feedback_to_clauses,
    char const * __restrict clause_output,
    int const number_of_states,
    int const response_error,
    bit_vector<bit_block_type> const & X,
    bool const boost_true_positive_feedback,
    IRNG & prng,
    EstimatorStateCacheBase::coin_tosser_type & ct
    )
{
    auto & ta_state_variant = ta_state.matrix;
    auto & ta_state_signum = ta_state.signum;

    std::visit(
        [&](auto & ta_state_values)
        {
            train_regressor_automata(
                ta_state_values,
                ta_state_signum,
                input_begin_ix,
                input_end_ix,
                feedback_to_clauses,
                clause_output,
                number_of_states,
                response_error,
                X,
                boost_true_positive_feedback,
                prng,
                ct
            );
        },
        ta_state_variant
    );
}


}  // anonymous namespace


}  // namespace Tsetlini


#endif /* LIB_SRC_TSETLINI_ALGO_BITWISE_HPP_ */
