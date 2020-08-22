#pragma once

#ifndef LIB_SRC_TSETLINI_ALGO_BITWISE_HPP_
#define LIB_SRC_TSETLINI_ALGO_BITWISE_HPP_

#include "ta_state.hpp"
#include "estimator_state_cache.hpp"
#include "tsetlini_types.hpp"
#include "basic_bit_vector.hpp"
#include "assume_aligned.hpp"


#ifndef TSETLINI_USE_OMP
#define TSETLINI_USE_OMP 1
#endif


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
#if TSETLINI_USE_OMP == 1
#pragma omp parallel for if (n_jobs > 1) num_threads(n_jobs)
#endif
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
#if TSETLINI_USE_OMP == 1
#pragma omp parallel for if (n_jobs > 1) num_threads(n_jobs)
#endif
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


template<unsigned int BATCH_SZ, typename bit_block_type>
inline
void calculate_clause_output_T(
    bit_vector<bit_block_type> const & X,
    bit_vector<bit_block_type> & clause_output,
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
#if TSETLINI_USE_OMP == 1
#pragma omp parallel for if (n_jobs > 1) num_threads(n_jobs)
#endif
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

            clause_output.assign(oidx, output);
        }
    }
    else
    {
#if TSETLINI_USE_OMP == 1
#pragma omp parallel for if (n_jobs > 1) num_threads(n_jobs)
#endif
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

            clause_output.assign(oidx, !toggle_output);
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
#if TSETLINI_USE_OMP == 1
#pragma omp parallel for if (n_jobs > 1) num_threads(n_jobs)
#endif
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
#if TSETLINI_USE_OMP == 1
#pragma omp parallel for if (n_jobs > 1) num_threads(n_jobs)
#endif
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


template<unsigned int BATCH_SZ, typename bit_block_type>
inline
void calculate_clause_output_for_predict_T(
    bit_vector<bit_block_type> const & X,
    bit_vector<bit_block_type> clause_output,
    int const number_of_clauses,
    TAStateWithSignum::value_type const & ta_state,
    int const n_jobs)
{
    auto const & ta_state_signum = ta_state.signum;
    bit_block_type const * X_p = assume_aligned<alignment>(X.data());
    int const feature_blocks = X.content_blocks();

    if (feature_blocks < (int)BATCH_SZ)
    {
#if TSETLINI_USE_OMP == 1
#pragma omp parallel for if (n_jobs > 1) num_threads(n_jobs)
#endif
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

            clause_output.assign(oidx, output);
        }
    }
    else
    {
#if TSETLINI_USE_OMP == 1
#pragma omp parallel for if (n_jobs > 1) num_threads(n_jobs)
#endif
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

            clause_output.assign(oidx, !toggle_output);
        }
    }
}


/*
 * Feedback Type I, negative
 *
 * https://godbolt.org/z/WK9bGc
 */
template<typename state_type, typename bit_block_type>
inline
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

            if (UNLIKELY(cond))
            {
                auto const ta_state = ta_state_pos_j[fidx];

                if (ta_state == 0)
                {
                    ta_state_pos_signum_j.flip(fidx); // flip positive clause bit in signum matrix
                }

                if (ta_state > -number_of_states)
                {
                    --ta_state_pos_j[fidx];
                }
            }
        }

        {
            auto cond = ct_neg[fidx];

            if (UNLIKELY(cond))
            {
                auto const ta_state = ta_state_neg_j[fidx];

                if (ta_state == 0)
                {
                    ta_state_neg_signum_j.flip(fidx); // flip positive clause bit in signum matrix
                }

                if (ta_state > -number_of_states)
                {
                    --ta_state_neg_j[fidx];
                }
            }
        }
    }
}


template<typename state_type, typename bit_block_type, typename PRNG>
inline
void block1_sparse(
    int const number_of_features,
    int const number_of_sparse_features,
    int const number_of_states,
    state_type * __restrict ta_state_pos_j,
    state_type * __restrict ta_state_neg_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_pos_signum_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_neg_signum_j,
    PRNG & prng
)
{
    for (int sfidx = 0; sfidx < number_of_sparse_features; ++sfidx)
    {
        {
            auto const fidx = prng() % number_of_features;
            auto const ta_state = ta_state_pos_j[fidx];

            if (ta_state == 0)
            {
                ta_state_pos_signum_j.flip(fidx); // flip positive clause bit in signum matrix
            }

            if (ta_state > -number_of_states)
            {
                --ta_state_pos_j[fidx];
            }
        }

        {
            auto const fidx = prng() % number_of_features;
            auto const ta_state = ta_state_neg_j[fidx];

            if (ta_state == 0)
            {
                ta_state_neg_signum_j.flip(fidx); // flip positive clause bit in signum matrix
            }

            if (ta_state > -number_of_states)
            {
                --ta_state_neg_j[fidx];
            }
        }
    }
}


/*
 * https://godbolt.org/z/v8nrf1
 */
template<typename state_type, typename bit_block_type>
inline
void block1(
    int const number_of_features,
    int const number_of_states,
    state_type * __restrict ta_state_pos_j,
    state_type * __restrict ta_state_neg_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_pos_signum_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_neg_signum_j,
    bit_block_type const * __restrict ct_pos_p,
    bit_block_type const * __restrict ct_neg_p
)
{
    using bit_view_type = typename std::decay<decltype(ta_state_pos_signum_j)>::type;
    auto constexpr block_bits = bit_view_type::block_bits;
    int const full_feature_blocks = ta_state_pos_signum_j.content_blocks() - (ta_state_pos_signum_j.tail_bits() != 0);

    ta_state_pos_j = assume_aligned<alignment>(ta_state_pos_j);
    ta_state_neg_j = assume_aligned<alignment>(ta_state_neg_j);

    bit_block_type * ta_state_pos_signum_j_p = assume_aligned<alignment>(ta_state_pos_signum_j.data());
    bit_block_type * ta_state_neg_signum_j_p = assume_aligned<alignment>(ta_state_neg_signum_j.data());

    ct_pos_p = assume_aligned<alignment>(ct_pos_p);
    ct_neg_p = assume_aligned<alignment>(ct_neg_p);

    auto process_block = [&](auto fidx, auto this_block_bits)
    {
        bit_block_type pos_dec = ct_pos_p[fidx];
        bit_block_type neg_dec = ct_neg_p[fidx];

        bit_block_type pos_signum_flip = 0;
        bit_block_type neg_signum_flip = 0;

        for (auto bix = 0u; bix < this_block_bits; ++bix)
        {
            auto pos_dec_bit = pos_dec & (ta_state_pos_j[fidx * block_bits + bix] > -number_of_states);
            auto neg_dec_bit = neg_dec & (ta_state_neg_j[fidx * block_bits + bix] > -number_of_states);

            // set the flip bit if the state BEFORE decrementation was 0 and the decrementation will take place
            pos_signum_flip |= ((bit_block_type)(0 == ta_state_pos_j[fidx * block_bits + bix]) & pos_dec_bit) << bix;
            neg_signum_flip |= ((bit_block_type)(0 == ta_state_neg_j[fidx * block_bits + bix]) & neg_dec_bit) << bix;

            ta_state_pos_j[fidx * block_bits + bix] -= pos_dec_bit;
            ta_state_neg_j[fidx * block_bits + bix] -= neg_dec_bit;

            pos_dec >>= 1;
            neg_dec >>= 1;
        }

        ta_state_pos_signum_j_p[fidx] ^= pos_signum_flip;
        ta_state_neg_signum_j_p[fidx] ^= neg_signum_flip;

    };

    for (unsigned int fidx = 0; fidx < full_feature_blocks; ++fidx)
    {
        process_block(fidx, block_bits);
    }

    if (ta_state_pos_signum_j.tail_bits() != 0)
    {
        auto last_block_idx = full_feature_blocks;
        process_block(last_block_idx, ta_state_pos_signum_j.tail_bits());
    }
}


// Feedback Type I, positive
template<bool boost_true_positive_feedback, typename state_type, typename bit_block_type>
inline
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


/*
 * https://godbolt.org/z/aWTqzv
 */
template<bool boost_true_positive_feedback, typename state_type, typename bit_block_type>
inline
void block2(
    int const number_of_states,
    state_type * __restrict ta_state_pos_j,
    state_type * __restrict ta_state_neg_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_pos_signum_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_neg_signum_j,
    bit_vector<bit_block_type> const & X,
    bit_block_type const * ct_pos_p, // TODO restrict
    bit_block_type const * ct_neg_p
)
{
    auto constexpr block_bits = bit_vector<bit_block_type>::block_bits;
    int const full_feature_blocks = X.content_blocks() - (X.tail_bits() != 0);

    bit_block_type const * X_p = assume_aligned<alignment>(X.data());

    ta_state_pos_j = assume_aligned<alignment>(ta_state_pos_j);
    ta_state_neg_j = assume_aligned<alignment>(ta_state_neg_j);

    bit_block_type * ta_state_pos_signum_j_p = assume_aligned<alignment>(ta_state_pos_signum_j.data());
    bit_block_type * ta_state_neg_signum_j_p = assume_aligned<alignment>(ta_state_neg_signum_j.data());

    ct_pos_p = assume_aligned<alignment>(ct_pos_p);
    ct_neg_p = assume_aligned<alignment>(ct_neg_p);

    auto process_block = [&](auto fidx, auto this_block_bits)
    {
        bit_block_type X_0 = ~X_p[fidx];
        bit_block_type X_1 = X_p[fidx];

        bit_block_type pos_inc = X_1 & (boost_true_positive_feedback ? -1 : ~ct_pos_p[fidx]);
        bit_block_type pos_dec = X_0 & ct_pos_p[fidx];
        bit_block_type neg_inc = X_0 & (boost_true_positive_feedback ? -1 : ~ct_neg_p[fidx]);;
        bit_block_type neg_dec = X_1 & ct_neg_p[fidx];

        bit_block_type pos_signum_flip = 0;
        bit_block_type neg_signum_flip = 0;

        for (auto bix = 0u; bix < this_block_bits; ++bix)
        {
            auto pos_inc_bit = pos_inc & (ta_state_pos_j[fidx * block_bits + bix] < number_of_states - 1);
            auto pos_dec_bit = pos_dec & (ta_state_pos_j[fidx * block_bits + bix] > -number_of_states);

            auto neg_inc_bit = neg_inc & (ta_state_neg_j[fidx * block_bits + bix] < number_of_states - 1);
            auto neg_dec_bit = neg_dec & (ta_state_neg_j[fidx * block_bits + bix] > -number_of_states);

            // set the flip bit if the state BEFORE decrementation was 0 and the decrementation will take place
            pos_signum_flip |= ((bit_block_type)(0 == ta_state_pos_j[fidx * block_bits + bix]) & pos_dec_bit) << bix;
            neg_signum_flip |= ((bit_block_type)(0 == ta_state_neg_j[fidx * block_bits + bix]) & neg_dec_bit) << bix;

            ta_state_pos_j[fidx * block_bits + bix] += pos_inc_bit;
            ta_state_pos_j[fidx * block_bits + bix] -= pos_dec_bit;
            ta_state_neg_j[fidx * block_bits + bix] += neg_inc_bit;
            ta_state_neg_j[fidx * block_bits + bix] -= neg_dec_bit;

            // set the flip bit if the state AFTER incrementation was 0 and the incrementation took place
            pos_signum_flip |= ((bit_block_type)(0 == ta_state_pos_j[fidx * block_bits + bix]) & pos_inc_bit) << bix;
            neg_signum_flip |= ((bit_block_type)(0 == ta_state_neg_j[fidx * block_bits + bix]) & neg_inc_bit) << bix;

            pos_inc >>= 1;
            pos_dec >>= 1;
            neg_inc >>= 1;
            neg_dec >>= 1;
        }

        ta_state_pos_signum_j_p[fidx] ^= pos_signum_flip;
        ta_state_neg_signum_j_p[fidx] ^= neg_signum_flip;
    };

    for (unsigned int fidx = 0; fidx < full_feature_blocks; ++fidx)
    {
        process_block(fidx, block_bits);
    }

    if (X.tail_bits() != 0)
    {
        auto last_block_idx = full_feature_blocks;
        process_block(last_block_idx, X.tail_bits());
    }
}


// Feedback Type II
template<typename state_type, typename bit_block_type>
inline
void block3_(
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


/*
 * https://godbolt.org/z/hxWT7x
 */
template<typename state_type, typename bit_block_type>
inline
void block3(
    int const number_of_features,
    state_type * __restrict ta_state_pos_j,
    state_type * __restrict ta_state_neg_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_pos_signum_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_neg_signum_j,
    bit_vector<bit_block_type> const & X
)
{
    auto constexpr block_bits = bit_vector<bit_block_type>::block_bits;
    unsigned int const full_feature_blocks = X.content_blocks() - (X.tail_bits() != 0);

    bit_block_type const * X_p = assume_aligned<alignment>(X.data());
    ta_state_pos_j = assume_aligned<alignment>(ta_state_pos_j);
    ta_state_neg_j = assume_aligned<alignment>(ta_state_neg_j);

    bit_block_type * ta_state_pos_signum_j_p = assume_aligned<alignment>(ta_state_pos_signum_j.data());
    bit_block_type * ta_state_neg_signum_j_p = assume_aligned<alignment>(ta_state_neg_signum_j.data());

    auto process_block = [&](auto fidx, auto this_block_bits)
    {
        bit_block_type X_0 = ~X_p[fidx];
        bit_block_type X_1 = X_p[fidx];
        bit_block_type X_pos_inc = ~ta_state_pos_signum_j_p[fidx];
        bit_block_type X_neg_inc = ~ta_state_neg_signum_j_p[fidx];

        bit_block_type pos_inc = X_0 & X_pos_inc;
        bit_block_type neg_inc = X_1 & X_neg_inc;

        bit_block_type pos_signum_flip = 0;
        bit_block_type neg_signum_flip = 0;

        for (auto bix = 0u; bix < this_block_bits; ++bix)
        {
            auto pos_inc_bit = pos_inc & 1;
            auto neg_inc_bit = neg_inc & 1;

            ta_state_pos_j[fidx * block_bits + bix] += pos_inc_bit;
            ta_state_neg_j[fidx * block_bits + bix] += neg_inc_bit;

            pos_signum_flip |= ((bit_block_type)(0 == ta_state_pos_j[fidx * block_bits + bix]) & pos_inc_bit) << bix;
            neg_signum_flip |= ((bit_block_type)(0 == ta_state_neg_j[fidx * block_bits + bix]) & neg_inc_bit) << bix;

            pos_inc >>= 1;
            neg_inc >>= 1;
        }

        ta_state_pos_signum_j_p[fidx] ^= pos_signum_flip;
        ta_state_neg_signum_j_p[fidx] ^= neg_signum_flip;
    };

    for (unsigned int fidx = 0; fidx < full_feature_blocks; ++fidx)
    {
        process_block(fidx, block_bits);
    }

    if (X.tail_bits() != 0)
    {
        auto last_block_idx = full_feature_blocks;
        process_block(last_block_idx, X.tail_bits());
    }
}


template<typename state_type, typename bit_block_type, typename PRNG>
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
    PRNG & prng,
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
                block1_sparse<state_type, bit_block_type>(number_of_features, ct.hits(), number_of_states,
                    ta_state_pos_j,
                    ta_state_neg_j,
                    ta_state_signum.row(2 * iidx + 0),
                    ta_state_signum.row(2 * iidx + 1),
                    prng);

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
                        X, ct.tosses1(prng), ct.tosses2(prng));
                }
                else
                {
                    block2<false>(number_of_states,
                        ta_state_pos_j,
                        ta_state_neg_j,
                        ta_state_signum.row(2 * iidx + 0),
                        ta_state_signum.row(2 * iidx + 1),
                        X, ct.tosses1(prng), ct.tosses2(prng));
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


template<typename bit_block_type, typename PRNG>
void train_classifier_automata(
    TAStateWithSignum::value_type & ta_state,
    int const input_begin_ix,
    int const input_end_ix,
    feedback_vector_type::value_type const * __restrict feedback_to_clauses,
    char const * __restrict clause_output,
    int const number_of_states,
    bit_vector<bit_block_type> const & X,
    bool const boost_true_positive_feedback,
    PRNG & prng,
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
    w_vector_type & weights,
    int const input_begin_ix,
    int const input_end_ix,
    char const * __restrict clause_output,
    int const number_of_states,
    int const response_error,
    bit_vector<bit_block_type> const & X,
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

        state_type * ta_state_pos_j = ::assume_aligned<alignment>(ta_state_matrix.row_data(2 * iidx + 0));
        state_type * ta_state_neg_j = ::assume_aligned<alignment>(ta_state_matrix.row_data(2 * iidx + 1));

        if (response_error < 0)
        {
            if (clause_output[iidx] == 0)
            {
                block1_sparse<state_type, bit_block_type>(number_of_features, ct.hits(), number_of_states,
                    ta_state_pos_j,
                    ta_state_neg_j,
                    ta_state_signum.row(2 * iidx + 0),
                    ta_state_signum.row(2 * iidx + 1),
                    prng);
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
                        X, ct.tosses1(prng), ct.tosses2(prng));
                }
                else
                {
                    block2<false>(number_of_states,
                        ta_state_pos_j,
                        ta_state_neg_j,
                        ta_state_signum.row(2 * iidx + 0),
                        ta_state_signum.row(2 * iidx + 1),
                        X, ct.tosses1(prng), ct.tosses2(prng));
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
                block3<state_type, bit_block_type>(number_of_features,
                    ta_state_pos_j,
                    ta_state_neg_j,
                    ta_state_signum.row(2 * iidx + 0),
                    ta_state_signum.row(2 * iidx + 1),
                    X);

                if (weights.size() != 0)
                {
                    weights[iidx] -= (weights[iidx] != 0);
                }
            }
        }
    }
}


template<typename bit_block_type>
void train_regressor_automata(
    TAStateWithSignum::value_type & ta_state,
    int const input_begin_ix,
    int const input_end_ix,
    char const * __restrict clause_output,
    int const number_of_states,
    int const response_error,
    bit_vector<bit_block_type> const & X,
    bool const boost_true_positive_feedback,
    IRNG & prng,
    unsigned int const threshold,
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
        ta_state_variant
    );
}


}  // anonymous namespace


}  // namespace Tsetlini


#endif /* LIB_SRC_TSETLINI_ALGO_BITWISE_HPP_ */
