#pragma once

#ifndef LIB_SRC_TSETLINI_ALGO_BITWISE_HPP_
#define LIB_SRC_TSETLINI_ALGO_BITWISE_HPP_

#include "tsetlini_types.hpp"
#include "basic_bit_vector.hpp"
#include "assume_aligned.hpp"

namespace Tsetlini
{


namespace
{


template<typename state_type, typename signum_type>
void
signum_from_ta_state(numeric_matrix<state_type> const & ta_state, bit_matrix<signum_type> & signum_matrix)
{
    auto const [nrows, ncols] = ta_state.shape();

    for (auto rix = 0u; rix < nrows; ++rix)
    {
        for (auto cix = 0u; cix < ncols; ++cix)
        {
            // x >= 0  --> 1
            // x < 0   --> 0
            auto const negative = ta_state[{rix, cix}] < 0;

            if (negative)
            {
                signum_matrix.clear(rix, cix);
            }
            else
            {
                signum_matrix.set(rix, cix);
            }
        }
    }
}


template<typename bit_block_type, unsigned int BATCH_SZ>
inline
void calculate_clause_output_T(
    bit_vector<bit_block_type> const & X,
    aligned_vector_char & clause_output,
    int const output_begin_ix,
    int const output_end_ix,
    bit_matrix<bit_block_type> const & ta_state_signum,
    int const n_jobs)
{
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


template<typename bit_block_type>
inline
void calculate_clause_output(
    bit_vector<bit_block_type> const & X,
    aligned_vector_char & clause_output,
    int const output_begin_ix,
    int const output_end_ix,
    bit_matrix<bit_block_type> const & ta_state_sign,
    int const n_jobs,
    int const TILE_SZ)
{
    switch (TILE_SZ)
    {
        case 128:
            calculate_clause_output_T<bit_block_type, 128>(
                X,
                clause_output,
                output_begin_ix,
                output_end_ix,
                ta_state_sign,
                n_jobs
            );
            break;
        case 64:
            calculate_clause_output_T<bit_block_type, 64>(
                X,
                clause_output,
                output_begin_ix,
                output_end_ix,
                ta_state_sign,
                n_jobs
            );
            break;
        case 32:
            calculate_clause_output_T<bit_block_type, 32>(
                X,
                clause_output,
                output_begin_ix,
                output_end_ix,
                ta_state_sign,
                n_jobs
            );
            break;
        default:
//            LOG_(warn) << "calculate_clause_output: unrecognized clause_output_tile_size value "
//                       << clause_output_tile_size << ", fallback to 16.\n";
        case 16:
            calculate_clause_output_T<bit_block_type, 16>(
                X,
                clause_output,
                output_begin_ix,
                output_end_ix,
                ta_state_sign,
                n_jobs
            );
            break;
    }
}


// Feedback Type I, negative
template<typename state_type, typename bit_block_type>
int block1(
    int const number_of_features,
    int const number_of_states,
    float const S_inv,
    state_type * __restrict ta_state_pos_j,
    state_type * __restrict ta_state_neg_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_pos_signum_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_neg_signum_j,
    float const * __restrict fcache,
    int fcache_pos
)
{
    fcache = assume_aligned<alignment>(fcache);
    ta_state_pos_j = assume_aligned<alignment>(ta_state_pos_j);
    ta_state_neg_j = assume_aligned<alignment>(ta_state_neg_j);

    // TODO: check vectorization
    for (int fidx = 0; fidx < number_of_features; ++fidx)
    {
        {
            auto cond = fcache[fcache_pos++] <= S_inv;

            if (ta_state_pos_j[fidx] == 0 and cond)
            {
                ta_state_pos_signum_j.flip(fidx); // flip positive clause bit in signum matrix
            }

            ta_state_pos_j[fidx] = cond ? (ta_state_pos_j[fidx] > -number_of_states ? ta_state_pos_j[fidx] - 1 : ta_state_pos_j[fidx]) : ta_state_pos_j[fidx];
        }

        {
            auto cond = fcache[fcache_pos++] <= S_inv;

            if (ta_state_neg_j[fidx] == 0 and cond)
            {
                ta_state_neg_signum_j.flip(fidx); // flip negative clause bit in signum matrix
            }

            ta_state_neg_j[fidx] = cond ? (ta_state_neg_j[fidx] > -number_of_states ? ta_state_neg_j[fidx] - 1 : ta_state_neg_j[fidx]) : ta_state_neg_j[fidx];
        }
    }

    return fcache_pos;
}


// Feedback Type I, positive
template<bool boost_true_positive_feedback, typename state_type, typename bit_block_type>
int block2(
    int const number_of_states,
    float const S_inv,
    state_type * __restrict ta_state_pos_j,
    state_type * __restrict ta_state_neg_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_pos_signum_j,
    typename bit_matrix<bit_block_type>::bit_view && ta_state_neg_signum_j,
    bit_vector<bit_block_type> const & X,
    float const * __restrict fcache,
    int fcache_pos
)
{
    constexpr float ONE = 1.0f;

    int const number_of_features = X.size();

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

                if (ta_state_pos_j[fidx] == 0)
                {
                    ta_state_pos_signum_j.flip(fidx); // flip positive clause bit in signum matrix
                }

            }
            if (cond2)
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
            if (cond1)
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

            if (cond2)
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

    return fcache_pos;
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
void train_classifier_automata(
    numeric_matrix<state_type> & ta_state,
    bit_matrix<bit_block_type> & ta_state_signum,
    int const input_begin_ix,
    int const input_end_ix,
    feedback_vector_type::value_type const * __restrict feedback_to_clauses,
    char const * __restrict clause_output,
    int const number_of_states,
    float const S_inv,
    bit_vector<bit_block_type> const & X,
    bool const boost_true_positive_feedback,
    FRNG & frng,
    ClassifierState::cache_type::frand_cache_type & fcache
    )
{
    float const * fcache_ = assume_aligned<alignment>(fcache.m_fcache.data());
    int const number_of_features = X.size();

    for (int iidx = input_begin_ix; iidx < input_end_ix; ++iidx)
    {
        state_type * ta_state_pos_j = ::assume_aligned<alignment>(ta_state.row_data(2 * iidx + 0));
        state_type * ta_state_neg_j = ::assume_aligned<alignment>(ta_state.row_data(2 * iidx + 1));

        if (feedback_to_clauses[iidx] > 0)
        {
            if (clause_output[iidx] == 0)
            {
                fcache.refill(frng);

                fcache.m_pos = block1<state_type, bit_block_type>(number_of_features, number_of_states, S_inv,
                    ta_state_pos_j,
                    ta_state_neg_j,
                    ta_state_signum.row(2 * iidx + 0),
                    ta_state_signum.row(2 * iidx + 1),
                    fcache_, fcache.m_pos);
            }
            else // if (clause_output[iidx] == 1)
            {
                fcache.refill(frng);

                if (boost_true_positive_feedback)
                {
                    fcache.m_pos = block2<true>(number_of_states, S_inv,
                        ta_state_pos_j,
                        ta_state_neg_j,
                        ta_state_signum.row(2 * iidx + 0),
                        ta_state_signum.row(2 * iidx + 1),
                        X, fcache_, fcache.m_pos);
                }
                else
                {
                    fcache.m_pos = block2<false>(number_of_states, S_inv,
                        ta_state_pos_j,
                        ta_state_neg_j,
                        ta_state_signum.row(2 * iidx + 0),
                        ta_state_signum.row(2 * iidx + 1),
                        X, fcache_, fcache.m_pos);
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


}  // anonymous namespace


}  // namespace Tsetlini


#endif /* LIB_SRC_TSETLINI_ALGO_BITWISE_HPP_ */
