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


}  // anonymous namespace


}  // namespace Tsetlini


#endif /* LIB_SRC_TSETLINI_ALGO_BITWISE_HPP_ */
