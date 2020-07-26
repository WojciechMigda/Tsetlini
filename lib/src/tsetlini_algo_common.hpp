#pragma once

#ifndef LIB_SRC_TSETLINI_ALGO_COMMON_HPP_
#define LIB_SRC_TSETLINI_ALGO_COMMON_HPP_

#include "tsetlini_types.hpp"
#include "tsetlini_algo_classic.hpp"
#include "tsetlini_algo_bitwise.hpp"


namespace Tsetlini
{


/**
 * @param X
 *      Vector of 0s and 1s, with size equal to @c number_of_features .
 *
 * @param clause_output
 *      Output vector of 0s and 1s, with size equal to @c number_of_clauses .
 *
 * @param number_of_clauses
 *      Positive integer equal to size of @c clause_output .
 *
 * @param number_of_features
 *      Positive integer equal to size of @c X .
 *
 * @param ta_state
 *      @c numeric_matrix with 2 * @c number_of_clauses rows and
 *      @c number_of_features columns.
 *
 * @param n_jobs
 *      Number of parallel jobs.
 *
 * @param TILE_SZ
 *      Positive integer {16, 32, 64, 128} that specifies batch size of
 *      data processed in @c X .
 */
template<typename SampleType, typename TAStateType>
inline
void calculate_clause_output(
    SampleType const & X,
    aligned_vector_char & clause_output,
    int const output_begin_ix,
    int const output_end_ix,
    TAStateType const & ta_state,
    int const n_jobs,
    int const TILE_SZ)
{
    switch (TILE_SZ)
    {
        case 128:
            calculate_clause_output_T<128>(
                X,
                clause_output,
                output_begin_ix,
                output_end_ix,
                ta_state,
                n_jobs
            );
            break;
        case 64:
            calculate_clause_output_T<64>(
                X,
                clause_output,
                output_begin_ix,
                output_end_ix,
                ta_state,
                n_jobs
            );
            break;
        case 32:
            calculate_clause_output_T<32>(
                X,
                clause_output,
                output_begin_ix,
                output_end_ix,
                ta_state,
                n_jobs
            );
            break;
        default:
//            LOG_(warn) << "calculate_clause_output: unrecognized clause_output_tile_size value "
//                       << clause_output_tile_size << ", fallback to 16.\n";
        case 16:
            calculate_clause_output_T<16>(
                X,
                clause_output,
                output_begin_ix,
                output_end_ix,
                ta_state,
                n_jobs
            );
            break;
    }
}


template<typename SampleType, typename TAStateType>
inline
void calculate_clause_output_for_predict(
    SampleType const & X,
    aligned_vector_char & clause_output,
    int const number_of_clauses,
    TAStateType const & ta_state,
    int const n_jobs,
    int const TILE_SZ)
{
    switch (TILE_SZ)
    {
        case 128:
            calculate_clause_output_for_predict_T<128>(
                X,
                clause_output,
                number_of_clauses,
                ta_state,
                n_jobs
            );
            break;
        case 64:
            calculate_clause_output_for_predict_T<64>(
                X,
                clause_output,
                number_of_clauses,
                ta_state,
                n_jobs
            );
            break;
        case 32:
            calculate_clause_output_for_predict_T<32>(
                X,
                clause_output,
                number_of_clauses,
                ta_state,
                n_jobs
            );
            break;
        default:
//            LOG_(warn) << "calculate_clause_output_for_predict: unrecognized clause_output_tile_size value "
//                       << clause_output_tile_size << ", fallback to 16.\n";
        case 16:
            calculate_clause_output_for_predict_T<16>(
                X,
                clause_output,
                number_of_clauses,
                ta_state,
                n_jobs
            );
            break;
    }
}


}  // namespace Tsetlini


#endif /* LIB_SRC_TSETLINI_ALGO_COMMON_HPP_ */
