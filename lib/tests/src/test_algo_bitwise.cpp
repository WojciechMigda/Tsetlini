#include "cair_algos.hpp"

#include "tsetlini_types.hpp"
#include "tsetlini_algo.hpp"
#include "mt.hpp"
#include "assume_aligned.hpp"

#include <gtest/gtest.h>


/////////////////////
#include "basic_bit_matrix.hpp"
#include "assume_aligned.hpp"

template<typename T>
using bit_matrix = basic_bit_matrix<T, 32>;
using bit_matrix_int32 = bit_matrix<std::int32_t>;
using bit_matrix_int16 = bit_matrix<std::int16_t>;
using bit_matrix_int8 = bit_matrix<std::int8_t>;

#if 0
template<typename bit_block_type, unsigned int BATCH_SZ>
inline
void calculate_clause_output_T(
    aligned_vector_char const & X,
    Tsetlini::aligned_vector_char & clause_output,
    int const output_begin_ix,
    int const output_end_ix,
    int const number_of_features,
    bit_matrix<bit_block_type> const & ta_state,
    int const n_jobs)
{
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
#endif

//template<typename state_type, typename RowType>
//inline
//void calculate_clause_output(
//    RowType const & X,
//    aligned_vector_char & clause_output,
//    int const output_begin_ix,
//    int const output_end_ix,
//    int const number_of_features,
//    numeric_matrix<state_type> const & ta_state,
//    int const n_jobs,
//    int const TILE_SZ)
//{
//    switch (TILE_SZ)
//    {
//        case 128:
//            calculate_clause_output_T<state_type, 128>(
//                X,
//                clause_output,
//                output_begin_ix,
//                output_end_ix,
//                number_of_features,
//                ta_state,
//                n_jobs
//            );
//            break;
//        case 64:
//            calculate_clause_output_T<state_type, 64>(
//                X,
//                clause_output,
//                output_begin_ix,
//                output_end_ix,
//                number_of_features,
//                ta_state,
//                n_jobs
//            );
//            break;
//        case 32:
//            calculate_clause_output_T<state_type, 32>(
//                X,
//                clause_output,
//                output_begin_ix,
//                output_end_ix,
//                number_of_features,
//                ta_state,
//                n_jobs
//            );
//            break;
//        default:
////            LOG_(warn) << "calculate_clause_output: unrecognized clause_output_tile_size value "
////                       << clause_output_tile_size << ", fallback to 16.\n";
//        case 16:
//            calculate_clause_output_T<state_type, 16>(
//                X,
//                clause_output,
//                output_begin_ix,
//                output_end_ix,
//                number_of_features,
//                ta_state,
//                n_jobs
//            );
//            break;
//    }
//}



namespace
{


TEST(BitwiseCalculateClauseOutput, replicates_result_of_CAIR_code)
{
    IRNG    irng(1234);

    for (auto it = 0u; it < 1000; /* nop */)
    {
        int const number_of_features = irng.next(1, 200);
        int const number_of_clauses = irng.next(1, 10) * 2; // must be even

        Tsetlini::aligned_vector_char X(number_of_features);

        std::generate(X.begin(), X.end(), [&irng](){ return irng.next(0, 1); });

        Tsetlini::numeric_matrix_int8 ta_state(2 * number_of_clauses, number_of_features);

        auto ta_state_gen = [&irng](auto & ta_state)
        {
            for (auto rit = 0u; rit < ta_state.rows(); ++rit)
            {
                auto row_data = ta_state.row_data(rit);

                for (auto cit = 0u; cit < ta_state.cols(); ++cit)
                {
                    row_data[cit] = irng.next(-1, 0);
                }
            }
        };

        ta_state_gen(ta_state);

        Tsetlini::aligned_vector_char clause_output_CAIR(number_of_clauses);
//        Tsetlini::aligned_vector_char clause_output(number_of_clauses);

        CAIR::calculate_clause_output(X, clause_output_CAIR, number_of_clauses, number_of_features, ta_state, false);
//        Tsetlini::calculate_clause_output(X, clause_output, 0, number_of_clauses, number_of_features, ta_state, 1, 16);

        if (0 != std::accumulate(clause_output_CAIR.cbegin(), clause_output_CAIR.cend(), 0u))
        {
            ++it;
        }

//        EXPECT_TRUE(clause_output_CAIR == clause_output);
    }
}


} // anonymous namespace
