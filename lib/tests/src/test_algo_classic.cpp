#include "tsetlini_types.hpp"
#include "tsetlini_algo.hpp"
#include "mt.hpp"
#include "assume_aligned.hpp"

#include <gtest/gtest.h>


namespace
{


namespace CAIR
{


/*

# Calculate the output of each clause using the actions of each Tsetline Automaton.
# Output is stored an internal output array.
cdef void calculate_clause_output(self, int[:] X, int predict=0):
    cdef int j,k
    cdef int action_include, action_include_negated
    cdef int all_exclude

    for j in xrange(self.number_of_clauses):
        self.clause_output[j] = 1
        all_exclude = 1
        for k in xrange(self.number_of_features):
            action_include = self.action(self.ta_state[j,k,0])
            action_include_negated = self.action(self.ta_state[j,k,1])

            if action_include == 1 or action_include_negated == 1:
                all_exclude = 0

            if (action_include == 1 and X[k] == 0) or (action_include_negated == 1 and X[k] == 1):
                self.clause_output[j] = 0
                break

        if predict == 1 and all_exclude == 1:
            self.clause_output[j] = 0

 */
template<typename state_type>
inline
void calculate_clause_output(
    Tsetlini::aligned_vector_char const & X,
    Tsetlini::aligned_vector_char & clause_output,
    int const number_of_clauses,
    int const number_of_features,
    Tsetlini::numeric_matrix<state_type> const & ta_state,
    bool const predict)
{
    char const * X_p = assume_aligned<Tsetlini::alignment>(X.data());

    for (int j = 0; j < number_of_clauses; ++j)
    {
        state_type const * ta_state_pos_j = assume_aligned<Tsetlini::alignment>(ta_state.row_data(2 * j + 0));
        state_type const * ta_state_neg_j = assume_aligned<Tsetlini::alignment>(ta_state.row_data(2 * j + 1));

        clause_output[j] = 1;
        bool all_exclude = 1;

        for (int k = 0; k < number_of_features; ++k)
        {
            bool const action_include = Tsetlini::action(ta_state_pos_j[k]);
            bool const action_include_negated = Tsetlini::action(ta_state_neg_j[k]);

            if ((action_include == 1) or (action_include_negated == 1))
            {
                all_exclude = 0;
            }

            if (((action_include == 1) and (X_p[k] == 0)) or ((action_include_negated == 1) and (X_p[k] == 1)))
            {
                clause_output[j] = 0;
                break;
            }
        }

        if ((predict == 1) and (all_exclude == 1))
        {
            clause_output[j] = 0;
        }
    }
}


} // namespace CAIR


TEST(CalculateClauseOutput, gives_same_result_as_CAIR_version)
{
    IRNG    irng(1234);

    for (auto it = 0u; it < 1000; /* nop */)
    {
        int const number_of_features = irng.next(1, 200);
        int const number_of_clauses = irng.next(1, 20);

        Tsetlini::aligned_vector_char X(number_of_features);

        std::generate(X.begin(), X.end(), [&irng](){ return irng.next(0, 1); });

        Tsetlini::numeric_matrix_int8 ta_state(2 * number_of_clauses, number_of_features);

        auto ta_state_gen = [number_of_clauses, number_of_features, &irng](auto & ta_state)
        {
            for (auto rit = 0; rit < 2 * number_of_clauses; ++rit)
            {
                auto row_data = ta_state.row_data(rit);

                for (auto cit = 0; cit < number_of_features; ++cit)
                {
                    row_data[cit] = irng.next(-1, 0);
                }
            }
        };

        ta_state_gen(ta_state);

        Tsetlini::aligned_vector_char clause_output_CAIR(number_of_clauses);
        Tsetlini::aligned_vector_char clause_output(number_of_clauses);

        CAIR::calculate_clause_output(X, clause_output_CAIR, number_of_clauses, number_of_features, ta_state, false);
        Tsetlini::calculate_clause_output(X, clause_output, number_of_clauses, number_of_features, ta_state, 1, 16);

        if (0 != std::accumulate(clause_output_CAIR.cbegin(), clause_output_CAIR.cend(), 0u))
        {
            ++it;
        }

        EXPECT_TRUE(clause_output_CAIR == clause_output);
    }
}


TEST(CalculateClauseOutput, gives_same_prediction_result_as_CAIR_version)
{
    IRNG    irng(1234);

    for (auto it = 0u; it < 1000; /* nop */)
    {
        int const number_of_features = irng.next(1, 200);
        int const number_of_clauses = irng.next(1, 20);

        Tsetlini::aligned_vector_char X(number_of_features);

        std::generate(X.begin(), X.end(), [&irng](){ return irng.next(0, 1); });

        Tsetlini::numeric_matrix_int8 ta_state(2 * number_of_clauses, number_of_features);

        auto ta_state_gen = [number_of_clauses, number_of_features, &irng](auto & ta_state)
        {
            for (auto rit = 0; rit < 2 * number_of_clauses; ++rit)
            {
                auto row_data = ta_state.row_data(rit);

                for (auto cit = 0; cit < number_of_features; ++cit)
                {
                    row_data[cit] = irng.next(-1, 0);
                }
            }
        };

        ta_state_gen(ta_state);

        Tsetlini::aligned_vector_char clause_output_CAIR(number_of_clauses);
        Tsetlini::aligned_vector_char clause_output(number_of_clauses);

        CAIR::calculate_clause_output(X, clause_output_CAIR, number_of_clauses, number_of_features, ta_state, true);
        Tsetlini::calculate_clause_output_for_predict(X, clause_output, number_of_clauses, number_of_features, ta_state, 1, 16);

        if (0 != std::accumulate(clause_output_CAIR.cbegin(), clause_output_CAIR.cend(), 0u))
        {
            ++it;
        }

        EXPECT_TRUE(clause_output_CAIR == clause_output);
    }
}


} // anonymous namespace
