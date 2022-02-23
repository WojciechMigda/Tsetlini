#pragma once

#ifndef LIB_TESTS_SRC_CAIR_ALGOS_HPP_
#define LIB_TESTS_SRC_CAIR_ALGOS_HPP_

#include "tsetlini_types.hpp"
#include "tsetlini_algo_classic.hpp"


namespace CAIR
{


/*

for j in xrange(self.number_of_clauses):
    if self.feedback_to_clauses[j] > 0:
        ####################################################
        ### Type I Feedback (Combats False Negatives) ###
        ####################################################

        if self.clause_output[j] == 0:
            for k in xrange(self.number_of_features):
                if 1.0*rand()/RAND_MAX <= 1.0/self.s:
                    if self.ta_state[j,k,0] > 1:
                        self.ta_state[j,k,0] -= 1

                if 1.0*rand()/RAND_MAX <= 1.0/self.s:
                    if self.ta_state[j,k,1] > 1:
                        self.ta_state[j,k,1] -= 1

        elif self.clause_output[j] == 1:
            for k in xrange(self.number_of_features):
                if X[k] == 1:
                    if self.boost_true_positive_feedback == 1 or 1.0*rand()/RAND_MAX <= (self.s-1)/self.s:
                        if self.ta_state[j,k,0] < self.number_of_states*2:
                            self.ta_state[j,k,0] += 1

                    if 1.0*rand()/RAND_MAX <= 1.0/self.s:
                        if self.ta_state[j,k,1] > 1:
                            self.ta_state[j,k,1] -= 1

                elif X[k] == 0:
                    if self.boost_true_positive_feedback == 1 or 1.0*rand()/RAND_MAX <= (self.s-1)/self.s:
                        if self.ta_state[j,k,1] < self.number_of_states*2:
                            self.ta_state[j,k,1] += 1

                    if 1.0*rand()/RAND_MAX <= 1.0/self.s:
                        if self.ta_state[j,k,0] > 1:
                            self.ta_state[j,k,0] -= 1

    elif self.feedback_to_clauses[j] < 0:
        #####################################################
        ### Type II Feedback (Combats False Positives) ###
        #####################################################
        if self.clause_output[j] == 1:
            for k in xrange(self.number_of_features):
                action_include = self.action(self.ta_state[j,k,0])
                action_include_negated = self.action(self.ta_state[j,k,1])

                if X[k] == 0:
                    if action_include == 0 and self.ta_state[j,k,0] < self.number_of_states*2:
                        self.ta_state[j,k,0] += 1
                elif X[k] == 1:
                    if action_include_negated == 0 and self.ta_state[j,k,1] < self.number_of_states*2:
                        self.ta_state[j,k,1] += 1

 */
template<typename state_type>
void train_classifier_automata(
    Tsetlini::numeric_matrix<state_type> & ta_state,
    int const input_ix_begin,
    int const input_ix_end,
    Tsetlini::feedback_vector_type::value_type const * __restrict feedback_to_clauses,
    char const * __restrict clause_output,
    int const number_of_features,
    int const number_of_states,
    float const S_inv,
    char const * __restrict X,
    bool const boost_true_positive_feedback,
    FRNG & frng
)
{
    for (int j = input_ix_begin; j < input_ix_end; ++j)
    {
        state_type * ta_state_pos_j = assume_aligned<Tsetlini::alignment>(ta_state.row_data(2 * j + 0));
        state_type * ta_state_neg_j = assume_aligned<Tsetlini::alignment>(ta_state.row_data(2 * j + 1));

        if (feedback_to_clauses[j] > 0)
        {
            if (clause_output[j] == 0)
            {
                for (int k = 0; k < number_of_features; ++k)
                {
                    if (frng() <= S_inv)
                    {
                        if (ta_state_pos_j[k] > -number_of_states)
                        {
                            ta_state_pos_j[k] -= 1;
                        }
                    }
                    if (frng() <= S_inv)
                    {
                        if (ta_state_neg_j[k] > -number_of_states)
                        {
                            ta_state_neg_j[k] -= 1;
                        }
                    }
                }
            }
            else if (clause_output[j] == 1)
            {
                for (int k = 0; k < number_of_features; ++k)
                {
                    if (X[k] == 1)
                    {
                        if (boost_true_positive_feedback == 1 or frng() <= (1 - S_inv))
                        {
                            if (ta_state_pos_j[k] < number_of_states - 1)
                            {
                                ta_state_pos_j[k] += 1;
                            }
                        }
                        if (frng() <= S_inv)
                        {
                            if (ta_state_neg_j[k] > -number_of_states)
                            {
                                ta_state_neg_j[k] -= 1;
                            }
                        }
                    }
                    else if (X[k] == 0)
                    {
                        if (boost_true_positive_feedback == 1 or frng() <= (1 - S_inv))
                        {
                            if (ta_state_neg_j[k] < number_of_states - 1)
                                ta_state_neg_j[k] += 1;
                        }
                        if (frng() <= S_inv)
                        {
                            if (ta_state_pos_j[k] > -number_of_states)
                            {
                                ta_state_pos_j[k] -= 1;
                            }
                        }
                    }
                }
            }
        }
        else if (feedback_to_clauses[j] < 0)
        {
            if (clause_output[j] == 1)
            {
                for (int k = 0; k < number_of_features; ++k)
                {
                    auto const action_include = Tsetlini::action(ta_state_pos_j[k]);
                    auto const action_include_negated = Tsetlini::action(ta_state_neg_j[k]);

                    if (X[k] == 0)
                    {
                        if (action_include == 0 and ta_state_pos_j[k] < number_of_states - 1)
                        {
                            ta_state_pos_j[k] += 1;
                        }
                    }
                    else if (X[k] == 1)
                    {
                        if (action_include_negated == 0 and ta_state_neg_j[k] < number_of_states - 1)
                        {
                            ta_state_neg_j[k] += 1;
                        }
                    }
                }
            }
        }
    }
}


} // namespace CAIR


#endif /* LIB_TESTS_SRC_CAIR_ALGOS_HPP_ */
