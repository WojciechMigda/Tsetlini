#pragma once

#ifndef LIB_TESTS_SRC_CAIR_ALGOS_HPP_
#define LIB_TESTS_SRC_CAIR_ALGOS_HPP_

#include "tsetlini_types.hpp"
#include "tsetlini_algo_classic.hpp"


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


struct ClauseProxy
{
    ClauseProxy(int const number_of_classes, int const number_of_pos_neg_clauses_per_label)
        : number_of_classes(number_of_classes)
        , number_of_pos_neg_clauses_per_label(number_of_pos_neg_clauses_per_label)
        , clause_sign_0(number_of_classes)
        , clause_sign_1(number_of_classes)
        , clause_count(number_of_classes)
    {
        for (auto i = 0u; i < clause_sign_0.size(); ++i)
        {
            clause_sign_0[i].resize(number_of_pos_neg_clauses_per_label);
            clause_sign_1[i].resize(number_of_pos_neg_clauses_per_label);

            for (auto j = 0; j < number_of_pos_neg_clauses_per_label; ++j)
            {
                clause_sign_0[i][clause_count[i]] = i * number_of_pos_neg_clauses_per_label + j;

                if (j % 2 == 0)
                {
                    clause_sign_1[i][clause_count[i]] = 1;
                }
                else
                {
                    clause_sign_1[i][clause_count[i]] = -1;
                }

                clause_count[i] += 1;
            }
        }
    }

    int const number_of_classes;
    int const number_of_pos_neg_clauses_per_label;
    std::vector<std::vector<int>> clause_sign_0;
    std::vector<std::vector<int>> clause_sign_1;
    std::vector<int> clause_count;
};


/*

# Sum up the votes for each class (this is the multiclass version of the Tsetlin Machine)
cdef void sum_up_class_votes(self):
    cdef int target_class
    cdef int j

    for target_class in xrange(self.number_of_classes):
        self.class_sum[target_class] = 0

        for j in xrange(self.clause_count[target_class]):
            self.class_sum[target_class] += self.clause_output[self.clause_sign[target_class,j,0]]*self.clause_sign[target_class,j,1]

        if self.class_sum[target_class] > self.threshold:
            self.class_sum[target_class] = self.threshold
        elif self.class_sum[target_class] < -self.threshold:
            self.class_sum[target_class] = -self.threshold

 */
inline
void sum_up_class_votes(
    Tsetlini::aligned_vector_char const & clause_output,
    Tsetlini::aligned_vector_int & class_sum,

    int const number_of_classes,
    int const number_of_pos_neg_clauses_per_label,
    int const threshold)
{

    CAIR::ClauseProxy const proxy(number_of_classes, number_of_pos_neg_clauses_per_label);


    for (auto target_class = 0; target_class < number_of_classes; ++target_class)
    {
        class_sum[target_class] = 0;

        auto const clause_count = number_of_pos_neg_clauses_per_label;

        for (auto j = 0; j < clause_count; ++j)
        {
            class_sum[target_class] += clause_output[proxy.clause_sign_0[target_class][j]] * proxy.clause_sign_1[target_class][j];
        }

        if (class_sum[target_class] > threshold)
        {
            class_sum[target_class] = threshold;
        }
        else if (class_sum[target_class] < -threshold)
        {
            class_sum[target_class] = -threshold;
        }
    }
}


/*
 * Calculate feedback to clauses
 *

for j in xrange(self.clause_count[target_class]):
    if 1.0*rand()/RAND_MAX > (1.0/(self.threshold*2))*(self.threshold - self.class_sum[target_class]):
        continue

    if self.clause_sign[target_class,j,1] >= 0:
        # Type I Feedback
        self.feedback_to_clauses[self.clause_sign[target_class,j,0]] = 1
    else:
        # Type II Feedback
        self.feedback_to_clauses[self.clause_sign[target_class,j,0]] = -1

for j in xrange(self.clause_count[negative_target_class]):
    if 1.0*rand()/RAND_MAX > (1.0/(self.threshold*2))*(self.threshold + self.class_sum[negative_target_class]):
        continue

    if self.clause_sign[negative_target_class,j,1] >= 0:
        # Type II Feedback
        self.feedback_to_clauses[self.clause_sign[negative_target_class,j,0]] = -1
    else:
        # Type I Feedback
        self.feedback_to_clauses[self.clause_sign[negative_target_class,j,0]] = 1

 */

template<typename TFRNG>
inline
void calculate_feedback_to_clauses(
    Tsetlini::feedback_vector_type & feedback_to_clauses,
    Tsetlini::label_type const target_class,
    Tsetlini::label_type const negative_target_class,
    int const target_class_votes,
    int const negative_target_class_votes,
    int const number_of_pos_neg_clauses_per_class,
    int const number_of_classes,
    int const threshold,
    TFRNG & fgen)
{
    CAIR::ClauseProxy const proxy(number_of_classes, number_of_pos_neg_clauses_per_class);

    for (auto j = 0; j < proxy.clause_count[target_class]; ++j)
    {
        if (fgen.next() > (1.0 / (threshold * 2)) * (threshold - target_class_votes))
        {
            continue;
        }

        if (proxy.clause_sign_1[target_class][j] >= 0)
        {
            // # Type I Feedback
            feedback_to_clauses[proxy.clause_sign_0[target_class][j]] = 1;
        }
        else
        {
            // # Type II Feedback
            feedback_to_clauses[proxy.clause_sign_0[target_class][j]] = -1;
        }
    }

    for (auto j = 0; j < proxy.clause_count[negative_target_class]; ++j)
    {
        if (fgen.next() > (1.0 / (threshold * 2)) * (threshold + negative_target_class_votes))
        {
            continue;
        }

        if (proxy.clause_sign_1[negative_target_class][j] >= 0)
        {
            // # Type II Feedback
            feedback_to_clauses[proxy.clause_sign_0[negative_target_class][j]] = -1;
        }
        else
        {
            // # Type I Feedback
            feedback_to_clauses[proxy.clause_sign_0[negative_target_class][j]] = 1;
        }
    }
}


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
