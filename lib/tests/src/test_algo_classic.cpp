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
            clause_sign_0[i].resize(2 * number_of_pos_neg_clauses_per_label);
            clause_sign_1[i].resize(2 * number_of_pos_neg_clauses_per_label);

            for (auto j = 0; j < 2 * number_of_pos_neg_clauses_per_label; ++j)
            {
                clause_sign_0[i][clause_count[i]] = i * 2 * number_of_pos_neg_clauses_per_label + j;
    #if 0
                // Exact CAIR implementation orders positive and negative clauses
                // by interleaving them
                if (j % 2 == 0)
                {
                    clause_sign_1[i][clause_count[i]] = 1;
                }
                else
                {
                    clause_sign_1[i][clause_count[i]] = -1;
                }
    #endif
                // This implementation aggregates positive and negative clauses
                // together in two contiguous but separate regions.
                // See pos_clause_index and neg_clause_index
                if (j < number_of_pos_neg_clauses_per_label)
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

        auto const clause_count = 2 * number_of_pos_neg_clauses_per_label;

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

    for (auto j = 0; j < 2 * number_of_pos_neg_clauses_per_class; ++j)
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

    for (auto j = 0; j < 2 * number_of_pos_neg_clauses_per_class; ++j)
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


} // namespace CAIR


TEST(CalculateClauseOutput, replicates_result_of_CAIR_code)
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


TEST(CalculateClauseOutputForPredict, replicates_result_of_CAIR_code)
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


TEST(SumUpAllLabelVotes, replicates_result_of_CAIR_code)
{
    IRNG    irng(1234);

    for (auto it = 0u; it < 1000; ++it)
    {
        int const number_of_pos_neg_clauses = irng.next(1, 20);
        int const number_of_labels = irng.next(2, 12);
        int const threshold = irng.next(1, 127);

        Tsetlini::aligned_vector_char clause_output(2 * number_of_pos_neg_clauses * number_of_labels);
        std::generate(clause_output.begin(), clause_output.end(), [&irng](){ return irng.next(0, 1); });

        Tsetlini::aligned_vector_int label_sum(number_of_labels);
        Tsetlini::aligned_vector_int label_sum_CAIR(number_of_labels);

        CAIR::sum_up_class_votes(clause_output, label_sum_CAIR, number_of_labels, number_of_pos_neg_clauses, threshold);
        Tsetlini::sum_up_all_label_votes(clause_output, label_sum, number_of_labels, number_of_pos_neg_clauses, threshold);

        EXPECT_TRUE(label_sum_CAIR == label_sum);
    }
}


TEST(CalculateFeedbackToClauses, replicates_result_of_CAIR_code)
{
    IRNG    irng(1234);

    for (auto it = 0u; it < 1000; ++it)
    {
        int const number_of_labels = irng.next(2, 12);
        Tsetlini::label_type const target_label = irng.next(0, number_of_labels - 1);
        Tsetlini::label_type const opposite_label = (target_label + 1 + irng() % (number_of_labels - 1)) % number_of_labels;

        int const threshold = irng.next(1, 127);
        int const target_label_votes = irng.next(-threshold, threshold);
        int const opposite_label_votes = irng.next(-threshold, threshold);

        int const number_of_pos_neg_clauses_per_label = irng.next(1, 20);

        Tsetlini::feedback_vector_type feedback_to_clauses(2 * number_of_pos_neg_clauses_per_label * number_of_labels);
        Tsetlini::feedback_vector_type feedback_to_clauses_CAIR(2 * number_of_pos_neg_clauses_per_label * number_of_labels);

        FRNG fgen(4567);
        FRNG fgen_CAIR(4567);

        CAIR::calculate_feedback_to_clauses(
            feedback_to_clauses_CAIR,
            target_label,
            opposite_label,
            target_label_votes,
            opposite_label_votes,
            number_of_pos_neg_clauses_per_label,
            number_of_labels,
            threshold,
            fgen_CAIR);

        Tsetlini::calculate_feedback_to_clauses(
            feedback_to_clauses,
            target_label,
            opposite_label,
            target_label_votes,
            opposite_label_votes,
            number_of_pos_neg_clauses_per_label,
            threshold,
            fgen);

        EXPECT_TRUE(feedback_to_clauses_CAIR == feedback_to_clauses);
    }
}


} // anonymous namespace
