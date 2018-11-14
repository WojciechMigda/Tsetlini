#define LOG_MODULE "tsetlin"
#include "logger.hpp"

#include "tsetlin.hpp"
#include "assume_aligned.hpp"
#include "config_companion.hpp"
#include "tsetlin_types.hpp"

#include <utility>
#include <algorithm>


namespace Tsetlin
{

namespace
{


inline
int pos_feat_index(int k)
{
    return k;
}


inline
int neg_feat_index(int k, int number_of_features)
{
    return k + number_of_features;
}


inline
bool action(int state, int number_of_states)
{
    return state > number_of_states;
}


inline
int pos_clause_index(int target_class, int j, int number_of_pos_neg_clauses_per_class)
{
    return 2 * target_class * number_of_pos_neg_clauses_per_class + j;
}


inline
int neg_clause_index(int target_class, int j, int number_of_pos_neg_clauses_per_class)
{
    return pos_clause_index(target_class, j, number_of_pos_neg_clauses_per_class) + number_of_pos_neg_clauses_per_class;
}


inline
void sum_up_class_votes(
    aligned_vector_char const & clause_output,
    aligned_vector_int & class_sum,
    int target_class,

    int const number_of_pos_neg_clauses_per_class,
    int const threshold)
{
    class_sum[target_class] = 0;

    for (int j = 0; j < number_of_pos_neg_clauses_per_class; ++j)
    {
        class_sum[target_class] += clause_output[pos_clause_index(target_class, j, number_of_pos_neg_clauses_per_class)];
    }

    for (int j = 0; j < number_of_pos_neg_clauses_per_class; ++j)
    {
        class_sum[target_class] -= clause_output[neg_clause_index(target_class, j, number_of_pos_neg_clauses_per_class)];
    }
    class_sum[target_class] = std::clamp(class_sum[target_class], -threshold, threshold);
}


inline
void sum_up_all_class_votes(
    aligned_vector_char const & clause_output,
    aligned_vector_int & class_sum,

    int const number_of_classes,
    int const number_of_pos_neg_clauses_per_class,
    int const threshold)
{
    for (int target_class = 0; target_class < number_of_classes; ++target_class)
    {
        sum_up_class_votes(clause_output, class_sum, target_class, number_of_pos_neg_clauses_per_class, threshold);
    }
}


inline
void calculate_clause_output(
    aligned_vector_char const & X,
    aligned_vector_char & clause_output,
    bool predict,
    int const number_of_clauses,
    int const number_of_features,
    int const number_of_states,
    std::vector<aligned_vector_int> const & ta_state)
{
    char const * X_p = assume_aligned<alignment>(X.data());
//    char const * clause_output_p = assume_aligned<alignment>(clause_output.data());

    for (int j = 0; j < number_of_clauses; ++j)
    {
        bool output = true;
        bool all_exclude = true;

        int const * ta_state_j = assume_aligned<alignment>(ta_state[j].data());

        for (int k = 0; k < number_of_features and output == true; ++k)
        {
            bool const action_include = action(ta_state_j[pos_feat_index(k)], number_of_states);
            bool const action_include_negated = action(ta_state_j[neg_feat_index(k, number_of_features)], number_of_states);

            all_exclude = (action_include == true or action_include_negated == true) ? false : all_exclude;

            output = ((action_include == true and X_p[k] == 0) or (action_include_negated == true and X_p[k] != 0)) ? false : output;
        }

        output = (predict == true and all_exclude == true) ? false : output;

        clause_output[j] = output;
    }
}


} // anonymous




Classifier::Classifier(ClassifierState const & state) :
    state(state)
{

}


Classifier::Classifier(ClassifierState && state) :
    state(std::move(state))
{

}


int Classifier::predict(aligned_vector_char const & sample) const
{
    calculate_clause_output(
        sample,
        state.cache.clause_output,
        true,
        Config::number_of_clauses(state.config),
        Config::number_of_features(state.config),
        Config::number_of_states(state.config),
        state.ta_state
    );

    sum_up_all_class_votes(
        state.cache.clause_output,
        state.cache.class_sum,
        Config::number_of_classes(state.config),
        Config::number_of_pos_neg_clauses_per_class(state.config),
        Config::threshold(state.config));

    int rv = std::distance(
        state.cache.class_sum.cbegin(),
        std::max_element(state.cache.class_sum.cbegin(), state.cache.class_sum.cend()));

    return rv;
}


aligned_vector_int Classifier::predict_raw(aligned_vector_char const & sample) const
{
    calculate_clause_output(
        sample,
        state.cache.clause_output,
        true,
        Config::number_of_clauses(state.config),
        Config::number_of_features(state.config),
        Config::number_of_states(state.config),
        state.ta_state
    );

    sum_up_all_class_votes(
        state.cache.clause_output,
        state.cache.class_sum,
        Config::number_of_classes(state.config),
        Config::number_of_pos_neg_clauses_per_class(state.config),
        Config::threshold(state.config));

    return state.cache.class_sum;
}


void Classifier::predict_raw(aligned_vector_char const & sample, int * out_p) const
{
    if (out_p == nullptr)
    {
        return;
    }

    calculate_clause_output(
        sample,
        state.cache.clause_output,
        true,
        Config::number_of_clauses(state.config),
        Config::number_of_features(state.config),
        Config::number_of_states(state.config),
        state.ta_state
    );

    sum_up_all_class_votes(
        state.cache.clause_output,
        state.cache.class_sum,
        Config::number_of_classes(state.config),
        Config::number_of_pos_neg_clauses_per_class(state.config),
        Config::threshold(state.config));

    std::copy(state.cache.class_sum.cbegin(), state.cache.class_sum.cend(), out_p);
}



} // namespace Tsetlin
