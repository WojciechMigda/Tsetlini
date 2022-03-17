#pragma once

#ifndef LIB_SRC_PARAMS_COMPANION_HPP_
#define LIB_SRC_PARAMS_COMPANION_HPP_

#include "tsetlini_params.hpp"
#include "tsetlini_strong_params.hpp"
#include "tsetlini_strong_params_private.hpp"

#include "strong_type/strong_type.hpp"

#include <variant>
#include <string>

namespace Tsetlini
{


namespace Params
{


inline
number_of_jobs_t n_jobs(params_t const & params)
{
    return number_of_jobs_t{std::get<int>(params.at("n_jobs"))};
}


inline
counting_type_t counting_type(params_t const & params)
{
    return counting_type_t{std::get<std::string>(params.at("counting_type"))};
}


inline
clause_output_tile_size_t clause_output_tile_size(params_t const & params)
{
    return clause_output_tile_size_t{std::get<int>(params.at("clause_output_tile_size"))};
}


inline
random_seed_t random_state(params_t const & params)
{
    return random_seed_t{std::get<seed_type>(params.at("random_state"))};
}


inline
verbosity_t verbose(params_t const & params)
{
    return verbosity_t{std::get<bool>(params.at("verbose"))};
}


inline
number_of_labels_t number_of_labels(params_t const & params)
{
    return number_of_labels_t{std::get<int>(params.at("number_of_labels"))};
}


inline
number_of_features_t number_of_features(params_t const & params)
{
    return number_of_features_t{std::get<int>(params.at("number_of_features"))};
}


inline
specificity_t s(params_t const & params)
{
    return specificity_t{std::get<real_type>(params.at("s"))};
}


inline
threshold_t threshold(params_t const & params)
{
    return threshold_t{std::get<int>(params.at("threshold"))};
}


inline
boost_tpf_t boost_true_positive_feedback(params_t const & params)
{
    return boost_tpf_t{std::get<int>(params.at("boost_true_positive_feedback")) != 0};
}


inline
number_of_states_t number_of_states(params_t const & params)
{
    return number_of_states_t{std::get<int>(params.at("number_of_states"))};
}


inline
number_of_classifier_clause_outputs_per_label_t number_of_classifier_clause_outputs_per_label(params_t const & params)
{
    return number_of_classifier_clause_outputs_per_label_t{std::get<int>(params.at("number_of_clauses_per_label")) / 2};
}


inline
number_of_estimator_clause_outputs_t number_of_classifier_clause_outputs(params_t const & params)
{
    return number_of_estimator_clause_outputs_t{
        value_of(number_of_labels(params)) *
        value_of(number_of_classifier_clause_outputs_per_label(params))};
}


inline
number_of_physical_classifier_clauses_per_label_t number_of_physical_classifier_clauses_per_label(params_t const & params)
{
    return number_of_physical_classifier_clauses_per_label_t{std::get<int>(params.at("number_of_clauses_per_label"))};
}


inline
number_of_physical_classifier_clauses_t number_of_physical_classifier_clauses(params_t const & params)
{
    return number_of_physical_classifier_clauses_t{
        value_of(number_of_labels(params)) *
        value_of(number_of_physical_classifier_clauses_per_label(params))};
}


inline
number_of_physical_regressor_clauses_t number_of_physical_regressor_clauses(params_t const & params)
{
    return number_of_physical_regressor_clauses_t{std::get<int>(params.at("number_of_regressor_clauses"))};
}


inline
number_of_estimator_clause_outputs_t number_of_regressor_clause_outputs(params_t const & params)
{
    return number_of_estimator_clause_outputs_t{std::get<int>(params.at("number_of_regressor_clauses")) / 2};
}


inline
weighted_flag_t weighted(params_t const & params)
{
    return weighted_flag_t{std::get<bool>(params.at("weighted"))};
}


inline
box_muller_flag_t box_muller(params_t const & params)
{
    return box_muller_flag_t{std::get<bool>(params.at("box_muller"))};
}


inline
max_weight_t max_weight(params_t const & params)
{
    return max_weight_t{std::get<int>(params.at("max_weight"))};
}


inline
loss_fn_name_t loss_fn_name(params_t const & params)
{
    return loss_fn_name_t{std::get<std::string>(params.at("loss_fn"))};
}


inline
loss_fn_C1_t loss_fn_C1(params_t const & params)
{
    return loss_fn_C1_t{std::get<real_type>(params.at("loss_fn_C1"))};
}


} // namespace Params


} // namespace Tsetlini


#endif /* LIB_SRC_PARAMS_COMPANION_HPP_ */
