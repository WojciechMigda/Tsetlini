#pragma once

#ifndef LIB_SRC_PARAMS_COMPANION_HPP_
#define LIB_SRC_PARAMS_COMPANION_HPP_

#include "tsetlini_params.hpp"
#include "tsetlini_strong_params.hpp"

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
int clause_output_tile_size(params_t const & params)
{
    return std::get<int>(params.at("clause_output_tile_size"));
}


inline
seed_type random_state(params_t const & params)
{
    return std::get<seed_type>(params.at("random_state"));
}


inline
verbosity_t verbose(params_t const & params)
{
    return verbosity_t{std::get<bool>(params.at("verbose"))};
}


inline
int number_of_pos_neg_clauses_per_label(params_t const & params)
{
    return std::get<int>(params.at("number_of_pos_neg_clauses_per_label"));
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
bool boost_true_positive_feedback(params_t const & params)
{
    return std::get<int>(params.at("boost_true_positive_feedback")) != 0;
}


inline
number_of_states_t number_of_states(params_t const & params)
{
    return number_of_states_t{std::get<int>(params.at("number_of_states"))};
}


inline
int number_of_clauses_per_label(params_t const & params)
{
    return 2 * number_of_pos_neg_clauses_per_label(params);
}


inline
int number_of_classifier_clauses(params_t const & params)
{
    return number_of_clauses_per_label(params) * value_of(number_of_labels(params));
}


inline
int number_of_regressor_clauses(params_t const & params)
{
    return std::get<int>(params.at("number_of_regressor_clauses"));
}


inline
bool weighted(params_t const & params)
{
    return std::get<bool>(params.at("weighted"));
}


inline
bool box_muller(params_t const & params)
{
    return std::get<bool>(params.at("box_muller"));
}


inline
int max_weight(params_t const & params)
{
    return std::get<int>(params.at("max_weight"));
}


inline
std::string loss_fn_name(params_t const & params)
{
    return std::get<std::string>(params.at("loss_fn"));
}


inline
real_type loss_fn_C1(params_t const & params)
{
    return std::get<real_type>(params.at("loss_fn_C1"));
}


} // namespace Params


} // namespace Tsetlini


#endif /* LIB_SRC_PARAMS_COMPANION_HPP_ */
