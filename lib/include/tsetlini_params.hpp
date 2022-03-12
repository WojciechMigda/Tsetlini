#pragma once

#ifndef LIB_INCLUDE_TSETLINI_PARAMS_HPP_
#define LIB_INCLUDE_TSETLINI_PARAMS_HPP_

#include "tsetlini_types.hpp"
#include "tsetlini_status_code.hpp"
#include "tsetlini_strong_params.hpp"
#include "either.hpp"

#include <string>
#include <unordered_map>
#include <variant>
#include <optional>


namespace Tsetlini
{


using param_value_t = std::variant<int, seed_type, real_type, bool, none_type, std::string>;
using params_t = std::unordered_map<std::string, param_value_t>;

Either<status_message_t, params_t> make_classifier_params_from_json(std::string const & json_params = "{}");

Either<status_message_t, params_t> make_classifier_params_from_args(
    number_of_physical_classifier_clauses_per_label_t number_of_clauses_per_label,
    number_of_states_t number_of_states,
    specificity_t specificity,
    threshold_t threshold,
    weighted_flag_t weighted_flag,
    max_weight_t max_weight,
    boost_tpf_t boost_tpf,
    number_of_jobs_t n_jobs,
    verbosity_t verbose,
    counting_type_t counting_type,
    clause_output_tile_size_t clause_output_tile_size,
    std::optional<random_seed_t> maybe_random_seed);

Either<status_message_t, params_t> make_regressor_params_from_json(std::string const & json_params = "{}");


} // namespace Tsetlini


#endif /* LIB_INCLUDE_TSETLINI_PARAMS_HPP_ */
