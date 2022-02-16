#pragma once

#ifndef LIB_SRC_TSETLINI_STRONG_PARAMS_HPP_
#define LIB_SRC_TSETLINI_STRONG_PARAMS_HPP_

#include "tsetlini_types.hpp"
#include "strong_type/strong_type.hpp"

#include <cstddef>
#include <string>


namespace Tsetlini
{


using number_of_epochs_t = strong::type<
    unsigned int, struct _number_of_epochs_tag,
    strong::ordered_with<unsigned int>
>;

using threshold_t = strong::type<
    int, struct _threshold_tag,
    strong::ordered_with<Tsetlini::response_type>,
    strong::ostreamable
>;

using specificity_t = strong::type<
    Tsetlini::real_type, struct _specificity_tag,
    strong::ostreamable
>;

using number_of_features_t = strong::type<
    int, struct _number_of_features_tag,
    strong::ordered_with<int>,
    strong::ostreamable
>;

using number_of_labels_t = strong::type<
    int, struct _number_of_labels_tag,
    strong::ordered_with<int>,
    strong::ostreamable
>;

using number_of_states_t = strong::type<
    int, struct _number_of_states_tag,
    strong::ordered_with<int>, // covers comparison against int{8,16,32}
    strong::arithmetic, // negation
    strong::affine_point<int>, // +/- int
    strong::ostreamable
>;

using verbosity_t = strong::type<
    bool, struct _verbosity_tag,
    strong::convertible_to<bool>
>;

using number_of_jobs_t = strong::type<
    int, struct _number_of_jobs_tag,
    strong::ordered_with<int>,
    strong::equality_with<int>,
    strong::ostreamable
>;

using counting_type_t = strong::type<
    std::string, struct _counting_type_tag,
    strong::equality_with<char const *>,
    strong::ostreamable
>;

using clause_output_tile_size_t = strong::type<
    int, struct _clause_output_tile_size_tag,
    strong::equality_with<int>,
    strong::ostreamable
>;

using random_seed_t = strong::type<
    seed_type, struct _random_device_seed_tag,
    strong::ostreamable
>;

using boost_tpf_t = strong::type<
    bool, struct _boost_tpf_tag,
    strong::equality_with<bool>,
    strong::ostreamable
>;

using number_of_physical_regressor_clauses_t = strong::type<
    int, struct _number_of_physical_regressor_clauses_tag,
    strong::ostreamable
>;

using number_of_physical_classifier_clauses_per_label_t = strong::type<
    int, struct _number_of_physical_classifier_clauses_per_label_tag
>;

using number_of_classifier_clause_outputs_per_label_t = strong::type<
    int, struct _number_of_classifier_clause_outputs_per_label_tag
>;

using number_of_physical_classifier_clauses_t = strong::type<
    int, struct _number_of_physical_classifier_clauses_tag,
    strong::ostreamable
>;

using number_of_physical_estimator_clauses_t = strong::type<
    int, struct _number_of_physical_estimator_clauses_tag
>;

using number_of_estimator_clause_outputs_t = strong::type<
    int, struct _number_of_estimator_clause_outputs_tag,
    strong::ordered_with<int>
>;

using weighted_flag_t = strong::type<
    bool, struct _weighted_flag_tag,
    strong::equality_with<bool>,
    strong::ostreamable
>;

using box_muller_flag_t = strong::type<
    bool, struct _box_muller_flag_tag,
    strong::equality_with<bool>,
    strong::ostreamable
>;

using max_weight_t = strong::type<
    int, struct _max_weight_tag,
    strong::ordered_with<int>,
    strong::ostreamable
>;

using loss_fn_name_t = strong::type<
    std::string, struct _loss_fn_name_tag,
    strong::equality_with<char const *>,
    strong::ostreamable
>;

using loss_fn_C1_t = strong::type<
    Tsetlini::real_type, struct _loss_fn_C1_tag,
    strong::ordered_with<Tsetlini::real_type>,
    strong::arithmetic,
    strong::ostreamable
>;


}  // namespace Tsetlini


#endif /* LIB_SRC_TSETLINI_STRONG_PARAMS_HPP_ */
