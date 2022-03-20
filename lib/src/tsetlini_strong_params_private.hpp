#pragma once

#ifndef LIB_SRC_TSETLINI_STRONG_PARAMS_PRIVATE_HPP_
#define LIB_SRC_TSETLINI_STRONG_PARAMS_PRIVATE_HPP_

#include "strong_type/strong_type.hpp"


namespace Tsetlini
{


using number_of_labels_t = strong::type<
    int, struct _number_of_labels_tag,
    strong::ordered_with<int>,
    strong::ostreamable
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

using response_error_t = strong::type<
    int, struct _resonse_error_tag,
    strong::ordered_with<int>
>;


}  // namespace Tsetlini


#endif /* LIB_SRC_TSETLINI_STRONG_PARAMS_PRIVATE_HPP_ */
