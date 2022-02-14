#pragma once

#ifndef LIB_SRC_TSETLINI_STRONG_PARAMS_HPP_
#define LIB_SRC_TSETLINI_STRONG_PARAMS_HPP_

#include "tsetlini_types.hpp"
#include "strong_type/strong_type.hpp"

#include <cstddef>


namespace Tsetlini
{


using threshold_t = strong::type<
    int, struct _threshold_tag,
    strong::ordered_with<Tsetlini::response_type>
>;

using specificity_t = strong::type<
    Tsetlini::real_type, struct _specificity_tag
>;

using number_of_features_t = strong::type<
    int, struct _number_of_features_tag,
    strong::ordered_with<int>
>;

using number_of_labels_t = strong::type<
    int, struct _number_of_labels_tag,
    strong::ordered_with<int>
>;

using number_of_states_t = strong::type<
    int, struct _number_of_states_tag,
    strong::ordered_with<int>, // covers comparison against int{8,16,32}
    strong::arithmetic, // negation
    strong::affine_point<int> // +/- int
>;

using verbosity_t = strong::type<
    bool, struct _verbosity_tag,
    strong::convertible_to<bool>
>;


}  // namespace Tsetlini


#endif /* LIB_SRC_TSETLINI_STRONG_PARAMS_HPP_ */
