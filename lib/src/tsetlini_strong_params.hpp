#pragma once

#ifndef LIB_SRC_TSETLINI_STRONG_PARAMS_HPP_
#define LIB_SRC_TSETLINI_STRONG_PARAMS_HPP_

#include "tsetlini_types.hpp"
#include "strong_type/strong_type.hpp"


namespace Tsetlini
{


using threshold_t = strong::type<
    int, struct threshold_t_,
    strong::ordered_with<Tsetlini::response_type>
>;


}  // namespace Tsetlini


#endif /* LIB_SRC_TSETLINI_STRONG_PARAMS_HPP_ */
