#pragma once

#include "tsetlin_status_code.hpp"
#include "tsetlin_state.hpp"
#include "tsetlin_types.hpp"

#include <vector>

namespace Tsetlin
{

status_message_t
fit_impl(
    ClassifierState & state,
    std::vector<aligned_vector_char> const & X,
    label_vector_type const & y,
    int epochs);

} // namespace Tsetlin
