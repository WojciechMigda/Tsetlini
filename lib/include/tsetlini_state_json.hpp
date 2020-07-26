#pragma once

#ifndef LIB_INCLUDE_TSETLINI_STATE_JSON_HPP_
#define LIB_INCLUDE_TSETLINI_STATE_JSON_HPP_

#include "estimator_state_fwd.hpp"

#include <string>


namespace Tsetlini
{


std::string to_json_string(ClassifierState const & state);
void from_json_string(ClassifierState & state, std::string const & js);

std::string to_json_string(RegressorState const & state);
void from_json_string(RegressorState & state, std::string const & js);


} // namespace Tsetlini


#endif /* LIB_INCLUDE_TSETLINI_STATE_JSON_HPP_ */
