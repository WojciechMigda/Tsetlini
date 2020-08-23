#pragma once

#ifndef LIB_INCLUDE_TSETLINI_STATE_JSON_HPP_
#define LIB_INCLUDE_TSETLINI_STATE_JSON_HPP_

#include "estimator_state_fwd.hpp"

#include <string>


namespace Tsetlini
{


std::string to_json_string(ClassifierStateClassic const & state);
void from_json_string(ClassifierStateClassic & state, std::string const & js);

std::string to_json_string(RegressorStateClassic const & state);
void from_json_string(RegressorStateClassic & state, std::string const & js);

std::string to_json_string(ClassifierStateBitwise const & state);
void from_json_string(ClassifierStateBitwise & state, std::string const & js);

std::string to_json_string(RegressorStateBitwise const & state);
void from_json_string(RegressorStateBitwise & state, std::string const & js);


} // namespace Tsetlini


#endif /* LIB_INCLUDE_TSETLINI_STATE_JSON_HPP_ */
