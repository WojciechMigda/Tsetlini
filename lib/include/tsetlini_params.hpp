#pragma once

#ifndef LIB_INCLUDE_TSETLINI_PARAMS_HPP_
#define LIB_INCLUDE_TSETLINI_PARAMS_HPP_

#include "tsetlini_types.hpp"
#include "tsetlini_status_code.hpp"
#include "either.hpp"

#include <string>
#include <unordered_map>
#include <variant>


namespace Tsetlini
{


using param_value_t = std::variant<int, seed_type, real_type, bool, none_type, std::string>;
using params_t = std::unordered_map<std::string, param_value_t>;

Either<status_message_t, params_t> make_classifier_params_from_json(std::string const & json_params = "{}");
Either<status_message_t, params_t> make_regressor_params_from_json(std::string const & json_params = "{}");


} // namespace Tsetlini


#endif /* LIB_INCLUDE_TSETLINI_PARAMS_HPP_ */
