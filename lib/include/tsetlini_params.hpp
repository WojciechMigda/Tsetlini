#pragma once

#include "neither/either.hpp"
#include "tsetlini_types.hpp"
#include "tsetlini_status_code.hpp"

#include <string>
#include <unordered_map>
#include <variant>

namespace Tsetlini
{

using param_value_t = std::variant<int, seed_type, real_type, bool, none_type, std::string>;
using params_t = std::unordered_map<std::string, param_value_t>;

neither::Either<status_message_t, params_t> make_classifier_params_from_json(std::string const & json_params = "{}");
neither::Either<status_message_t, params_t> make_regressor_params_from_json(std::string const & json_params = "{}");

} // namespace Tsetlini
