#pragma once

#include "neither/either.hpp"
#include "tsetlin_types.hpp"
#include "tsetlin_status_code.hpp"

#include <string>
#include <unordered_map>
#include <variant>

namespace Tsetlin
{

using param_value_t = std::variant<int, seed_type, real_type, bool, none_type>;
using params_t = std::unordered_map<std::string, param_value_t>;

neither::Either<status_message_t, params_t> make_params_from_json(std::string const & json_params = "{}");

} // namespace Tsetlin
