#pragma once

#include "tsetlin_types.hpp"

#include <string>
#include <unordered_map>
#include <variant>

namespace Tsetlin
{

using config_value_t = std::variant<int, seed_type, real_type, bool>;
using config_patch_t = std::unordered_map<std::string, config_value_t>;
using config_t = std::unordered_map<std::string, config_value_t>;

config_patch_t const config_patch_from_json(std::string const & json_params, bool verbose=false);

} // namespace Tsetlin
