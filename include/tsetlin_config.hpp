#pragma once

#include <string>
#include <unordered_map>
#include <any>

namespace Tsetlin
{

using config_patch_t = std::unordered_map<std::string, std::any>;
using config_t = std::unordered_map<std::string, std::any>;

config_patch_t const config_patch_from_json(std::string const & json_params, bool verbose=false);

} // namespace Tsetlin
