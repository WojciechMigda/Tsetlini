#pragma once

#include <string>


namespace Tsetlini
{


struct ClassifierState;

std::string to_json_string(ClassifierState const & state);
void from_json_string(ClassifierState & state, std::string const & js);


} // namespace Tsetlini
