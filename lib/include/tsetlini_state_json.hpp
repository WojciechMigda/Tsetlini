#pragma once

#include <string>


namespace Tsetlini
{


struct ClassifierState;
struct RegressorState;

std::string to_json_string(ClassifierState const & state);
void from_json_string(ClassifierState & state, std::string const & js);

std::string to_json_string(RegressorState const & state);
void from_json_string(RegressorState & state, std::string const & js);


} // namespace Tsetlini
