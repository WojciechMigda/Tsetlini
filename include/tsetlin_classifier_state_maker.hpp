#pragma once

#include "tsetlin_state.hpp"
#include "tsetlin_status_code.hpp"

#include "neither/either.hpp"

#include <string>
#include <memory>


namespace Tsetlin
{


neither::Either<status_message_t, std::unique_ptr<ClassifierState>>
make_classifier_state_ptr(std::string const & json_params = "{}");


} // namespace Tsetlin
