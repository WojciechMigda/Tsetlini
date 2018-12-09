#include "tsetlin_params.hpp"
#include "tsetlin_state.hpp"
#include "tsetlin_status_code.hpp"

#include "neither/either.hpp"

#include <string>
#include <new>
#include <memory>


namespace Tsetlin
{


neither::Either<status_message_t, std::unique_ptr<ClassifierState>>
make_classifier_state_ptr(std::string const & json_params)
{
    auto rv =
        make_params_from_json(json_params)
        .rightMap([](params_t && params)
        {
            return std::unique_ptr<ClassifierState>(new(std::nothrow) ClassifierState(params));
        });

    return rv;
}


} // namespace Tsetlin
