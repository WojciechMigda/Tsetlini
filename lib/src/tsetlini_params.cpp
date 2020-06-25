#include "tsetlini_params.hpp"
#include "tsetlini_status_code.hpp"
#include "params_companion.hpp"

#include "neither/either.hpp"
#include "json.hpp"

#include <optional>
#include <string>
#include <exception>
#include <thread>
#include <algorithm>
#include <random>


using namespace neither;
using namespace std::string_literals;
using json = nlohmann::json;

namespace Tsetlini
{


static const params_t default_classifier_params =
{
    {"number_of_pos_neg_clauses_per_label", param_value_t(5)},
    {"number_of_states", param_value_t(100)},
    {"s", param_value_t(2.0f)},
    {"threshold", param_value_t(15)},
    {"boost_true_positive_feedback", param_value_t(0)},
    {"n_jobs", param_value_t(-1)},
    {"verbose", param_value_t(false)},

    {"counting_type", param_value_t("auto"s)},
    {"clause_output_tile_size", param_value_t(16)},

    {"random_state", param_value_t(std::nullopt)},

    {"number_of_labels", param_value_t(std::nullopt)},
    {"number_of_features", param_value_t(std::nullopt)},
};


/*
 * Safely convert string into json container
 */
static
Either<status_message_t, json>
json_parse(std::string const & json_params)
{
    try
    {
        return Either<status_message_t, json>::rightOf(json::parse(json_params));
    }
    catch (const std::exception & ex)
    {
        return Either<status_message_t, json>::leftOf({S_BAD_JSON, ex.what()});
    }
}

/*
 * Merge two map-like containers
 *
 * Rhs is merged into lhs.
 */
template<typename LHS, typename RHS>
params_t merge(LHS && lhs, RHS && rhs)
{
    params_t rv(std::forward<RHS>(rhs));

    rv.merge(std::forward<LHS>(lhs));

    return rv;
}


/*
 * Assert json to be a dictionary (object)
 */
static
Either<status_message_t, json>
assert_json_dictionary(json const & js)
{
    if (js.is_object())
    {
        return Either<status_message_t, json>::rightOf(js);
    }
    else
    {
        return Either<status_message_t, json>::leftOf({S_BAD_JSON, "type error: passed json is not a dictionary"s});
    }
}


static
Either<status_message_t, params_t>
json_to_params(json const & js)
{
    params_t rv;

    for (auto const & kv : js.items())
    {
        auto const key = kv.key();
        auto const value = kv.value();

        if (
            (key == "number_of_pos_neg_clauses_per_label") or
            (key == "number_of_states") or
            (key == "boost_true_positive_feedback") or
            (key == "threshold") or
            (key == "n_jobs") or
            (key == "clause_output_tile_size")
            )
        {
            rv[key] = value.get<int>();
        }
        else if (key == "random_state")
        {
            if (!value.is_null())
            {
                rv[key] = value.get<seed_type>();
            }
        }
        else if (key == "s")
        {
            rv[key] = value.get<real_type>();
        }
        else if (key == "verbose")
        {
            rv[key] = value.get<bool>();
        }
        else if (key == "counting_type")
        {
            rv[key] = value.get<std::string>();
        }
        else
        {
            return Either<status_message_t, params_t>::leftOf({S_BAD_JSON, "Unknown key [" + key + "] in config\n"});
        }
    }

    return Either<status_message_t, params_t>::rightOf(rv);
}


static
params_t normalize_n_jobs(params_t params)
{
    if (Params::n_jobs(params) == -1)
    {
        params.at("n_jobs") = std::max<int>(1, std::thread::hardware_concurrency());
    }

    return params;
}


static
params_t normalize_random_state(params_t params)
{
    if (std::holds_alternative<none_type>(params.at("random_state")))
    {
        params.at("random_state") = param_value_t(std::random_device{}());
    }

    return params;
}

static
Either<status_message_t, params_t>
assert_counting_type_enumeration(params_t const & params)
{
    auto value = std::get<std::string>(params.at("counting_type"));

    if (not (value == "auto"
        or value == "int8"
        or value == "int16"
        or value == "int32"))
    {
        return Either<status_message_t, params_t>::leftOf({S_BAD_JSON,
            "Param 'counting_type' got value " + value + ", instead of allowed int8, int16, int32, or auto\n"});
    }
    else
    {
        return Either<status_message_t, params_t>::rightOf(params);
    }
}


static
Either<status_message_t, params_t>
assert_clause_output_tile_size_enumeration(params_t const & params)
{
    auto value = std::get<int>(params.at("clause_output_tile_size"));

    if (not (value == 16
        or value == 32
        or value == 64
        or value == 128))
    {
        return Either<status_message_t, params_t>::leftOf({S_BAD_JSON,
            "Param 'clause_output_tile_size' got value " + std::to_string(value) + ", instead of allowed 16, 32, 64, or 128\n"});
    }
    else
    {
        return Either<status_message_t, params_t>::rightOf(params);
    }
}


Either<status_message_t, params_t>
make_classifier_params_from_json(std::string const & json_params)
{
    auto rv =
        json_parse(json_params)
        .rightFlatMap(assert_json_dictionary)
        .rightFlatMap(json_to_params)
        .rightMap([](auto p){ return merge(params_t{default_classifier_params}, p); })
        .rightMap(normalize_n_jobs)
        .rightMap(normalize_random_state)
        .rightFlatMap(assert_counting_type_enumeration)
        .rightFlatMap(assert_clause_output_tile_size_enumeration)
        ;

    return rv;
}


} // namespace Tsetlini
