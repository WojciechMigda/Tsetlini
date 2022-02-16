#include "tsetlini_params.hpp"
#include "tsetlini_status_code.hpp"
#include "params_companion.hpp"

#include "neither/either.hpp"
#include "json/json.hpp"
#include "strong_type/strong_type.hpp"

#include <optional>
#include <string>
#include <exception>
#include <thread>
#include <algorithm>
#include <random>
#include <limits>


using namespace neither;
using namespace std::string_literals;
using json = nlohmann::json;

namespace Tsetlini
{


static const params_t default_classifier_params =
{
    {"number_of_clauses_per_label", param_value_t(12)},
    {"number_of_states", param_value_t(100)},
    {"s", param_value_t(2.0f)},
    {"threshold", param_value_t(15)},
    {"weighted", param_value_t(false)},
    {"max_weight", param_value_t(std::numeric_limits<int>::max())},
    {"boost_true_positive_feedback", param_value_t(0)},
    {"n_jobs", param_value_t(-1)},
    {"verbose", param_value_t(false)},

    {"counting_type", param_value_t("auto"s)},
    {"clause_output_tile_size", param_value_t(16)},

    {"random_state", param_value_t(std::nullopt)},

    // internal
    {"number_of_labels", param_value_t(std::nullopt)},
    {"number_of_features", param_value_t(std::nullopt)},
};


static const params_t default_regressor_params =
{
    {"number_of_regressor_clauses", param_value_t(20)},
    {"number_of_states", param_value_t(100)},
    {"s", param_value_t(2.0f)},
    {"threshold", param_value_t(15)},
    {"weighted", param_value_t(true)},
    {"max_weight", param_value_t(std::numeric_limits<int>::max())},
    {"boost_true_positive_feedback", param_value_t(0)},
    {"n_jobs", param_value_t(-1)},
    {"verbose", param_value_t(false)},

    {"counting_type", param_value_t("auto"s)},
    {"clause_output_tile_size", param_value_t(16)},

    {"loss_fn", param_value_t("MSE"s)},
    {"loss_fn_C1", param_value_t(0.f)},

    {"box_muller", param_value_t(false)},

    {"random_state", param_value_t(std::nullopt)},

    // internal
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
            (key == "number_of_clauses_per_label") or
            (key == "number_of_regressor_clauses") or
            (key == "number_of_states") or
            (key == "boost_true_positive_feedback") or
            (key == "threshold") or
            (key == "n_jobs") or
            (key == "clause_output_tile_size") or
            (key == "max_weight")
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
        else if (
            (key == "s") or
            (key == "loss_fn_C1"))
        {
            rv[key] = value.get<real_type>();
        }
        else if (
            (key == "verbose") or
            (key == "box_muller") or
            (key == "weighted"))
        {
            rv[key] = value.get<bool>();
        }
        else if (
            (key == "counting_type") or
            (key == "loss_fn"))
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
    auto str = Params::counting_type(params);

    if (not (str == "auto"
        or str == "int8"
        or str == "int16"
        or str == "int32"))
    {
        return Either<status_message_t, params_t>::leftOf({S_BAD_JSON,
            "Param 'counting_type' got value " + value_of(str) + ", instead of allowed int8, int16, int32, or auto\n"});
    }
    else
    {
        return Either<status_message_t, params_t>::rightOf(params);
    }
}


static
Either<status_message_t, params_t>
assert_n_jobs(params_t const & params)
{
    auto num = Params::n_jobs(params);

    if (not ((num == -1) or (num >= 1)))
    {
        return Either<status_message_t, params_t>::leftOf({S_BAD_JSON,
            "Param 'n_jobs' got value " + std::to_string(value_of(num)) + ", instead of natural integer or -1.\n"});
    }
    else
    {
        return Either<status_message_t, params_t>::rightOf(params);
    }
}


static
Either<status_message_t, params_t>
assert_boost_true_positive_feedback(params_t const & params)
{
    auto value = std::get<int>(params.at("boost_true_positive_feedback"));

    if ((value < 0) or (1 < value))
    {
        return Either<status_message_t, params_t>::leftOf({S_BAD_JSON,
            "Param 'boost_true_positive_feedback' got value " + std::to_string(value) + ", instead of either 0 or 1.\n"});
    }
    else
    {
        return Either<status_message_t, params_t>::rightOf(params);
    }
}


static
Either<status_message_t, params_t>
assert_number_of_states(params_t const & params)
{
    auto value = std::get<int>(params.at("number_of_states"));

    if (value < 1)
    {
        return Either<status_message_t, params_t>::leftOf({S_BAD_JSON,
            "Param 'number_of_states' got value " + std::to_string(value) + ", instead of a natural integer.\n"});
    }
    else
    {
        return Either<status_message_t, params_t>::rightOf(params);
    }
}


static
Either<status_message_t, params_t>
assert_specificity(params_t const & params)
{
    auto value = std::get<real_type>(params.at("s"));

    if (value < 1.0)
    {
        return Either<status_message_t, params_t>::leftOf({S_BAD_JSON,
            "Param 's' got value " + std::to_string(value) + ", instead of a value >= 1.0 .\n"});
    }
    else
    {
        return Either<status_message_t, params_t>::rightOf(params);
    }
}


static
Either<status_message_t, params_t>
assert_max_weight(params_t const & params)
{
    auto value = std::get<int>(params.at("max_weight"));

    if (value < 1)
    {
        return Either<status_message_t, params_t>::leftOf({S_BAD_JSON,
            "Param 'max_weight' got value " + std::to_string(value) + ", instead of a natural integer.\n"});
    }
    else
    {
        return Either<status_message_t, params_t>::rightOf(params);
    }
}


static
Either<status_message_t, params_t>
assert_threshold(params_t const & params)
{
    auto value = std::get<int>(params.at("threshold"));

    if (value < 1)
    {
        return Either<status_message_t, params_t>::leftOf({S_BAD_JSON,
            "Param 'threshold' got value " + std::to_string(value) + ", instead of a natural integer.\n"});
    }
    else
    {
        return Either<status_message_t, params_t>::rightOf(params);
    }
}


static
Either<status_message_t, params_t>
assert_number_of_physical_clauses_per_label(params_t const & params)
{
    auto num = Params::number_of_physical_classifier_clauses_per_label(params);

    if ((value_of(num) < 1) or ((value_of(num) % 4) != 0))
    {
        return Either<status_message_t, params_t>::leftOf({S_BAD_JSON,
            "Param 'number_of_clauses_per_label' got value " + std::to_string(value_of(num)) + ", instead of a natural integer divisible by 4.\n"});
    }
    else
    {
        return Either<status_message_t, params_t>::rightOf(params);
    }
}


static
Either<status_message_t, params_t>
assert_number_of_regressor_clauses(params_t const & params)
{
    auto num = Params::number_of_physical_regressor_clauses(params);

    if ((value_of(num) < 1) or ((value_of(num) % 2) != 0))
    {
        return Either<status_message_t, params_t>::leftOf({S_BAD_JSON,
            "Param 'number_of_regressor_clauses' got value " + std::to_string(value_of(num)) + ", instead of an even natural integer.\n"});
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
    auto size = Params::clause_output_tile_size(params);

    if (not (size == 16
        or size == 32
        or size == 64
        or size == 128))
    {
        return Either<status_message_t, params_t>::leftOf({S_BAD_JSON,
            "Param 'clause_output_tile_size' got value " + std::to_string(value_of(size)) + ", instead of allowed 16, 32, 64, or 128\n"});
    }
    else
    {
        return Either<status_message_t, params_t>::rightOf(params);
    }
}


static
Either<status_message_t, params_t>
assert_loss_function(params_t const & params)
{
    auto value = std::get<std::string>(params.at("loss_fn"));

    if (not (value == "L2"
        or value == "MSE"
        or value == "MAE"
        or value == "L1"
        or value == "berHu"
        or value == "L1+2"))
    {
        return Either<status_message_t, params_t>::leftOf({S_BAD_JSON,
            "Param 'loss_fn' got value " + value + ", instead of allowed MSE, MAE, L2, L1, L1+2, or berHu\n"});
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
        .rightFlatMap(assert_n_jobs)
        .rightFlatMap(assert_number_of_states)
        .rightFlatMap(assert_specificity)
        .rightFlatMap(assert_number_of_physical_clauses_per_label)
        .rightFlatMap(assert_boost_true_positive_feedback)
        .rightFlatMap(assert_threshold)
        .rightFlatMap(assert_max_weight)
        .rightMap(normalize_n_jobs)
        .rightMap(normalize_random_state)
        .rightFlatMap(assert_counting_type_enumeration)
        .rightFlatMap(assert_clause_output_tile_size_enumeration)
        ;

    return rv;
}


Either<status_message_t, params_t>
make_regressor_params_from_json(std::string const & json_params)
{
    auto rv =
        json_parse(json_params)
        .rightFlatMap(assert_json_dictionary)
        .rightFlatMap(json_to_params)
        .rightMap([](auto p){ return merge(params_t{default_regressor_params}, p); })
        .rightFlatMap(assert_n_jobs)
        .rightFlatMap(assert_number_of_states)
        .rightFlatMap(assert_specificity)
        .rightFlatMap(assert_number_of_regressor_clauses)
        .rightFlatMap(assert_boost_true_positive_feedback)
        .rightFlatMap(assert_threshold)
        .rightFlatMap(assert_max_weight)
        .rightMap(normalize_n_jobs)
        .rightMap(normalize_random_state)
        .rightFlatMap(assert_counting_type_enumeration)
        .rightFlatMap(assert_loss_function)
        .rightFlatMap(assert_clause_output_tile_size_enumeration)
        ;

    return rv;
}


} // namespace Tsetlini
