#include "tsetlini_params.hpp"
#include "tsetlini_types.hpp"

#include <gtest/gtest.h>
#include <variant>
#include <thread>
#include <algorithm>

namespace
{


TEST(RegressorParams, can_be_created)
{
    auto const rv = Tsetlini::make_regressor_params_from_json();

    EXPECT_TRUE(rv);
}


TEST(RegressorParams, cannot_be_created_from_empty_string_json)
{
    auto const rv = Tsetlini::make_regressor_params_from_json("");

    EXPECT_FALSE(rv);
}


TEST(RegressorParams, can_be_created_from_empty_dict_json)
{
    auto const rv = Tsetlini::make_regressor_params_from_json("{}");

    EXPECT_TRUE(rv);
}


TEST(RegressorParams, cannot_be_created_from_invalid_json)
{
    auto const rv = Tsetlini::make_regressor_params_from_json("[]");

    EXPECT_FALSE(rv);
}


TEST(RegressorParams, cannot_be_created_from_malformed_json)
{
    auto const rv = Tsetlini::make_regressor_params_from_json("5\"}");

    EXPECT_FALSE(rv);
}


TEST(RegressorParams, can_be_created_from_json_with_one_integer_item)
{
    auto const rv = Tsetlini::make_regressor_params_from_json(R"({"number_of_states": 200})");

    EXPECT_TRUE(rv);

    auto params = rv.right().value;

    EXPECT_EQ(200, std::get<int>(params.at("number_of_states")));
}


TEST(RegressorParams, can_be_created_from_json_with_one_float_item)
{
    auto const rv = Tsetlini::make_regressor_params_from_json(R"({"s": 3.9})");

    EXPECT_TRUE(rv);

    auto params = rv.right().value;

    EXPECT_FLOAT_EQ(3.9, std::get<Tsetlini::real_type>(params.at("s")));
}


TEST(RegressorParams, can_be_created_from_json_with_one_boolean_item)
{
    auto const rv = Tsetlini::make_regressor_params_from_json(R"({"verbose": true})");

    EXPECT_TRUE(rv);

    auto params = rv.right().value;

    EXPECT_EQ(true, std::get<bool>(params.at("verbose")));

}


TEST(RegressorParams, can_be_created_from_json_with_string_item)
{
    auto const rv = Tsetlini::make_regressor_params_from_json(R"({"counting_type": "int16"})");

    EXPECT_TRUE(rv);

    auto params = rv.right().value;

    EXPECT_EQ("int16", std::get<std::string>(params.at("counting_type")));
}


TEST(RegressorParams, can_be_created_from_json_with_null_random_state)
{
    auto const rv = Tsetlini::make_regressor_params_from_json(R"({"random_state": null})");

    EXPECT_TRUE(rv);

    auto params = rv.right().value;

    // null random_state is normalized with random seed
    EXPECT_TRUE(std::holds_alternative<Tsetlini::seed_type>(params.at("random_state")));
}


TEST(RegressorParams, cannot_be_created_from_json_with_unrecognized_item)
{
    auto const rv = Tsetlini::make_regressor_params_from_json(R"({"foobar": true})");

    EXPECT_FALSE(rv);
}


TEST(RegressorParams, can_be_created_from_json_with_full_config)
{
    auto const rv = Tsetlini::make_regressor_params_from_json(R"(
{
"verbose": true,
"number_of_regressor_clauses": 36,
"number_of_states": 125,
"s": 6.3 ,
"threshold": 8,
"weighted": true,
"max_weight": 7,
"loss_fn": "L2",
"boost_true_positive_feedback": 1,
"counting_type": "int32",
"clause_output_tile_size": 32,
"n_jobs": 3,
"random_state": 123
}
)");

    EXPECT_TRUE(rv);

    auto params = rv.right().value;

    EXPECT_EQ(true, std::get<bool>(params.at("verbose")));
    EXPECT_EQ(36, std::get<int>(params.at("number_of_regressor_clauses")));
    EXPECT_EQ(125, std::get<int>(params.at("number_of_states")));
    EXPECT_EQ(8, std::get<int>(params.at("threshold")));
    EXPECT_EQ(true, std::get<bool>(params.at("weighted")));
    EXPECT_EQ(7, std::get<int>(params.at("max_weight")));
    EXPECT_EQ("L2", std::get<std::string>(params.at("loss_fn")));
    EXPECT_EQ(3, std::get<int>(params.at("n_jobs")));
    EXPECT_EQ(1, std::get<int>(params.at("boost_true_positive_feedback")));
    EXPECT_EQ("int32", std::get<std::string>(params.at("counting_type")));
    EXPECT_EQ(32, std::get<int>(params.at("clause_output_tile_size")));
    EXPECT_FLOAT_EQ(6.3, std::get<Tsetlini::real_type>(params.at("s")));
    EXPECT_EQ(123u, std::get<Tsetlini::seed_type>(params.at("random_state")));
}


TEST(RegressorParams, default_max_weight_is_max_int)
{
    auto const rv = Tsetlini::make_regressor_params_from_json(R"({})");

    EXPECT_TRUE(rv);

    auto params = rv.right().value;

    EXPECT_EQ(std::numeric_limits<int>::max(), std::get<int>(params.at("max_weight")));
}


TEST(RegressorParams, n_jobs_equal_neg_one_is_normalized)
{
    auto const rv = Tsetlini::make_regressor_params_from_json(R"({"n_jobs": -1})");

    EXPECT_TRUE(rv);

    auto params = rv.right().value;

    EXPECT_FLOAT_EQ(
        std::max<int>(1, std::thread::hardware_concurrency()),
        std::get<int>(params.at("n_jobs")));
}


TEST(RegressorParams, unspecified_random_state_is_initialized)
{
    auto const rv = Tsetlini::make_regressor_params_from_json(R"({})");

    EXPECT_TRUE(rv);

    auto params = rv.right().value;

    auto random_state = params.at("random_state");

    EXPECT_TRUE(std::holds_alternative<Tsetlini::seed_type>(random_state));
}


}
