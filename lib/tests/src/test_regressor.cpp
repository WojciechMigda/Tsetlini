#include "tsetlini.hpp"
#include "tsetlini_types.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace
{


TEST(TsetlinRegressorClassic, can_be_created)
{
    auto const rgr = Tsetlini::make_regressor_classic();

    EXPECT_TRUE(rgr);
}


TEST(TsetlinRegressorClassic, cannot_be_created_from_empty_json)
{
    auto const rgr = Tsetlini::make_regressor_classic("");

    EXPECT_FALSE(rgr);
}


TEST(TsetlinRegressorClassic, cannot_be_created_from_bad_json)
{
    auto const rgr = Tsetlini::make_regressor_classic("[]");

    EXPECT_FALSE(rgr);
}


TEST(TsetlinRegressorClassic, can_be_created_from_empty_json_dict)
{
    auto const rgr = Tsetlini::make_regressor_classic("{}");

    EXPECT_TRUE(rgr);
}


TEST(TsetlinRegressorClassic, cannot_be_created_from_json_with_unrecognized_param)
{
    auto const rgr = Tsetlini::make_regressor_classic(R"({"gotcha": 564})");

    EXPECT_FALSE(rgr);
}


TEST(TsetlinRegressorClassic, can_be_created_from_json_with_counting_type_int8)
{
    auto const rgr = Tsetlini::make_regressor_classic(R"({"counting_type": "int8"})");

    EXPECT_TRUE(rgr);
}


TEST(TsetlinRegressorClassic, can_be_created_from_json_with_counting_type_int16)
{
    auto const rgr = Tsetlini::make_regressor_classic(R"({"counting_type": "int16"})");

    EXPECT_TRUE(rgr);
}


TEST(TsetlinRegressorClassic, can_be_created_from_json_with_counting_type_int32)
{
    auto const rgr = Tsetlini::make_regressor_classic(R"({"counting_type": "int32"})");

    EXPECT_TRUE(rgr);
}


TEST(TsetlinRegressorClassic, can_be_created_from_json_with_counting_type_auto)
{
    auto const rgr = Tsetlini::make_regressor_classic(R"({"counting_type": "auto"})");

    EXPECT_TRUE(rgr);
}


TEST(TsetlinRegressorClassic, can_be_created_from_json_with_clause_output_tile_size_16)
{
    auto const rgr = Tsetlini::make_regressor_classic(R"({"clause_output_tile_size": 16})");

    EXPECT_TRUE(rgr);
}


TEST(TsetlinRegressorClassic, can_be_created_from_json_with_clause_output_tile_size_32)
{
    auto const rgr = Tsetlini::make_regressor_classic(R"({"clause_output_tile_size": 32})");

    EXPECT_TRUE(rgr);
}


TEST(TsetlinRegressorClassic, can_be_created_from_json_with_clause_output_tile_size_64)
{
    auto const rgr = Tsetlini::make_regressor_classic(R"({"clause_output_tile_size": 64})");

    EXPECT_TRUE(rgr);
}


TEST(TsetlinRegressorClassic, can_be_created_from_json_with_clause_output_tile_size_128)
{
    auto const rgr = Tsetlini::make_regressor_classic(R"({"clause_output_tile_size": 128})");

    EXPECT_TRUE(rgr);
}


TEST(TsetlinRegressorClassic, cannot_be_created_from_json_with_bad_clause_output_tile_size)
{
    auto const rgr = Tsetlini::make_regressor_classic(R"({"clause_output_tile_size": 24})");

    EXPECT_FALSE(rgr);
}


} // anonymous namespace
