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


TEST(TsetlinRegressorClassic, can_be_created_from_json_with_weighted_param)
{
    auto const rgr = Tsetlini::make_regressor_classic(R"({"weighted": false})");

    EXPECT_TRUE(rgr);
}


///     Fit

TEST(TsetlinRegressorClassicFit, rejects_empty_X)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X;
            Tsetlini::response_vector_type y{1, 0, 1, 0};

            auto const rv = reg.fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicFit, rejects_empty_y)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y;

            auto const rv = reg.fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicFit, rejects_X_with_uneven_row_sizes)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y{1, 0, 0};

            auto const rv = reg.fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicFit, rejects_X_with_values_not_0_1)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, -1}, {0, 2, 0}};
            Tsetlini::response_vector_type y{1, 0, 0};

            auto const rv = reg.fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicFit, rejects_X_and_y_with_different_lengths)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y{1, 0, 0, 1};

            auto const rv = reg.fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicFit, rejects_y_with_negative_response)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 1}, {0, 1, 0}};
            Tsetlini::response_vector_type y{1, 0, -1};

            auto const rv = reg.fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicFit, rejects_y_with_response_over_threshold)
{
    Tsetlini::make_regressor_classic("{'threshold': 15}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 1}, {0, 1, 0}};
            Tsetlini::response_vector_type y{1, 15 + 1, 1};

            auto const rv = reg.fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicFit, accepts_y_within_valid_range)
{
    Tsetlini::make_regressor_classic("{'threshold': 15}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 1}, {0, 1, 0}};
            Tsetlini::response_vector_type y{0, 1, 15};

            auto const rv = reg.fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_OK, rv.first);
            return reg;
        });
}


///     Partial Fit

TEST(TsetlinRegressorClassicPartialFit, rejects_empty_X)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X;
            Tsetlini::response_vector_type y{1, 0, 1, 0};

            auto const rv = reg.partial_fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicPartialFit, rejects_empty_y)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y;

            auto const rv = reg.partial_fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicPartialFit, rejects_X_with_uneven_row_sizes)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y{1, 0, 0};

            auto const rv = reg.partial_fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicPartialFit, rejects_X_with_values_not_0_1)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, -1}, {0, 2, 0}};
            Tsetlini::response_vector_type y{1, 0, 0};

            auto const rv = reg.partial_fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicPartialFit, rejects_X_and_y_with_different_lengths)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y{1, 0, 0, 1};

            auto const rv = reg.partial_fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicPartialFit, rejects_y_with_negative_response)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 1}, {0, 1, 0}};
            Tsetlini::response_vector_type y{1, 0, -1};

            auto const rv = reg.partial_fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicPartialFit, rejects_y_with_response_over_threshold)
{
    Tsetlini::make_regressor_classic("{'threshold': 15}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 1}, {0, 1, 0}};
            Tsetlini::response_vector_type y{1, 15 + 1, 1};

            auto const rv = reg.partial_fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicPartialFit, accepts_y_within_valid_range)
{
    Tsetlini::make_regressor_classic("{'threshold': 15}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 1}, {0, 1, 0}};
            Tsetlini::response_vector_type y{0, 1, 15};

            auto const rv = reg.partial_fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_OK, rv.first);
            return reg;
        });
}


///     Next Partial Fit

TEST(TsetlinRegressorClassicNextPartialFit, rejects_empty_X)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y0{1, 0, 1};
            auto const rv0 = reg.partial_fit(X0, y0);

            std::vector<Tsetlini::aligned_vector_char> X;
            Tsetlini::response_vector_type y{1, 0, 1, 0};

            auto const rv = reg.partial_fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicNextPartialFit, rejects_empty_y)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y0{1, 0, 1};
            auto const rv0 = reg.partial_fit(X0, y0);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y;

            auto const rv = reg.partial_fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicNextPartialFit, rejects_X_with_uneven_row_sizes)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y0{1, 0, 1};
            auto const rv0 = reg.partial_fit(X0, y0);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y{1, 0, 0};

            auto const rv = reg.partial_fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicNextPartialFit, rejects_X_with_values_not_0_1)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y0{1, 0, 1};
            auto const rv0 = reg.partial_fit(X0, y0);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, -1}, {0, 2, 0}};
            Tsetlini::response_vector_type y{1, 0, 0};

            auto const rv = reg.partial_fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicNextPartialFit, rejects_X_and_y_with_different_lengths)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y0{1, 0, 1};
            auto const rv0 = reg.partial_fit(X0, y0);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y{1, 0, 0, 1};

            auto const rv = reg.partial_fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicNextPartialFit, rejects_y_with_negative_response)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y0{1, 0, 1};
            auto const rv0 = reg.partial_fit(X0, y0);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 1, 1}, {0, 1, 0}};
            Tsetlini::response_vector_type y{1, 0, -1};

            auto const rv = reg.partial_fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicNextPartialFit, rejects_y_with_response_over_threshold)
{
    Tsetlini::make_regressor_classic("{'threshold': 15}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y0{1, 0, 1};
            auto const rv0 = reg.partial_fit(X0, y0);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 1, 1}, {0, 1, 0}};
            Tsetlini::response_vector_type y{15 + 1, 1, 0};

            auto const rv = reg.partial_fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return reg;
        });
}


TEST(TsetlinRegressorClassicNextPartialFit, accepts_y_within_valid_range)
{
    Tsetlini::make_regressor_classic("{'threshold': 15}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y0{1, 0, 1};
            auto const rv0 = reg.partial_fit(X0, y0);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 1, 1}, {0, 1, 0}};
            Tsetlini::response_vector_type y{15, 1, 0};

            auto const rv = reg.partial_fit(X, y);

            EXPECT_EQ(Tsetlini::StatusCode::S_OK, rv.first);
            return reg;
        });
}


///     Predict matrix

TEST(TsetlinRegressorClassicPredictMatrix, fails_without_train)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};

            auto const rv = reg.predict(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_NOT_FITTED_ERROR, sm.first); return sm; });

            return reg;
        });
}


TEST(TsetlinRegressorClassicPredictMatrix, fails_for_empty_X)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y0{1, 0, 1};

            auto const rv0 = reg.fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X;

            auto const rv = reg.predict(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return reg;

        });
}


TEST(TsetlinRegressorClassicPredictMatrix, rejects_X_with_uneven_row_sizes)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y0{1, 0, 1};

            auto const rv0 = reg.fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0}, {0, 0, 0}};

            auto const rv = reg.predict(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return reg;

        });
}


TEST(TsetlinRegressorClassicPredictMatrix, rejects_X_with_invalid_number_of_features)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y0{1, 0, 1};

            auto const rv0 = reg.fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}};

            auto const rv = reg.predict(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return reg;

        });
}


TEST(TsetlinRegressorClassicPredictMatrix, rejects_X_with_values_not_0_1)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y0{1, 0, 1};

            auto const rv0 = reg.fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, -1}, {0, 0, 2}};

            auto const rv = reg.predict(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return reg;

        });
}


///     Predict sample

TEST(TsetlinRegressorClassicPredictSample, fails_without_train)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            Tsetlini::aligned_vector_char sample{1, 0, 1};

            auto const rv = reg.predict(sample);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_NOT_FITTED_ERROR, sm.first); return sm; });

            return reg;
        });
}


TEST(TsetlinRegressorClassicPredictSample, rejects_sample_with_invalid_number_of_features)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y0{1, 0, 1};

            auto const rv0 = reg.fit(X0, y0);

            Tsetlini::aligned_vector_char sample{1, 0, 1, 0};

            auto const rv = reg.predict(sample);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return reg;
        });
}


TEST(TsetlinRegressorClassicPredictSample, rejects_sample_with_values_not_0_1)
{
    Tsetlini::make_regressor_classic("{}")
        .rightMap(
        [](auto && reg)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::response_vector_type y0{1, 0, 1};

            auto const rv0 = reg.fit(X0, y0);

            Tsetlini::aligned_vector_char sample{1, -1, 2};

            auto const rv = reg.predict(sample);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return reg;
        });
}


} // anonymous namespace
