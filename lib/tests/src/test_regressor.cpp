#include "tsetlini.hpp"
#include "tsetlini_types.hpp"

#include "gtest/gtest.h"

#include <memory>
#include <vector>


namespace
{


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
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_NOT_FITTED_ERROR, sm.first); return std::move(sm); });

            return std::move(reg);
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
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(reg);

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
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(reg);

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
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(reg);

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
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(reg);

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
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_NOT_FITTED_ERROR, sm.first); return std::move(sm); });

            return std::move(reg);
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
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(reg);
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
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(reg);
        });
}


} // anonymous namespace
