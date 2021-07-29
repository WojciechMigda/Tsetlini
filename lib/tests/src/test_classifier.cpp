#include "tsetlini.hpp"
#include "tsetlini_types.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace
{


TEST(TsetlinClassifierClassic, can_be_created)
{
    auto const clf = Tsetlini::make_classifier_classic();

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifierClassic, cannot_be_created_from_empty_json)
{
    auto const clf = Tsetlini::make_classifier_classic("");

    EXPECT_FALSE(clf);
}


TEST(TsetlinClassifierClassic, cannot_be_created_from_bad_json)
{
    auto const clf = Tsetlini::make_classifier_classic("[]");

    EXPECT_FALSE(clf);
}


TEST(TsetlinClassifierClassic, can_be_created_from_empty_json_dict)
{
    auto const clf = Tsetlini::make_classifier_classic("{}");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifierClassic, cannot_be_created_from_json_with_unrecognized_param)
{
    auto const clf = Tsetlini::make_classifier_classic(R"({"gotcha": 564})");

    EXPECT_FALSE(clf);
}


TEST(TsetlinClassifierClassic, can_be_created_from_json_with_counting_type_int8)
{
    auto const clf = Tsetlini::make_classifier_classic(R"({"counting_type": "int8"})");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifierClassic, can_be_created_from_json_with_counting_type_int16)
{
    auto const clf = Tsetlini::make_classifier_classic(R"({"counting_type": "int16"})");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifierClassic, can_be_created_from_json_with_counting_type_int32)
{
    auto const clf = Tsetlini::make_classifier_classic(R"({"counting_type": "int32"})");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifierClassic, can_be_created_from_json_with_counting_type_auto)
{
    auto const clf = Tsetlini::make_classifier_classic(R"({"counting_type": "auto"})");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifierClassic, can_be_created_from_json_with_clause_output_tile_size_16)
{
    auto const clf = Tsetlini::make_classifier_classic(R"({"clause_output_tile_size": 16})");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifierClassic, can_be_created_from_json_with_clause_output_tile_size_32)
{
    auto const clf = Tsetlini::make_classifier_classic(R"({"clause_output_tile_size": 32})");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifierClassic, can_be_created_from_json_with_clause_output_tile_size_64)
{
    auto const clf = Tsetlini::make_classifier_classic(R"({"clause_output_tile_size": 64})");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifierClassic, can_be_created_from_json_with_clause_output_tile_size_128)
{
    auto const clf = Tsetlini::make_classifier_classic(R"({"clause_output_tile_size": 128})");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifierClassic, cannot_be_created_from_json_with_bad_clause_output_tile_size)
{
    auto const clf = Tsetlini::make_classifier_classic(R"({"clause_output_tile_size": 24})");

    EXPECT_FALSE(clf);
}


///     Fit

TEST(TsetlinClassifierClassicFit, rejects_empty_X)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X;
            Tsetlini::label_vector_type y{1, 0, 1, 0};

            auto const rv = clf.fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicFit, rejects_empty_y)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y;

            auto const rv = clf.fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicFit, rejects_X_with_uneven_row_sizes)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y{1, 0, 0};

            auto const rv = clf.fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicFit, rejects_X_with_values_not_0_1)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, -1}, {0, 2, 0}};
            Tsetlini::label_vector_type y{1, 0, 0};

            auto const rv = clf.fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicFit, rejects_X_and_y_with_different_lengths)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y{1, 0, 0, 1};

            auto const rv = clf.fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicFit, rejects_y_with_negative_label)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y{1, 0, -21};

            auto const rv = clf.fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


///     Partial Fit

TEST(TsetlinClassifierClassicPartialFit, rejects_empty_X)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X;
            Tsetlini::label_vector_type y{1, 0, 1, 0};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicPartialFit, rejects_empty_y)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y;

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicPartialFit, rejects_X_with_uneven_row_sizes)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicPartialFit, rejects_X_with_values_not_0_1)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, -1}, {0, 2, 0}};
            Tsetlini::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicPartialFit, rejects_X_and_y_with_different_lengths)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y{1, 0, 0, 1};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicPartialFit, rejects_y_with_negative_label)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y{1, 0, -21};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


///     Next Partial Fit

TEST(TsetlinClassifierClassicNextPartialFit, rejects_empty_X)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};
            auto const rv0 = clf.partial_fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X;
            Tsetlini::label_vector_type y{1, 0, 1, 0};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicNextPartialFit, rejects_empty_y)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};
            auto const rv0 = clf.partial_fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y;

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicNextPartialFit, rejects_X_with_uneven_row_sizes)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};
            auto const rv0 = clf.partial_fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicNextPartialFit, rejects_X_with_values_not_0_1)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};
            auto const rv0 = clf.partial_fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, -1}, {0, 2, 0}};
            Tsetlini::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicNextPartialFit, rejects_X_and_y_with_different_lengths)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};
            auto const rv0 = clf.partial_fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y{1, 0, 0, 1};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicNextPartialFit, rejects_y_with_negative_label)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};
            auto const rv0 = clf.partial_fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y{1, 0, -21};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicNextPartialFit, rejects_y_with_label_outside_range)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};
            auto const rv0 = clf.partial_fit(X0, y0, 3);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y{1, 0, 4};

            auto const rv = clf.partial_fit(X, y, 4);

            EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, rv.first);
            return std::move(clf);
        });
}


///     Predict matrix

TEST(TsetlinClassifierClassicPredictMatrix, fails_without_train)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};

            auto const rv = clf.predict(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_NOT_FITTED_ERROR, sm.first); return std::move(sm); });

            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicPredictMatrix, fails_for_empty_X)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X;

            auto const rv = clf.predict(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicPredictMatrix, rejects_X_with_uneven_row_sizes)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0}, {0, 0, 0}};

            auto const rv = clf.predict(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicPredictMatrix, rejects_X_with_invalid_number_of_features)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}};

            auto const rv = clf.predict(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicPredictMatrix, rejects_X_with_values_not_0_1)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, -1}, {0, 0, 2}};

            auto const rv = clf.predict(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(clf);
        });
}


///     Predict sample

TEST(TsetlinClassifierClassicPredictSample, fails_without_train)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            Tsetlini::aligned_vector_char sample{1, 0, 1};

            auto const rv = clf.predict(sample);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_NOT_FITTED_ERROR, sm.first); return std::move(sm); });

            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicPredictSample, rejects_sample_with_invalid_number_of_features)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            Tsetlini::aligned_vector_char sample{1, 0, 1, 0};

            auto const rv = clf.predict(sample);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicPredictSample, rejects_sample_with_values_not_0_1)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            Tsetlini::aligned_vector_char sample{1, -1, 2};

            auto const rv = clf.predict(sample);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(clf);
        });
}


///     PredictRaw matrix

TEST(TsetlinClassifierClassicPredictRawMatrix, fails_without_train)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};

            auto const rv = clf.predict_raw(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_NOT_FITTED_ERROR, sm.first); return std::move(sm); });

            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicPredictRawMatrix, fails_for_empty_X)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X;

            auto const rv = clf.predict_raw(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicPredictRawMatrix, rejects_X_with_uneven_row_sizes)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0}, {0, 0, 0}};

            auto const rv = clf.predict_raw(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicPredictRawMatrix, rejects_X_with_invalid_number_of_features)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}};

            auto const rv = clf.predict_raw(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicPredictRawMatrix, rejects_X_with_values_not_0_1)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            std::vector<Tsetlini::aligned_vector_char> X{{1, 0, 1}, {1, 0, -1}, {0, 0, 2}};

            auto const rv = clf.predict_raw(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(clf);
        });
}


///     Predict sample

TEST(TsetlinClassifierClassicPredictRawSample, fails_without_train)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            Tsetlini::aligned_vector_char sample{1, 0, 1};

            auto const rv = clf.predict_raw(sample);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_NOT_FITTED_ERROR, sm.first); return std::move(sm); });

            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicPredictRawSample, rejects_sample_with_invalid_number_of_features)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            Tsetlini::aligned_vector_char sample{1, 0, 1, 0};

            auto const rv = clf.predict_raw(sample);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(clf);
        });
}


TEST(TsetlinClassifierClassicPredictRawSample, rejects_sample_with_values_not_0_1)
{
    Tsetlini::make_classifier_classic("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlini::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlini::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            Tsetlini::aligned_vector_char sample{1, -1, 2};

            auto const rv = clf.predict_raw(sample);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlini::StatusCode::S_VALUE_ERROR, sm.first); return std::move(sm); });

            return std::move(clf);
        });
}


} // anonymous namespace
