#include "tsetlin.hpp"
#include "tsetlin_types.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace
{


TEST(TsetlinClassifier, can_be_created)
{
    auto const clf = Tsetlin::make_classifier();

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifier, cannot_be_created_from_empty_json)
{
    auto const clf = Tsetlin::make_classifier("");

    EXPECT_FALSE(clf);
}


TEST(TsetlinClassifier, cannot_be_created_from_bad_json)
{
    auto const clf = Tsetlin::make_classifier("[]");

    EXPECT_FALSE(clf);
}


TEST(TsetlinClassifier, can_be_created_from_empty_json_dict)
{
    auto const clf = Tsetlin::make_classifier("{}");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifier, cannot_be_created_from_json_with_unrecognized_param)
{
    auto const clf = Tsetlin::make_classifier(R"({"gotcha": 564})");

    EXPECT_FALSE(clf);
}


TEST(TsetlinClassifier, can_be_created_from_json_with_counting_type_int8)
{
    auto const clf = Tsetlin::make_classifier(R"({"counting_type": "int8"})");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifier, can_be_created_from_json_with_counting_type_int16)
{
    auto const clf = Tsetlin::make_classifier(R"({"counting_type": "int16"})");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifier, can_be_created_from_json_with_counting_type_int32)
{
    auto const clf = Tsetlin::make_classifier(R"({"counting_type": "int32"})");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifier, can_be_created_from_json_with_counting_type_auto)
{
    auto const clf = Tsetlin::make_classifier(R"({"counting_type": "auto"})");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifier, can_be_created_from_json_with_clause_output_tile_size_16)
{
    auto const clf = Tsetlin::make_classifier(R"({"clause_output_tile_size": 16})");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifier, can_be_created_from_json_with_clause_output_tile_size_32)
{
    auto const clf = Tsetlin::make_classifier(R"({"clause_output_tile_size": 32})");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifier, can_be_created_from_json_with_clause_output_tile_size_64)
{
    auto const clf = Tsetlin::make_classifier(R"({"clause_output_tile_size": 64})");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifier, can_be_created_from_json_with_clause_output_tile_size_128)
{
    auto const clf = Tsetlin::make_classifier(R"({"clause_output_tile_size": 128})");

    EXPECT_TRUE(clf);
}


TEST(TsetlinClassifier, cannot_be_created_from_json_with_bad_clause_output_tile_size)
{
    auto const clf = Tsetlin::make_classifier(R"({"clause_output_tile_size": 24})");

    EXPECT_FALSE(clf);
}


///     Partial Fit

TEST(TsetlinClassifierFit, rejects_empty_X)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X;
            Tsetlin::label_vector_type y{1, 0, 1, 0};

            auto const rv = clf.fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


TEST(TsetlinClassifierFit, rejects_empty_y)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y;

            auto const rv = clf.fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


TEST(TsetlinClassifierFit, rejects_X_with_uneven_row_sizes)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y{1, 0, 0};

            auto const rv = clf.fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


TEST(TsetlinClassifierFit, rejects_X_with_values_not_0_1)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0, -1}, {0, 2, 0}};
            Tsetlin::label_vector_type y{1, 0, 0};

            auto const rv = clf.fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


TEST(TsetlinClassifierFit, rejects_X_and_y_with_different_lengths)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y{1, 0, 0, 1};

            auto const rv = clf.fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


TEST(TsetlinClassifierFit, rejects_y_with_negative_label)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y{1, 0, -21};

            auto const rv = clf.fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


///     Partial Fit

TEST(TsetlinClassifierPartialFit, rejects_empty_X)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X;
            Tsetlin::label_vector_type y{1, 0, 1, 0};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


TEST(TsetlinClassifierPartialFit, rejects_empty_y)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y;

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


TEST(TsetlinClassifierPartialFit, rejects_X_with_uneven_row_sizes)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


TEST(TsetlinClassifierPartialFit, rejects_X_with_values_not_0_1)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0, -1}, {0, 2, 0}};
            Tsetlin::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


TEST(TsetlinClassifierPartialFit, rejects_X_and_y_with_different_lengths)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y{1, 0, 0, 1};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


TEST(TsetlinClassifierPartialFit, rejects_y_with_negative_label)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y{1, 0, -21};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


///     Next Partial Fit

TEST(TsetlinClassifierNextPartialFit, rejects_empty_X)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};
            auto const rv0 = clf.partial_fit(X0, y0, 2);

            std::vector<Tsetlin::aligned_vector_char> X;
            Tsetlin::label_vector_type y{1, 0, 1, 0};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


TEST(TsetlinClassifierNextPartialFit, rejects_empty_y)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};
            auto const rv0 = clf.partial_fit(X0, y0, 2);

            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y;

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


TEST(TsetlinClassifierNextPartialFit, rejects_X_with_uneven_row_sizes)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};
            auto const rv0 = clf.partial_fit(X0, y0, 2);

            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


TEST(TsetlinClassifierNextPartialFit, rejects_X_with_values_not_0_1)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};
            auto const rv0 = clf.partial_fit(X0, y0, 2);

            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0, -1}, {0, 2, 0}};
            Tsetlin::label_vector_type y{1, 0, 0};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


TEST(TsetlinClassifierNextPartialFit, rejects_X_and_y_with_different_lengths)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};
            auto const rv0 = clf.partial_fit(X0, y0, 2);

            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y{1, 0, 0, 1};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


TEST(TsetlinClassifierNextPartialFit, rejects_y_with_negative_label)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};
            auto const rv0 = clf.partial_fit(X0, y0, 2);

            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y{1, 0, -21};

            auto const rv = clf.partial_fit(X, y, 2);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


TEST(TsetlinClassifierNextPartialFit, rejects_y_with_label_outside_range)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};
            auto const rv0 = clf.partial_fit(X0, y0, 3);

            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y{1, 0, 4};

            auto const rv = clf.partial_fit(X, y, 4);

            EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, rv.first);
            return clf;
        });
}


///     Predict matrix

TEST(TsetlinClassifierPredictMatrix, fails_without_train)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};

            auto const rv = clf.predict(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlin::StatusCode::S_NOT_FITTED_ERROR, sm.first); return sm; });

            return clf;
        });
}


TEST(TsetlinClassifierPredictMatrix, fails_for_empty_X)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            std::vector<Tsetlin::aligned_vector_char> X;

            auto const rv = clf.predict(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return clf;
        });
}


TEST(TsetlinClassifierPredictMatrix, rejects_X_with_uneven_row_sizes)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0}, {0, 0, 0}};

            auto const rv = clf.predict(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return clf;
        });
}


TEST(TsetlinClassifierPredictMatrix, rejects_X_with_invalid_number_of_features)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}};

            auto const rv = clf.predict(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return clf;
        });
}


TEST(TsetlinClassifierPredictMatrix, rejects_X_with_values_not_0_1)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0, -1}, {0, 0, 2}};

            auto const rv = clf.predict(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return clf;
        });
}


///     Predict sample

TEST(TsetlinClassifierPredictSample, fails_without_train)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            Tsetlin::aligned_vector_char sample{1, 0, 1};

            auto const rv = clf.predict(sample);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlin::StatusCode::S_NOT_FITTED_ERROR, sm.first); return sm; });

            return clf;
        });
}


TEST(TsetlinClassifierPredictSample, rejects_sample_with_invalid_number_of_features)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            Tsetlin::aligned_vector_char sample{1, 0, 1, 0};

            auto const rv = clf.predict(sample);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return clf;
        });
}


TEST(TsetlinClassifierPredictSample, rejects_sample_with_values_not_0_1)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            Tsetlin::aligned_vector_char sample{1, -1, 2};

            auto const rv = clf.predict(sample);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return clf;
        });
}


///     PredictRaw matrix

TEST(TsetlinClassifierPredictRawMatrix, fails_without_train)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};

            auto const rv = clf.predict_raw(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlin::StatusCode::S_NOT_FITTED_ERROR, sm.first); return sm; });

            return clf;
        });
}


TEST(TsetlinClassifierPredictRawMatrix, fails_for_empty_X)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            std::vector<Tsetlin::aligned_vector_char> X;

            auto const rv = clf.predict_raw(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return clf;
        });
}


TEST(TsetlinClassifierPredictRawMatrix, rejects_X_with_uneven_row_sizes)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0}, {0, 0, 0}};

            auto const rv = clf.predict_raw(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return clf;
        });
}


TEST(TsetlinClassifierPredictRawMatrix, rejects_X_with_invalid_number_of_features)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}};

            auto const rv = clf.predict_raw(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return clf;
        });
}


TEST(TsetlinClassifierPredictRawMatrix, rejects_X_with_values_not_0_1)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            std::vector<Tsetlin::aligned_vector_char> X{{1, 0, 1}, {1, 0, -1}, {0, 0, 2}};

            auto const rv = clf.predict_raw(X);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return clf;
        });
}


///     Predict sample

TEST(TsetlinClassifierPredictRawSample, fails_without_train)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            Tsetlin::aligned_vector_char sample{1, 0, 1};

            auto const rv = clf.predict_raw(sample);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlin::StatusCode::S_NOT_FITTED_ERROR, sm.first); return sm; });

            return clf;
        });
}


TEST(TsetlinClassifierPredictRawSample, rejects_sample_with_invalid_number_of_features)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            Tsetlin::aligned_vector_char sample{1, 0, 1, 0};

            auto const rv = clf.predict_raw(sample);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return clf;
        });
}


TEST(TsetlinClassifierPredictRawSample, rejects_sample_with_values_not_0_1)
{
    Tsetlin::make_classifier("{}")
        .rightMap(
        [](auto && clf)
        {
            std::vector<Tsetlin::aligned_vector_char> X0{{1, 0, 1}, {1, 0, 0}, {0, 0, 0}};
            Tsetlin::label_vector_type y0{1, 0, 1};

            auto const rv0 = clf.fit(X0, y0, 2);

            Tsetlin::aligned_vector_char sample{1, -1, 2};

            auto const rv = clf.predict_raw(sample);

            EXPECT_FALSE(rv);
            rv.leftMap([](auto && sm){ EXPECT_EQ(Tsetlin::StatusCode::S_VALUE_ERROR, sm.first); return sm; });

            return clf;
        });
}


}
