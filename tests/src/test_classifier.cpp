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


}
