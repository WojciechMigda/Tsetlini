#include "tsetlini.hpp"
#include "tsetlini_types.hpp"

#include "gtest/gtest.h"

#include <vector>
#include <memory>


namespace
{


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
